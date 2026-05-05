"""Stress-edit benchmark — sequential edits with MoME expansion.

Replays the scalability experiment in Table 2 / Figure 2 at a small scale.
For each edit we randomly pick Insert, Modify, or Delete and apply it;
every K steps we log ESR (edit success rate), MP (memory preservation),
and the current MoME shard count.

Run:
    # Small smoke test (~seconds on CPU)
    python scripts/stress_edit.py --config configs/tiny.yaml --num-edits 200

    # Larger run
    python scripts/stress_edit.py --config configs/tiny.yaml --num-edits 2000 \
        --log-every 200 --output-json stress.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from memedit import MemEditConfig, MemEditor, MemoryTrace
from memedit.models.mlp_memory import MLPMemory
from memedit.utils.config_loader import load_config
from memedit.utils.logging_utils import get_logger

_log = get_logger("stress_edit")


@dataclass
class StressSnapshot:
    step: int
    esr: float
    mp: float
    num_shards: int
    null_fractions: list
    insert_count: int
    modify_count: int
    delete_count: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--num-edits", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    # Probabilities of each op type for the random workload.
    p.add_argument("--p-insert", type=float, default=0.6)
    p.add_argument("--p-modify", type=float, default=0.3)
    p.add_argument("--p-delete", type=float, default=0.1)
    # Number of "preserved" memories we sample when measuring MP.
    p.add_argument("--mp-sample-size", type=int, default=20)
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


def _mk_trace(tid: str, d: int, V: int) -> MemoryTrace:
    target = torch.full((V,), 1e-4)
    target[random.randrange(V)] = 1.0
    target = target / target.sum()
    return MemoryTrace(
        trace_id=tid,
        content=f"auto-{tid}",
        probe_hidden=torch.randn(d),
        target_distribution=target,
    )


def _measure_mp(
    editor: MemEditor,
    pre_snapshots: dict[str, torch.Tensor],
    sample_ids: list[str],
    active_memories: dict[str, MemoryTrace],
    tol: float = 0.01,
) -> float:
    """Fraction of preserved memories whose output KL drift < `tol`."""
    if not sample_ids:
        return 1.0
    ok = 0
    for tid in sample_ids:
        if tid not in active_memories:
            continue
        trace = active_memories[tid]
        with torch.no_grad():
            logp_now = F.log_softmax(editor.predict(trace.probe_hidden).clamp_min(1e-12).log(), dim=-1) \
                if False else None     # predict already returns probs
            probs = editor.predict(trace.probe_hidden)
            logp_now = probs.clamp_min(1e-12).log()
        logp_before = pre_snapshots[tid]
        kl = (logp_before.exp() * (logp_before - logp_now)).sum().item()
        if abs(kl) < tol:
            ok += 1
    # Normalize by how many of the sampled were still active.
    denom = sum(1 for tid in sample_ids if tid in active_memories)
    return ok / max(denom, 1)


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    base = MLPMemory(cfg.memory)
    if args.checkpoint:
        base.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])
    editor = MemEditor(base, cfg)
    editor.set_baseline_from_samples(torch.randn(100, cfg.memory.hidden_dim))

    d = cfg.memory.hidden_dim
    V = cfg.memory.vocab_size

    # Active memory pool (for MODIFY/DELETE targets).
    active: dict[str, MemoryTrace] = {}
    # Snapshots of pre-edit output log-probs per active memory (for MP).
    snapshots: dict[str, torch.Tensor] = {}

    # Normalize op probabilities.
    total_p = args.p_insert + args.p_modify + args.p_delete
    p_ins = args.p_insert / total_p
    p_mod = args.p_modify / total_p

    succ = 0
    total = 0
    counts = {"INSERT": 0, "MODIFY": 0, "DELETE": 0}
    history: list[dict] = []

    for step in range(1, args.num_edits + 1):
        r = random.random()
        if not active or r < p_ins:
            op = "INSERT"
        elif r < p_ins + p_mod:
            op = "MODIFY"
        else:
            op = "DELETE"

        if op == "INSERT":
            tid = f"t{step}"
            trace = _mk_trace(tid, d, V)
            res = editor.insert(trace)
            if res.success:
                active[tid] = trace
                # Record baseline distribution for MP measurement.
                with torch.no_grad():
                    probs = editor.predict(trace.probe_hidden)
                    snapshots[tid] = probs.clamp_min(1e-12).log()
        elif op == "MODIFY":
            tid = random.choice(list(active.keys()))
            old = active[tid]
            new = _mk_trace(tid, d, V)
            new.probe_hidden = old.probe_hidden  # same probe, new target
            res = editor.modify(old, new, preserved=None)
            if res.success:
                active[tid] = new
                with torch.no_grad():
                    probs = editor.predict(new.probe_hidden)
                    snapshots[tid] = probs.clamp_min(1e-12).log()
        else:  # DELETE
            tid = random.choice(list(active.keys()))
            trace = active[tid]
            res = editor.delete(trace)
            if res.success:
                del active[tid]
                snapshots.pop(tid, None)

        counts[op] += 1
        total += 1
        if res.success:
            succ += 1

        if step % args.log_every == 0 or step == args.num_edits:
            sample_ids = random.sample(
                list(active.keys()),
                min(args.mp_sample_size, len(active)),
            )
            mp = _measure_mp(editor, snapshots, sample_ids, active)
            esr = succ / max(total, 1)
            stats = editor.stats()
            snap = StressSnapshot(
                step=step,
                esr=esr,
                mp=mp,
                num_shards=stats["num_shards"],
                null_fractions=[float(f) for f in stats["null_fractions"]],
                insert_count=counts["INSERT"],
                modify_count=counts["MODIFY"],
                delete_count=counts["DELETE"],
            )
            _log.info(
                "step=%d ESR=%.3f MP=%.3f shards=%d null=%s",
                step, esr, mp, stats["num_shards"],
                [f"{f:.2f}" for f in stats["null_fractions"]],
            )
            history.append(snap.__dict__)

    if args.output_json:
        payload = {
            "config": args.config,
            "num_edits": args.num_edits,
            "final_counts": counts,
            "final_esr": succ / max(total, 1),
            "history": history,
            "final_stats": editor.stats(),
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        _log.info("Wrote stress results to %s", args.output_json)


if __name__ == "__main__":
    main()
