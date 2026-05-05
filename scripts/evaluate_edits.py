"""Evaluate a memory editor on a set of prescribed edits.

Computes the four metrics reported in the paper (Section 4.1):
  * ESR — Edit Success Rate
  * MP  — Memory Preservation
  * FR  — Forgetting Ratio (for DELETE)
  * Latency per operation

Input JSON format:

    {
      "hidden_dim": 64,
      "vocab_size": 256,
      "memories": [                    # the pool of memories used throughout
        {"trace_id": "m0", "probe": [...], "target_peak": 7},
        ...
      ],
      "edits": [                       # prescribed edit sequence
        {"op": "INSERT", "new": "m0"},
        {"op": "MODIFY", "old": "m0", "new_target_peak": 42},
        {"op": "DELETE", "target": "m0"},
        ...
      ]
    }

If probes are omitted, they're sampled from a fixed RNG so the numbers
are reproducible.

Run:
    python scripts/evaluate_edits.py --config configs/tiny.yaml \
        --edits data/sample_edits.json --output-json eval_out.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from memedit import MemEditConfig, MemEditor, MemoryTrace
from memedit.data.trace import OperationType
from memedit.models.mlp_memory import MLPMemory
from memedit.utils.config_loader import load_config
from memedit.utils.logging_utils import get_logger

_log = get_logger("evaluate")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--edits", type=str, required=True)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--preservation-tol", type=float, default=0.01,
                   help="KL threshold for MP (paper: ε=0.01).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _mk_trace(tid: str, probe: torch.Tensor, peak: int, V: int, content: str = "") -> MemoryTrace:
    target = torch.full((V,), 1e-4)
    target[peak % V] = 1.0
    target = target / target.sum()
    return MemoryTrace(
        trace_id=tid, content=content or tid,
        probe_hidden=probe, target_distribution=target,
    )


def _build_editor(cfg: MemEditConfig, checkpoint: str | None) -> MemEditor:
    base = MLPMemory(cfg.memory)
    if checkpoint:
        base.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])
    editor = MemEditor(base, cfg)
    editor.set_baseline_from_samples(torch.randn(100, cfg.memory.hidden_dim))
    return editor


def _measure_preservation(
    editor: MemEditor,
    preserved: dict[str, MemoryTrace],
    snapshots: dict[str, torch.Tensor],
    tol: float,
) -> tuple[int, int]:
    """Return (num_preserved, total_checked)."""
    ok = 0
    tot = 0
    for tid, tr in preserved.items():
        if tid not in snapshots:
            continue
        tot += 1
        with torch.no_grad():
            probs = editor.predict(tr.probe_hidden)
            logp_now = probs.clamp_min(1e-12).log()
        kl = (snapshots[tid].exp() * (snapshots[tid] - logp_now)).sum().item()
        if abs(kl) < tol:
            ok += 1
    return ok, tot


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    editor = _build_editor(cfg, args.checkpoint)

    spec = json.loads(Path(args.edits).read_text())
    d = int(spec.get("hidden_dim", cfg.memory.hidden_dim))
    V = int(spec.get("vocab_size", cfg.memory.vocab_size))
    if d != cfg.memory.hidden_dim or V != cfg.memory.vocab_size:
        _log.warning("spec dims (%d, %d) differ from config (%d, %d) — using config",
                     d, V, cfg.memory.hidden_dim, cfg.memory.vocab_size)
        d, V = cfg.memory.hidden_dim, cfg.memory.vocab_size

    # Build initial memory pool.
    pool: dict[str, MemoryTrace] = {}
    for rec in spec.get("memories", []):
        tid = rec["trace_id"]
        probe_raw = rec.get("probe")
        if probe_raw is None:
            g = torch.Generator().manual_seed(hash((args.seed, tid)) & 0xFFFFFFFF)
            probe = torch.randn(d, generator=g)
        else:
            probe = torch.tensor(probe_raw, dtype=torch.float32)
        peak = int(rec.get("target_peak", abs(hash(tid)) % V))
        pool[tid] = _mk_trace(tid, probe, peak, V, content=rec.get("content", ""))

    active: dict[str, MemoryTrace] = {}
    snapshots: dict[str, torch.Tensor] = {}

    # Counters.
    succ = 0
    total_non_query = 0
    latencies_ms: dict[str, list[float]] = {"INSERT": [], "MODIFY": [], "DELETE": [], "QUERY": []}
    mp_running_num = 0
    mp_running_den = 0
    fr_num = 0
    fr_den = 0

    per_edit_records: list[dict] = []

    for i, edit in enumerate(spec.get("edits", [])):
        op = str(edit["op"]).upper()
        t0 = time.perf_counter()

        if op == "INSERT":
            tid = edit["new"]
            trace = pool.get(tid)
            if trace is None:
                _log.warning("edit %d: unknown memory id %s", i, tid)
                continue
            r = editor.insert(trace)
            if r.success:
                active[tid] = trace
                with torch.no_grad():
                    snapshots[tid] = editor.predict(trace.probe_hidden).clamp_min(1e-12).log()
        elif op == "MODIFY":
            tid = edit["old"]
            if tid not in active:
                _log.warning("edit %d: MODIFY target %s not active", i, tid)
                continue
            old = active[tid]
            new_peak = int(edit.get("new_target_peak", abs(hash(f"new-{tid}-{i}")) % V))
            new = _mk_trace(tid, old.probe_hidden, new_peak, V)
            r = editor.modify(old, new, preserved=None)
            if r.success:
                active[tid] = new
                with torch.no_grad():
                    snapshots[tid] = editor.predict(new.probe_hidden).clamp_min(1e-12).log()
        elif op == "DELETE":
            tid = edit["target"]
            if tid not in active:
                _log.warning("edit %d: DELETE target %s not active", i, tid)
                continue
            trace = active[tid]
            r = editor.delete(trace)
            fr_den += 1
            if r.success:
                fr_num += 1
                del active[tid]
                snapshots.pop(tid, None)
        elif op == "QUERY":
            tid = edit["target"]
            if tid not in active:
                _log.warning("edit %d: QUERY target %s not active", i, tid)
                continue
            r = editor.query(active[tid])
        else:
            _log.warning("edit %d: unknown op %s", i, op)
            continue

        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms[op].append(dt_ms)

        if op != "QUERY":
            total_non_query += 1
            if r.success:
                succ += 1

        # MP check using preserved set after THIS edit.
        preserved_here = {k: v for k, v in active.items()
                          if k != edit.get("new") and k != edit.get("old") and k != edit.get("target")}
        ok, denom = _measure_preservation(
            editor, preserved_here, snapshots, args.preservation_tol,
        )
        mp_running_num += ok
        mp_running_den += denom

        per_edit_records.append({
            "i": i,
            "op": op,
            "trace_id": getattr(r, "trace_id", None),
            "success": bool(r.success),
            "shard_idx": r.shard_idx,
            "kl_before": r.kl_before,
            "kl_after": r.kl_after,
            "latency_ms": dt_ms,
            "mp_ok": ok, "mp_checked": denom,
        })

    esr = succ / max(total_non_query, 1)
    mp = mp_running_num / max(mp_running_den, 1)
    fr = fr_num / max(fr_den, 1) if fr_den > 0 else float("nan")

    mean_lat = {
        k: (sum(v) / len(v) if v else None) for k, v in latencies_ms.items()
    }

    report = {
        "config": args.config,
        "num_edits": len(per_edit_records),
        "ESR": esr,
        "MP": mp,
        "FR": fr,
        "mean_latency_ms": mean_lat,
        "final_stats": editor.stats(),
        "records": per_edit_records,
    }
    _log.info("ESR=%.3f MP=%.3f FR=%.3f shards=%d",
              esr, mp, fr, editor.stats()["num_shards"])

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2))
        _log.info("Wrote evaluation report to %s", args.output_json)


if __name__ == "__main__":
    main()
