"""End-to-end CRUD demo with an optional pretrained checkpoint.

Exercises every atomic operation and prints a structured report.
Mirrors the "CRUD cycle" depicted in Figure 1 (center panel) of the paper.

Run:
    # With synthetic weights (no pretraining):
    python scripts/run_crud_demo.py --config configs/tiny.yaml

    # With a pretrained checkpoint produced by scripts/pretrain_memory.py:
    python scripts/run_crud_demo.py --config configs/tiny.yaml --checkpoint memory.ckpt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from memedit import MemEditConfig, MemEditor, MemoryTrace
from memedit.data.trace import EditResult
from memedit.models.mlp_memory import MLPMemory
from memedit.utils.config_loader import load_config
from memedit.utils.logging_utils import get_logger

_log = get_logger("crud_demo")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional pretrained memory checkpoint.")
    p.add_argument("--num-memories", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=str, default=None,
                   help="Write structured results to this JSON file.")
    return p.parse_args()


def _build_editor(cfg: MemEditConfig, checkpoint: str | None) -> MemEditor:
    base = MLPMemory(cfg.memory)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        base.load_state_dict(ckpt["state_dict"])
        _log.info("Loaded checkpoint from %s", checkpoint)
        editor = MemEditor(base, cfg)
        if "corpus_mean_hidden" in ckpt:
            editor._baseline_hidden = ckpt["corpus_mean_hidden"].to(
                device=editor.device, dtype=editor.dtype,
            )
            _log.info("Set attribution baseline from checkpoint's corpus_mean_hidden")
        else:
            editor.set_baseline_from_samples(torch.randn(100, cfg.memory.hidden_dim))
    else:
        editor = MemEditor(base, cfg)
        editor.set_baseline_from_samples(torch.randn(100, cfg.memory.hidden_dim))
        _log.info("Using freshly initialized memory module (no checkpoint).")
    return editor


def _make_memory(tid: str, hidden_dim: int, vocab_size: int, peak: int) -> MemoryTrace:
    target = torch.full((vocab_size,), 1e-4)
    target[peak % vocab_size] = 1.0
    target = target / target.sum()
    return MemoryTrace(
        trace_id=tid,
        content=f"fact {tid} → token {peak}",
        probe_hidden=torch.randn(hidden_dim),
        target_distribution=target,
    )


def _summarize(res: EditResult) -> dict:
    return {
        "op": res.op_type.value,
        "trace_id": res.trace_id,
        "success": bool(res.success),
        "shard_idx": res.shard_idx,
        "kl_before": res.kl_before,
        "kl_after": res.kl_after,
        "preservation_kl": res.preservation_kl,
        "num_iterations": res.num_iterations,
        "message": res.message,
    }


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    editor = _build_editor(cfg, args.checkpoint)

    d = cfg.memory.hidden_dim
    V = cfg.memory.vocab_size
    mems = [_make_memory(f"m{i}", d, V, peak=(i * 7) + 3) for i in range(args.num_memories)]

    results: list[dict] = []

    # 1) INSERT
    _log.info("=== INSERT %d memories ===", len(mems))
    for m in mems:
        r = editor.insert(m)
        _log.info("  insert %s: %s", m.trace_id, r.message)
        results.append(_summarize(r))

    # 2) QUERY each memory
    _log.info("=== QUERY ===")
    for m in mems:
        r = editor.query(m)
        _log.info("  query %s: footprint=%d neurons  confidence=%.3f",
                  m.trace_id, len(r.footprint.neurons), r.footprint.confidence)
        results.append(_summarize(r))

    # 3) MODIFY one memory
    _log.info("=== MODIFY first memory ===")
    mod_target = _make_memory(mems[0].trace_id, d, V, peak=99)
    mod_target.probe_hidden = mems[0].probe_hidden       # same probe
    r = editor.modify(mems[0], mod_target, preserved=mems[1:])
    _log.info("  modify %s: %s", mems[0].trace_id, r.message)
    results.append(_summarize(r))

    # 4) DELETE the last memory
    _log.info("=== DELETE last memory ===")
    r = editor.delete(mems[-1])
    _log.info("  delete %s: %s", mems[-1].trace_id, r.message)
    results.append(_summarize(r))

    # 5) Final stats
    stats = editor.stats()
    _log.info("=== Final MoME stats ===  %s", stats)

    if args.output_json:
        payload = {"results": results, "stats": stats}
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        _log.info("Wrote results to %s", args.output_json)


if __name__ == "__main__":
    main()
