"""Minimal CRUD demo on a tiny CPU-sized memory.

Runs in a few seconds on CPU. No real LLM backbone — we fake hidden
states with random vectors to exercise every code path.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import torch

from memedit import (
    MemEditConfig,
    MemEditor,
    MemoryModuleConfig,
    MLPMemory,
    MemoryTrace,
)
from memedit.core.config import (
    AttributionConfig,
    DeleteConfig,
    InsertConfig,
    ModifyConfig,
    MoMEConfig,
)


def make_tiny_config() -> MemEditConfig:
    return MemEditConfig(
        memory=MemoryModuleConfig(
            num_layers=2, hidden_dim=64, intermediate_dim=128, vocab_size=256,
        ),
        attribution=AttributionConfig(riemann_steps=8, sparsity_tau=0.05),
        insert=InsertConfig(),
        modify=ModifyConfig(num_sgd_steps=10, learning_rate=1e-2),
        delete=DeleteConfig(num_steps=5, initial_lr=5e-3, kl_threshold=1.0),
        mome=MoMEConfig(initial_num_shards=2, expansion_threshold=0.3),
        device="cpu", dtype="float32", seed=0,
    )


def random_trace(tid: str, d: int, V: int, peak: int) -> MemoryTrace:
    """Build a toy memory trace: random probe + peaked target distribution."""
    h = torch.randn(d)
    # Target distribution concentrated on `peak`.
    target = torch.full((V,), 1e-4)
    target[peak] = 1.0
    target = target / target.sum()
    return MemoryTrace(trace_id=tid, content=f"fact #{tid}",
                       probe_hidden=h, target_distribution=target)


def main() -> None:
    torch.manual_seed(0)
    cfg = make_tiny_config()
    base = MLPMemory(cfg.memory)

    editor = MemEditor(base, cfg)
    # Set a baseline for attribution (paper: mean over corpus samples).
    editor.set_baseline_from_samples(torch.randn(100, cfg.memory.hidden_dim))

    # 1) INSERT — add three new memories.
    traces = [random_trace(f"t{i}", cfg.memory.hidden_dim,
                           cfg.memory.vocab_size, peak=i * 7) for i in range(3)]
    print("\n=== INSERT ===")
    for t in traces:
        r = editor.insert(t)
        print(f"  {r.message}")

    # 2) QUERY — inspect footprint.
    print("\n=== QUERY ===")
    r = editor.query(traces[0])
    print(f"  footprint has {len(r.footprint.neurons)} neurons, "
          f"confidence={r.footprint.confidence:.3f}")

    # 3) MODIFY — change trace[1]'s target peak.
    print("\n=== MODIFY ===")
    new_trace = MemoryTrace(
        trace_id=traces[1].trace_id,
        content=traces[1].content + " (updated)",
        probe_hidden=traces[1].probe_hidden,
        target_distribution=random_trace("x", cfg.memory.hidden_dim,
                                         cfg.memory.vocab_size, peak=42).target_distribution,
    )
    r = editor.modify(traces[1], new_trace, preserved=[traces[0], traces[2]])
    print(f"  {r.message}")

    # 4) DELETE — erase trace[2].
    print("\n=== DELETE ===")
    r = editor.delete(traces[2])
    print(f"  {r.message}")

    # 5) Final stats.
    print("\n=== Stats ===")
    print(editor.stats())


if __name__ == "__main__":
    main()
