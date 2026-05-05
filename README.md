# MemEdit: Fine-Grained Memory Surgery for Expandable Parametric Agent Memory

The code implementation of **MemEdit**: a framework that brings full CRUD
(Create, Read, Update, Delete) operations to parametric agent memory modules.

This repo implements the method described in the paper
*"MemEdit: Fine-Grained Memory Surgery for Expandable Parametric Agent Memory"*,
including:

- **Parametric Memory Module** — an all-MLP memory that produces a token
  distribution over the LLM vocabulary, interpolated with the backbone LLM at
  inference time.
- **Memory Attribution** — integrated-gradient based localization of the
  neuron subset responsible for a given memory trace.
- **Four atomic CRUD operations:**
  - `QUERY` — inspect the memory footprint of a trace.
  - `INSERT` — add a new memory via null-space-projected rank-one update.
  - `MODIFY` — update an existing memory while preserving locality
    (constrained low-rank perturbation).
  - `DELETE` — gradient-ascent unlearning with null-space projection.
- **Mixture-of-Memory-Experts (MoME)** — dynamic expansion that spawns a
  fresh shard when the null-space capacity of the active shard is exhausted.
- **Automatic operation selector** — an LLM-prompt-based controller that
  classifies each dialogue turn into one of `{INSERT, MODIFY, DELETE, NONE}`.

## Repo Layout

```
MemEdit/
├── memedit/                    # library source
│   ├── core/                   #   config, editor facade, key buffer
│   ├── models/                 #   MLPMemory + InterpolatedMemoryLM
│   ├── attribution/            #   integrated-gradient attribution
│   ├── operations/             #   Query / Insert / Modify / Delete + selector
│   ├── mome/                   #   gate + dynamic shard manager
│   ├── data/                   #   MemoryTrace / EditOperation dataclasses
│   └── utils/                  #   null-space math, logging, config loader
├── configs/                    # default.yaml (paper) + tiny.yaml (CPU smoke)
├── scripts/                    # CLI entry points (pretrain, demo, stress, eval, replay)
├── examples/                   # quickstart.py — minimal end-to-end demo
├── data/                       # sample_dialogue.json, sample_edits.json
└── tests/                      # pytest suite for every component
```

## Quick Start

```bash
pip install -e .

# 1. Minimal sanity demo — tiny config, CPU, ~seconds
python examples/quickstart.py

# 2. Pretrain a synthetic memory checkpoint
python scripts/pretrain_memory.py --config configs/tiny.yaml \
    --synthetic --epochs 5 --output artifacts/tiny_memory.ckpt

# 3. Full CRUD cycle using the checkpoint
python scripts/run_crud_demo.py --config configs/tiny.yaml \
    --checkpoint artifacts/tiny_memory.ckpt --output-json artifacts/crud.json

# 4. Sequential-edit stress test (triggers MoME expansion)
python scripts/stress_edit.py --config configs/tiny.yaml --num-edits 500 \
    --log-every 50 --output-json artifacts/stress.json

# 5. Replay a dialogue through the LLM operation selector (Appendix D)
python scripts/replay_dialogue.py --config configs/tiny.yaml \
    --dialogue data/sample_dialogue.json --output-json artifacts/replay.json

# 6. ESR/MP/FR evaluation on a prescribed edit sequence
python scripts/evaluate_edits.py --config configs/tiny.yaml \
    --edits data/sample_edits.json --output-json artifacts/eval.json

# 7. Run the unit tests
pytest tests -q
```

## Design Notes

A few places where this implementation makes explicit engineering choices
beyond what the paper specifies:

1. **Key collection for null-space.** We maintain a rolling buffer of recent
   probe keys per shard. The paper assumes `K_0` is known; we populate it by
   forwarding a representative probe set through the backbone once at setup
   time, and append new keys as inserts happen.
2. **Input-side vs output-side projection.** The proof in Appendix A.1 points
   out that the correct formulation projects `k_new` onto the null space
   (Eq. 15), not the outer perturbation. We implement that form.
3. **Modify uses the constrained closed-form least-squares solution from
   MEMIT** when the target is a single linear layer; otherwise it falls back
   to a few steps of constrained SGD.
4. **Delete** uses `T=5` gradient-ascent steps with cosine-decayed step size
   and per-step null-space re-projection, matching Appendix B.5.
5. **MoME** tracks the rank of `K_0` per shard via truncated SVD and
   spawns a new shard when `rank(P_⊥) < δ · d`.

This is a research implementation. It is not optimized for
production throughput, but every component is self-contained and testable.

## Citation

```bibtex
@inproceedings{memedit2026,
  title     = {MemEdit: Fine-Grained Memory Surgery for Expandable Parametric Agent Memory},
  author    = {Anonymous},
  booktitle = {NeurIPS},
  year      = {2026}
}
```
