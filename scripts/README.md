# scripts/

Command-line entry points for the full MemEdit pipeline. All scripts
read a YAML config (`configs/default.yaml` or `configs/tiny.yaml`) and
accept an optional pretrained checkpoint produced by `pretrain_memory.py`.

| Script | Purpose |
| --- | --- |
| `pretrain_memory.py` | Pretrain the MLP memory module on kNN-style soft targets (or hard labels, or synthetic data for smoke tests). Produces a checkpoint. |
| `run_crud_demo.py` | Walk through Insert → Query → Modify → Delete end-to-end with structured logging — the script behind Figure 1's CRUD cycle. |
| `stress_edit.py` | Sequential random-edit benchmark with MoME expansion; reproduces the shape of Table 2 / Figure 2. |
| `replay_dialogue.py` | Drive a dialogue JSON through the LLM-based operation selector (Appendix D). Ships a keyword-based `MockLLM` for CI; plug in a real LLM via `--llm-module pkg.mod:callable`. |
| `evaluate_edits.py` | Apply a prescribed edit sequence and compute ESR / MP / FR / per-op latency (paper §4.1 metrics). |

## Typical workflow

```bash
# 1. Pretrain a tiny memory module on synthetic data (seconds on CPU).
python scripts/pretrain_memory.py \
    --config configs/tiny.yaml --synthetic --epochs 5 \
    --output artifacts/tiny_memory.ckpt

# 2. Smoke-test CRUD.
python scripts/run_crud_demo.py \
    --config configs/tiny.yaml \
    --checkpoint artifacts/tiny_memory.ckpt \
    --output-json artifacts/crud.json

# 3. Stress sequential edits.
python scripts/stress_edit.py \
    --config configs/tiny.yaml \
    --checkpoint artifacts/tiny_memory.ckpt \
    --num-edits 500 --log-every 50 \
    --output-json artifacts/stress.json

# 4. Run a prescribed evaluation set.
python scripts/evaluate_edits.py \
    --config configs/tiny.yaml \
    --checkpoint artifacts/tiny_memory.ckpt \
    --edits data/sample_edits.json \
    --output-json artifacts/eval.json
```

Sample input JSONs for `replay_dialogue.py` and `evaluate_edits.py` live
under `data/` (provided alongside the code so the scripts are runnable
out of the box).
