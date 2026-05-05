"""Pretrain a parametric memory module.

This is a simplified reference implementation of the pretraining stage
described in MLP Memory (Wei et al., 2025) and cited in Sec. 3.1 /
Appendix B.1:

    Loss = β · KL(p_kNN || p_M) + (1 - β) · CE(y || p_M)
    p_kNN = softmax over the k nearest neighbor tokens in the corpus.

We do NOT ship a real kNN index here. Instead the script takes:

  (a) A tensor of hidden states `H ∈ R^{N, d}` (from a frozen backbone's
      layer at depth `backbone_layer_frac`).
  (b) Either:
        * a tensor of target distributions `P ∈ R^{N, V}` (full kNN soft
          targets) — use this if you already built a kNN index, OR
        * a tensor of integer labels `Y ∈ {0, ..., V-1}^N` — use this
          if you just want a language-modeling-style pretraining.
  (c) An optional corpus-mean hidden state for attribution baseline
      initialization (saved alongside the checkpoint).

If neither P nor Y is provided, we generate synthetic targets (peaked
distributions) so the script can be smoke-tested without real data.

Run:
    python scripts/pretrain_memory.py \
        --config configs/tiny.yaml \
        --hidden-states hidden.pt \
        --targets targets.pt \
        --output memory.ckpt \
        --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from memedit.models.mlp_memory import MLPMemory
from memedit.utils.config_loader import load_config
from memedit.utils.logging_utils import get_logger

_log = get_logger("pretrain")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config (e.g., configs/tiny.yaml).")
    p.add_argument("--hidden-states", type=str, default=None,
                   help="torch.save-d tensor of shape (N, d).")
    p.add_argument("--targets", type=str, default=None,
                   help="torch.save-d tensor of shape (N, V) [soft] or (N,) [hard].")
    p.add_argument("--output", type=str, required=True,
                   help="Where to save the pretrained memory checkpoint.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--kl-weight", type=float, default=0.5,
                   help="β: mixing between KL(kNN||model) and CE(y||model).")
    p.add_argument("--synthetic", action="store_true",
                   help="Generate synthetic (h, p) pairs — for smoke testing.")
    p.add_argument("--synthetic-n", type=int, default=2000)
    p.add_argument("--device", type=str, default=None,
                   help="Override cfg.device.")
    return p.parse_args()


def _load_or_synth(
    args: argparse.Namespace, hidden_dim: int, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if args.synthetic or (args.hidden_states is None and args.targets is None):
        _log.info("Generating synthetic pretraining data: N=%d", args.synthetic_n)
        torch.manual_seed(0)
        H = torch.randn(args.synthetic_n, hidden_dim)
        # Peaked targets: a random token per sample, softened.
        peaks = torch.randint(0, vocab_size, (args.synthetic_n,))
        P = torch.full((args.synthetic_n, vocab_size), 1e-4)
        P[torch.arange(args.synthetic_n), peaks] = 1.0
        P = P / P.sum(dim=-1, keepdim=True)
        return H, P

    if args.hidden_states is None:
        raise ValueError("--hidden-states is required (or use --synthetic)")
    if args.targets is None:
        raise ValueError("--targets is required (or use --synthetic)")

    H = torch.load(args.hidden_states, map_location="cpu")
    T = torch.load(args.targets, map_location="cpu")

    if T.ndim == 1:
        # Hard labels → one-hot soft targets.
        P = F.one_hot(T.long(), num_classes=vocab_size).float()
    else:
        P = T.float()
    return H, P


def _loss(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    kl_weight: float,
) -> torch.Tensor:
    """β·KL(target || model) + (1-β)·CE against the argmax of target."""
    logq = F.log_softmax(logits, dim=-1)
    kl = (target_probs.clamp_min(1e-12) * (target_probs.clamp_min(1e-12).log() - logq)).sum(dim=-1).mean()
    labels = target_probs.argmax(dim=-1)
    ce = F.cross_entropy(logits, labels)
    return kl_weight * kl + (1.0 - kl_weight) * ce


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device
    device = torch.device(cfg.device)

    _log.info("Config: %s", args.config)
    _log.info("Building MLPMemory: num_layers=%d hidden=%d vocab=%d",
              cfg.memory.num_layers, cfg.memory.hidden_dim, cfg.memory.vocab_size)

    memory = MLPMemory(cfg.memory).to(device=device)
    H, P = _load_or_synth(args, cfg.memory.hidden_dim, cfg.memory.vocab_size)
    assert H.shape[1] == cfg.memory.hidden_dim, (
        f"hidden-state dim {H.shape[1]} != cfg.memory.hidden_dim {cfg.memory.hidden_dim}"
    )
    assert P.shape[1] == cfg.memory.vocab_size
    _log.info("Pretraining data: N=%d", H.shape[0])

    dataset = TensorDataset(H, P)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(memory.parameters(), lr=args.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(args.epochs):
        running = 0.0
        n_seen = 0
        for hb, pb in loader:
            hb = hb.to(device=device, dtype=torch.float32)
            pb = pb.to(device=device, dtype=torch.float32)
            logits = memory(hb)
            loss = _loss(logits, pb, args.kl_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * hb.shape[0]
            n_seen += hb.shape[0]
        sched.step()
        _log.info("epoch %d/%d  avg_loss=%.4f  lr=%.2e",
                  epoch + 1, args.epochs, running / max(1, n_seen),
                  opt.param_groups[0]["lr"])

    # Save checkpoint: weights + a corpus-mean baseline derived from the
    # pretraining data, so downstream scripts can wire attribution.
    corpus_mean = H.to(device=device).mean(dim=0).detach().cpu()
    out = {
        "state_dict": memory.state_dict(),
        "memory_config": cfg.memory.__dict__,
        "corpus_mean_hidden": corpus_mean,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    _log.info("Saved pretrained memory to %s", output_path)


if __name__ == "__main__":
    main()
