"""MoME gating network (Sec. 3.4, Eq. 8).

A lightweight linear gate over hidden states that routes each input to
the top-k memory shards. The gate is dynamically expanded when a new
shard is spawned (Appendix B.4).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoMEGate(nn.Module):
    """Linear gate: logits = W_g @ h + b_g; softmax for routing."""

    def __init__(
        self,
        hidden_dim: int,
        num_shards: int,
        top_k: int = 1,
        noise_std: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k = int(top_k)
        self.noise_std = float(noise_std)

        self.weight = nn.Parameter(torch.zeros(num_shards, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(num_shards))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

    @property
    def num_shards(self) -> int:
        return int(self.weight.shape[0])

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (topk_weights, topk_indices).

        h: (B, hidden_dim) or (hidden_dim,)
        topk_weights: (B, k) — softmax-normalized gate weights
        topk_indices: (B, k) — shard indices
        """
        squeezed = h.ndim == 1
        if squeezed:
            h = h.unsqueeze(0)
        logits = h @ self.weight.t() + self.bias      # (B, N)
        probs = F.softmax(logits, dim=-1)
        k = min(self.top_k, self.num_shards)
        topk_vals, topk_idx = probs.topk(k, dim=-1)
        # Re-normalize over the chosen shards so weights sum to 1.
        topk_w = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if squeezed:
            return topk_w.squeeze(0), topk_idx.squeeze(0)
        return topk_w, topk_idx

    # ------------------------------------------------------------------
    # Shard-count expansion (Appendix B.4).
    # ------------------------------------------------------------------

    def expand(self, init_from_mean: bool = True) -> None:
        """Add one output unit. Its weight is the mean of existing rows
        plus Gaussian noise, per Appendix B.4."""
        N, d = self.weight.shape
        if init_from_mean:
            mean_row = self.weight.data.mean(dim=0, keepdim=True)
            noise = torch.randn_like(mean_row) * self.noise_std
            new_row = mean_row + noise
            new_bias = self.bias.data.mean(keepdim=True)
        else:
            new_row = torch.randn(1, d, device=self.weight.device, dtype=self.weight.dtype) * 0.02
            new_bias = torch.zeros(1, device=self.bias.device, dtype=self.bias.dtype)
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_row], dim=0))
        self.bias = nn.Parameter(torch.cat([self.bias.data, new_bias], dim=0))
