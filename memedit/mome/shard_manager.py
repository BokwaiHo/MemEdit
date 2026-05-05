"""MoME Shard Manager (Sec. 3.4).

Owns a collection of (MLPMemory, KeyBuffer) pairs and a gating network.
Monitors each shard's null-space rank; when the *active* shard drops below
δ·d (configurable), spawns a new shard cloned from the pretrained base.

Key API:
  * `route(h)`          — return (shard_idx, weight) for inference / editing.
  * `capacity_ok(idx)`  — True iff the shard still has null-space budget.
  * `expand()`          — add a new shard.
  * `forward(h)`        — mixture-of-experts probability output (Eq. 9).

Expansion algorithm (Eq. capacity, Prop. 3):
    if null_rank(active_shard) / d < δ  →  expand()
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from memedit.core.config import MemoryModuleConfig, MoMEConfig
from memedit.core.key_buffer import KeyBuffer
from memedit.models.mlp_memory import MLPMemory
from memedit.mome.gate import MoMEGate
from memedit.utils.logging_utils import get_logger

_log = get_logger(__name__)


class MoMEShardManager:
    """Holds N memory shards plus a gate; supports dynamic expansion."""

    def __init__(
        self,
        base_memory: MLPMemory,
        mem_cfg: MemoryModuleConfig,
        mome_cfg: MoMEConfig,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.mem_cfg = mem_cfg
        self.mome_cfg = mome_cfg
        self.device = torch.device(device)
        self.dtype = dtype

        # Keep a deep-copied pretrained template so new shards start from
        # the same weights as the original base memory.
        self._template = copy.deepcopy(base_memory)

        self.shards: List[MLPMemory] = []
        self.key_buffers: List[KeyBuffer] = []
        self.expansion_history: List[dict] = []  # for diagnostics

        # Initialize with `initial_num_shards` shards (default 2).
        for _ in range(max(1, mome_cfg.initial_num_shards)):
            self._add_shard(initial=True)

        self.gate = MoMEGate(
            hidden_dim=mem_cfg.hidden_dim,
            num_shards=len(self.shards),
            top_k=mome_cfg.top_k,
            noise_std=mome_cfg.gate_noise_std,
        ).to(device=self.device, dtype=dtype)

    # ------------------------------------------------------------------
    # Shard creation / expansion
    # ------------------------------------------------------------------

    def _fresh_memory(self) -> MLPMemory:
        return copy.deepcopy(self._template).to(device=self.device, dtype=self.dtype)

    def _add_shard(self, initial: bool = False) -> None:
        new_mem = self._fresh_memory()
        new_buf = KeyBuffer(
            hidden_dim=self.mem_cfg.hidden_dim,
            max_size=self.mome_cfg.max_key_buffer,
            device=str(self.device),
            dtype=self.dtype,
        )
        self.shards.append(new_mem)
        self.key_buffers.append(new_buf)
        if not initial:
            _log.info(
                "MoME expansion: added shard %d (total now %d)",
                len(self.shards) - 1, len(self.shards),
            )
            self.expansion_history.append({
                "event": "add_shard",
                "total_shards": len(self.shards),
            })

    def expand(self) -> int:
        """Add one shard and grow the gate. Returns the new shard's index."""
        self._add_shard(initial=False)
        self.gate.expand()
        return len(self.shards) - 1

    # ------------------------------------------------------------------
    # Capacity checks
    # ------------------------------------------------------------------

    def null_fraction(self, shard_idx: int) -> float:
        return self.key_buffers[shard_idx].null_fraction()

    def capacity_ok(self, shard_idx: int) -> bool:
        """True if shard still has null-space budget above δ·d."""
        return self.null_fraction(shard_idx) >= self.mome_cfg.expansion_threshold

    def maybe_expand_for(self, shard_idx: int) -> Optional[int]:
        """Spawn a new shard if the given one is exhausted. Returns new idx
        (or None if no expansion happened)."""
        if not self.capacity_ok(shard_idx):
            old_frac = self.null_fraction(shard_idx)
            _log.info(
                "Shard %d null-space fraction %.3f < δ=%.3f — expanding",
                shard_idx, old_frac, self.mome_cfg.expansion_threshold,
            )
            return self.expand()
        return None

    # ------------------------------------------------------------------
    # Routing and forward
    # ------------------------------------------------------------------

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    def route(self, h: torch.Tensor) -> Tuple[List[int], List[float]]:
        """Return (shard_indices, weights) chosen by the gate for a single h.

        h: (hidden_dim,). For a batch, use `forward` directly.
        """
        with torch.no_grad():
            w, idx = self.gate(h.to(device=self.device, dtype=self.dtype))
        # (k,) and (k,)
        return [int(i.item()) for i in idx], [float(v.item()) for v in w]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Mixture-of-experts probability (Eq. 9).

        Runs top-k shards and combines their probability outputs with gate
        weights. h: (B, d) or (d,).
        """
        squeezed = h.ndim == 1
        if squeezed:
            h = h.unsqueeze(0)
        h = h.to(device=self.device, dtype=self.dtype)

        weights, indices = self.gate(h)       # (B, k), (B, k)
        B, k = indices.shape
        V = self.mem_cfg.vocab_size
        out = torch.zeros(B, V, device=self.device, dtype=self.dtype)

        # Straightforward loop over (batch, k). For larger batches you'd
        # group by shard, but this impl favors clarity.
        for b in range(B):
            for j in range(k):
                s = int(indices[b, j].item())
                w = weights[b, j]
                p = self.shards[s].prob(h[b])
                out[b] = out[b] + w * p
        if squeezed:
            return out.squeeze(0)
        return out

    # ------------------------------------------------------------------
    # Edit-target selection
    # ------------------------------------------------------------------

    def select_shard_for_edit(self, h: torch.Tensor) -> int:
        """Pick which shard an edit should target.

        Policy:
          1) If any shard containing this key (by gate vote) has capacity,
             pick the highest-weighted such shard.
          2) Otherwise, pick the shard with the most remaining null-space.
        """
        indices, weights = self.route(h)
        # weights are sorted DESCENDING by the topk call.
        for idx in indices:
            if self.capacity_ok(idx):
                return idx
        # All routed shards exhausted → find the freshest one globally.
        best = max(range(self.num_shards), key=lambda i: self.null_fraction(i))
        return best

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "num_shards": self.num_shards,
            "null_fractions": [self.null_fraction(i) for i in range(self.num_shards)],
            "key_counts": [self.key_buffers[i].size for i in range(self.num_shards)],
            "expansion_events": len(self.expansion_history),
        }
