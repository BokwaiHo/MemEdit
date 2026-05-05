"""MemEditor — the public facade that ties everything together.

Usage pattern:

    cfg = MemEditConfig()
    base = MLPMemory(cfg.memory)          # optionally load pretrained weights

    editor = MemEditor(base, cfg)
    editor.set_baseline_from_samples(hidden_samples)   # once at setup

    # One-shot CRUD
    editor.insert(new_trace)
    editor.modify(old_trace, new_trace, preserved=some_list)
    editor.delete(trace)
    result = editor.query(trace)

    # Dialogue-driven edits
    result = editor.apply(EditOperation(op_type=..., target_memory=..., new_memory=...))

    # Inference through the mixture
    probs = editor.predict(h)

The editor transparently routes each operation to a shard chosen by MoME
and triggers expansion when capacity runs out.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch

from memedit.attribution.integrated_gradients import MemoryAttributor
from memedit.core.config import MemEditConfig
from memedit.data.trace import (
    EditOperation,
    EditResult,
    MemoryTrace,
    OperationType,
)
from memedit.models.mlp_memory import MLPMemory
from memedit.mome.shard_manager import MoMEShardManager
from memedit.operations.delete import delete_memory
from memedit.operations.insert import insert_memory
from memedit.operations.modify import modify_memory
from memedit.operations.query import query_memory
from memedit.utils.logging_utils import get_logger

_log = get_logger(__name__)


class MemEditor:
    """CRUD editor over a MoME-managed collection of memory shards."""

    def __init__(self, base_memory: MLPMemory, cfg: MemEditConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[cfg.dtype]
        torch.manual_seed(cfg.seed)

        base_memory = base_memory.to(device=self.device, dtype=self.dtype)

        # One MoME manager owns all shards.
        self.mome = MoMEShardManager(
            base_memory=base_memory,
            mem_cfg=cfg.memory,
            mome_cfg=cfg.mome,
            device=cfg.device,
            dtype=self.dtype,
        )

        # One attributor per shard; since the shards share architecture we
        # instantiate lazily and swap the bound memory at call time. The
        # baseline hidden state is shared across shards.
        self._baseline_hidden: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_baseline_from_samples(self, hidden_samples: torch.Tensor) -> None:
        """Set the corpus-mean baseline h̄ shared by all attributors."""
        self._baseline_hidden = hidden_samples.mean(dim=0).detach().to(
            device=self.device, dtype=self.dtype,
        )

    def _attributor_for(self, shard_idx: int) -> MemoryAttributor:
        return MemoryAttributor(
            memory=self.mome.shards[shard_idx],
            cfg=self.cfg.attribution,
            baseline_hidden=self._baseline_hidden,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        """Return the mixture-of-shards probability distribution."""
        return self.mome.forward(h)

    # ------------------------------------------------------------------
    # Shard-routing helpers
    # ------------------------------------------------------------------

    def _pick_shard_for(self, trace: MemoryTrace, allow_expand: bool = True) -> int:
        """Choose shard. Optionally trigger expansion if capacity is low."""
        idx = self.mome.select_shard_for_edit(trace.probe_hidden)
        if allow_expand:
            new_idx = self.mome.maybe_expand_for(idx)
            if new_idx is not None:
                idx = new_idx
        return idx

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def query(self, trace: MemoryTrace) -> EditResult:
        idx = self.mome.select_shard_for_edit(trace.probe_hidden)
        attr = self._attributor_for(idx)
        res = query_memory(self.mome.shards[idx], trace, attr,
                           tau=self.cfg.attribution.sparsity_tau)
        res.shard_idx = idx
        return res

    def insert(self, trace: MemoryTrace) -> EditResult:
        idx = self._pick_shard_for(trace, allow_expand=True)
        res = insert_memory(
            memory=self.mome.shards[idx],
            trace=trace,
            key_buffer=self.mome.key_buffers[idx],
            cfg=self.cfg.insert,
        )
        res.shard_idx = idx
        return res

    def modify(
        self,
        old_trace: MemoryTrace,
        new_trace: MemoryTrace,
        preserved: Optional[Iterable[MemoryTrace]] = None,
    ) -> EditResult:
        idx = self._pick_shard_for(old_trace, allow_expand=False)
        attr = self._attributor_for(idx)
        preserved_list = list(preserved) if preserved else None
        res = modify_memory(
            memory=self.mome.shards[idx],
            old_trace=old_trace,
            new_trace=new_trace,
            attributor=attr,
            cfg=self.cfg.modify,
            preserved_traces=preserved_list,
        )
        res.shard_idx = idx
        return res

    def delete(self, trace: MemoryTrace) -> EditResult:
        idx = self._pick_shard_for(trace, allow_expand=False)
        res = delete_memory(
            memory=self.mome.shards[idx],
            trace=trace,
            key_buffer=self.mome.key_buffers[idx],
            cfg=self.cfg.delete,
        )
        res.shard_idx = idx
        return res

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def apply(self, op: EditOperation, **kwargs) -> EditResult:
        """Dispatch an EditOperation to the right atomic handler."""
        op.validate()
        ot = op.op_type
        if ot is OperationType.NONE:
            return EditResult(
                op_type=ot, trace_id=None, success=True,
                message="NONE — no-op",
            )
        if ot is OperationType.QUERY:
            return self.query(op.target_memory)
        if ot is OperationType.INSERT:
            return self.insert(op.new_memory)
        if ot is OperationType.MODIFY:
            return self.modify(
                op.target_memory, op.new_memory,
                preserved=kwargs.get("preserved"),
            )
        if ot is OperationType.DELETE:
            return self.delete(op.target_memory)
        raise ValueError(f"unhandled op {ot}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return self.mome.stats()
