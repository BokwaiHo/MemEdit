"""Data structures for representing memories and edit operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch


class OperationType(str, Enum):
    """Operation kinds dispatched by MemEditor."""

    QUERY = "QUERY"
    INSERT = "INSERT"
    MODIFY = "MODIFY"
    DELETE = "DELETE"
    NONE = "NONE"


@dataclass
class MemoryTrace:
    """A single memory entry.

    In the paper (Sec. 3.1), a memory trace m_i is tied to:
      * a probe context c_i,
      * its hidden representation h_i = f(c_i) from the backbone LLM,
      * a target output distribution p*_i.

    We keep all three plus any free-form metadata for bookkeeping.
    """

    trace_id: str
    content: str                       # human-readable description
    probe_hidden: torch.Tensor         # (d,) — h_i
    target_distribution: Optional[torch.Tensor] = None   # (V,) — p*_i
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.probe_hidden.ndim != 1:
            raise ValueError(
                f"probe_hidden must be 1-D (hidden_dim,), got shape "
                f"{tuple(self.probe_hidden.shape)}"
            )

    @property
    def hidden_dim(self) -> int:
        return self.probe_hidden.shape[0]


@dataclass
class MemoryFootprint:
    """Output of Memory Attribution (Sec. 3.2, Eq. 3).

    `neurons` is a list of (layer_idx, neuron_idx) tuples forming F_τ(m_i).
    `scores` are the raw attribution magnitudes for each entry in `neurons`.
    """

    trace_id: str
    neurons: List[tuple]            # list of (layer_idx, neuron_idx)
    scores: torch.Tensor            # (len(neurons),)
    total_score: float              # sum of attribution across *all* neurons
    confidence: float               # ratio of attributed score to total


@dataclass
class EditOperation:
    """A dispatchable edit instruction.

    Produced either directly by the caller or by the LLM-prompt-based
    operation selector (`memedit.operations.selector`).
    """

    op_type: OperationType
    target_memory: Optional[MemoryTrace] = None   # for MODIFY / DELETE / QUERY
    new_memory: Optional[MemoryTrace] = None       # for INSERT / MODIFY
    reason: str = ""

    def validate(self) -> None:
        """Sanity-check that the operation has the fields it needs."""
        ot = self.op_type
        if ot is OperationType.INSERT:
            if self.new_memory is None:
                raise ValueError("INSERT requires new_memory")
        elif ot is OperationType.DELETE:
            if self.target_memory is None:
                raise ValueError("DELETE requires target_memory")
        elif ot is OperationType.MODIFY:
            if self.target_memory is None or self.new_memory is None:
                raise ValueError("MODIFY requires both target_memory and new_memory")
        elif ot is OperationType.QUERY:
            if self.target_memory is None:
                raise ValueError("QUERY requires target_memory")
        elif ot is OperationType.NONE:
            pass
        else:
            raise ValueError(f"unknown op_type {ot}")


@dataclass
class EditResult:
    """Return value from any of the four atomic operations."""

    op_type: OperationType
    trace_id: Optional[str]
    success: bool
    # Operation-specific diagnostics
    footprint: Optional[MemoryFootprint] = None
    kl_before: Optional[float] = None
    kl_after: Optional[float] = None
    preservation_kl: Optional[float] = None
    num_iterations: int = 0
    shard_idx: Optional[int] = None
    message: str = ""
