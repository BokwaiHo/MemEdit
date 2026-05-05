"""QUERY: retrieve the memory footprint F_τ(m_i).

A read-only operation that runs Memory Attribution on the probe and
returns the top-τ neurons plus a confidence estimate.

Sec. 3.3 (Query paragraph) describes this as the entry point to every
edit cycle.
"""

from __future__ import annotations

from memedit.attribution.integrated_gradients import MemoryAttributor
from memedit.data.trace import EditResult, MemoryTrace, OperationType


def query_memory(
    memory_module,
    trace: MemoryTrace,
    attributor: MemoryAttributor,
    tau: float = None,
) -> EditResult:
    """Run attribution and return a footprint as an EditResult."""
    footprint = attributor.footprint(trace, tau=tau)
    return EditResult(
        op_type=OperationType.QUERY,
        trace_id=trace.trace_id,
        success=True,
        footprint=footprint,
        message=(
            f"QUERY: {len(footprint.neurons)} neurons, "
            f"confidence={footprint.confidence:.3f}"
        ),
    )
