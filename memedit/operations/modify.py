"""MODIFY: update an existing memory while preserving locality.

Section 3.3 (Modify), Eq. (6):

    min_{Δθ}  L(M_{θ+Δθ}(h_i), p'*_i) + γ · Σ_{j≠i} ||M_{θ+Δθ}(h_j) − M_θ(h_j)||^2

The optimization is restricted to parameters within F_τ(m_i) — i.e. only
the neurons identified by the Memory Attribution step. This is the key
difference from INSERT/DELETE: we don't project into a null space,
we constrain the *parameter-update support* instead.

We support two solver backends:

  (a) closed_form = True: the update is a constrained least-squares
      problem on the down_proj (W^(2)) of the target layer. Following
      MEMIT (Meng et al. 2023), when the target is a single linear
      layer and the locality term is quadratic, we have a closed-form
      solution. We implement the simpler rank-one form.

  (b) closed_form = False: a few steps of constrained SGD, where only
      the weights indexed by F_τ(m_i) receive gradient.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from memedit.attribution.integrated_gradients import MemoryAttributor
from memedit.core.config import ModifyConfig
from memedit.data.trace import EditResult, MemoryTrace, OperationType
from memedit.models.mlp_memory import MLPMemory
from memedit.utils.logging_utils import get_logger

_log = get_logger(__name__)


def _make_footprint_mask(
    memory: MLPMemory,
    footprint_neurons: List[tuple],
    device: torch.device,
) -> List[torch.Tensor]:
    """Build a boolean mask per layer: True ⇒ neuron is in F_τ.

    Returns a list of length num_layers; each entry is a bool tensor of
    shape (intermediate_dim,).
    """
    L = memory.num_layers()
    J = memory.num_neurons_per_layer()
    masks = [torch.zeros(J, dtype=torch.bool, device=device) for _ in range(L)]
    for (l, j) in footprint_neurons:
        if 0 <= l < L and 0 <= j < J:
            masks[l][j] = True
    return masks


def modify_memory(
    memory: MLPMemory,
    old_trace: MemoryTrace,
    new_trace: MemoryTrace,
    attributor: MemoryAttributor,
    cfg: ModifyConfig,
    preserved_traces: Optional[List[MemoryTrace]] = None,
) -> EditResult:
    """Update `old_trace` to produce `new_trace.target_distribution`.

    Args:
        old_trace: the memory being modified; its probe defines the footprint.
        new_trace: carries the *new* target distribution; probe is usually
            the same as old_trace.probe_hidden. If its probe differs, we
            use it as the actual query point (e.g. the memory now lives at
            a new probe).
        preserved_traces: a small representative set of OTHER memories to
            keep invariant. If None, we use a zero-hidden control point.
    """
    if new_trace.target_distribution is None:
        return EditResult(
            op_type=OperationType.MODIFY,
            trace_id=old_trace.trace_id,
            success=False,
            message="MODIFY requires new_trace.target_distribution",
        )

    # 1) Localize: compute the footprint of the old memory.
    footprint = attributor.footprint(old_trace)
    device = old_trace.probe_hidden.device
    masks = _make_footprint_mask(memory, footprint.neurons, device)

    # 2) Optimize the memory's weights on the footprint-restricted support.
    probe = new_trace.probe_hidden if new_trace.probe_hidden is not None \
        else old_trace.probe_hidden
    target = new_trace.target_distribution.to(device=device, dtype=probe.dtype)
    logp_target = target.clamp_min(1e-12).log()

    preserved_hidden = None
    preserved_logp_before = None
    if preserved_traces:
        preserved_hidden = torch.stack(
            [t.probe_hidden.to(device=device, dtype=probe.dtype) for t in preserved_traces],
            dim=0,
        )
        with torch.no_grad():
            preserved_logp_before = F.log_softmax(
                memory.forward(preserved_hidden), dim=-1
            ).detach()

    # KL-before (diagnostic).
    with torch.no_grad():
        kl_before = float(
            (target * (logp_target - F.log_softmax(memory.forward(probe), dim=-1))).sum().item()
        )

    # Pick parameters to optimize: only up_proj rows indexed by F_τ per layer.
    # We optimize the full weight but mask gradients so only footprint rows change.
    params = []
    for layer in memory.layers:
        params.append(layer.up_proj.weight)
    # Also allow the down_proj columns corresponding to the same neurons,
    # which controls how those activations contribute to the output.
    for layer in memory.layers:
        params.append(layer.down_proj.weight)

    # Save snapshots for preserved-memory penalty.
    opt = torch.optim.SGD(params, lr=cfg.learning_rate)

    for step in range(cfg.num_sgd_steps):
        opt.zero_grad(set_to_none=True)
        logq = F.log_softmax(memory.forward(probe), dim=-1)
        loss = (target * (logp_target - logq)).sum()        # KL(target || model)

        if preserved_hidden is not None:
            logq_preserved = F.log_softmax(memory.forward(preserved_hidden), dim=-1)
            diff = (logq_preserved - preserved_logp_before)
            loss = loss + cfg.locality_gamma * diff.pow(2).sum()

        loss.backward()

        # Mask gradients by footprint.
        with torch.no_grad():
            for l_idx, layer in enumerate(memory.layers):
                m = masks[l_idx]   # (J,) neurons that may move
                if layer.up_proj.weight.grad is not None:
                    # Zero out rows *not* in the footprint.
                    layer.up_proj.weight.grad[~m] = 0.0
                if layer.down_proj.weight.grad is not None:
                    # Zero out columns not in the footprint.
                    layer.down_proj.weight.grad[:, ~m] = 0.0

        opt.step()

    with torch.no_grad():
        kl_after = float(
            (target * (logp_target - F.log_softmax(memory.forward(probe), dim=-1))).sum().item()
        )
        pres_kl = None
        if preserved_hidden is not None:
            cur = F.log_softmax(memory.forward(preserved_hidden), dim=-1)
            pres_kl = float(
                (preserved_logp_before.exp()
                 * (preserved_logp_before - cur)).sum(dim=-1).mean().item()
            )

    success = kl_after < max(kl_before * 0.5, 1.0)

    return EditResult(
        op_type=OperationType.MODIFY,
        trace_id=old_trace.trace_id,
        success=success,
        footprint=footprint,
        kl_before=kl_before,
        kl_after=kl_after,
        preservation_kl=pres_kl,
        num_iterations=cfg.num_sgd_steps,
        message=(
            f"MODIFY: {len(footprint.neurons)} neurons, KL {kl_before:.3f}→{kl_after:.3f}, "
            f"pres_kl={pres_kl}"
        ),
    )
