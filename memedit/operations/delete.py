"""DELETE: erase a memory trace via gradient-ascent unlearning.

Section 3.3 (Delete), Eq. (7), Appendix B.5:

    θ^{(t+1)} = θ^{(t)} + η_t · P_⊥^{(i)} · ∇_θ L(M_{θ^{(t)}}(h_i), p*_i)

where P_⊥^{(i)} is the null-space projector built from the preserved key
set with k_i excluded. The objective is *maximized* (ascent) so the
module is pushed away from recalling m_i, while the projection ensures
zero first-order impact on other memories.

Following Appendix B.5:
  * T = 5 steps by default
  * Initial η_0 = 1e-2, cosine-decayed
  * Gradient clipping at max norm 1.0
  * If the deletion check (KL(p*_i || M_{θ'}(h_i)) > τ_del) fails after T
    steps, we continue for up to max_extra_steps at η_0/2.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from memedit.core.config import DeleteConfig
from memedit.core.key_buffer import KeyBuffer
from memedit.data.trace import EditResult, MemoryTrace, OperationType
from memedit.models.mlp_memory import MLPMemory
from memedit.utils.logging_utils import get_logger

_log = get_logger(__name__)


def _target_layer_index(memory: MLPMemory, override: Optional[int]) -> int:
    if override is None:
        return memory.num_layers() - 1
    return int(override)


def _cosine_lr(step: int, total: int, lr0: float) -> float:
    return lr0 * 0.5 * (1.0 + math.cos(math.pi * step / max(total, 1)))


def _forward_to_layer_input(memory: MLPMemory, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
    x = h.unsqueeze(0) if h.ndim == 1 else h
    for i, layer in enumerate(memory.layers):
        if i == layer_idx:
            break
        x = layer(x, capture=False)
    return x.squeeze(0) if x.shape[0] == 1 else x


def delete_memory(
    memory: MLPMemory,
    trace: MemoryTrace,
    key_buffer: KeyBuffer,
    cfg: DeleteConfig,
    target_layer: Optional[int] = None,
) -> EditResult:
    """Apply a DELETE edit for `trace`.

    The target of the gradient ascent is the up_proj weight of the target
    layer (same as INSERT). The gradient is projected at each step onto
    the null space of K_0^{(−i)} (the preserved keys with k_i excluded).
    """
    if trace.target_distribution is None:
        return EditResult(
            op_type=OperationType.DELETE,
            trace_id=trace.trace_id,
            success=False,
            message="DELETE requires trace.target_distribution",
        )

    layer_idx = _target_layer_index(memory, target_layer)
    layer = memory.layers[layer_idx]
    W_ref = layer.up_proj.weight

    target = trace.target_distribution.to(
        device=trace.probe_hidden.device, dtype=trace.probe_hidden.dtype
    )
    logp_target = target.clamp_min(1e-12).log()

    # Compute k_i once and derive P_⊥^{(-i)}: null-space projector with
    # this memory's key excluded from the preserved set.
    with torch.no_grad():
        k_i = _forward_to_layer_input(memory, trace.probe_hidden.detach(), layer_idx).detach()
    P_perp_minus_i = key_buffer.projector_excluding(k_i)

    # Baseline loss (forgetting KL).
    with torch.no_grad():
        logq0 = F.log_softmax(memory.forward(trace.probe_hidden), dim=-1)
        kl_before = float((target * (logp_target - logq0)).sum().item())

    total_steps = cfg.num_steps
    step_idx = 0

    def one_step(lr: float) -> None:
        memory.zero_grad(set_to_none=True)
        # Need grad on W_ref only
        logq = F.log_softmax(memory.forward(trace.probe_hidden), dim=-1)
        loss = (target * (logp_target - logq)).sum()   # KL; we ASCEND this
        grad, = torch.autograd.grad(loss, W_ref, retain_graph=False)

        # Grad clipping (max-norm).
        gnorm = grad.norm()
        if gnorm > cfg.grad_clip:
            grad = grad * (cfg.grad_clip / gnorm.clamp_min(1e-12))

        # Null-space projection on the INPUT side of W ∈ R^{out, in}.
        # For every preserved key k_j ∈ rowspan(K_0^{(-i)}): we want
        # (ΔW) k_j = 0. A sufficient condition is ΔW = G · P_⊥.
        dW = grad @ P_perp_minus_i       # (out, in)
        with torch.no_grad():
            W_ref.add_(dW, alpha=lr)     # GRADIENT ASCENT (positive step)

    while step_idx < total_steps:
        lr = _cosine_lr(step_idx, total_steps, cfg.initial_lr)
        one_step(lr)
        step_idx += 1

    # Extra steps if KL not yet large enough (Appendix B.5 tail condition).
    with torch.no_grad():
        logq = F.log_softmax(memory.forward(trace.probe_hidden), dim=-1)
        kl_after = float((target * (logp_target - logq)).sum().item())

    extra = 0
    while kl_after < cfg.kl_threshold and extra < cfg.max_extra_steps:
        one_step(cfg.initial_lr * 0.5)
        extra += 1
        with torch.no_grad():
            logq = F.log_softmax(memory.forward(trace.probe_hidden), dim=-1)
            kl_after = float((target * (logp_target - logq)).sum().item())
        step_idx += 1

    # Remove the deleted key from the preserved buffer (if present).
    key_buffer.remove_key(k_i)

    success = kl_after >= cfg.kl_threshold

    return EditResult(
        op_type=OperationType.DELETE,
        trace_id=trace.trace_id,
        success=success,
        kl_before=kl_before,
        kl_after=kl_after,
        num_iterations=step_idx,
        message=(
            f"DELETE @ layer {layer_idx}: KL {kl_before:.3f}→{kl_after:.3f} "
            f"(threshold {cfg.kl_threshold}), steps={step_idx}"
        ),
    )
