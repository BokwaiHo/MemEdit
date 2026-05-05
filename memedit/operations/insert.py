"""INSERT: encode a new memory via null-space-projected rank-one update.

Section 3.3 (Insert) and Appendix A.1.

The correct formulation (Eq. 15 in Appendix A.1) projects the *key*
onto the null space before forming the rank-one update:

    k_⊥  = P_⊥ @ k_new
    ΔW   = (v_new − W @ k_new) @ k_⊥^T / ||k_⊥||^2
    W'   = W + ΔW

This guarantees W' @ k_j = W @ k_j for any k_j in row(K_0) (Prop. 1).

We target a single linear layer of the memory module. By default this is
the last `up_proj` (W^(1)) of the last MLP layer, which in the paper's
all-MLP architecture is a well-studied place for factual editing. Users
can override via `cfg.insert.target_layer`.

Strategy for the "key" and "value":
  * The key is the hidden-state input at the target layer. We get it by
    forwarding the probe through earlier layers of the memory module.
  * The value is what we *want* the target layer to output when given
    this key. We derive it by one-step supervised least-squares on the
    desired output distribution: specifically, we set v_new = (current
    target-layer output) + (residual that drives the final logits toward
    the target distribution). A simple approximation sets v_new to be
    the output that maximizes log p*_i.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from memedit.core.config import InsertConfig
from memedit.core.key_buffer import KeyBuffer
from memedit.data.trace import EditResult, MemoryTrace, OperationType
from memedit.models.mlp_memory import MLPMemory
from memedit.utils.linalg import apply_projected_rank_one
from memedit.utils.logging_utils import get_logger

_log = get_logger(__name__)


def _resolve_target_layer(memory: MLPMemory, cfg: InsertConfig) -> int:
    if cfg.target_layer is None:
        return memory.num_layers() - 1
    idx = int(cfg.target_layer)
    if not 0 <= idx < memory.num_layers():
        raise ValueError(f"target_layer {idx} out of range")
    return idx


def _forward_to_layer_input(memory: MLPMemory, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Run the first `layer_idx` layers, return the input to layer `layer_idx`."""
    x = h.unsqueeze(0) if h.ndim == 1 else h
    for i, layer in enumerate(memory.layers):
        if i == layer_idx:
            break
        x = layer(x, capture=False)
    return x.squeeze(0) if x.shape[0] == 1 else x


def _desired_up_proj_value(
    memory: MLPMemory,
    layer_idx: int,
    k_in: torch.Tensor,
    target_distribution: torch.Tensor,
    steps: int = 15,
    lr: float = 1e-2,
) -> torch.Tensor:
    """Find a replacement value v_new for up_proj(k_in) that drives the
    module's final output toward `target_distribution`.

    We optimize v_new directly by gradient descent on the downstream
    cross-entropy, keeping everything else fixed. This is the analog of
    the "optimize-v" step in ROME/MEMIT adapted to our setting.
    """
    layer = memory.layers[layer_idx]
    v_init = layer.up_proj(k_in.unsqueeze(0)).squeeze(0).detach()
    v_new = v_init.clone().requires_grad_(True)

    target = target_distribution.to(device=k_in.device, dtype=k_in.dtype)
    logp_target = target.clamp_min(1e-12).log()

    opt = torch.optim.Adam([v_new], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        # Run the remaining layers starting from the substituted activation.
        a = layer.act(v_new)
        x = layer.down_proj(a)
        # Match the residual connection in MLPMemoryLayer.forward
        x = x + k_in
        for j in range(layer_idx + 1, memory.num_layers()):
            x = memory.layers[j](x.unsqueeze(0), capture=False).squeeze(0)
        logits = memory.output_head(x)
        logq = F.log_softmax(logits, dim=-1)
        loss = (target * (logp_target - logq)).sum()   # KL(target || model)
        loss.backward()
        opt.step()
    return v_new.detach()


def insert_memory(
    memory: MLPMemory,
    trace: MemoryTrace,
    key_buffer: KeyBuffer,
    cfg: InsertConfig,
    v_optimization_steps: int = 15,
) -> EditResult:
    """Apply an INSERT edit for `trace` and update the key buffer.

    Writes the new memory to the up_proj of the target layer by a
    null-space-projected rank-one update.
    """
    if trace.target_distribution is None:
        return EditResult(
            op_type=OperationType.INSERT,
            trace_id=trace.trace_id,
            success=False,
            message="INSERT requires trace.target_distribution",
        )

    layer_idx = _resolve_target_layer(memory, cfg)
    layer = memory.layers[layer_idx]

    with torch.no_grad():
        k_new = _forward_to_layer_input(memory, trace.probe_hidden.detach(), layer_idx)
    # KL-before for diagnostics.
    with torch.no_grad():
        logq_before = F.log_softmax(memory.forward(trace.probe_hidden), dim=-1)
        logp_t = trace.target_distribution.clamp_min(1e-12).log().to(
            device=logq_before.device, dtype=logq_before.dtype)
        kl_before = float(
            (logp_t.exp() * (logp_t - logq_before)).sum().item()
        )

    # Optimize v_new against the target distribution.
    v_new = _desired_up_proj_value(
        memory, layer_idx, k_new, trace.target_distribution,
        steps=v_optimization_steps,
    )

    # Null-space-projected rank-one update on up_proj.
    W = layer.up_proj.weight.data     # (intermediate, hidden)
    P_perp = key_buffer.projector()   # (hidden, hidden)

    # Check that the projected key isn't degenerate.
    k_perp = P_perp @ k_new
    if k_perp.norm() < 1e-8:
        _log.warning(
            "k_⊥ nearly zero for trace %s — null space exhausted. "
            "Consider spawning a new MoME shard.",
            trace.trace_id,
        )

    W_new = apply_projected_rank_one(W, k_new, v_new, P_perp)
    with torch.no_grad():
        layer.up_proj.weight.copy_(W_new)

    # Update key buffer so future inserts see this memory as "preserved".
    key_buffer.add(k_new.detach())

    # KL-after for diagnostics.
    with torch.no_grad():
        logq_after = F.log_softmax(memory.forward(trace.probe_hidden), dim=-1)
        kl_after = float(
            (logp_t.exp() * (logp_t - logq_after)).sum().item()
        )

    success = kl_after < max(kl_before * 0.5, 1.0)    # loose threshold

    return EditResult(
        op_type=OperationType.INSERT,
        trace_id=trace.trace_id,
        success=success,
        kl_before=kl_before,
        kl_after=kl_after,
        num_iterations=v_optimization_steps,
        message=(
            f"INSERT @ layer {layer_idx}: KL {kl_before:.3f}→{kl_after:.3f}, "
            f"keys={key_buffer.size}"
        ),
    )
