"""All-MLP Parametric Memory Module.

Implements the architecture described in:
  * Wei et al. 2025 (MLP Memory) — the primary reference in the paper.
  * Paper Sec. 3.1, Appendix B.1 for hyperparameters.

A layer applies two linear projections with an activation in between:

    a_l = σ(W_l^(1) @ h_in + b_l^(1))           # intermediate activation
    h_out = W_l^(2) @ a_l + b_l^(2)              # down-projection

The module's output is a logit vector over the LLM vocabulary, which is
later softmaxed and interpolated with the LLM's own distribution (Eq. 1).

We expose a `capture_activations=True` mode so the attribution pass can
record per-neuron activations for integrated gradients.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from memedit.core.config import MemoryModuleConfig


def _get_activation(name: str) -> nn.Module:
    return {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
    }[name]


class MLPMemoryLayer(nn.Module):
    """One MLP memory layer: two linear projections with a nonlinearity.

    Exposes both weight matrices so external edit operations can modify
    them in place.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int, activation: str,
                 dropout: float = 0.0):
        super().__init__()
        # W1: (intermediate, hidden), W2: (hidden, intermediate)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=True)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=True)
        self.act = _get_activation(activation)
        self.dropout = nn.Dropout(dropout)

        # Cached activations for attribution. Populated only when the parent
        # module is in attribution mode.
        self._cached_pre_act: Optional[torch.Tensor] = None
        self._cached_post_act: Optional[torch.Tensor] = None

    @property
    def W1(self) -> torch.Tensor:
        return self.up_proj.weight  # (intermediate_dim, hidden_dim)

    @property
    def W2(self) -> torch.Tensor:
        return self.down_proj.weight  # (hidden_dim, intermediate_dim)

    def forward(self, h: torch.Tensor, capture: bool = False) -> torch.Tensor:
        # h: (B, hidden_dim)  OR  (hidden_dim,)
        squeezed = False
        if h.ndim == 1:
            h = h.unsqueeze(0)
            squeezed = True

        pre_act = self.up_proj(h)              # (B, intermediate_dim)
        post_act = self.act(pre_act)
        if capture:
            # Keep references so gradients can flow through these tensors when
            # `requires_grad_(True)` is set on them by the attribution engine.
            self._cached_pre_act = pre_act
            self._cached_post_act = post_act
        else:
            self._cached_pre_act = None
            self._cached_post_act = None

        out = self.down_proj(self.dropout(post_act))
        out = out + h   # residual connection keeps the depth stable
        if squeezed:
            out = out.squeeze(0)
        return out


class MLPMemory(nn.Module):
    """Stack of L MLP memory layers + an output head to the vocabulary.

    Input: hidden state h ∈ R^{hidden_dim} from some intermediate backbone layer.
    Output: logit vector ∈ R^{vocab_size}.

    Call `module.forward(h)` for a normal forward pass, or
    `module.forward(h, capture_activations=True)` to let the attribution
    engine record per-layer activations.
    """

    def __init__(self, cfg: MemoryModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([
            MLPMemoryLayer(cfg.hidden_dim, cfg.intermediate_dim,
                           cfg.activation, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.output_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        capture_activations: bool = False,
        return_logits: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            h: (B, hidden_dim) or (hidden_dim,)
            capture_activations: store per-layer activations for attribution.
            return_logits: if True, returns logits; else the final hidden.
        """
        x = h
        for layer in self.layers:
            x = layer(x, capture=capture_activations)
        if return_logits:
            return self.output_head(x)
        return x

    def log_prob(self, h: torch.Tensor) -> torch.Tensor:
        """Return log-softmax over the vocabulary."""
        return F.log_softmax(self.forward(h), dim=-1)

    def prob(self, h: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(h), dim=-1)

    # ------------------------------------------------------------------
    # Introspection helpers used by the attribution engine.
    # ------------------------------------------------------------------

    def num_neurons_per_layer(self) -> int:
        return self.cfg.intermediate_dim

    def num_layers(self) -> int:
        return self.cfg.num_layers

    def cached_activations(self) -> List[torch.Tensor]:
        """Return list of captured post-activation tensors in layer order.

        Must be called after a forward pass with `capture_activations=True`.
        """
        out = []
        for layer in self.layers:
            if layer._cached_post_act is None:
                raise RuntimeError(
                    "Activations not captured — run forward(capture_activations=True) first"
                )
            out.append(layer._cached_post_act)
        return out


# ----------------------------------------------------------------------
# Interpolated memory-LM wrapper
# ----------------------------------------------------------------------


class InterpolatedMemoryLM(nn.Module):
    """Wraps a memory module + a callable LLM-logit function with Eq. (1).

    The LLM-logit function must take whatever inputs you want and return logits
    of shape (..., vocab_size) *on the same vocabulary* as the memory module.

    Example:
        llm_logits_fn = lambda ctx: backbone_lm(ctx).logits
        wrapped = InterpolatedMemoryLM(memory, llm_logits_fn, lam=0.3)
        probs = wrapped(h, ctx)        # returns interpolated probs
    """

    def __init__(
        self,
        memory: MLPMemory,
        llm_logits_fn: Optional[callable] = None,
        lam: float = 0.3,
    ):
        super().__init__()
        self.memory = memory
        self.llm_logits_fn = llm_logits_fn
        self.lam = float(lam)

    def forward(
        self,
        hidden: torch.Tensor,
        llm_ctx=None,
    ) -> torch.Tensor:
        """Returns the interpolated probability distribution (Eq. 1)."""
        mem_probs = self.memory.prob(hidden)
        if self.llm_logits_fn is None or llm_ctx is None:
            return mem_probs
        llm_logits = self.llm_logits_fn(llm_ctx)
        llm_probs = F.softmax(llm_logits, dim=-1)
        return self.lam * mem_probs + (1.0 - self.lam) * llm_probs
