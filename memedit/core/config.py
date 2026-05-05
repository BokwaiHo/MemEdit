"""Configuration dataclasses for MemEdit.

These mirror the hyperparameters reported in the paper
(Appendix B, Table 6; Section 4.1 Implementation Details).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryModuleConfig:
    """Hyperparameters for a single parametric memory shard (all-MLP)."""

    num_layers: int = 4
    hidden_dim: int = 4096          # d in the paper
    intermediate_dim: int = 11008
    vocab_size: int = 32000         # output distribution over LLM vocab
    activation: str = "gelu"        # 'gelu' | 'relu' | 'silu'
    dropout: float = 0.0

    # Which backbone layer's hidden state feeds the memory module.
    # 0.75 = 75% depth (Appendix B.1).
    backbone_layer_frac: float = 0.75

    # Inference-time interpolation coefficient p = λ·p_M + (1-λ)·p_LLM.
    interpolation_lambda: float = 0.3

    def __post_init__(self):
        if self.activation not in {"gelu", "relu", "silu"}:
            raise ValueError(f"unknown activation '{self.activation}'")
        if not 0.0 <= self.interpolation_lambda <= 1.0:
            raise ValueError("interpolation_lambda must be in [0,1]")


@dataclass
class AttributionConfig:
    """Integrated-gradient attribution settings (Sec. 3.2)."""

    riemann_steps: int = 20            # S in Eq. (2)
    sparsity_tau: float = 0.03         # top-τ fraction (Eq. 3)
    baseline: str = "corpus_mean"      # 'corpus_mean' | 'zero'
    eps: float = 1e-8                  # numerical stability


@dataclass
class InsertConfig:
    """Insert operation (Sec. 3.3, Eq. 4–5)."""

    # Singular values below eps_svd * sigma_max are treated as zero for the
    # null-space projector (Appendix B.3).
    eps_svd: float = 1e-5
    # The paper picks l* as the layer with highest edit success on a held-out
    # set. For a reference impl we let the user override it, or use the last
    # linear projection of the last MLP layer by default.
    target_layer: Optional[int] = None


@dataclass
class ModifyConfig:
    """Modify operation (Sec. 3.3, Eq. 6)."""

    locality_gamma: float = 0.1
    closed_form: bool = True        # least-squares solution when single linear
    num_sgd_steps: int = 20         # fallback if closed_form=False
    learning_rate: float = 1e-3


@dataclass
class DeleteConfig:
    """Delete operation (Sec. 3.3, Eq. 7; Appendix B.5)."""

    num_steps: int = 5
    initial_lr: float = 1e-2
    grad_clip: float = 1.0
    max_extra_steps: int = 5        # beyond num_steps if KL threshold not hit
    kl_threshold: float = 2.0


@dataclass
class MoMEConfig:
    """Mixture-of-Memory-Experts expansion (Sec. 3.4)."""

    initial_num_shards: int = 2
    top_k: int = 1
    expansion_threshold: float = 0.15   # δ; spawn when null-space < δ·d
    gating_hidden: int = 0              # 0 ⇒ linear gate
    gate_noise_std: float = 0.01
    max_key_buffer: int = 20000         # keys retained per shard for SVD


@dataclass
class MemEditConfig:
    """Top-level config bundling every component."""

    memory: MemoryModuleConfig = field(default_factory=MemoryModuleConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    insert: InsertConfig = field(default_factory=InsertConfig)
    modify: ModifyConfig = field(default_factory=ModifyConfig)
    delete: DeleteConfig = field(default_factory=DeleteConfig)
    mome: MoMEConfig = field(default_factory=MoMEConfig)

    device: str = "cpu"
    dtype: str = "float32"              # 'float32' | 'float16' | 'bfloat16'
    seed: int = 42

    # Used by InterpolatedMemoryLM when mixing with a backbone LLM.
    # If None, the memory module is used standalone (no LLM interpolation).
    use_llm_interpolation: bool = True
