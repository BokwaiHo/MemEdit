"""Linear-algebra helpers for null-space-constrained editing.

The core construction (Eq. 4) is

        P_⊥ = I − K_0^T (K_0 K_0^T)^(-1) K_0

but in practice we use the equivalent SVD form (Appendix B.3):

        K_0 = U Σ V^T
        V_r = right-singular-vectors with σ_r > ε · σ_max
        P_⊥ = I − V_r V_r^T

because it is numerically stable when K_0 has near-singular rows.

We also expose a helper to project a single vector onto the null space,
which is what the actual Insert/Delete code uses (see the `Proof of
Proposition 1` discussion for why we project the key, not the update).
"""

from __future__ import annotations

from typing import Tuple

import torch


def compute_null_space_projector(
    K0: torch.Tensor,
    eps_svd: float = 1e-5,
) -> torch.Tensor:
    """Return P_⊥ ∈ R^{d×d} that projects onto the null space of rows(K0).

    Args:
        K0: (n, d) — key matrix of existing memory probes.
        eps_svd: singular values below eps_svd * σ_max are treated as zero.

    Returns:
        P_⊥ ∈ R^{d, d} such that P_⊥ @ k = 0 for every k ∈ rowspan(K0),
        and P_⊥ @ v = v for any v ⊥ rowspan(K0).
    """
    if K0.numel() == 0 or K0.shape[0] == 0:
        # No keys yet — null space is all of R^d.
        d = K0.shape[-1] if K0.ndim == 2 else 0
        return torch.eye(d, dtype=K0.dtype, device=K0.device)

    # We want the right-singular basis of K0 (which spans row(K0)).
    # torch.linalg.svd returns K0 = U diag(S) Vh with Vh: (min(n,d), d).
    U, S, Vh = torch.linalg.svd(K0, full_matrices=False)
    if S.numel() == 0:
        d = K0.shape[-1]
        return torch.eye(d, dtype=K0.dtype, device=K0.device)

    sigma_max = S.max().clamp_min(1e-30)
    keep = S > (eps_svd * sigma_max)
    Vr = Vh[keep]                        # (r, d)

    d = K0.shape[-1]
    I = torch.eye(d, dtype=K0.dtype, device=K0.device)
    if Vr.shape[0] == 0:
        return I
    return I - Vr.t() @ Vr               # (d, d)


def null_space_rank(
    K0: torch.Tensor,
    eps_svd: float = 1e-5,
) -> Tuple[int, int]:
    """Return (null_space_dim, total_dim) for the row-space of K0."""
    d = K0.shape[-1] if K0.ndim == 2 else 0
    if K0.numel() == 0 or K0.shape[0] == 0:
        return d, d
    S = torch.linalg.svdvals(K0)
    sigma_max = S.max().clamp_min(1e-30)
    row_rank = int((S > (K0.new_tensor(1e-5) * sigma_max)).sum().item())
    if eps_svd != 1e-5:                 # recompute if caller used a custom eps
        row_rank = int((S > (eps_svd * sigma_max)).sum().item())
    return d - row_rank, d


def project_onto_null_space(
    vec: torch.Tensor,
    P_perp: torch.Tensor,
) -> torch.Tensor:
    """Project a (d,) vector onto the null space represented by P_perp."""
    return P_perp @ vec


def apply_projected_rank_one(
    W: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    P_perp: torch.Tensor,
) -> torch.Tensor:
    """Input-side-projected rank-one update (Eq. 15 in Appendix A.1).

    ΔW = (v_new - W k_new) (P_⊥ k_new)^T / ||P_⊥ k_new||^2

    Returns the new weight matrix W' = W + ΔW. Leaves W unchanged.

    The projection is applied to k_new *before* forming the outer product,
    which is the correct formulation for the proof of Proposition 1.
    """
    k_proj = P_perp @ k_new                          # (d_in,)
    denom = (k_proj @ k_proj).clamp_min(1e-12)
    residual = v_new - W @ k_new                    # (d_out,)
    dW = torch.outer(residual, k_proj) / denom      # (d_out, d_in)
    return W + dW


def kl_divergence(
    logp: torch.Tensor,
    logq: torch.Tensor,
    reduction: str = "sum",
) -> torch.Tensor:
    """KL(P || Q) with both inputs in log-space (any shape, last dim = vocab).

    reduction: 'sum' or 'mean' over the vocabulary.
    """
    p = logp.exp()
    kl = (p * (logp - logq)).sum(dim=-1)
    if reduction == "mean":
        kl = kl.mean()
    elif reduction == "sum":
        pass
    else:
        raise ValueError(reduction)
    return kl


def top_tau_mask(values: torch.Tensor, tau: float) -> torch.Tensor:
    """Boolean mask that keeps the top-τ fraction of `values`.

    Handles arbitrary shape — flattens internally. Returns a bool tensor of
    the same shape as `values`.
    """
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    flat = values.flatten()
    n = flat.numel()
    k = max(1, int(round(tau * n)))
    # topk is faster than quantile for small τ.
    thresh = flat.topk(k).values[-1]
    return values >= thresh
