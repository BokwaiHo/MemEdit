"""Tests for memedit.utils.linalg.

These exercise the null-space machinery that underpins Propositions 1-2
of the paper. If these pass, INSERT and DELETE are on firm ground.
"""

from __future__ import annotations

import pytest
import torch

from memedit.utils.linalg import (
    apply_projected_rank_one,
    compute_null_space_projector,
    kl_divergence,
    null_space_rank,
    project_onto_null_space,
    top_tau_mask,
)


class TestNullSpaceProjector:
    def test_empty_keys_gives_identity(self):
        d = 8
        K0 = torch.empty(0, d)
        P = compute_null_space_projector(K0)
        assert torch.allclose(P, torch.eye(d), atol=1e-6)

    def test_projector_kills_row_space(self):
        """P_⊥ @ k_j == 0 for any k_j in rowspan(K_0)."""
        torch.manual_seed(0)
        n, d = 5, 16
        K0 = torch.randn(n, d)
        P = compute_null_space_projector(K0)
        for i in range(n):
            projected = P @ K0[i]
            assert projected.norm().item() < 1e-4, (
                f"P_⊥ @ k_{i} should be ~0, got norm {projected.norm().item()}"
            )

    def test_projector_preserves_null_space(self):
        """P_⊥ acts as identity on vectors orthogonal to rowspan(K_0)."""
        torch.manual_seed(1)
        n, d = 3, 12
        K0 = torch.randn(n, d)
        # Pick a vector, then subtract its row-space component.
        v = torch.randn(d)
        # Gram-Schmidt-ish: remove projections onto each row of K0
        U, S, Vh = torch.linalg.svd(K0, full_matrices=False)
        # Project v onto row space: V V^T v
        v_row = Vh.t() @ (Vh @ v)
        v_null = v - v_row
        P = compute_null_space_projector(K0)
        restored = P @ v_null
        assert torch.allclose(restored, v_null, atol=1e-4)

    def test_projector_is_idempotent(self):
        torch.manual_seed(2)
        K0 = torch.randn(4, 10)
        P = compute_null_space_projector(K0)
        assert torch.allclose(P @ P, P, atol=1e-4)

    def test_projector_is_symmetric(self):
        torch.manual_seed(3)
        K0 = torch.randn(3, 9)
        P = compute_null_space_projector(K0)
        assert torch.allclose(P, P.t(), atol=1e-5)


class TestNullSpaceRank:
    def test_empty_keys_full_rank(self):
        null_dim, d = null_space_rank(torch.empty(0, 7))
        assert null_dim == 7
        assert d == 7

    def test_nonzero_keys_reduce_null_dim(self):
        torch.manual_seed(0)
        K0 = torch.randn(4, 10)
        null_dim, d = null_space_rank(K0)
        assert d == 10
        # 4 generically-independent rows → null dim = 6
        assert null_dim == 6

    def test_linearly_dependent_keys(self):
        K0 = torch.zeros(3, 5)
        K0[0, 0] = 1.0
        K0[1, 0] = 2.0          # dependent
        K0[2, 1] = 1.0
        null_dim, d = null_space_rank(K0)
        assert d == 5
        # Rank = 2 → null dim = 3
        assert null_dim == 3


class TestProjectedRankOneUpdate:
    def test_insert_invariance_on_existing_keys(self):
        """Proposition 1: W' @ k_j == W @ k_j for every existing k_j."""
        torch.manual_seed(0)
        d_in, d_out, n = 12, 16, 4
        W = torch.randn(d_out, d_in)
        K0 = torch.randn(n, d_in)
        P = compute_null_space_projector(K0)

        k_new = torch.randn(d_in)
        v_new = torch.randn(d_out)
        W_new = apply_projected_rank_one(W, k_new, v_new, P)

        for j in range(n):
            before = W @ K0[j]
            after = W_new @ K0[j]
            assert torch.allclose(before, after, atol=1e-4), (
                f"Preserved-key invariance failed at j={j}: "
                f"max diff {(before - after).abs().max().item():.2e}"
            )

    def test_update_has_effect_on_new_key(self):
        """The update should actually do *something* on the new direction."""
        torch.manual_seed(0)
        d_in, d_out = 10, 8
        W = torch.randn(d_out, d_in)
        K0 = torch.randn(3, d_in)
        P = compute_null_space_projector(K0)

        # Pick k_new with a significant null-space component
        k_new = torch.randn(d_in)
        v_new = torch.randn(d_out)

        W_new = apply_projected_rank_one(W, k_new, v_new, P)
        # On the projected key direction, the output should have changed.
        k_perp = P @ k_new
        assert k_perp.norm().item() > 1e-3, "test precondition: k_⊥ not degenerate"
        diff = (W_new - W) @ k_perp
        assert diff.norm().item() > 1e-6


class TestKLDivergence:
    def test_same_distribution_is_zero(self):
        logp = torch.log_softmax(torch.randn(10), dim=-1)
        assert kl_divergence(logp, logp).item() == pytest.approx(0.0, abs=1e-5)

    def test_kl_nonnegative(self):
        torch.manual_seed(0)
        logp = torch.log_softmax(torch.randn(16), dim=-1)
        logq = torch.log_softmax(torch.randn(16), dim=-1)
        assert kl_divergence(logp, logq).item() >= 0.0

    def test_batch_mean_reduction(self):
        torch.manual_seed(0)
        logp = torch.log_softmax(torch.randn(4, 16), dim=-1)
        logq = torch.log_softmax(torch.randn(4, 16), dim=-1)
        mean_kl = kl_divergence(logp, logq, reduction="mean").item()
        sum_kl = kl_divergence(logp, logq, reduction="sum")   # (4,)
        assert sum_kl.shape == (4,)
        assert mean_kl == pytest.approx(sum_kl.mean().item(), abs=1e-6)


class TestTopTauMask:
    def test_picks_correct_fraction(self):
        values = torch.arange(100).float()
        mask = top_tau_mask(values, tau=0.1)
        assert mask.sum().item() == 10

    def test_picks_largest(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = top_tau_mask(values, tau=0.4)
        # Top-2 = [4.0, 5.0]
        picked = values[mask]
        assert set(picked.tolist()) == {4.0, 5.0}

    def test_multidim_shape_preserved(self):
        values = torch.randn(3, 4)
        mask = top_tau_mask(values, tau=0.5)
        assert mask.shape == values.shape
        assert mask.dtype == torch.bool

    def test_invalid_tau_raises(self):
        with pytest.raises(ValueError):
            top_tau_mask(torch.zeros(5), tau=0.0)
        with pytest.raises(ValueError):
            top_tau_mask(torch.zeros(5), tau=1.5)


class TestProjectOntoNullSpace:
    def test_single_vector_projection(self):
        torch.manual_seed(0)
        K0 = torch.randn(3, 8)
        P = compute_null_space_projector(K0)
        v = torch.randn(8)
        proj = project_onto_null_space(v, P)
        # Result should be orthogonal to every row of K0.
        for k in K0:
            assert abs((k @ proj).item()) < 1e-4
