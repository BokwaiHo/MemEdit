"""Tests for memedit.core.key_buffer.KeyBuffer."""

from __future__ import annotations

import pytest
import torch

from memedit.core.key_buffer import KeyBuffer


class TestKeyBufferBasics:
    def test_starts_empty(self):
        buf = KeyBuffer(hidden_dim=16)
        assert buf.size == 0
        assert buf.keys.shape == (0, 16)

    def test_add_grows_buffer(self):
        buf = KeyBuffer(hidden_dim=8)
        for _ in range(3):
            buf.add(torch.randn(8))
        assert buf.size == 3

    def test_extend_grows_buffer(self):
        buf = KeyBuffer(hidden_dim=8)
        buf.extend(torch.randn(5, 8))
        assert buf.size == 5

    def test_wrong_dim_raises(self):
        buf = KeyBuffer(hidden_dim=8)
        with pytest.raises(ValueError):
            buf.add(torch.randn(16))
        with pytest.raises(ValueError):
            buf.extend(torch.randn(3, 16))

    def test_clear_resets(self):
        buf = KeyBuffer(hidden_dim=8)
        buf.extend(torch.randn(3, 8))
        buf.clear()
        assert buf.size == 0


class TestKeyBufferEviction:
    def test_respects_max_size(self):
        buf = KeyBuffer(hidden_dim=4, max_size=3)
        for _ in range(10):
            buf.add(torch.randn(4))
        assert buf.size == 3

    def test_eviction_is_fifo(self):
        buf = KeyBuffer(hidden_dim=2, max_size=2)
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        c = torch.tensor([1.0, 1.0])
        buf.add(a)
        buf.add(b)
        buf.add(c)      # evicts a
        # buffer should now contain b, c
        assert buf.size == 2
        assert torch.allclose(buf.keys[0], b) or torch.allclose(buf.keys[1], b)
        assert not any(torch.allclose(buf.keys[i], a) for i in range(buf.size))


class TestKeyBufferRemove:
    def test_remove_existing_key(self):
        buf = KeyBuffer(hidden_dim=4)
        target = torch.tensor([1.0, 0.0, 0.0, 0.0])
        buf.add(target)
        buf.add(torch.tensor([0.0, 1.0, 0.0, 0.0]))
        assert buf.remove_key(target) is True
        assert buf.size == 1

    def test_remove_missing_key_returns_false(self):
        buf = KeyBuffer(hidden_dim=4)
        buf.add(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        assert buf.remove_key(torch.tensor([0.0, 1.0, 0.0, 0.0])) is False
        assert buf.size == 1

    def test_remove_from_empty(self):
        buf = KeyBuffer(hidden_dim=4)
        assert buf.remove_key(torch.zeros(4)) is False


class TestKeyBufferProjector:
    def test_empty_projector_is_identity(self):
        buf = KeyBuffer(hidden_dim=6)
        P = buf.projector()
        assert torch.allclose(P, torch.eye(6), atol=1e-6)

    def test_projector_cached_until_dirty(self):
        buf = KeyBuffer(hidden_dim=4)
        buf.add(torch.randn(4))
        P1 = buf.projector()
        P2 = buf.projector()           # same, cached
        assert P1 is P2

        buf.add(torch.randn(4))         # marks dirty
        P3 = buf.projector()
        # Identity semantics: a new object is computed.
        assert P3 is not P1

    def test_projector_excluding_matches_manual(self):
        torch.manual_seed(0)
        buf = KeyBuffer(hidden_dim=6)
        k1 = torch.randn(6)
        k2 = torch.randn(6)
        k3 = torch.randn(6)
        for k in (k1, k2, k3):
            buf.add(k)
        P_minus_2 = buf.projector_excluding(k2)
        # P_{-2} @ k_1 ≈ 0, P_{-2} @ k_3 ≈ 0, but P_{-2} @ k_2 != 0 in general
        assert (P_minus_2 @ k1).norm().item() < 1e-4
        assert (P_minus_2 @ k3).norm().item() < 1e-4
        # k_2 was excluded, so it's NOT guaranteed to be killed.

    def test_projector_excluding_missing_key_falls_back(self):
        buf = KeyBuffer(hidden_dim=4)
        buf.add(torch.randn(4))
        # If key isn't present, function returns the default projector.
        P = buf.projector_excluding(torch.zeros(4) + 99.0)
        assert P.shape == (4, 4)


class TestKeyBufferRank:
    def test_null_rank_empty(self):
        buf = KeyBuffer(hidden_dim=5)
        assert buf.null_rank() == 5
        assert buf.null_fraction() == pytest.approx(1.0)

    def test_null_rank_decreases_with_keys(self):
        torch.manual_seed(0)
        buf = KeyBuffer(hidden_dim=8)
        r0 = buf.null_rank()
        buf.extend(torch.randn(3, 8))
        r1 = buf.null_rank()
        assert r1 < r0
        assert r1 == 8 - 3  # generic rank
