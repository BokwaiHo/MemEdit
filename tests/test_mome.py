"""Tests for memedit.mome — the gate and the shard manager."""

from __future__ import annotations

import pytest
import torch

from memedit.core.config import MemoryModuleConfig, MoMEConfig
from memedit.models.mlp_memory import MLPMemory
from memedit.mome.gate import MoMEGate
from memedit.mome.shard_manager import MoMEShardManager


# ----------------------------------------------------------------------
# MoMEGate
# ----------------------------------------------------------------------


class TestMoMEGate:
    def test_output_shapes_single(self):
        gate = MoMEGate(hidden_dim=16, num_shards=3, top_k=2)
        h = torch.randn(16)
        w, idx = gate(h)
        assert w.shape == (2,)
        assert idx.shape == (2,)

    def test_output_shapes_batch(self):
        gate = MoMEGate(hidden_dim=16, num_shards=4, top_k=2)
        h = torch.randn(5, 16)
        w, idx = gate(h)
        assert w.shape == (5, 2)
        assert idx.shape == (5, 2)

    def test_weights_sum_to_one(self):
        gate = MoMEGate(hidden_dim=8, num_shards=5, top_k=3)
        h = torch.randn(4, 8)
        w, _ = gate(h)
        totals = w.sum(dim=-1)
        assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)

    def test_top_k_caps_at_num_shards(self):
        """If top_k > num_shards, gate should still produce num_shards outputs."""
        gate = MoMEGate(hidden_dim=4, num_shards=2, top_k=5)
        h = torch.randn(4)
        w, idx = gate(h)
        assert w.shape == (2,)
        assert idx.shape == (2,)

    def test_indices_in_range(self):
        gate = MoMEGate(hidden_dim=8, num_shards=6, top_k=2)
        w, idx = gate(torch.randn(3, 8))
        assert (idx >= 0).all() and (idx < 6).all()

    def test_expand_adds_one_shard(self):
        gate = MoMEGate(hidden_dim=4, num_shards=2, top_k=1)
        assert gate.num_shards == 2
        gate.expand()
        assert gate.num_shards == 3
        # New-row parameter should be registered and gradient-capable.
        assert gate.weight.shape == (3, 4)
        assert gate.bias.shape == (3,)
        assert gate.weight.requires_grad

    def test_expand_preserves_existing_rows(self):
        gate = MoMEGate(hidden_dim=4, num_shards=2, top_k=1)
        old_rows = gate.weight.data.clone()
        gate.expand()
        assert torch.allclose(gate.weight.data[:2], old_rows, atol=1e-6)


# ----------------------------------------------------------------------
# MoMEShardManager
# ----------------------------------------------------------------------


def _make_manager(d: int = 16, N: int = 2, delta: float = 0.3) -> MoMEShardManager:
    mem_cfg = MemoryModuleConfig(
        num_layers=2, hidden_dim=d, intermediate_dim=32, vocab_size=64,
    )
    base = MLPMemory(mem_cfg)
    mome_cfg = MoMEConfig(
        initial_num_shards=N, top_k=1, expansion_threshold=delta,
        max_key_buffer=500,
    )
    return MoMEShardManager(base, mem_cfg, mome_cfg, device="cpu")


class TestShardManagerBasics:
    def test_initial_shard_count(self):
        mgr = _make_manager(N=3)
        assert mgr.num_shards == 3
        assert len(mgr.key_buffers) == 3
        assert mgr.gate.num_shards == 3

    def test_route_returns_valid_indices(self):
        mgr = _make_manager(d=8, N=2)
        idx, w = mgr.route(torch.randn(8))
        assert len(idx) == len(w) == 1
        assert 0 <= idx[0] < 2

    def test_forward_probability_shape(self):
        mgr = _make_manager(d=8, N=2)
        p = mgr.forward(torch.randn(8))
        assert p.shape == (64,)
        assert p.sum().item() == pytest.approx(1.0, abs=1e-4)

    def test_forward_batch_probability(self):
        mgr = _make_manager(d=8, N=2)
        p = mgr.forward(torch.randn(3, 8))
        assert p.shape == (3, 64)
        totals = p.sum(dim=-1)
        assert torch.allclose(totals, torch.ones_like(totals), atol=1e-4)


class TestShardManagerExpansion:
    def test_capacity_ok_initially(self):
        mgr = _make_manager(d=8, N=2, delta=0.1)
        for i in range(mgr.num_shards):
            assert mgr.capacity_ok(i) is True

    def test_expand_increases_shard_count(self):
        mgr = _make_manager(d=8, N=2)
        new_idx = mgr.expand()
        assert mgr.num_shards == 3
        assert new_idx == 2
        assert mgr.gate.num_shards == 3
        assert len(mgr.expansion_history) == 1

    def test_maybe_expand_triggers_when_capacity_low(self):
        mgr = _make_manager(d=8, N=1, delta=0.9)  # very strict δ
        # Add one key → null_fraction drops from 1.0 to 7/8 = 0.875 < 0.9
        mgr.key_buffers[0].add(torch.randn(8))
        before = mgr.num_shards
        result = mgr.maybe_expand_for(0)
        assert result is not None
        assert mgr.num_shards == before + 1

    def test_maybe_expand_noop_when_plenty_of_capacity(self):
        mgr = _make_manager(d=32, N=1, delta=0.1)
        before = mgr.num_shards
        result = mgr.maybe_expand_for(0)
        assert result is None
        assert mgr.num_shards == before


class TestShardManagerSelection:
    def test_select_shard_for_edit_picks_a_valid_index(self):
        mgr = _make_manager(d=8, N=3)
        idx = mgr.select_shard_for_edit(torch.randn(8))
        assert 0 <= idx < mgr.num_shards

    def test_stats_contains_expected_fields(self):
        mgr = _make_manager(d=8, N=2)
        s = mgr.stats()
        assert "num_shards" in s
        assert "null_fractions" in s
        assert "key_counts" in s
        assert "expansion_events" in s
        assert len(s["null_fractions"]) == 2
        assert len(s["key_counts"]) == 2
