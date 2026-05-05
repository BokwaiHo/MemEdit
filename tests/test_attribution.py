"""Tests for memedit.attribution.integrated_gradients.MemoryAttributor."""

from __future__ import annotations

import pytest
import torch

from memedit.attribution.integrated_gradients import MemoryAttributor
from memedit.core.config import AttributionConfig


class TestAttributionShapes:
    def test_score_shape(self, memory, tiny_memory_cfg, trace_factory):
        attr = MemoryAttributor(memory, AttributionConfig(riemann_steps=4))
        trace = trace_factory("t0", peak=5)
        scores = attr.compute_attribution(trace)
        L = tiny_memory_cfg.num_layers
        J = tiny_memory_cfg.intermediate_dim
        assert scores.shape == (L, J)

    def test_scores_nonnegative(self, memory, trace_factory):
        """Attribution uses |·|, so every entry should be ≥ 0."""
        attr = MemoryAttributor(memory, AttributionConfig(riemann_steps=4))
        scores = attr.compute_attribution(trace_factory("t", peak=3))
        assert (scores >= 0).all()

    def test_works_without_target_distribution(self, memory, tiny_memory_cfg):
        """QUERY path must work even if the trace has no target set."""
        from memedit.data.trace import MemoryTrace

        attr = MemoryAttributor(memory, AttributionConfig(riemann_steps=4))
        trace = MemoryTrace(
            trace_id="no_target",
            content="placeholder",
            probe_hidden=torch.randn(tiny_memory_cfg.hidden_dim),
            target_distribution=None,
        )
        scores = attr.compute_attribution(trace)
        assert scores.shape == (
            tiny_memory_cfg.num_layers,
            tiny_memory_cfg.intermediate_dim,
        )
        assert (scores >= 0).all()


class TestAttributionBaseline:
    def test_set_corpus_mean_baseline(self, memory, tiny_memory_cfg):
        attr = MemoryAttributor(memory, AttributionConfig())
        samples = torch.randn(50, tiny_memory_cfg.hidden_dim)
        attr.set_corpus_mean_baseline(samples)
        assert attr._baseline is not None
        assert attr._baseline.shape == (tiny_memory_cfg.hidden_dim,)
        assert torch.allclose(attr._baseline, samples.mean(dim=0), atol=1e-5)

    def test_wrong_shape_baseline_raises(self, memory):
        attr = MemoryAttributor(memory, AttributionConfig())
        with pytest.raises(ValueError):
            attr.set_corpus_mean_baseline(torch.randn(10))    # 1-D

    def test_zero_baseline_fallback(self, memory, tiny_memory_cfg):
        """With no baseline set and cfg.baseline='zero', fallback is zeros."""
        attr = MemoryAttributor(memory, AttributionConfig(baseline="zero"))
        b = attr.baseline(torch.randn(tiny_memory_cfg.hidden_dim))
        assert torch.allclose(b, torch.zeros_like(b))


class TestFootprint:
    def test_footprint_has_expected_count(self, memory, tiny_memory_cfg, trace_factory):
        L = tiny_memory_cfg.num_layers
        J = tiny_memory_cfg.intermediate_dim
        total = L * J
        tau = 0.1
        attr = MemoryAttributor(memory, AttributionConfig(riemann_steps=4, sparsity_tau=tau))
        fp = attr.footprint(trace_factory("t", peak=3))
        # Top-τ fraction, rounded to at least 1 neuron.
        expected = max(1, int(round(tau * total)))
        assert len(fp.neurons) == expected

    def test_footprint_coords_in_bounds(self, memory, tiny_memory_cfg, trace_factory):
        attr = MemoryAttributor(memory, AttributionConfig(riemann_steps=4, sparsity_tau=0.2))
        fp = attr.footprint(trace_factory("t", peak=0))
        for (l, j) in fp.neurons:
            assert 0 <= l < tiny_memory_cfg.num_layers
            assert 0 <= j < tiny_memory_cfg.intermediate_dim

    def test_confidence_between_zero_and_one(self, memory, trace_factory):
        attr = MemoryAttributor(memory, AttributionConfig(riemann_steps=4, sparsity_tau=0.1))
        fp = attr.footprint(trace_factory("t", peak=7))
        assert 0.0 <= fp.confidence <= 1.0 + 1e-6

    def test_scores_match_neurons(self, memory, trace_factory):
        attr = MemoryAttributor(memory, AttributionConfig(riemann_steps=4, sparsity_tau=0.1))
        fp = attr.footprint(trace_factory("t", peak=2))
        assert fp.scores.shape == (len(fp.neurons),)

    def test_higher_tau_gives_more_neurons(self, memory, trace_factory):
        attr_low = MemoryAttributor(memory, AttributionConfig(riemann_steps=4, sparsity_tau=0.05))
        attr_high = MemoryAttributor(memory, AttributionConfig(riemann_steps=4, sparsity_tau=0.3))
        trace = trace_factory("t", peak=5)
        fp_low = attr_low.footprint(trace)
        fp_high = attr_high.footprint(trace)
        assert len(fp_high.neurons) >= len(fp_low.neurons)


class TestAttributionReproducibility:
    def test_same_trace_same_scores(self, memory, trace_factory):
        """Integrated gradients over α is deterministic for a fixed model."""
        attr = MemoryAttributor(memory, AttributionConfig(riemann_steps=4))
        trace = trace_factory("t", peak=3)
        s1 = attr.compute_attribution(trace)
        s2 = attr.compute_attribution(trace)
        assert torch.allclose(s1, s2, atol=1e-5)
