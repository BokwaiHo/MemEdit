"""Tests for the MLPMemory module and InterpolatedMemoryLM wrapper."""

from __future__ import annotations

import pytest
import torch

from memedit.core.config import MemoryModuleConfig
from memedit.models.mlp_memory import (
    InterpolatedMemoryLM,
    MLPMemory,
    MLPMemoryLayer,
)


class TestMLPMemoryLayer:
    def test_output_shape_2d(self):
        layer = MLPMemoryLayer(hidden_dim=32, intermediate_dim=64, activation="gelu")
        h = torch.randn(5, 32)
        out = layer(h)
        assert out.shape == (5, 32)

    def test_output_shape_1d(self):
        layer = MLPMemoryLayer(hidden_dim=32, intermediate_dim=64, activation="gelu")
        h = torch.randn(32)
        out = layer(h)
        assert out.shape == (32,)

    def test_capture_stores_activations(self):
        layer = MLPMemoryLayer(hidden_dim=16, intermediate_dim=24, activation="gelu")
        h = torch.randn(1, 16)
        _ = layer(h, capture=True)
        assert layer._cached_post_act is not None
        assert layer._cached_post_act.shape == (1, 24)

    def test_capture_false_clears_activations(self):
        layer = MLPMemoryLayer(hidden_dim=16, intermediate_dim=24, activation="gelu")
        _ = layer(torch.randn(1, 16), capture=True)
        _ = layer(torch.randn(1, 16), capture=False)
        assert layer._cached_post_act is None


class TestMLPMemory:
    def test_forward_logits_shape(self, memory, tiny_memory_cfg):
        h = torch.randn(tiny_memory_cfg.hidden_dim)
        logits = memory(h)
        assert logits.shape == (tiny_memory_cfg.vocab_size,)

    def test_forward_batch_logits_shape(self, memory, tiny_memory_cfg):
        h = torch.randn(3, tiny_memory_cfg.hidden_dim)
        logits = memory(h)
        assert logits.shape == (3, tiny_memory_cfg.vocab_size)

    def test_prob_is_distribution(self, memory, tiny_memory_cfg):
        h = torch.randn(tiny_memory_cfg.hidden_dim)
        p = memory.prob(h)
        assert p.shape == (tiny_memory_cfg.vocab_size,)
        assert (p >= 0).all()
        assert p.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_log_prob_matches_prob(self, memory, tiny_memory_cfg):
        h = torch.randn(tiny_memory_cfg.hidden_dim)
        logp = memory.log_prob(h)
        p = memory.prob(h)
        assert torch.allclose(logp.exp(), p, atol=1e-5)

    def test_cached_activations_raises_without_capture(self, memory, tiny_memory_cfg):
        _ = memory(torch.randn(tiny_memory_cfg.hidden_dim), capture_activations=False)
        with pytest.raises(RuntimeError):
            memory.cached_activations()

    def test_cached_activations_available_with_capture(self, memory, tiny_memory_cfg):
        _ = memory(torch.randn(tiny_memory_cfg.hidden_dim), capture_activations=True)
        acts = memory.cached_activations()
        assert len(acts) == tiny_memory_cfg.num_layers
        for a in acts:
            # activations were captured for a batch-1 forward → (1, J)
            assert a.shape[-1] == tiny_memory_cfg.intermediate_dim

    def test_num_layers_and_neurons(self, memory, tiny_memory_cfg):
        assert memory.num_layers() == tiny_memory_cfg.num_layers
        assert memory.num_neurons_per_layer() == tiny_memory_cfg.intermediate_dim

    def test_weight_matrix_shapes(self, memory, tiny_memory_cfg):
        for layer in memory.layers:
            assert layer.W1.shape == (
                tiny_memory_cfg.intermediate_dim,
                tiny_memory_cfg.hidden_dim,
            )
            assert layer.W2.shape == (
                tiny_memory_cfg.hidden_dim,
                tiny_memory_cfg.intermediate_dim,
            )


class TestInterpolatedMemoryLM:
    def test_standalone_returns_memory_probs(self, memory, tiny_memory_cfg):
        wrapped = InterpolatedMemoryLM(memory, llm_logits_fn=None, lam=0.5)
        h = torch.randn(tiny_memory_cfg.hidden_dim)
        p = wrapped(h)
        assert p.shape == (tiny_memory_cfg.vocab_size,)
        assert p.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_interpolation_matches_formula(self, memory, tiny_memory_cfg):
        """p = λ p_M + (1-λ) p_LLM (Eq. 1)."""
        h = torch.randn(tiny_memory_cfg.hidden_dim)
        V = tiny_memory_cfg.vocab_size

        # Dummy LLM that returns fixed logits.
        fixed_logits = torch.randn(V)

        def llm_fn(_ctx):
            return fixed_logits

        lam = 0.3
        wrapped = InterpolatedMemoryLM(memory, llm_logits_fn=llm_fn, lam=lam)
        p = wrapped(h, llm_ctx="dummy")

        mem_p = memory.prob(h)
        llm_p = torch.softmax(fixed_logits, dim=-1)
        expected = lam * mem_p + (1 - lam) * llm_p
        assert torch.allclose(p, expected, atol=1e-6)

    def test_interpolation_boundary_lambda_1(self, memory, tiny_memory_cfg):
        h = torch.randn(tiny_memory_cfg.hidden_dim)
        V = tiny_memory_cfg.vocab_size

        def llm_fn(_):
            return torch.randn(V)

        wrapped = InterpolatedMemoryLM(memory, llm_logits_fn=llm_fn, lam=1.0)
        p = wrapped(h, llm_ctx="x")
        assert torch.allclose(p, memory.prob(h), atol=1e-6)


class TestConfigValidation:
    def test_bad_activation_raises(self):
        with pytest.raises(ValueError):
            MemoryModuleConfig(activation="sigmoid")

    def test_bad_lambda_raises(self):
        with pytest.raises(ValueError):
            MemoryModuleConfig(interpolation_lambda=-0.1)
        with pytest.raises(ValueError):
            MemoryModuleConfig(interpolation_lambda=1.1)
