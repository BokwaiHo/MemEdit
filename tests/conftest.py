"""Shared pytest fixtures for the MemEdit test suite."""

from __future__ import annotations

import pytest
import torch

from memedit.core.config import (
    AttributionConfig,
    DeleteConfig,
    InsertConfig,
    MemEditConfig,
    MemoryModuleConfig,
    ModifyConfig,
    MoMEConfig,
)
from memedit.data.trace import MemoryTrace
from memedit.models.mlp_memory import MLPMemory


@pytest.fixture(autouse=True)
def _fix_seed():
    """Ensure deterministic results across tests."""
    torch.manual_seed(0)
    yield


@pytest.fixture
def tiny_memory_cfg() -> MemoryModuleConfig:
    return MemoryModuleConfig(
        num_layers=2,
        hidden_dim=32,
        intermediate_dim=64,
        vocab_size=128,
        activation="gelu",
        dropout=0.0,
    )


@pytest.fixture
def tiny_cfg(tiny_memory_cfg) -> MemEditConfig:
    return MemEditConfig(
        memory=tiny_memory_cfg,
        attribution=AttributionConfig(riemann_steps=4, sparsity_tau=0.1),
        insert=InsertConfig(),
        modify=ModifyConfig(num_sgd_steps=6, learning_rate=1e-2),
        delete=DeleteConfig(num_steps=4, initial_lr=5e-3, kl_threshold=0.5),
        mome=MoMEConfig(initial_num_shards=2, expansion_threshold=0.3),
        device="cpu",
        dtype="float32",
        seed=0,
    )


@pytest.fixture
def memory(tiny_memory_cfg) -> MLPMemory:
    return MLPMemory(tiny_memory_cfg)


def make_trace(
    tid: str,
    hidden_dim: int,
    vocab_size: int,
    peak: int,
    probe: torch.Tensor | None = None,
) -> MemoryTrace:
    """Helper: build a toy probe + peaked target distribution."""
    h = probe if probe is not None else torch.randn(hidden_dim)
    target = torch.full((vocab_size,), 1e-4)
    target[peak % vocab_size] = 1.0
    target = target / target.sum()
    return MemoryTrace(
        trace_id=tid,
        content=f"fact {tid} -> peak {peak}",
        probe_hidden=h,
        target_distribution=target,
    )


@pytest.fixture
def trace_factory(tiny_memory_cfg):
    """Callable that produces a MemoryTrace for the tiny config."""

    def _make(tid: str, peak: int, probe: torch.Tensor | None = None) -> MemoryTrace:
        return make_trace(
            tid,
            tiny_memory_cfg.hidden_dim,
            tiny_memory_cfg.vocab_size,
            peak,
            probe,
        )

    return _make
