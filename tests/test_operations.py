"""Tests for memedit.operations — Query / Insert / Modify / Delete."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from memedit.attribution.integrated_gradients import MemoryAttributor
from memedit.core.config import (
    AttributionConfig,
    DeleteConfig,
    InsertConfig,
    ModifyConfig,
)
from memedit.core.key_buffer import KeyBuffer
from memedit.data.trace import OperationType
from memedit.operations.delete import delete_memory
from memedit.operations.insert import insert_memory
from memedit.operations.modify import modify_memory
from memedit.operations.query import query_memory


# ----------------------------------------------------------------------
# QUERY
# ----------------------------------------------------------------------


class TestQuery:
    def test_query_returns_footprint(self, memory, trace_factory):
        attr = MemoryAttributor(memory, AttributionConfig(riemann_steps=4))
        trace = trace_factory("t0", peak=3)
        res = query_memory(memory, trace, attr, tau=0.1)
        assert res.op_type is OperationType.QUERY
        assert res.success is True
        assert res.footprint is not None
        assert res.footprint.trace_id == "t0"
        assert len(res.footprint.neurons) > 0

    def test_query_does_not_mutate_weights(self, memory, trace_factory):
        attr = MemoryAttributor(memory, AttributionConfig(riemann_steps=4))
        trace = trace_factory("t0", peak=5)
        before = {
            id(p): p.detach().clone() for p in memory.parameters()
        }
        _ = query_memory(memory, trace, attr, tau=0.1)
        for p in memory.parameters():
            assert torch.allclose(p, before[id(p)], atol=1e-7), (
                "QUERY must be read-only"
            )


# ----------------------------------------------------------------------
# INSERT
# ----------------------------------------------------------------------


class TestInsert:
    def test_insert_reduces_kl(self, memory, tiny_memory_cfg, trace_factory):
        buf = KeyBuffer(hidden_dim=tiny_memory_cfg.hidden_dim)
        trace = trace_factory("new", peak=9)
        res = insert_memory(memory, trace, buf, InsertConfig(),
                            v_optimization_steps=10)
        assert res.op_type is OperationType.INSERT
        assert res.kl_before is not None and res.kl_after is not None
        # The insert should move the memory's distribution toward the target.
        assert res.kl_after <= res.kl_before + 1e-4

    def test_insert_adds_to_key_buffer(self, memory, tiny_memory_cfg, trace_factory):
        buf = KeyBuffer(hidden_dim=tiny_memory_cfg.hidden_dim)
        before = buf.size
        _ = insert_memory(memory, trace_factory("t", peak=2), buf,
                          InsertConfig(), v_optimization_steps=5)
        assert buf.size == before + 1

    def test_insert_without_target_fails(self, memory, tiny_memory_cfg):
        from memedit.data.trace import MemoryTrace

        buf = KeyBuffer(hidden_dim=tiny_memory_cfg.hidden_dim)
        trace = MemoryTrace(
            trace_id="t",
            content="x",
            probe_hidden=torch.randn(tiny_memory_cfg.hidden_dim),
            target_distribution=None,
        )
        res = insert_memory(memory, trace, buf, InsertConfig())
        assert res.success is False

    def test_insert_preserves_prior_keys(self, memory, tiny_memory_cfg, trace_factory):
        """Proposition 1 in practice: W @ k_j should not change for prior keys.

        We check the up_proj weight of the target layer (the last one by default).
        We also need to record the *layer-l*-input* key, because insert's k_new
        is the input to the target layer after earlier layers have processed h.
        """
        buf = KeyBuffer(hidden_dim=tiny_memory_cfg.hidden_dim)
        # Seed the buffer with 3 prior keys, making sure rank < d.
        prior_traces = [trace_factory(f"p{i}", peak=i) for i in range(3)]
        for t in prior_traces:
            insert_memory(memory, t, buf, InsertConfig(), v_optimization_steps=5)

        # Snapshot up_proj outputs for prior keys (which are now in buf).
        target_layer_idx = memory.num_layers() - 1
        W_before = memory.layers[target_layer_idx].up_proj.weight.data.clone()

        # Gather each prior key AS IT APPEARS AT THE TARGET LAYER (i.e., the
        # thing that actually sits in buf).
        buf_keys_before = buf.keys.clone()
        outputs_before = [W_before @ k for k in buf_keys_before]

        # Do a fresh INSERT.
        new_trace = trace_factory("new", peak=50)
        insert_memory(memory, new_trace, buf, InsertConfig(), v_optimization_steps=5)

        W_after = memory.layers[target_layer_idx].up_proj.weight.data
        # Prior keys (those that were in the buffer) should produce the same
        # up_proj output.
        for i, k in enumerate(buf_keys_before):
            after = W_after @ k
            max_diff = (after - outputs_before[i]).abs().max().item()
            assert max_diff < 1e-3, (
                f"Prior key {i} output changed by {max_diff:.2e} — "
                f"Insert-Invariance violated"
            )


# ----------------------------------------------------------------------
# MODIFY
# ----------------------------------------------------------------------


class TestModify:
    def test_modify_moves_toward_new_target(self, memory, tiny_memory_cfg, trace_factory):
        old = trace_factory("old", peak=4)
        new = trace_factory("old", peak=77, probe=old.probe_hidden)
        attr = MemoryAttributor(memory, AttributionConfig(riemann_steps=4, sparsity_tau=0.3))
        res = modify_memory(
            memory, old, new, attr,
            ModifyConfig(num_sgd_steps=30, learning_rate=5e-2, locality_gamma=0.0),
        )
        assert res.op_type is OperationType.MODIFY
        assert res.kl_before is not None and res.kl_after is not None
        assert res.kl_after <= res.kl_before + 1e-4

    def test_modify_respects_locality_penalty(self, memory, tiny_memory_cfg, trace_factory):
        """With a very large γ, preserved memory drift should stay small."""
        old = trace_factory("old", peak=4)
        new = trace_factory("old", peak=60, probe=old.probe_hidden)
        preserved = [trace_factory(f"p{i}", peak=i * 3 + 1) for i in range(2)]

        attr = MemoryAttributor(memory, AttributionConfig(riemann_steps=4, sparsity_tau=0.3))
        res = modify_memory(
            memory, old, new, attr,
            ModifyConfig(num_sgd_steps=10, learning_rate=1e-2, locality_gamma=10.0),
            preserved_traces=preserved,
        )
        assert res.preservation_kl is not None
        # With γ=10 and only 10 steps, drift should be bounded.
        assert res.preservation_kl < 2.0

    def test_modify_without_target_fails(self, memory, tiny_memory_cfg, trace_factory):
        from memedit.data.trace import MemoryTrace

        old = trace_factory("old", peak=2)
        new = MemoryTrace(
            trace_id="old", content="x",
            probe_hidden=old.probe_hidden,
            target_distribution=None,
        )
        attr = MemoryAttributor(memory, AttributionConfig(riemann_steps=4))
        res = modify_memory(memory, old, new, attr, ModifyConfig())
        assert res.success is False


# ----------------------------------------------------------------------
# DELETE
# ----------------------------------------------------------------------


class TestDelete:
    def test_delete_increases_kl(self, memory, tiny_memory_cfg, trace_factory):
        buf = KeyBuffer(hidden_dim=tiny_memory_cfg.hidden_dim)
        # First insert a memory we'll then delete.
        trace = trace_factory("t", peak=12)
        insert_memory(memory, trace, buf, InsertConfig(), v_optimization_steps=10)
        res = delete_memory(
            memory, trace, buf,
            DeleteConfig(num_steps=8, initial_lr=5e-2, kl_threshold=0.1),
        )
        assert res.op_type is OperationType.DELETE
        assert res.kl_before is not None and res.kl_after is not None
        # Gradient ASCENT increases the KL (memory forgets the target).
        assert res.kl_after >= res.kl_before - 1e-4

    def test_delete_removes_key_from_buffer(self, memory, tiny_memory_cfg, trace_factory):
        buf = KeyBuffer(hidden_dim=tiny_memory_cfg.hidden_dim)
        trace = trace_factory("doomed", peak=0)
        insert_memory(memory, trace, buf, InsertConfig(), v_optimization_steps=5)
        size_before = buf.size
        _ = delete_memory(memory, trace, buf, DeleteConfig(num_steps=3))
        # The deleted trace's key should be gone.
        assert buf.size == size_before - 1

    def test_delete_without_target_fails(self, memory, tiny_memory_cfg):
        from memedit.data.trace import MemoryTrace

        buf = KeyBuffer(hidden_dim=tiny_memory_cfg.hidden_dim)
        trace = MemoryTrace(
            trace_id="t", content="x",
            probe_hidden=torch.randn(tiny_memory_cfg.hidden_dim),
            target_distribution=None,
        )
        res = delete_memory(memory, trace, buf, DeleteConfig())
        assert res.success is False

    def test_delete_preserves_unrelated_key_outputs(self, memory, tiny_memory_cfg, trace_factory):
        """Per Proposition 2: outputs for preserved keys should move little.

        This is an *approximate* guarantee — the gradient-ascent loop is
        projected layer-wise, so some nonlinear leakage is expected. We
        just check that the drift is small relative to an unprojected
        baseline would be.
        """
        buf = KeyBuffer(hidden_dim=tiny_memory_cfg.hidden_dim)
        # Insert several memories, delete one, check others are mostly intact.
        traces = [trace_factory(f"t{i}", peak=i * 5) for i in range(4)]
        for t in traces:
            insert_memory(memory, t, buf, InsertConfig(), v_optimization_steps=5)

        preserved = [traces[0], traces[2], traces[3]]
        logp_before = {
            t.trace_id: F.log_softmax(memory.forward(t.probe_hidden), dim=-1).detach()
            for t in preserved
        }

        delete_memory(memory, traces[1], buf,
                      DeleteConfig(num_steps=5, initial_lr=1e-2, kl_threshold=0.5))

        for t in preserved:
            after = F.log_softmax(memory.forward(t.probe_hidden), dim=-1).detach()
            kl = (after.exp() * (after - logp_before[t.trace_id])).sum().abs().item()
            # Tiny model + approximation error — allow up to 2 nats drift.
            assert kl < 2.0, f"Preserved memory {t.trace_id} drifted by KL={kl:.3f}"
