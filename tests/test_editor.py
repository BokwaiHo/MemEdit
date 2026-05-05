"""End-to-end tests for memedit.core.editor.MemEditor."""

from __future__ import annotations

import pytest
import torch

from memedit.core.editor import MemEditor
from memedit.data.trace import EditOperation, MemoryTrace, OperationType
from memedit.models.mlp_memory import MLPMemory


@pytest.fixture
def editor(tiny_cfg) -> MemEditor:
    base = MLPMemory(tiny_cfg.memory)
    ed = MemEditor(base, tiny_cfg)
    ed.set_baseline_from_samples(torch.randn(30, tiny_cfg.memory.hidden_dim))
    return ed


class TestEditorSetup:
    def test_editor_has_initial_shards(self, editor, tiny_cfg):
        assert editor.mome.num_shards == tiny_cfg.mome.initial_num_shards

    def test_baseline_is_set(self, editor, tiny_cfg):
        assert editor._baseline_hidden is not None
        assert editor._baseline_hidden.shape == (tiny_cfg.memory.hidden_dim,)

    def test_stats_are_reportable(self, editor):
        s = editor.stats()
        assert "num_shards" in s


class TestEditorCRUD:
    def test_insert_then_query(self, editor, trace_factory):
        trace = trace_factory("t0", peak=7)
        r_ins = editor.insert(trace)
        assert r_ins.shard_idx is not None
        r_q = editor.query(trace)
        assert r_q.success
        assert r_q.footprint is not None

    def test_modify_then_stats(self, editor, trace_factory):
        old = trace_factory("old", peak=3)
        editor.insert(old)
        new = trace_factory("old", peak=19, probe=old.probe_hidden)
        r = editor.modify(old, new)
        assert r.op_type is OperationType.MODIFY

    def test_delete_after_insert(self, editor, trace_factory):
        trace = trace_factory("doomed", peak=11)
        editor.insert(trace)
        r = editor.delete(trace)
        assert r.op_type is OperationType.DELETE
        assert r.kl_after is not None

    def test_predict_returns_distribution(self, editor, tiny_cfg):
        h = torch.randn(tiny_cfg.memory.hidden_dim)
        p = editor.predict(h)
        assert p.shape == (tiny_cfg.memory.vocab_size,)
        assert p.sum().item() == pytest.approx(1.0, abs=1e-4)


class TestEditorApplyDispatch:
    def test_apply_none_is_noop(self, editor):
        r = editor.apply(EditOperation(op_type=OperationType.NONE))
        assert r.op_type is OperationType.NONE
        assert r.success

    def test_apply_insert_dispatches(self, editor, trace_factory):
        op = EditOperation(op_type=OperationType.INSERT, new_memory=trace_factory("x", peak=1))
        r = editor.apply(op)
        assert r.op_type is OperationType.INSERT

    def test_apply_delete_dispatches(self, editor, trace_factory):
        trace = trace_factory("z", peak=4)
        editor.insert(trace)
        op = EditOperation(op_type=OperationType.DELETE, target_memory=trace)
        r = editor.apply(op)
        assert r.op_type is OperationType.DELETE

    def test_apply_modify_dispatches(self, editor, trace_factory):
        old = trace_factory("m", peak=5)
        editor.insert(old)
        new = trace_factory("m", peak=25, probe=old.probe_hidden)
        op = EditOperation(
            op_type=OperationType.MODIFY, target_memory=old, new_memory=new,
        )
        r = editor.apply(op)
        assert r.op_type is OperationType.MODIFY

    def test_apply_query_dispatches(self, editor, trace_factory):
        trace = trace_factory("q", peak=6)
        editor.insert(trace)
        op = EditOperation(op_type=OperationType.QUERY, target_memory=trace)
        r = editor.apply(op)
        assert r.op_type is OperationType.QUERY
        assert r.footprint is not None

    def test_apply_validates_required_fields(self, editor):
        # INSERT without new_memory should raise on validate().
        op = EditOperation(op_type=OperationType.INSERT, new_memory=None)
        with pytest.raises(ValueError):
            editor.apply(op)


class TestEditorExpansionUnderLoad:
    def test_repeated_inserts_may_trigger_expansion(self, tiny_cfg, trace_factory):
        """Stress a single shard with many inserts; expansion should fire."""
        # Force δ very high so expansion kicks in quickly.
        tiny_cfg.mome.expansion_threshold = 0.9
        tiny_cfg.mome.initial_num_shards = 1
        base = MLPMemory(tiny_cfg.memory)
        editor = MemEditor(base, tiny_cfg)
        editor.set_baseline_from_samples(torch.randn(30, tiny_cfg.memory.hidden_dim))

        start_n = editor.mome.num_shards
        # Insert enough keys to push null-space below δ
        for i in range(10):
            editor.insert(trace_factory(f"t{i}", peak=i))
        # At least one expansion should have happened.
        assert editor.mome.num_shards > start_n
