"""Tests for memedit.operations.selector (Appendix D)."""

from __future__ import annotations

from memedit.data.trace import OperationType
from memedit.operations.selector import (
    OperationSelector,
    SELECTOR_PROMPT_TEMPLATE,
    parse_selector_response,
)


class TestParseSelectorResponse:
    def test_well_formed_insert(self):
        raw = """{
            "operation": "INSERT",
            "target_memory": null,
            "new_memory": "User started learning piano in March 2025",
            "reason": "New factual information about hobbies."
        }"""
        out = parse_selector_response(raw)
        assert out.operation is OperationType.INSERT
        assert out.target_memory is None
        assert "piano" in out.new_memory
        assert "factual" in out.reason.lower()

    def test_well_formed_modify(self):
        raw = """{
            "operation": "MODIFY",
            "target_memory": "User lives in Tokyo",
            "new_memory": "User lives in Osaka",
            "reason": "User reported moving."
        }"""
        out = parse_selector_response(raw)
        assert out.operation is OperationType.MODIFY
        assert out.target_memory == "User lives in Tokyo"
        assert out.new_memory == "User lives in Osaka"

    def test_well_formed_delete(self):
        raw = """{
            "operation": "DELETE",
            "target_memory": "User's medical history",
            "new_memory": null,
            "reason": "Explicit retraction."
        }"""
        out = parse_selector_response(raw)
        assert out.operation is OperationType.DELETE
        assert out.target_memory is not None
        assert out.new_memory is None

    def test_none_operation(self):
        raw = '{"operation": "NONE", "target_memory": null, "new_memory": null, "reason": "greeting"}'
        out = parse_selector_response(raw)
        assert out.operation is OperationType.NONE

    def test_markdown_fenced_json(self):
        raw = """```json
{"operation": "INSERT", "target_memory": null, "new_memory": "x", "reason": "y"}
```"""
        out = parse_selector_response(raw)
        assert out.operation is OperationType.INSERT
        assert out.new_memory == "x"

    def test_plain_markdown_fence(self):
        raw = """```
{"operation": "NONE", "target_memory": null, "new_memory": null, "reason": "r"}
```"""
        out = parse_selector_response(raw)
        assert out.operation is OperationType.NONE

    def test_extra_prose_around_json(self):
        raw = """Here's my classification:
{"operation": "INSERT", "target_memory": null, "new_memory": "fact", "reason": "r"}
Hope that helps!"""
        out = parse_selector_response(raw)
        assert out.operation is OperationType.INSERT
        assert out.new_memory == "fact"

    def test_malformed_json_falls_back_to_none(self):
        raw = "this is not json at all"
        out = parse_selector_response(raw)
        assert out.operation is OperationType.NONE
        assert "parse_failed" in out.reason

    def test_empty_response_safe(self):
        out = parse_selector_response("")
        assert out.operation is OperationType.NONE

    def test_unknown_operation_string_falls_back(self):
        raw = '{"operation": "FROB", "target_memory": null, "new_memory": null, "reason": "r"}'
        out = parse_selector_response(raw)
        assert out.operation is OperationType.NONE

    def test_legacy_field_name(self):
        """The original paper prompt uses 'new/updated memory' — accept both."""
        raw = """{
            "operation": "INSERT",
            "target_memory": null,
            "new/updated memory": "legacy field",
            "reason": "r"
        }"""
        out = parse_selector_response(raw)
        assert out.operation is OperationType.INSERT
        assert out.new_memory == "legacy field"


class TestOperationSelector:
    def test_calls_llm_with_formatted_prompt(self):
        captured = {}

        def fake_llm(prompt: str) -> str:
            captured["prompt"] = prompt
            return '{"operation": "NONE", "target_memory": null, "new_memory": null, "reason": "r"}'

        sel = OperationSelector(fake_llm)
        out = sel(
            session_id="s1",
            turn_id=4,
            timestamp="2025-06-01",
            speaker="user",
            utterance="Hi there",
        )
        assert out.operation is OperationType.NONE
        assert "s1" in captured["prompt"]
        assert "Hi there" in captured["prompt"]
        assert "Turn: 4" in captured["prompt"]

    def test_end_to_end_returns_parsed_output(self):
        def llm(_p):
            return '{"operation":"INSERT","target_memory":null,"new_memory":"loves cats","reason":"r"}'

        out = OperationSelector(llm)(
            session_id="s", turn_id=0, timestamp="t",
            speaker="u", utterance="I love cats",
        )
        assert out.operation is OperationType.INSERT
        assert "cats" in out.new_memory

    def test_prompt_template_contains_key_phrases(self):
        """Spot-check that the template carries the paper's controlling language."""
        for phrase in ("INSERT", "MODIFY", "DELETE", "NONE", "JSON object"):
            assert phrase in SELECTOR_PROMPT_TEMPLATE
