"""LLM-prompt-based Operation Selector (Appendix D).

For benchmarks where edit operations are not pre-annotated (LoCoMo,
LongMemEval), the paper prompts the backbone LLM once per dialogue turn
to classify the turn into one of {INSERT, MODIFY, DELETE, NONE} and
identify the target memory entry.

This module provides:

  * `SELECTOR_PROMPT_TEMPLATE`  — the exact prompt template from App. D.
  * `parse_selector_response`   — robust JSON parsing with graceful fallback.
  * `OperationSelector`         — thin wrapper around any callable that
                                  takes a prompt string and returns a
                                  model response string.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from memedit.data.trace import OperationType


SELECTOR_PROMPT_TEMPLATE = """\
You are the memory controller of a personal AI assistant. Given a new
dialogue turn from the ongoing conversation, decide whether and how to update
the assistant's parametric memory.

== New Dialogue Turn ==
Session: {session_id}, Turn: {turn_id}
Timestamp: {timestamp}
Speaker: {speaker}
Content: "{utterance}"

== Task ==
Analyze the new turn and select exactly ONE of the following operations:

INSERT: The turn introduces factual information (personal details,
preferences, events, plans, or stated facts) that is genuinely new--not
present in or inferable from any existing memory entry. Specify the factual
content to memorize.

MODIFY: The turn updates, corrects, or supersedes an existing memory entry.
For example, the user previously said "I live in Tokyo" and now says "I
actually moved to Osaka last month." Specify the previous content to modify
and the corrected content.

DELETE: The turn explicitly retracts, denies, or requests removal of a
previously stated fact. For example, "Please forget what I told you about my
medical history" or "That's no longer true, ignore it." Specify the previous
content to delete.

NONE: The turn is a greeting, acknowledgment, follow-up question, emotional
expression, or casual conversation that does not contain new factual content
worth memorizing.

== Guidelines ==
- Only INSERT genuinely new facts; do not memorize opinions, filler, or
  pleasantries.
- Prefer MODIFY over INSERT when the new information overlaps with or refines
  an existing entry, to avoid duplicates.
- Use DELETE only when there is an explicit retraction or removal request, not
  merely a topic change.
- When uncertain between INSERT and NONE, prefer INSERT to avoid information
  loss.
- When uncertain between MODIFY and INSERT, prefer MODIFY to maintain memory
  consistency.

Respond with a JSON object and nothing else:
{{
  "operation": "<INSERT | MODIFY | DELETE | NONE>",
  "target_memory": "<existing memory contents to modify or delete; null for INSERT and NONE>",
  "new_memory": "<factual content to insert, or the updated content after modification; null for DELETE and NONE>",
  "reason": "<one-sentence justification>"
}}
"""


@dataclass
class SelectorOutput:
    operation: OperationType
    target_memory: Optional[str]
    new_memory: Optional[str]
    reason: str
    raw: str


def _strip_markdown_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # Remove optional language tag then trailing fence.
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def parse_selector_response(raw: str) -> SelectorOutput:
    """Parse the LLM's JSON response robustly.

    Falls back to OperationType.NONE if parsing fails rather than raising —
    better to miss an edit than to crash the pipeline.
    """
    raw = raw or ""
    cleaned = _strip_markdown_fence(raw)
    # Try to find the first {...} block.
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    blob = match.group(0) if match else cleaned

    try:
        data: Dict[str, Any] = json.loads(blob)
    except (ValueError, json.JSONDecodeError):
        return SelectorOutput(OperationType.NONE, None, None, "parse_failed", raw)

    op_name = str(data.get("operation", "NONE")).strip().upper()
    try:
        op = OperationType(op_name)
    except ValueError:
        op = OperationType.NONE

    tgt = data.get("target_memory")
    new = data.get("new_memory") or data.get("new/updated memory")
    reason = str(data.get("reason", ""))
    return SelectorOutput(op, tgt if tgt else None, new if new else None, reason, raw)


class OperationSelector:
    """Adapter around any LLM-call function.

    Usage:
        selector = OperationSelector(lambda prompt: llm.generate(prompt))
        out = selector(session_id="s1", turn_id=3, timestamp="2025-06-01",
                       speaker="user", utterance="I moved to Osaka.")
        # out.operation is an OperationType
    """

    def __init__(self, llm_call: Callable[[str], str],
                 prompt_template: str = SELECTOR_PROMPT_TEMPLATE):
        self.llm_call = llm_call
        self.prompt_template = prompt_template

    def __call__(
        self,
        session_id: str,
        turn_id: int,
        timestamp: str,
        speaker: str,
        utterance: str,
    ) -> SelectorOutput:
        prompt = self.prompt_template.format(
            session_id=session_id, turn_id=turn_id, timestamp=timestamp,
            speaker=speaker, utterance=utterance,
        )
        raw = self.llm_call(prompt)
        return parse_selector_response(raw)
