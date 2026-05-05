"""Replay a dialogue (sessions → turns) through the LLM Operation Selector.

Mirrors the LoCoMo / LongMemEval evaluation pipeline described in
Appendix D: for each turn, an LLM classifies the operation type
(INSERT / MODIFY / DELETE / NONE) and supplies the memory content. We
then apply the decision through MemEditor.apply().

Input JSON format:

    {
      "sessions": [
        {
          "session_id": "s1",
          "turns": [
            {"turn_id": 0, "timestamp": "2024-03-01", "speaker": "user",
             "utterance": "I just moved to Osaka last month."}
          ]
        }
      ]
    }

Because a real LLM API is out of scope for this repo, we ship a stub
`MockLLM` that produces reasonable JSON classifications based on keyword
heuristics (useful for CI / debugging). To plug in a real LLM, pass
`--llm-module my_pkg.my_llm:my_callable`: that callable must take a
prompt string and return a response string.

Run:
    python scripts/replay_dialogue.py \
        --config configs/tiny.yaml \
        --dialogue data/sample_dialogue.json \
        --output-json replay_out.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path
from typing import Callable

import torch

from memedit import MemEditConfig, MemEditor, MemoryTrace
from memedit.data.trace import EditOperation, OperationType
from memedit.models.mlp_memory import MLPMemory
from memedit.operations.selector import OperationSelector, parse_selector_response
from memedit.utils.config_loader import load_config
from memedit.utils.logging_utils import get_logger

_log = get_logger("replay")


# ----------------------------------------------------------------------
# Mock LLM + mock hidden-state encoder
# ----------------------------------------------------------------------


class MockLLM:
    """Keyword-based fallback classifier.

    Not a substitute for a real LLM — it exists so CI and smoke tests can
    run without any model download.
    """

    DELETE_PATTERNS = [
        r"\b(forget|delete|retract|ignore|no longer)\b",
        r"\bdisregard\b",
    ]
    MODIFY_PATTERNS = [
        r"\b(actually|now|instead|moved|changed|updated|correction)\b",
    ]
    INSERT_PATTERNS = [
        r"\b(I am|I'm|I like|I love|I have|my|started|began)\b",
    ]

    def __call__(self, prompt: str) -> str:
        text = prompt.lower()
        # Look for the content line; the prompt puts it inside `Content: "..."`
        m = re.search(r'content:\s*"([^"]*)"', text)
        content = m.group(1) if m else text
        op = "NONE"
        if any(re.search(p, content) for p in self.DELETE_PATTERNS):
            op = "DELETE"
        elif any(re.search(p, content) for p in self.MODIFY_PATTERNS):
            op = "MODIFY"
        elif any(re.search(p, content) for p in self.INSERT_PATTERNS):
            op = "INSERT"
        payload = {
            "operation": op,
            "target_memory": content if op in ("MODIFY", "DELETE") else None,
            "new_memory": content if op in ("INSERT", "MODIFY") else None,
            "reason": f"matched {op} heuristics" if op != "NONE" else "no factual content",
        }
        return json.dumps(payload)


def _import_callable(spec: str) -> Callable[[str], str]:
    """Import `pkg.mod:attr` style spec."""
    mod_name, _, attr = spec.partition(":")
    if not mod_name or not attr:
        raise ValueError(f"Expected 'pkg.mod:attr', got {spec!r}")
    module = importlib.import_module(mod_name)
    return getattr(module, attr)


# ----------------------------------------------------------------------
# Fake hidden-state encoder
# ----------------------------------------------------------------------


class HashHiddenEncoder:
    """Deterministic fake encoder: hashes the utterance → random hidden.

    In a real deployment this is replaced by a forward pass through the
    backbone LLM's layer-at-depth-75%.
    """

    def __init__(self, hidden_dim: int, seed: int = 0):
        self.hidden_dim = hidden_dim
        self.seed = seed

    def __call__(self, text: str) -> torch.Tensor:
        h = hash((self.seed, text)) & 0xFFFFFFFF
        g = torch.Generator().manual_seed(h)
        return torch.randn(self.hidden_dim, generator=g)


def _uniform_target(vocab_size: int) -> torch.Tensor:
    return torch.full((vocab_size,), 1.0 / vocab_size)


def _peaked_target(vocab_size: int, text: str) -> torch.Tensor:
    """Pick a single peak token by hashing the text — deterministic stand-in for
    a real kNN / tokenization-based target."""
    peak = hash(text) % vocab_size
    p = torch.full((vocab_size,), 1e-4)
    p[peak] = 1.0
    return p / p.sum()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--dialogue", type=str, required=True,
                   help="Path to a JSON file with {'sessions': [...]}.")
    p.add_argument("--llm-module", type=str, default=None,
                   help="Import spec 'pkg.mod:attr' for a real LLM callable. "
                        "If omitted, MockLLM is used.")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _build_editor(cfg: MemEditConfig, checkpoint: str | None) -> MemEditor:
    base = MLPMemory(cfg.memory)
    if checkpoint:
        base.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])
    editor = MemEditor(base, cfg)
    editor.set_baseline_from_samples(torch.randn(100, cfg.memory.hidden_dim))
    return editor


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    editor = _build_editor(cfg, args.checkpoint)

    llm_call: Callable[[str], str] = (
        _import_callable(args.llm_module) if args.llm_module else MockLLM()
    )
    selector = OperationSelector(llm_call)

    encoder = HashHiddenEncoder(cfg.memory.hidden_dim, seed=args.seed)

    dialogue = json.loads(Path(args.dialogue).read_text())
    op_counts = {ot.value: 0 for ot in OperationType}
    results: list[dict] = []

    # Map content-string → last-seen trace, so MODIFY/DELETE can refer back.
    known: dict[str, MemoryTrace] = {}

    for sess in dialogue["sessions"]:
        sid = sess["session_id"]
        for turn in sess["turns"]:
            out = selector(
                session_id=sid,
                turn_id=int(turn["turn_id"]),
                timestamp=str(turn.get("timestamp", "")),
                speaker=str(turn.get("speaker", "user")),
                utterance=str(turn["utterance"]),
            )
            op_counts[out.operation.value] += 1

            if out.operation is OperationType.NONE:
                results.append({"session": sid, "turn": turn["turn_id"],
                                "op": "NONE", "reason": out.reason})
                continue

            # Build MemoryTraces from the selector's content fields.
            if out.operation is OperationType.INSERT:
                content = out.new_memory or turn["utterance"]
                trace = MemoryTrace(
                    trace_id=f"{sid}-{turn['turn_id']}",
                    content=content,
                    probe_hidden=encoder(content),
                    target_distribution=_peaked_target(cfg.memory.vocab_size, content),
                )
                r = editor.apply(EditOperation(op_type=OperationType.INSERT, new_memory=trace))
                known[content] = trace
            elif out.operation is OperationType.MODIFY:
                old_content = out.target_memory or ""
                old = known.get(old_content)
                new_content = out.new_memory or turn["utterance"]
                if old is None:
                    # Fallback: treat as INSERT to avoid losing info.
                    trace = MemoryTrace(
                        trace_id=f"{sid}-{turn['turn_id']}",
                        content=new_content,
                        probe_hidden=encoder(new_content),
                        target_distribution=_peaked_target(cfg.memory.vocab_size, new_content),
                    )
                    r = editor.apply(EditOperation(op_type=OperationType.INSERT, new_memory=trace))
                    known[new_content] = trace
                else:
                    new_trace = MemoryTrace(
                        trace_id=old.trace_id,
                        content=new_content,
                        probe_hidden=old.probe_hidden,     # same probe
                        target_distribution=_peaked_target(cfg.memory.vocab_size, new_content),
                    )
                    r = editor.apply(EditOperation(
                        op_type=OperationType.MODIFY,
                        target_memory=old, new_memory=new_trace,
                    ))
                    known.pop(old_content, None)
                    known[new_content] = new_trace
            elif out.operation is OperationType.DELETE:
                old_content = out.target_memory or ""
                old = known.get(old_content)
                if old is None:
                    results.append({"session": sid, "turn": turn["turn_id"],
                                    "op": "DELETE", "skipped": True,
                                    "reason": "target memory not in cache"})
                    continue
                r = editor.apply(EditOperation(
                    op_type=OperationType.DELETE, target_memory=old,
                ))
                known.pop(old_content, None)
            else:
                continue

            results.append({
                "session": sid,
                "turn": turn["turn_id"],
                "op": r.op_type.value,
                "success": bool(r.success),
                "shard_idx": r.shard_idx,
                "kl_before": r.kl_before,
                "kl_after": r.kl_after,
                "message": r.message,
            })
            _log.info("%s/%s  %s  success=%s", sid, turn["turn_id"],
                      r.op_type.value, r.success)

    summary = {
        "op_counts": op_counts,
        "final_stats": editor.stats(),
        "num_applied": len([x for x in results if x.get("success")]),
        "num_total": len(results),
    }
    _log.info("Summary: %s", summary)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(
            {"summary": summary, "results": results}, indent=2,
        ))
        _log.info("Wrote replay results to %s", args.output_json)


if __name__ == "__main__":
    main()
