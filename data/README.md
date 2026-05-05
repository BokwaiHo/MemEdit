# data/

Small sample inputs so the CLI scripts in `scripts/` work out of the box.

| File | Consumed by | Notes |
| --- | --- | --- |
| `sample_dialogue.json` | `scripts/replay_dialogue.py` | 2 sessions, 7 turns. Covers Insert (new fact), Modify ("actually moved on from Chopin"), and Delete ("please forget..."). |
| `sample_edits.json` | `scripts/evaluate_edits.py` | 5 seed memories + 12 edits spanning Insert/Modify/Delete/Query. |

For real experiments, replace these with LoCoMo / MemBench / LongMemEval
adaptations following the same schemas.
