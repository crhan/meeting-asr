# Transcript Polish Eval

Transcript polish is not trusted just because it produced a diff. It must pass a replayable eval set.

## Why This Exists

Recent meeting ingestion showed the same pattern:

1. `project run --polish` produced a large proposal.
2. The downstream note still used the unaccepted transcript.
3. Human review accepted the idea of some fixes, but not the whole diff.

That means polish quality is a product behavior, not a prompt preference. The tool needs cases that say what should be changed, what must stay unchanged, and what must be rejected.

## Eval Shape

Cases live in `evals/polish_cases.jsonl`. Each line is one small transcript span:

- `id`: stable case id.
- `original_text`: raw ASR sentence.
- `expected_decision`: `change`, `no_change`, or `reject`.
- `expected_text`: required output for `change`.
- `proposed_text`: offline candidate proposal used by deterministic eval.
- `change_type`: expected polish operation, such as `dup`, `term`, `filler`, `restart`.
- `previous_text` / `next_text`: optional neighbor context for cross-sentence borrow checks.
- `must_keep`: words that must survive, such as `我觉得`, `可能`, `对吧`.
- `category`, `source`, `rationale`: audit metadata.

`reject` and `no_change` cases are first-class. A polish system that changes everything is broken.

## Run

Offline guard eval:

```bash
uv run meeting-asr project correct eval-polish
```

Show every case:

```bash
uv run meeting-asr project correct eval-polish --show-passed
```

Live model eval:

```bash
uv run meeting-asr project correct eval-polish --model qwen3.6-plus
```

## Current Standard

Good polish:

- removes ASR noise like repeated characters, filler loops, and failed restarts;
- fixes obvious term errors when the source sentence has enough evidence;
- preserves numbers, task ownership, speaker boundaries, timestamps, and domain terms;
- preserves uncertainty and consensus markers, such as `我觉得`, `可能`, `对吧`.

Bad polish:

- rewrites a sentence into a summary;
- borrows words from neighboring timestamps;
- invents English terms from Chinese phonetics without evidence;
- deletes uncertainty markers and turns a proposal into a decision;
- changes unclear ASR into a confident guess.

## Flywheel

Use the eval set as the version gate:

1. Add raw span and current proposal as a case.
2. Mark the gold decision: `change`, `no_change`, or `reject`.
3. Run offline eval.
4. If changing prompt/model/guard, run live model eval.
5. Only trust broader auto-accept after repeated cases pass.

This mirrors the workspace evaluation method: snapshot input, keep goodcase and badcase, replay the same cases across versions, and optimize from failures rather than from vibes.
