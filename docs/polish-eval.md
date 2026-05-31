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

## Independent vs Circular Gold (de-circularization)

The reviewed gold (`evals/local/polish_reviewed_gold.jsonl`, 2096 rows) is not one
uniform signal. `assemble_gold.py` records, per row, **how** the verdict was decided
(`gold_source`) and whether that decision is **independent of the guard** under test
(`gold_independent`):

| `gold_source` | independent? | how the verdict was decided |
| --- | --- | --- |
| `audio_human` | ✅ | user listened to the source audio (ground truth) |
| `panel` | ✅ | 508-row blind panel, majority-of-3 |
| `codex_keep` / `codex_reject` | ✅ | codex adjudication |
| `destutter` | ❌ **circular** | `_is_destutter_only` — the guard's own early-accept |
| `despace` | ❌ **circular** | the guard's own ascii re-segmentation exemption |
| `ascii_vocab` | ❌ **circular** | the guard's own lexicon vocab whitelist |

On a circular row the gold was computed by the very function the scoreboard then
replays, so `guard == gold` is near-tautological — it measures self-consistency,
not correctness. About **53% of the gold is circular**, so a single blended number
hides the truth.

`polish_scoreboard.py` therefore splits **every** metric into an independent column
(real signal) and a circular column (self-confirming). The headline that motivated
this split:

```
误杀救回 518/755 (68.6%)
  ├─ 独立金标救回 119/335 (35.5%)   ← real
  └─ 循环金标救回 399/420 (95.0%)   ← self-confirming
```

Read the **independent** column as the guard's true score; treat the circular column
as a consistency check, not evidence.

### Iron rules

1. **Always score on `gold_verdict`.** `polish_scoreboard.py --gold-field gold_verdict`.
   The default field is the weaker `codex_verdict`; the board now prints a warning
   when you score on anything other than `gold_verdict`, and no longer silently falls
   back to `codex_verdict` when a row lacks the requested field.
2. **After any guard or gold-rule change, re-run `python -m evals.assemble_gold`**
   so `gold_source` / `gold_independent` stay in sync, then re-run the scoreboard.
3. **The circular axioms must be audio-verified, not just asserted.** The biggest one
   (destutter→keep) is spot-checked by `python -m evals.verify_destutter_audio`,
   which cuts real audio for sampled rows so the user can confirm none deleted real
   content.

### Honest denominators for model comparison

`python -m evals.divergence_denominator` re-derives what "3.7 胜 63%" is a fraction
of: total compared 33969, divergences 26235 (77.2%), but codex judged only 370
(1.4% of divergences). It also runs a directional-bias self-check showing codex
favors whichever model edited more (92% vs 15% on the two opposite-direction
buckets). `python -m evals.codex_judge_variance` re-judges a subset to put a
self-consistency and position-bias error bar on that single-run win rate.
