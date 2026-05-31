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

## 模型对比（记录，不是默认）

Polish 用哪个 DashScope 模型由**用户自己决定**（`config.json` 的
`dashscope.correction_model`，或 `--model` 临时覆盖）。代码内置默认是 `qwen-plus`，
本节只是把跑过的对比沉淀下来供选型参考，**不是把某个模型写死成强制项**。

下面这组数字来自一次全量真实对比（约 33969 句 / 71 个会议项目）的脱敏聚合，
challenger 是 `qwen3.7-max`，baseline 是 `qwen3.6-plus`。原始逐句文本属用户数据，
落在 gitignore 的 `evals/local/`，不进库；这里只留聚合指标和复现方法。

| 维度 | qwen3.7-max vs qwen3.6-plus | 说明 |
| --- | --- | --- |
| 质量 | 3.7 胜 **63.0%** / 3.6 胜 35.7%（codex 跨源盲判 370 句；轻量 120 句 62.5%，方向一致） | 3.7 更激进，多做的编辑里 ~92% 是改进，唯一弱点是偶尔过于保守 |
| 速度 | 3.7 ~**2.2x** 更快 | 3.7 走 generation 端点，3.6 在本仓库被 `resolve_chat_endpoint` 判成 multimodal |
| 成本 | 3.7 ~**3.4x** 更贵 | 贵在 max 档输出单价；绝对值小，约 ¥1/千句会议 |
| 拒绝率 | 3.7 5.4% vs 3.6 3.5% | 3.7 更激进的好编辑撞上长度/protected 过度拒绝护栏——好模型 + guard 救回一起上才最优 |

**别把 63% 当成铁口直断**——`docs` 上一节的诚实分母警告同样适用于这次对比：

- 这 63% 是 codex 判的 **370 句**之上的胜率，而 370 句只占全部分歧（26235 句，
  占比 77.2%）的 **1.4%**。绝大多数分歧没人判过，样本是被抽出来的一小撮。
- codex 作为裁判有**方向偏好**：在两个相反方向的桶上，它偏向「编辑更多」的那个模型
  92% vs 15%。3.7 本来就更激进，这个偏好会系统性地抬高它的胜率。
- 想给这个单次胜率加误差棒，用 `python -m evals.codex_judge_variance`
  （自洽性 + 位置偏置）和 `python -m evals.divergence_denominator`（重算分母 + 方向偏置自检）。

**复现 / 自己选型**：

```bash
# 单模型在本机真实项目上跑 polish，看采纳与分歧
uv run meeting-asr project correct eval-polish --model qwen3.7-max

# 两模型护栏对比 + 分歧导出 + 成本
python -m evals.model_compare
python -m evals.model_cost_compare      # 计时 / token / 成本
python -m evals.codex_quality_judge     # 盲 A/B 跨源判质量
```

结论：质量和速度上 `qwen3.7-max` 更好，代价是更贵；但模型质量与确定性 guard 是**互补**的
（见上文「误杀救回」），换更激进的模型时 guard 的去口癖/保护词救回同样重要。是否切换、
切到哪个，由用户按自己的成本/质量取舍决定。
