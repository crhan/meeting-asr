"""Build a human-adjudication Markdown file for the Polish gold standard.

Linus is the final arbiter. This assembles, for every CONTESTED / HIGH-RISK
candidate, all three independent signals side by side:

  * codex (cross-model, always 'reject' in this region by construction)
  * 严尺A — strict downstream-fidelity blind lens
  * 宽尺B — lenient editor-naturalness blind lens

plus full text + neighbour context + the auto-reconciled verdict (majority-of-3,
splits -> reject). He answers a few global "判定方向" (policy levers) and/or marks
per-case `判定[aid]>` lines to override.

Inputs (all under evals/local/, git-ignored):
  audit_joined.json  — d* (172 disputes) panel verdicts {aid,category,kind,A,B}
  hr_joined.json     — h* (336 high-risk) panel verdicts {aid,category,kind,A,B}
  d_text.json        — d* full text + codex_reason
  hr_text.json       — h* full text + codex_reason
  audit_keymap.json  — aid -> {category,kind}

Output: evals/local/POLISH_GOLD_REVIEW.md (git-ignored — it embeds real text).

Run: PYTHONPATH=src .venv/bin/python evals/build_review_file.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

LOCAL = Path(__file__).resolve().parent / "local"
OUT = LOCAL / "POLISH_GOLD_REVIEW.md"

KIND_CN = {"reject": "现在被拦", "accept": "现在放行"}
CATEGORY_CN = {
    "ascii_hallucination": "英文/术语改动",
    "protected_word_deleted": "删了保护词",
    "len_ratio": "删太多(长度骤减)",
    "len_delta": "长度变化过大",
    "cross_sentence_borrow": "疑似借用邻句",
    "control": "本来就放行的(抽样)",
}
_ASCII_RE = re.compile(r"[A-Za-z0-9]+")


def _cat_cn(category: str) -> str:
    """Human-readable Chinese label for a guard category."""
    return CATEGORY_CN.get(category, category)


def _load(name: str) -> object:
    """Load a JSON file from the local dir."""
    return json.loads((LOCAL / name).read_text(encoding="utf-8"))


def _is_subsequence(small: str, big: str) -> bool:
    """True if ``small`` is ``big`` with only deletions (a pure deletion)."""
    it = iter(big)
    return all(ch in it for ch in small)


def _ascii_blob(text: str) -> str:
    """Lowercased concatenation of all ascii runs (spacing/segmentation removed)."""
    return "".join(_ASCII_RE.findall(text)).lower()


def _auto(a: str, b: str) -> tuple[str, str]:
    """Majority-of-3 (codex always reject) -> (verdict, tier)."""
    if a == "keep" and b == "keep":
        return "keep", "T2_both_keep"
    if a == "reject" and b == "reject":
        return "reject", "T3_both_reject"
    return "reject", "T1_split"


def _collect() -> list[dict]:
    """Join panel verdicts with full text into one list of case dicts."""
    keymap = _load("audit_keymap.json")
    d_text = _load("d_text.json")
    hr_text = _load("hr_text.json")
    d_panel = {j["aid"]: j for j in _load("audit_joined.json") if j["aid"].startswith("d")}
    hr_panel = {j["aid"]: j for j in _load("hr_joined.json")}

    cases: list[dict] = []
    for aid, panel in list(d_panel.items()) + list(hr_panel.items()):
        text = d_text.get(aid) or hr_text.get(aid)
        if text is None or not panel.get("A") or not panel.get("B"):
            continue
        a, b = panel["A"]["verdict"], panel["B"]["verdict"]
        verdict, tier = _auto(a, b)
        orig, prop = text["original_text"], text["proposed_text"]
        otok, ptok = _ASCII_RE.findall(orig), _ASCII_RE.findall(prop)
        # Real segmentation-only change: ascii characters identical after removing
        # spaces, but the token boundaries changed (casebycase -> case by case).
        # Exclude cases where ascii is byte-identical (otok == ptok) — those are
        # incidental, the edit is elsewhere.
        despace_eq = (
            _ascii_blob(orig) == _ascii_blob(prop) != "" and otok != ptok
        )
        cases.append(
            {
                "aid": aid,
                "category": keymap[aid]["category"],
                "kind": keymap[aid]["kind"],
                "orig": orig,
                "prop": prop,
                "prev": text.get("previous_text", ""),
                "next": text.get("next_text", ""),
                "codex_reason": text.get("codex_reason", ""),
                "a": panel["A"],
                "b": panel["B"],
                "auto": verdict,
                "tier": tier,
                "is_sub": _is_subsequence(prop, orig),
                "despace_eq": despace_eq,
            }
        )
    return cases


def _full_block(c: dict) -> list[str]:
    """Render one case as a full detail block with a mark line."""
    ctx = []
    if c["prev"]:
        ctx.append(f"上文…{c['prev'][-22:]}")
    if c["next"]:
        ctx.append(f"下文{c['next'][:22]}…")
    return [
        f"#### [{c['aid']}] {_cat_cn(c['category'])} · {KIND_CN.get(c['kind'], c['kind'])} · 自动={c['auto']}",
        f"- 原句: {c['orig']}",
        f"- 润色: {c['prop']}",
        *([f"- 语境: {' ┊ '.join(ctx)}"] if ctx else []),
        f"- codex 说 reject — {c['codex_reason']}",
        f"- 严尺A 说 {c['a']['verdict']}({c['a']['confidence']}) — {c['a']['reason']}",
        f"- 宽尺B 说 {c['b']['verdict']}({c['b']['confidence']}) — {c['b']['reason']}",
        f"判定[{c['aid']}]> ",
        "",
    ]


def _compact_block(c: dict) -> list[str]:
    """Render one case as a compact two-liner with a mark line."""
    return [
        f"- `{c['aid']}` 原:{c['orig'][:32]} → 改:{c['prop'][:32]}",
        f"    codex拦因:{c['codex_reason']} ┊ 严A:{c['a']['verdict']} ┊ 宽B:{c['b']['verdict']}"
        f"  判定[{c['aid']}]> ",
    ]


def _diverse_examples(cases: list[dict], n: int) -> list[dict]:
    """Pick up to n cases spread across distinct categories (round-robin)."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_cat[c["category"]].append(c)
    picked: list[dict] = []
    while len(picked) < n and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat]:
                picked.append(by_cat[cat].pop(0))
                if len(picked) >= n:
                    break
    return picked


def _example_lines(cases: list[dict], pred, n: int) -> list[str]:
    """Pick up to n cases matching pred and render them as short example lines."""
    picked = [c for c in cases if pred(c)][:n]
    out = []
    for c in picked:
        out.append(
            f"  - `{c['aid']}` 原:{c['orig'][:30]} → 改:{c['prop'][:30]}  "
            f"(codex拦因:{c['codex_reason']} ┊ 严A说:{c['a']['verdict']} ┊ 宽B说:{c['b']['verdict']})"
        )
    return out


def _tier_section(title: str, note: str, cases: list[dict], full: bool, lines: list[str]) -> None:
    """Emit one tier section, grouped by category, full or compact."""
    lines.append(f"### {title} （{len(cases)} 条）")
    lines.append(note)
    lines.append("")
    if not cases:
        lines.append("（无）\n")
        return
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_cat[c["category"]].append(c)
    for cat in sorted(by_cat):
        group = sorted(by_cat[cat], key=lambda x: x["aid"])
        lines.append(f"<details><summary>{_cat_cn(cat)} · {len(group)} 条</summary>\n")
        for c in group:
            lines.extend(_full_block(c) if full else _compact_block(c))
        lines.append("</details>\n")


def main() -> None:
    """Assemble and write the human-adjudication Markdown file."""
    cases = _collect()
    tiers: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        tiers[c["tier"]].append(c)
    t1, t2, t3 = tiers["T1_split"], tiers["T2_both_keep"], tiers["T3_both_reject"]
    n_keep = len(t2)
    n_reject = len(t1) + len(t3)

    L: list[str] = []
    L.append("# 润色拦截 — 人工裁定（哥来拍最终的尺子）")
    L.append("")
    L.append("## 这是什么")
    L.append("我们的 guard 会拦掉一部分 LLM 的润色。用户老抱怨“润色总被拒”，")
    L.append("所以要查清楚：被拦的里面，哪些是**拦错了**（本该放行）、哪些是**拦对了**（确实危险）。")
    L.append("")
    L.append(f"这份文件挑出 **{len(cases)} 条有争议或高风险**的润色，")
    L.append("每条都请了三个**互相独立、互相看不到对方判断**的评审看过：")
    L.append("- **codex**：另一个 AI 模型，风格偏严。")
    L.append("- **严尺A**：我另派的评审，立场是“宁可拦错也别放过”。")
    L.append("- **宽尺B**：我另派的评审，立场是“宁可放过也别拦错”。")
    L.append("")
    L.append("每条润色，三个评审各表态 `keep`（该放行）或 `reject`（该拦）。")
    L.append("")
    L.append("## 三个评审怎么读")
    L.append("- **A 和 B 都说 keep** → 基本能确定该放行：连最严的评审都放它，多半是 codex 拦错了。")
    L.append("- **A 和 B 都说 reject** → 确定该拦：连最宽的评审都拦它。")
    L.append("- **A 和 B 打架（一个 keep 一个 reject）** → 真的难判，最需要你来定。")
    L.append("")
    L.append("## 为什么不直接信 codex")
    L.append("本来想直接拿 codex 当标准答案，但实测它太爱拦：")
    L.append("- 在“我和 codex 不一致”的 172 条里，**有 57% 是严尺和宽尺都觉得 codex 拦错了**。")
    L.append("- 高风险那批里，codex 说“该拦”的，严+宽两把尺只认同 ascii 类的 32%、protected 类的 41%。")
    L.append("- 所以直接用 codex 当标准，等于把“润色总被拒”这个毛病当成正确答案，会越修越糟。")
    L.append("")
    L.append("但 codex 也不是没用：真正危险的——把“不是…”删成“是”（否定反转）、")
    L.append("把“我觉得”改成“你觉得”（说话人变了）、整句被换成别的话、凭空造术语（“底码”→“Dima”）——")
    L.append("三个评审都说该拦。这些是真要守住的。")
    L.append("")
    L.append("## 你要做什么")
    L.append("**先答下面“判定方向”的几道选择题**（一次定一类，最省事），剩下不放心的再逐条翻。")
    L.append("- 方向题：在 `你选>` 后写 a / b / c。")
    L.append("- 逐条：在 `判定[aid]>` 后写 keep 或 reject；**留空就是采纳我给的“自动”判断**，只标你不同意的。")
    L.append("- 标完把文件存回原路径告诉我，我据此定稿，作为以后改 guard 的回归标准。")
    L.append("")
    L.append(
        f"我给每条的“自动”判断：A、B 都说 keep 才算 keep；否则算 reject（打架的边界先保守归拒）。"
        f"现在自动判出 **该放行 {n_keep} 条 / 该拦 {n_reject} 条**。"
    )
    L.append("")
    L.append("---")
    L.append("## 第一部分：先答这几道方向题（一次定一类，最省事）")
    L.append("")

    # D1 split default
    L.append(f"### 方向1 — 两个评审打架的边界，默认怎么处理（{len(t1)} 条）")
    L.append("就是上面说的“严尺A 和宽尺B 一个说放、一个说拦”的那批。这种最难判，先定个默认：")
    L.append("- (a) 默认拦掉 [当前]：宁可错拦，安全优先")
    L.append("- (b) 默认放行：能多救回一些被错拦的，但会放过一点危险的")
    L.append("- (c) 这批我自己逐条看")
    L.append("- 几个例子（来自不同类型）：")
    L.extend(_example_lines(_diverse_examples(t1, 5), lambda c: True, 5))
    L.append("你选> ")
    L.append("")

    # D2 protected
    prot = [c for c in cases if c["category"] == "protected_word_deleted"]
    prot_bk = [c for c in prot if c["tier"] == "T2_both_keep"]
    prot_br = [c for c in prot if c["tier"] == "T3_both_reject"]
    L.append(f"### 方向2 — “保护词”被删，怎么算（{len(prot)} 条）")
    L.append("保护词＝“我觉得 / 可能 / 对吧 / 同意”这类。删了它们，纪要就分不清“是决定还是提议、")
    L.append("是共识还是某个人单方面说的”。但同一个词重复说了好几遍、删掉重复的应该没问题。怎么区分：")
    L.append("- (a) 紧挨着的重复（“可以可以”→“可以”）放行；分散在不同句子里各删一个（削弱了语气）拦掉 [我推荐]")
    L.append("- (b) 只要删完还剩至少一个，就放行（前一个 agent 的做法，偏松——正是误判的来源）")
    L.append("- (c) 只要删了一个保护词就拦（现在 guard 的做法，最严——正是被抱怨的痛点）")
    L.append(f"- codex 拦了、但严+宽都说该放的例子（这类共 {len(prot_bk)} 条）：")
    L.extend(_example_lines(prot_bk, lambda c: True, 3))
    L.append(f"- codex 拦对了、三个评审都说该拦的例子（这类共 {len(prot_br)} 条）：")
    L.extend(_example_lines(prot_br, lambda c: True, 3))
    L.append("你选> ")
    L.append("")

    # D3 ascii despacing
    despace = [c for c in cases if c["despace_eq"]]
    L.append(f"### 方向3 — 英文只是加了空格/断词（{len(despace)} 条）")
    L.append("把连在一起的英文拆开：casebycase→case by case、把 top12343 排版成 Top 12343。")
    L.append("字母数字一个没变，只是断词和空格变了。")
    L.append("- (a) 算放行 [我推荐]：字符没变，是合理排版")
    L.append("- (b) 还是当成乱改、拦掉")
    L.append("- 例子：")
    L.extend(_example_lines(despace, lambda c: True, 4))
    L.append("你选> ")
    L.append("")

    # D4 pure deletion that still deleted meaning
    pure_del_rej = [c for c in cases if c["is_sub"] and c["auto"] == "reject"]
    L.append(f"### 方向4 — 只删字没加字，但删掉了关键内容（{len(pure_del_rej)} 条）")
    L.append("润色只做了删除（没加任何字），但删掉的是要紧的东西：比如")
    L.append("“不是…不合适”删成“是”（意思反了）、删掉“8月份的图”（丢了信息）。")
    L.append("- (a) 拦掉 [我推荐]：纯删除也可能删坏意思")
    L.append("- (b) 只要是纯删除就一律放行（前一个 agent 的做法，会漏掉上面这些）")
    L.append("- 例子（都是三个评审一致说该拦的）：")
    pure_del_examples = sorted(pure_del_rej, key=lambda c: (c["tier"] != "T3_both_reject", c["aid"]))
    L.extend(_example_lines(pure_del_examples, lambda c: True, 5))
    L.append("你选> ")
    L.append("")

    # D5 ascii term/name restoration (the big ascii split driver)
    ascii_restore = [
        c for c in cases
        if c["category"] == "ascii_hallucination" and not c["despace_eq"] and c["tier"] != "T3_both_reject"
    ]
    L.append(f"### 方向5 — 把听不清的音“还原”成英文术语或人名（约 {len(ascii_restore)} 条）")
    L.append("模型把含糊的发音猜成像样的词：店→demo、武一→WuYi、A证→A-Proof、稳→SRE、扣扣→Docker。")
    L.append("严尺A 多半判“凭空造词”拦掉，宽尺B 多半判“合理还原”放行——这是两把尺打架最多的地方。")
    L.append("本质是个取舍：**忠实**（拿不准就别改、保留原样）还是 **可读**（信模型把它还原对了）。")
    L.append("- (a) 一律拦：拿不准就当编造，最安全（代价：少了一些正确的还原）")
    L.append("- (b) 一律放：相信模型还原（代价：还原错了会写进纪要，比如把人名猜错）")
    L.append("- (c) 我逐条看")
    L.append("- 例子：")
    L.extend(_example_lines(ascii_restore, lambda c: True, 5))
    L.append("你选> ")
    L.append("")

    L.append("---")
    L.append("## 第二部分：逐条明细（想覆盖哪条就标哪条）")
    L.append("")
    _tier_section(
        "第一类：两个评审打架的（最需你定，给了全文）",
        "严尺A 和宽尺B 不一致的硬骨头。按类型折叠，展开你想看的那类。",
        t1, True, L,
    )
    _tier_section(
        "第二类：严+宽都说该放、只有 codex 拦的（我认为 codex 拦错了，帮我抽查）",
        "这批我自动判成“放行”。扫一眼有没有看着不对劲的，挑出来标 reject 就行。",
        t2, False, L,
    )
    _tier_section(
        "第三类：三个评审都说该拦的（最稳，略读即可）",
        "codex、严尺、宽尺一致说该拦（否定/人称反转、整句被换、凭空造词、删掉数字人名）。",
        t3, False, L,
    )

    OUT.write_text("\n".join(L), encoding="utf-8")
    cat_tier = Counter((c["category"], c["tier"]) for c in cases)
    print(f"写出 {OUT}")
    print(f"总 {len(cases)} 条：T1 split {len(t1)} / T2 both_keep {len(t2)} / T3 both_reject {len(t3)}")
    print(f"自动判定：keep {n_keep} / reject {n_reject}")
    print(f"方向命中：despace {len(despace)} / 纯删删实义 {len(pure_del_rej)} / ascii还原 {len(ascii_restore)}")
    print("分类×层:")
    for k in sorted(cat_tier):
        print(f"  {k[0]:24s}{k[1]:16s}{cat_tier[k]:4d}")


if __name__ == "__main__":
    main()
