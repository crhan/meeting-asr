"""Build an audio-verification page for disputed Polish gold labels.

For every row where the gold says reject but the CURRENT guard would keep (the
"leaks"), this finds the original sentence's timestamp in its project's polish
sidecar, cuts the matching slice out of the project's source audio, and writes a
self-contained interactive HTML so the user can PLAY the real recording, read the
ASR original vs the polish, and rule keep/reject from the ground-truth audio —
instead of trusting ASR text or codex. Verdicts export to a pasteable JSON block.

Audio + page land under evals/local/audio_verify/ (git-ignored). Run:
    python -m evals.audio_verify
"""

from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path

from app.lexicon_store import list_lexicon_correction_rules, list_lexicon_known_texts
from app.models import SentenceSegment
from app.transcript_corrections import (
    _apply_rules_to_text,
    _is_change_type_allowed,
    _polish_guard,
)

from evals._log import log

PROJ = Path.home() / ".local" / "share" / "meeting-asr" / "projects"
LOCAL = Path(__file__).resolve().parent / "local"
GOLD = LOCAL / "polish_reviewed_gold.jsonl"
OUTDIR = LOCAL / "audio_verify"
PAD_MS = 500
VOCAB = list_lexicon_known_texts()


def guard_keep(row: dict) -> bool:
    """True if the current guard would KEEP this row's proposal."""
    o, p = row["original_text"], row["proposed_text"]
    if not _is_change_type_allowed(row.get("change_type", "")):
        return False
    s = [
        SentenceSegment(0, 1000, row.get("previous_text", ""), None, 0),
        SentenceSegment(1000, 2000, o, None, 1),
        SentenceSegment(2000, 3000, row.get("next_text", ""), None, 2),
    ]
    return _polish_guard(1, s, o, p, VOCAB) is None


def sidecar_times(project: str) -> dict[str, tuple[int, int]]:
    """Map each original_text -> (begin_ms, end_ms) from a project's sidecars."""
    out: dict[str, tuple[int, int]] = {}
    for sc in sorted((PROJ / project).glob("tmp/corrections/polish_strict_meta_*.json")):
        try:
            items = json.loads(sc.read_text(encoding="utf-8")).get("items", [])
        except (OSError, json.JSONDecodeError):
            continue
        for it in items:
            text = str(it.get("original_text", "")).strip()
            b, e = it.get("begin_time_ms"), it.get("end_time_ms")
            if text and b is not None and e is not None and text not in out:
                out[text] = (int(b), int(e))
    return out


def cut_clip(project: str, begin_ms: int, end_ms: int, dest: Path) -> bool:
    """Extract [begin-pad, end+pad] from the project audio into an mp3 clip."""
    audio = next((PROJ / project / "audio").glob("audio.*"), None)
    if audio is None:
        return False
    start = max(0, begin_ms - PAD_MS) / 1000
    duration = (end_ms - begin_ms + 2 * PAD_MS) / 1000
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
         "-i", str(audio), "-ac", "1", "-ar", "16000", "-b:a", "64k", str(dest)],
        capture_output=True, text=True,
    )
    return dest.exists() and dest.stat().st_size > 0


def collect_disputed(rows: list[dict]) -> list[dict]:
    """Rows where gold=reject but the current guard keeps — the labels to verify."""
    return [
        r for r in rows
        if r.get("gold_verdict") == "reject" and r.get("_kind", "reject") == "reject"
        and guard_keep(r)
    ]


_LEAK_TITLE = "Polish 争议句音频复核"
_LEAK_INTRO = "（gold 判 reject，但听原音可能是对的）"


def build(
    cases: list[dict], *, title: str = _LEAK_TITLE, intro: str = _LEAK_INTRO
) -> str:
    """Render the interactive verification HTML.

    ``title``/``intro`` let other eval pages (e.g. the destutter spot-check)
    reuse the exact same card UI + verdict export with their own framing,
    instead of duplicating the 40-line page template.
    """
    cards = []
    for i, c in enumerate(cases):
        clip = f"clips/{c['_clip']}" if c.get("_clip") else ""
        player = (f'<audio controls preload="none" src="{clip}"></audio>'
                  if clip else '<em style="color:#c00">无音频(未匹配到时间戳)</em>')
        cards.append(f"""
  <div class="card" data-i="{i}">
    <div class="hd"><b>#{i + 1}</b> <span class="proj">{html.escape(c['source'])}</span></div>
    {player}
    <div class="row"><span class="lbl">ASR 原文</span><div class="txt">{html.escape(c['original_text'])}</div></div>
    <div class="row"><span class="lbl">polish</span><div class="txt prop">{html.escape(c['proposed_text'])}</div></div>
    <div class="row"><span class="lbl">我的判读</span><div class="note">{html.escape(c.get('_note', ''))}</div></div>
    <div class="verdict">
      <label><input type="radio" name="v{i}" value="keep"> 保留(polish 对)</label>
      <label><input type="radio" name="v{i}" value="reject"> 拒绝(polish 改坏)</label>
      <label><input type="radio" name="v{i}" value="both_wrong"> 都错</label>
      <label><input type="radio" name="v{i}" value="unsure"> 拿不准</label>
    </div>
    <div class="row"><span class="lbl">正确文本</span><input class="fix" data-i="{i}" type="text" placeholder="原文和 polish 都错时，填你听到的正确文本"></div>
  </div>""")
    meta = [{"i": i, "source": c["source"], "original": c["original_text"],
             "proposed": c["proposed_text"]} for i, c in enumerate(cases)]
    return (
        _PAGE.replace("__CARDS__", "\n".join(cards))
        .replace("__META__", json.dumps(meta, ensure_ascii=False))
        .replace("__TITLE__", html.escape(title))
        .replace("__INTRO__", html.escape(intro))
    )


def main() -> None:
    """Collect disputed rows, cut their audio, and write the verification page."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "clips").mkdir(exist_ok=True)
    rows = [
        json.loads(line)
        for line in GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = collect_disputed(rows)
    log.info("disputed_found", count=len(cases))

    times_cache: dict[str, dict] = {}
    ok = 0
    for i, c in enumerate(cases):
        proj = c["source"]
        times_cache.setdefault(proj, sidecar_times(proj))
        match = times_cache[proj].get(c["original_text"].strip())
        c["_note"] = _classify(c)
        if not match:
            log.warning("no_timestamp", i=i, proj=proj)
            continue
        clip_name = f"{i:02d}_{proj}.mp3"
        if cut_clip(proj, match[0], match[1], OUTDIR / "clips" / clip_name):
            c["_clip"] = clip_name
            ok += 1
        else:
            log.warning("cut_failed", i=i, proj=proj)

    # The gold snapshot froze the pre-lexicon text (P叉一); show it as the live
    # corrected transcript does (PXE) by replaying current rules on the display
    # text. Timestamp matching above already used the raw text, so do this last.
    rules = list_lexicon_correction_rules()
    for c in cases:
        c["original_text"] = _apply_rules_to_text(c["original_text"], rules)
        c["proposed_text"] = _apply_rules_to_text(c["proposed_text"], rules)

    (OUTDIR / "verify.html").write_text(build(cases), encoding="utf-8")
    log.info("written", cases=len(cases), clips=ok,
             page=str(OUTDIR / "verify.html"))


def _classify(c: dict) -> str:
    """One-line read of why this looks like a stutter/substring artifact."""
    o, p = c["original_text"], c["proposed_text"]
    if "是不是" not in p and "不是不是" in o:
        return "「是不是」子串误匹配：原文是「不是不是」(口吃)，含确认词子串"
    for w in ("可以可以", "我觉得我觉得", "应该应该", "就是就是", "不是不是"):
        if w in o.replace(" ", ""):
            return f"相邻 stutter「{w}」折叠"
    return "需听音频确认"


_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Polish 争议句音频复核</title>
<style>
 body{font-family:-apple-system,system-ui,sans-serif;max-width:820px;margin:0 auto;padding:16px;background:#faf9f7;color:#222}
 h1{font-size:18px} .card{background:#fff;border:1px solid #e3e0db;border-radius:10px;padding:14px;margin:14px 0}
 .hd{margin-bottom:8px} .proj{color:#888;font-size:12px}
 audio{width:100%;margin:6px 0}
 .row{display:flex;gap:10px;margin:6px 0;align-items:flex-start}
 .lbl{flex:0 0 64px;color:#888;font-size:12px;padding-top:2px} .txt{flex:1;line-height:1.5}
 .prop{color:#0a6} .note{flex:1;color:#b26a00;font-size:13px}
 .verdict{margin-top:8px;display:flex;gap:16px;font-size:14px}
 .fix{flex:1;padding:6px 8px;border:1px solid #ccc;border-radius:6px;font-size:14px}
 input.fix:not(:placeholder-shown){border-color:#0a6;background:#f0fff8}
 .bar{position:sticky;bottom:0;background:#fff;border-top:1px solid #ddd;padding:10px;margin-top:16px;display:flex;gap:12px;align-items:center}
 button{padding:8px 14px;border-radius:8px;border:1px solid #0a6;background:#0a6;color:#fff;cursor:pointer}
 #out{width:100%;height:120px;font-family:monospace;font-size:12px;margin-top:8px;display:none}
 #prog{color:#888;font-size:13px}
</style></head><body>
<h1>__TITLE__ <span style="font-size:13px;color:#888">__INTRO__</span></h1>
<p style="color:#666;font-size:13px">点播放听原话 → 看 polish 改得对不对 → 勾选。全部勾完点「导出」把结果贴回给我。</p>
__CARDS__
<div class="bar"><button onclick="exportV()">导出裁定</button><span id="prog"></span></div>
<textarea id="out" readonly></textarea>
<script>
const META=__META__;
function exportV(){
 const res=META.map(m=>{const el=document.querySelector('input[name="v'+m.i+'"]:checked');
   const fx=document.querySelector('.fix[data-i="'+m.i+'"]');
   return {...m, verdict: el?el.value:null,
           corrected: (fx&&fx.value.trim())?fx.value.trim():null};});
 const o=document.getElementById('out'); o.style.display='block';
 o.value=JSON.stringify(res,null,1); o.select();
 try{document.execCommand('copy');}catch(e){}
}
function upd(){const n=META.filter(m=>{
   const v=document.querySelector('input[name="v'+m.i+'"]:checked');
   const fx=document.querySelector('.fix[data-i="'+m.i+'"]');
   return v||(fx&&fx.value.trim());}).length;
 document.getElementById('prog').textContent=n+'/'+META.length+' 已判';}
document.addEventListener('change',upd);upd();
</script></body></html>"""


if __name__ == "__main__":
    main()
