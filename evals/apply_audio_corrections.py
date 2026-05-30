"""Apply human audio-review corrections back into project transcripts.

The audio verification page (evals/audio_verify.py) lets the user type the
ground-truth text for rows where both the ASR original and the polish were wrong
(verdict 'both_wrong'). This reads those verdicts and writes each corrected
sentence into its project's sentences_corrected.json, matching the sentence by
its exact original text and replacing ONLY the text field (timestamps/speaker
kept). The raw asr/sentences.json is never touched, so it stays recoverable.

Real meeting text; reads evals/local/. Run:
    python -m evals.apply_audio_corrections
"""

from __future__ import annotations

import json
from pathlib import Path

from evals._log import log

PROJ = Path.home() / ".local" / "share" / "meeting-asr" / "projects"
VERDICTS = Path(__file__).resolve().parent / "local" / "audio_verdicts.json"


def main() -> None:
    """Write each 'corrected' audio verdict into its project transcript."""
    verdicts = json.loads(VERDICTS.read_text(encoding="utf-8"))
    fixes = [v for v in verdicts if v.get("corrected")]
    log.info("start", corrections=len(fixes))

    applied = 0
    for v in fixes:
        path = PROJ / v["source"] / "asr" / "sentences_corrected.json"
        if not path.exists():
            log.warning("no_transcript", proj=v["source"])
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        sentences = payload.get("sentences", [])
        original, corrected = v["original"].strip(), v["corrected"].strip()
        hits = [s for s in sentences if str(s.get("text", "")).strip() == original]
        if len(hits) != 1:
            log.warning("match_ambiguous", proj=v["source"], i=v["i"], hits=len(hits))
            continue
        if hits[0].get("text", "").strip() == corrected:
            log.info("already_applied", proj=v["source"], i=v["i"])
            continue
        raw = hits[0]["text"]
        hits[0]["text"] = raw.replace(original, corrected) if original in raw else corrected
        # Keep the concatenated full_text consistent with the edited sentence.
        if isinstance(payload.get("full_text"), str) and original in payload["full_text"]:
            payload["full_text"] = payload["full_text"].replace(original, corrected)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        applied += 1
        log.info("corrected", proj=v["source"], i=v["i"],
                 was=original[:24], now=corrected[:24])
    log.info("done", applied=applied, of=len(fixes))


if __name__ == "__main__":
    main()
