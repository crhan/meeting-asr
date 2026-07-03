import type { ReviewSpeaker } from "../api/client";
import { tr } from "../lib/i18n";
import { Modal } from "./Modal";

interface Props {
  speakers: ReviewSpeaker[];
  currentSpeakerId: number;
  sentencePreview: string;
  title?: string;
  onPick: (speakerId: number) => void;
  /** Offered for reassignment (not merge): mint a brand-new speaker as the target,
   *  the web-side rescue for ASR under-split tracks. */
  onCreate?: () => void;
  onClose: () => void;
}

/** Choose which speaker a sentence should be reassigned to. */
export function SpeakerPicker({
  speakers,
  currentSpeakerId,
  sentencePreview,
  title,
  onPick,
  onCreate,
  onClose,
}: Props) {
  return (
    <Modal title={title ?? tr("Reassign sentence to…", "把这句重新指派给…")} onClose={onClose}>
      <div className="sentence-preview">{sentencePreview}</div>
      <div className="people-list">
        {speakers.map((s) => (
          <button
            key={s.speaker_id}
            className={`person-row ${s.speaker_id === currentSpeakerId ? "current" : ""}`}
            disabled={s.speaker_id === currentSpeakerId}
            onClick={() => onPick(s.speaker_id)}
          >
            <span className="person-name">
              {s.current_name || s.label}
              {s.speaker_id === currentSpeakerId && (
                <span className="subtle"> · {tr("current", "当前")}</span>
              )}
            </span>
            <span className="person-id mono">{s.label}</span>
          </button>
        ))}
        {onCreate && (
          <button className="person-row" onClick={onCreate}>
            <span className="person-name">
              + {tr("New speaker (split an under-split track)", "新建 speaker（拆分误合并的轨道）")}
            </span>
          </button>
        )}
      </div>
    </Modal>
  );
}
