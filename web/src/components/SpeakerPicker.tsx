import type { ReviewSpeaker } from "../api/client";
import { tr } from "../lib/i18n";
import { Modal } from "./Modal";

interface Props {
  speakers: ReviewSpeaker[];
  currentSpeakerId: number;
  sentencePreview: string;
  title?: string;
  onPick: (speakerId: number) => void;
  onClose: () => void;
}

/** Choose which speaker a sentence should be reassigned to. */
export function SpeakerPicker({
  speakers,
  currentSpeakerId,
  sentencePreview,
  title,
  onPick,
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
      </div>
    </Modal>
  );
}
