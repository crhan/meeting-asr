import { useMemo, useState } from "react";
import type { Person, ReviewSpeaker } from "../api/client";
import { tr } from "../lib/i18n";
import { Modal } from "./Modal";

export interface IdentitySelection {
  name: string;
  person_id: number | null;
  person_public_id: string | null;
  ignored: boolean;
}

interface Props {
  speaker: ReviewSpeaker;
  people: Person[];
  onSelect: (selection: IdentitySelection) => void;
  onClose: () => void;
}

/** Bind a speaker to a known voiceprint person, rename freely, or keep anonymous. */
export function IdentityPicker({ speaker, people, onSelect, onClose }: Props) {
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return people;
    return people.filter(
      (p) =>
        p.name.toLowerCase().includes(q) ||
        p.public_id.toLowerCase().includes(q),
    );
  }, [people, query]);

  const match = speaker.match;
  const canAccept = !!match?.best_name && (match?.best_score ?? 0) > 0;

  const pickPerson = (p: Person) =>
    onSelect({
      name: p.name,
      person_id: p.person_id,
      person_public_id: p.public_id || null,
      ignored: false,
    });

  const useFreeName = () => {
    const name = query.trim();
    if (!name) return;
    onSelect({ name, person_id: null, person_public_id: null, ignored: false });
  };

  return (
    <Modal
      title={tr(
        `Identify ${speaker.label}`,
        `指认 ${speaker.label}`,
      )}
      onClose={onClose}
      footer={
        <div className="row gap">
          <button
            className="btn ghost"
            onClick={() =>
              onSelect({
                name: speaker.label,
                person_id: null,
                person_public_id: null,
                ignored: true,
              })
            }
          >
            {tr("Keep anonymous (ignore)", "保持匿名（忽略）")}
          </button>
          {query.trim() && (
            <button className="btn" onClick={useFreeName}>
              {tr(`Use name "${query.trim()}"`, `用名字「${query.trim()}」`)}
            </button>
          )}
        </div>
      }
    >
      {canAccept && (
        <button
          className="match-suggest"
          onClick={() =>
            onSelect({
              name: match!.best_name!,
              person_id:
                match!.candidates.find((c) => c.name === match!.best_name)
                  ?.person_id ?? null,
              person_public_id:
                match!.candidates.find((c) => c.name === match!.best_name)
                  ?.person_public_id ?? null,
              ignored: false,
            })
          }
        >
          <span className="kbd">A</span>
          {tr("Accept voiceprint match:", "接受声纹匹配：")}{" "}
          <strong>{match!.best_name}</strong>
          <span className="score">
            {match!.best_score != null ? match!.best_score.toFixed(3) : ""}
          </span>
        </button>
      )}
      <input
        className="search"
        autoFocus
        placeholder={tr(
          "Search people, or type a new name…",
          "搜索人物，或输入新名字…",
        )}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <div className="people-list">
        {filtered.length === 0 ? (
          <div className="subtle pad">{tr("No matching people.", "没有匹配的人物。")}</div>
        ) : (
          filtered.map((p) => (
            <button
              key={p.person_id}
              className={`person-row ${p.person_id === speaker.person_id ? "current" : ""}`}
              onClick={() => pickPerson(p)}
            >
              <span className="person-name">{p.name}</span>
              {p.public_id && <span className="person-id mono">{p.public_id}</span>}
            </button>
          ))
        )}
      </div>
    </Modal>
  );
}
