import { useMemo, useState } from "react";
import type { Person, ReviewSpeaker } from "../api/client";
import { tr } from "../lib/i18n";
import { Modal } from "./Modal";

export interface IdentitySelection {
  name: string;
  person_id: number | null;
  person_public_id: string | null;
  ignored: boolean;
  create_person?: boolean;
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
  const newName = query.trim();
  const filtered = useMemo(() => {
    const q = newName.toLowerCase();
    if (!q) return people;
    return people.filter(
      (p) =>
        p.name.toLowerCase().includes(q) ||
        p.public_id.toLowerCase().includes(q),
    );
  }, [people, newName]);
  const exactPerson = useMemo(() => {
    const q = newName.toLowerCase();
    if (!q) return null;
    return people.find((p) => p.name.trim().toLowerCase() === q) ?? null;
  }, [people, newName]);

  const match = speaker.match;
  const canAccept = !!match?.best_name && (match?.best_score ?? 0) > 0;
  // The runner-up candidates are the key evidence when the best match is wrong;
  // show the top few directly instead of making the user dig through the people list.
  const rankedCandidates = useMemo(() => {
    const rows = (match?.candidates ?? []).filter((c) => c.name);
    return [...rows].sort((a, b) => (b.score ?? 0) - (a.score ?? 0)).slice(0, 3);
  }, [match]);

  const pickPerson = (p: Person) =>
    onSelect({
      name: p.name,
      person_id: p.person_id,
      person_public_id: p.public_id || null,
      ignored: false,
      create_person: false,
    });

  const useFreeName = () => {
    if (!newName) return;
    if (exactPerson) {
      pickPerson(exactPerson);
      return;
    }
    onSelect({
      name: newName,
      person_id: null,
      person_public_id: null,
      ignored: false,
      create_person: true,
    });
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
                create_person: false,
              })
            }
          >
            {tr("Keep anonymous (ignore)", "保持匿名（忽略）")}
          </button>
          {newName && (
            <button className="btn" onClick={useFreeName}>
              {exactPerson
                ? tr(`Use existing person "${newName}"`, `使用已有人物「${newName}」`)
                : tr(`Create person "${newName}"`, `新建人物「${newName}」`)}
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
              create_person: false,
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
      {rankedCandidates.length > 1 && (
        <div className="candidate-list">
          {rankedCandidates.map((candidate) => (
            <button
              key={`${candidate.person_id ?? candidate.name}`}
              className="person-row candidate-row"
              onClick={() =>
                onSelect({
                  name: candidate.name,
                  person_id: candidate.person_id,
                  person_public_id: candidate.person_public_id,
                  ignored: false,
                  create_person: false,
                })
              }
            >
              <span className="person-name">{candidate.name}</span>
              <span className="person-id mono">
                {candidate.score != null ? candidate.score.toFixed(3) : "—"}
                {candidate.name === match?.best_name ? ` · ${tr("best", "最佳")}` : ""}
              </span>
            </button>
          ))}
        </div>
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
        // Enter picks the exact-name person or creates one -- same as the footer
        // button; guarded against IME composition commits.
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.nativeEvent.isComposing) useFreeName();
        }}
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
