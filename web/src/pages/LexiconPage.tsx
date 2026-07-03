import { useState } from "react";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  deleteLexiconTerm,
  getDisambiguations,
  getHotwords,
  getLexiconStats,
  getLexiconTerm,
  getLexiconTerms,
  setDisambiguation,
  upsertLexiconTerm,
  type LexiconTerm,
} from "../api/client";
import type { Disambiguation } from "../api/client";
import { tr } from "../lib/i18n";
import { confirmDialog } from "../lib/confirm";
import { promptDialog } from "../lib/prompt";
import { Modal } from "../components/Modal";

type Tab = "terms" | "disambiguations" | "hotwords";
type TermStatus = "active" | "inactive" | "all";

export function LexiconPage() {
  const [tab, setTab] = useState<Tab>("terms");
  const { data: stats } = useQuery({ queryKey: ["lex-stats"], queryFn: getLexiconStats });

  return (
    <div>
      <h1>{tr("Lexicon", "纠错词库")}</h1>
      {stats && (
        <div className="subtle mono" style={{ marginBottom: 12 }}>
          {stats.active_terms} {tr("terms", "词条")}
          {stats.inactive_terms > 0 && (
            <span>
              {" "}
              (+{stats.inactive_terms} {tr("inactive", "已停用")})
            </span>
          )}{" "}
          · {stats.aliases} {tr("aliases", "别名")} · {stats.contexts}{" "}
          {tr("contexts", "上下文")} · {stats.hotwords} {tr("hotwords", "热词")}
        </div>
      )}
      <div className="row gap" style={{ marginBottom: 14 }}>
        {(["terms", "disambiguations", "hotwords"] as const).map((t) => (
          <button key={t} className={`chip ${tab === t ? "on" : ""}`} onClick={() => setTab(t)}>
            {t === "terms"
              ? tr("Terms", "词条")
              : t === "disambiguations"
                ? tr("Disambiguations", "消歧")
                : tr("Hotwords", "热词")}
          </button>
        ))}
      </div>
      {tab === "terms" && <TermsTab />}
      {tab === "disambiguations" && <DisambiguationsTab />}
      {tab === "hotwords" && <HotwordsTab />}
    </div>
  );
}

/** Single-form replacement for the old chained prompts: everything visible at once,
 *  and a failed submit keeps what was typed. */
function TermFormModal(props: {
  onSubmit: (body: {
    canonical: string;
    aliases: string[];
    category?: string;
    description?: string;
  }) => void;
  onClose: () => void;
}) {
  const [canonical, setCanonical] = useState("");
  const [aliases, setAliases] = useState("");
  const [category, setCategory] = useState("");
  const [description, setDescription] = useState("");
  const canSubmit = canonical.trim().length > 0;
  const submit = () => {
    if (!canSubmit) return;
    props.onSubmit({
      canonical: canonical.trim(),
      aliases: aliases
        .split(/[,，]/)
        .map((a) => a.trim())
        .filter(Boolean),
      // Empty = omit: the backend preserves existing values on update and
      // falls back to unknown/empty for brand-new terms.
      category: category.trim() || undefined,
      description: description.trim() || undefined,
    });
  };
  return (
    <Modal
      title={tr("Add term", "新增词条")}
      onClose={props.onClose}
      footer={
        <div className="row gap">
          <button className="btn ghost" onClick={props.onClose}>
            {tr("Cancel", "取消")}
          </button>
          <button className="btn primary" disabled={!canSubmit} onClick={submit}>
            {tr("Save", "保存")}
          </button>
        </div>
      }
    >
      <input
        className="search"
        autoFocus
        placeholder={tr("Canonical term…", "标准词…")}
        value={canonical}
        onChange={(e) => setCanonical(e.target.value)}
      />
      <input
        className="search"
        placeholder={tr("Aliases (comma separated, optional)", "别名（逗号分隔，可选）")}
        value={aliases}
        onChange={(e) => setAliases(e.target.value)}
      />
      <input
        className="search"
        placeholder={tr("Category (optional, e.g. person / system)", "类别（可选，如 person / system）")}
        value={category}
        onChange={(e) => setCategory(e.target.value)}
      />
      <textarea
        className="text-edit-area"
        style={{ minHeight: 64 }}
        placeholder={tr("Description (optional)", "描述（可选）")}
        value={description}
        onChange={(e) => setDescription(e.target.value)}
      />
    </Modal>
  );
}

/** Single-form mark/edit-disambiguation dialog. The term field autocompletes from
 *  existing canonical terms so a typo can't 404 after the guidance was typed. */
function DisambigFormModal(props: {
  initial?: { term: string; alias: string; guidance: string };
  onSubmit: (body: { term: string; alias: string; guidance: string }) => void;
  onClose: () => void;
}) {
  const [term, setTerm] = useState(props.initial?.term ?? "");
  const [alias, setAlias] = useState(props.initial?.alias ?? "");
  const [guidance, setGuidance] = useState(props.initial?.guidance ?? "");
  const editing = props.initial != null;
  const termsQuery = useQuery({
    queryKey: ["lex-terms", "", "active"],
    queryFn: () => getLexiconTerms(undefined, "active"),
  });
  const canSubmit = term.trim().length > 0 && alias.trim().length > 0;
  return (
    <Modal
      title={editing ? tr("Edit guidance", "编辑判别指引") : tr("Mark alias ambiguous", "标记歧义别名")}
      onClose={props.onClose}
      footer={
        <div className="row gap">
          <button className="btn ghost" onClick={props.onClose}>
            {tr("Cancel", "取消")}
          </button>
          <button
            className="btn primary"
            disabled={!canSubmit}
            onClick={() =>
              props.onSubmit({ term: term.trim(), alias: alias.trim(), guidance: guidance.trim() })
            }
          >
            {tr("Save", "保存")}
          </button>
        </div>
      }
    >
      <div className="subtle" style={{ marginBottom: 8 }}>
        {tr(
          "The polish LLM resolves an ambiguous alias per sentence using this guidance; empty guidance clears the mark (back to blanket replacement).",
          "润色 LLM 会按语境+指引逐句判别歧义别名；指引留空则清除标记，回到无条件替换。",
        )}
      </div>
      <input
        className="search"
        list="lexicon-canonical-terms"
        placeholder={tr("Term (canonical / id / alias)…", "词条（标准词/ID/别名）…")}
        value={term}
        disabled={editing}
        onChange={(e) => setTerm(e.target.value)}
      />
      <datalist id="lexicon-canonical-terms">
        {(termsQuery.data?.terms ?? []).map((t) => (
          <option key={t.public_id} value={t.canonical} />
        ))}
      </datalist>
      <input
        className="search"
        placeholder={tr("Ambiguous alias…", "歧义别名…")}
        value={alias}
        disabled={editing}
        onChange={(e) => setAlias(e.target.value)}
      />
      <textarea
        className="text-edit-area"
        style={{ minHeight: 96 }}
        autoFocus={editing}
        placeholder={tr(
          "Context guidance for the polish LLM…",
          "给润色 LLM 的语境判别指引…",
        )}
        value={guidance}
        onChange={(e) => setGuidance(e.target.value)}
      />
    </Modal>
  );
}

function TermDetailModal(props: {
  termRef: string;
  editable: boolean;
  onClose: () => void;
}) {
  const { termRef, editable } = props;
  const queryClient = useQueryClient();
  const [disambigFor, setDisambigFor] = useState<{
    term: string;
    alias: string;
    guidance: string;
  } | null>(null);
  const detailQuery = useQuery({
    queryKey: ["lex-term", termRef],
    queryFn: () => getLexiconTerm(termRef),
  });
  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["lex-term", termRef] });
    queryClient.invalidateQueries({ queryKey: ["lex-terms"] });
    queryClient.invalidateQueries({ queryKey: ["lex-stats"] });
    queryClient.invalidateQueries({ queryKey: ["lex-disambig"] });
  };
  const upsertMut = useMutation({ mutationFn: upsertLexiconTerm, onSuccess: invalidate });
  const disambigMut = useMutation({
    mutationFn: setDisambiguation,
    onSuccess: () => {
      setDisambigFor(null);
      invalidate();
    },
  });

  const detail = detailQuery.data;
  const term = detail?.term;

  const editCategory = async () => {
    if (!term) return;
    const value = await promptDialog({
      title: tr("Edit category", "编辑类别"),
      message: tr(`Category of "${term.canonical}":`, `「${term.canonical}」的类别：`),
      defaultValue: term.category,
    });
    if (value?.trim() && value.trim() !== term.category)
      upsertMut.mutate({ canonical: term.canonical, category: value.trim() });
  };
  const editDescription = async () => {
    if (!term) return;
    const value = await promptDialog({
      title: tr("Edit description", "编辑描述"),
      message: tr(`Description of "${term.canonical}":`, `「${term.canonical}」的描述：`),
      defaultValue: term.description,
      multiline: true,
    });
    if (value != null && value.trim() !== term.description)
      upsertMut.mutate({ canonical: term.canonical, description: value.trim() });
  };
  const addAlias = async () => {
    if (!term) return;
    const value = await promptDialog({
      title: tr("Add aliases", "新增别名"),
      message: tr(
        "Aliases / common ASR mistakes (comma separated):",
        "别名或常见 ASR 误识（逗号分隔）：",
      ),
    });
    const aliases = (value ?? "")
      .split(/[,，]/)
      .map((a) => a.trim())
      .filter(Boolean);
    if (aliases.length) upsertMut.mutate({ canonical: term.canonical, aliases });
  };

  return (
    <Modal title={tr("Term detail", "词条详情")} onClose={props.onClose}>
      {detailQuery.isLoading && <div className="placeholder">{tr("Loading…", "加载中…")}</div>}
      {detailQuery.error != null && (
        <div className="error-box">{(detailQuery.error as Error).message}</div>
      )}
      {detail && term && (
        <div className="term-detail">
          <div className="row gap" style={{ alignItems: "baseline" }}>
            <h2 style={{ margin: 0 }}>{term.canonical}</h2>
            <span className="badge">{term.category}</span>
            {term.status !== "active" && (
              <span className="badge state-broken">{tr("inactive", "已停用")}</span>
            )}
          </div>
          <div className="subtle" style={{ margin: "6px 0 10px" }}>
            {term.description || tr("(no description)", "（无描述）")}
          </div>
          {editable && (
            <div className="row gap" style={{ marginBottom: 12 }}>
              <button className="chip" onClick={editCategory} disabled={upsertMut.isPending}>
                {tr("Edit category", "改类别")}
              </button>
              <button className="chip" onClick={editDescription} disabled={upsertMut.isPending}>
                {tr("Edit description", "改描述")}
              </button>
              <button className="chip" onClick={addAlias} disabled={upsertMut.isPending}>
                + {tr("Aliases", "别名")}
              </button>
            </div>
          )}
          <div className="term-detail-section">
            <div className="subtle">{tr("Aliases", "别名")}</div>
            {detail.aliases.length === 0 ? (
              <div className="subtle pad">{tr("No aliases.", "暂无别名。")}</div>
            ) : (
              detail.aliases.map((alias) => (
                <div key={`${alias.alias}:${alias.alias_type}`} className="term-alias-row">
                  <span className="mono">{alias.alias}</span>
                  <span className="subtle mono">{alias.alias_type}</span>
                  {alias.disambiguation ? (
                    <span
                      className="badge crosstalk"
                      title={alias.disambiguation}
                    >
                      {tr("ambiguous", "歧义")}
                    </span>
                  ) : null}
                  {editable && (
                    <button
                      className="chip"
                      style={{ marginLeft: "auto" }}
                      onClick={() =>
                        setDisambigFor({
                          term: term.canonical,
                          alias: alias.alias,
                          guidance: alias.disambiguation ?? "",
                        })
                      }
                    >
                      {alias.disambiguation
                        ? tr("Edit guidance", "编辑指引")
                        : tr("Mark ambiguous", "标歧义")}
                    </button>
                  )}
                </div>
              ))
            )}
          </div>
          <div className="term-detail-section">
            <div className="subtle">
              {tr("Recent correction contexts", "最近纠错上下文")} ({term.context_count})
            </div>
            {detail.contexts.length === 0 ? (
              <div className="subtle pad">{tr("No contexts yet.", "暂无上下文。")}</div>
            ) : (
              detail.contexts.slice(0, 10).map((ctx, index) => (
                <div key={index} className="term-context-row">
                  <span className="mono">
                    <del>{ctx.wrong_text}</del> → <ins>{ctx.corrected_text}</ins>
                  </span>
                  <span className="subtle mono">
                    {ctx.project_id}
                    {ctx.speaker_name ? ` · ${ctx.speaker_name}` : ""}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      )}
      {disambigFor && (
        <DisambigFormModal
          initial={disambigFor}
          onSubmit={(body) => disambigMut.mutate(body)}
          onClose={() => setDisambigFor(null)}
        />
      )}
    </Modal>
  );
}

function TermsTab() {
  const queryClient = useQueryClient();
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<TermStatus>("active");
  const [showAdd, setShowAdd] = useState(false);
  const [detailRef, setDetailRef] = useState<string | null>(null);
  const { data, isLoading } = useQuery({
    queryKey: ["lex-terms", query, status],
    queryFn: () => getLexiconTerms(query || undefined, status),
    // Keep the table on screen while typing instead of flashing "Loading…" per keystroke.
    placeholderData: keepPreviousData,
  });
  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["lex-terms"] });
    queryClient.invalidateQueries({ queryKey: ["lex-stats"] });
    // Term edits change the derived hotwords and disambiguation lists too.
    queryClient.invalidateQueries({ queryKey: ["lex-hotwords"] });
    queryClient.invalidateQueries({ queryKey: ["lex-disambig"] });
  };
  const upsertMut = useMutation({
    mutationFn: upsertLexiconTerm,
    onSuccess: () => {
      setShowAdd(false);
      invalidate();
    },
  });
  const deleteMut = useMutation({
    mutationFn: ({ ref, permanent }: { ref: string; permanent: boolean }) =>
      deleteLexiconTerm(ref, permanent),
    onSuccess: invalidate,
  });
  const restoreMut = useMutation({
    // Reactivating goes through upsert with only status: category/description are
    // preserved server-side (None-preserve semantics).
    mutationFn: (term: LexiconTerm) =>
      upsertLexiconTerm({ canonical: term.canonical, status: "active" }),
    onSuccess: invalidate,
  });

  const deactivate = async (term: LexiconTerm) => {
    if (
      await confirmDialog({
        message: tr(
          `Deactivate "${term.canonical}"? It stops applying to corrections but is recoverable from the "inactive" filter (not physically deleted).`,
          `停用「${term.canonical}」？它将不再参与纠错，可在「已停用」筛选里恢复（不会物理删除）。`,
        ),
        confirmLabel: tr("Deactivate", "停用"),
        danger: true,
      })
    )
      deleteMut.mutate({ ref: term.public_id, permanent: false });
  };
  const purge = async (term: LexiconTerm) => {
    if (
      await confirmDialog({
        message: tr(
          `Permanently delete "${term.canonical}" with its ${term.alias_count} alias(es) and ${term.context_count} learned context(s)? This cannot be undone.`,
          `永久删除「${term.canonical}」及其 ${term.alias_count} 个别名、${term.context_count} 条已学习上下文？此操作不可恢复。`,
        ),
        confirmLabel: tr("Delete permanently", "永久删除"),
        danger: true,
      })
    )
      deleteMut.mutate({ ref: term.public_id, permanent: true });
  };

  return (
    <div>
      <div className="row gap" style={{ marginBottom: 10 }}>
        <input
          className="search"
          style={{ marginBottom: 0, maxWidth: 320 }}
          placeholder={tr("Search terms…", "搜索词条…")}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {(["active", "inactive", "all"] as const).map((s) => (
          <button
            key={s}
            className={`chip ${status === s ? "on" : ""}`}
            onClick={() => setStatus(s)}
          >
            {s === "active" ? tr("Active", "启用中") : s === "inactive" ? tr("Inactive", "已停用") : tr("All", "全部")}
          </button>
        ))}
        <button className="btn" onClick={() => setShowAdd(true)}>
          + {tr("Add term", "新增词条")}
        </button>
      </div>
      {showAdd && (
        <TermFormModal onSubmit={(body) => upsertMut.mutate(body)} onClose={() => setShowAdd(false)} />
      )}
      {detailRef && (
        <TermDetailModal termRef={detailRef} editable onClose={() => setDetailRef(null)} />
      )}
      {isLoading ? (
        <div className="placeholder">{tr("Loading…", "加载中…")}</div>
      ) : (data?.terms ?? []).length === 0 ? (
        <div className="placeholder">
          {query
            ? tr("No terms match the search.", "没有匹配搜索的词条。")
            : tr(
                "No terms yet. Terms are also learned automatically when you accept corrections.",
                "还没有词条。接受纠错提案时也会自动学习词条。",
              )}
        </div>
      ) : (
        <div className="table-scroll">
      <table className="projects">
          <thead>
            <tr>
              <th>{tr("Canonical", "标准词")}</th>
              <th>{tr("Category", "类别")}</th>
              <th>{tr("Aliases", "别名")}</th>
              <th>{tr("Contexts", "上下文")}</th>
              <th>{tr("Status", "状态")}</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {(data?.terms ?? []).map((t) => (
              <tr key={t.public_id} className="clickable" onClick={() => setDetailRef(t.public_id)}>
                <td>{t.canonical}</td>
                <td className="mono subtle">{t.category}</td>
                <td className="mono">
                  {t.alias_count}
                  {t.ambiguous_alias_count > 0 && (
                    <span className="badge crosstalk" style={{ marginLeft: 6 }}>
                      {t.ambiguous_alias_count} {tr("ambiguous", "歧义")}
                    </span>
                  )}
                </td>
                <td className="mono">{t.context_count}</td>
                <td>
                  {t.status === "active" ? (
                    <span className="badge state-done">{tr("active", "启用")}</span>
                  ) : (
                    <span className="badge">{tr("inactive", "停用")}</span>
                  )}
                </td>
                <td onClick={(e) => e.stopPropagation()}>
                  {t.status === "active" ? (
                    <button
                      className="icon-btn"
                      title={tr("Deactivate (recoverable)", "停用（可恢复）")}
                      onClick={() => deactivate(t)}
                    >
                      🗑
                    </button>
                  ) : (
                    <span className="row gap">
                      <button
                        className="chip"
                        disabled={restoreMut.isPending}
                        onClick={() => restoreMut.mutate(t)}
                      >
                        {tr("Restore", "恢复")}
                      </button>
                      <button
                        className="chip danger"
                        disabled={deleteMut.isPending}
                        onClick={() => purge(t)}
                      >
                        {tr("Delete permanently", "永久删除")}
                      </button>
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      )}
    </div>
  );
}

function DisambiguationsTab() {
  const queryClient = useQueryClient();
  const [showForm, setShowForm] = useState(false);
  const [editing, setEditing] = useState<Disambiguation | null>(null);
  const { data, isLoading } = useQuery({
    queryKey: ["lex-disambig"],
    queryFn: getDisambiguations,
  });
  const invalidate = () => {
    // Marking/clearing an ambiguous alias changes the alias's correction routing and the
    // term's ambiguous_alias_count, so refresh the disambiguation list, the terms table, and
    // the header stats together.
    queryClient.invalidateQueries({ queryKey: ["lex-disambig"] });
    queryClient.invalidateQueries({ queryKey: ["lex-terms"] });
    queryClient.invalidateQueries({ queryKey: ["lex-stats"] });
  };
  const mut = useMutation({
    mutationFn: setDisambiguation,
    onSuccess: () => {
      setShowForm(false);
      setEditing(null);
      invalidate();
    },
  });

  if (isLoading) return <div className="placeholder">{tr("Loading…", "加载中…")}</div>;
  return (
    <div>
      <div className="row gap" style={{ marginBottom: 10 }}>
        <button className="btn" onClick={() => setShowForm(true)} disabled={mut.isPending}>
          {tr("Mark alias ambiguous", "标记歧义别名")}
        </button>
      </div>
      {showForm && (
        <DisambigFormModal onSubmit={(body) => mut.mutate(body)} onClose={() => setShowForm(false)} />
      )}
      {editing && (
        <DisambigFormModal
          initial={{ term: editing.canonical, alias: editing.alias, guidance: editing.guidance }}
          onSubmit={(body) => mut.mutate(body)}
          onClose={() => setEditing(null)}
        />
      )}
      {(data ?? []).length === 0 ? (
        <div className="placeholder">
          {tr(
            "No ambiguous aliases. Mark one when a surface form needs per-sentence judgement instead of blanket replacement.",
            "暂无歧义别名。当某个写法需要按语境逐句判别、而不是无条件替换时，在这里标记它。",
          )}
        </div>
      ) : (
        <div className="table-scroll">
      <table className="projects">
          <thead>
            <tr>
              <th>{tr("Alias", "别名")}</th>
              <th>{tr("Canonical", "标准词")}</th>
              <th>{tr("Guidance", "判别指引")}</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {(data ?? []).map((d) => (
              <tr key={`${d.alias} ${d.canonical}`}>
                <td className="mono">{d.alias}</td>
                <td>{d.canonical}</td>
                <td className="subtle">{d.guidance}</td>
                <td>
                  <button className="btn ghost" onClick={() => setEditing(d)} disabled={mut.isPending}>
                    {tr("Edit", "编辑")}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      )}
    </div>
  );
}

function HotwordsTab() {
  const { data, isLoading } = useQuery({ queryKey: ["lex-hotwords"], queryFn: getHotwords });
  if (isLoading) return <div className="placeholder">{tr("Loading…", "加载中…")}</div>;
  if ((data ?? []).length === 0)
    return (
      <div className="placeholder">
        {tr(
          "No hotwords yet; they are derived from accepted corrections.",
          "暂无热词；热词由已接受的纠错自动衍生。",
        )}
      </div>
    );
  return (
    <div className="table-scroll">
      <table className="projects">
      <thead>
        <tr>
          <th>{tr("Text", "词")}</th>
          <th>{tr("Weight", "权重")}</th>
          <th>{tr("Category", "类别")}</th>
          <th>{tr("Source", "来源")}</th>
        </tr>
      </thead>
      <tbody>
        {(data ?? []).map((h) => (
          <tr key={`${h.text} ${h.source} ${h.category}`}>
            <td>{h.text}</td>
            <td className="mono">{h.weight}</td>
            <td className="mono subtle">{h.category}</td>
            <td className="mono subtle">{h.source}</td>
          </tr>
        ))}
      </tbody>
    </table>
      </div>
  );
}
