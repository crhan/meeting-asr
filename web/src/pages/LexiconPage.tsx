import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  deleteLexiconTerm,
  getDisambiguations,
  getHotwords,
  getLexiconStats,
  getLexiconTerms,
  upsertLexiconTerm,
} from "../api/client";
import { tr } from "../lib/i18n";

type Tab = "terms" | "disambiguations" | "hotwords";

export function LexiconPage() {
  const [tab, setTab] = useState<Tab>("terms");
  const { data: stats } = useQuery({ queryKey: ["lex-stats"], queryFn: getLexiconStats });

  return (
    <div>
      <h1>{tr("Lexicon", "纠错词库")}</h1>
      {stats && (
        <div className="subtle mono" style={{ marginBottom: 12 }}>
          {stats.active_terms} {tr("terms", "词条")} · {stats.aliases} {tr("aliases", "别名")} ·{" "}
          {stats.contexts} {tr("contexts", "上下文")} · {stats.hotwords} {tr("hotwords", "热词")}
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

function TermsTab() {
  const queryClient = useQueryClient();
  const [query, setQuery] = useState("");
  const { data, isLoading } = useQuery({
    queryKey: ["lex-terms", query],
    queryFn: () => getLexiconTerms(query || undefined),
  });
  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["lex-terms"] });
    queryClient.invalidateQueries({ queryKey: ["lex-stats"] });
  };
  const upsertMut = useMutation({
    mutationFn: upsertLexiconTerm,
    onSuccess: invalidate,
  });
  const deleteMut = useMutation({
    mutationFn: (ref: string) => deleteLexiconTerm(ref),
    onSuccess: invalidate,
  });

  const addTerm = () => {
    const canonical = window.prompt(tr("Canonical term:", "标准词："));
    if (!canonical?.trim()) return;
    const aliasStr = window.prompt(tr("Aliases (comma separated):", "别名（逗号分隔）："), "");
    const aliases = (aliasStr ?? "")
      .split(",")
      .map((a) => a.trim())
      .filter(Boolean);
    upsertMut.mutate({ canonical: canonical.trim(), category: "unknown", aliases });
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
        <button className="btn" onClick={addTerm}>
          + {tr("Add term", "新增词条")}
        </button>
      </div>
      {isLoading ? (
        <div className="placeholder">{tr("Loading…", "加载中…")}</div>
      ) : (
        <table className="projects">
          <thead>
            <tr>
              <th>{tr("Canonical", "标准词")}</th>
              <th>{tr("Category", "类别")}</th>
              <th>{tr("Aliases", "别名")}</th>
              <th>{tr("Contexts", "上下文")}</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {(data?.terms ?? []).map((t) => (
              <tr key={t.public_id}>
                <td>{t.canonical}</td>
                <td className="mono subtle">{t.category}</td>
                <td className="mono">{t.alias_count}</td>
                <td className="mono">{t.context_count}</td>
                <td>
                  <button
                    className="icon-btn"
                    title={tr("Delete", "删除")}
                    onClick={() => {
                      if (window.confirm(tr(`Delete "${t.canonical}"?`, `删除「${t.canonical}」？`)))
                        deleteMut.mutate(t.public_id);
                    }}
                  >
                    🗑
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function DisambiguationsTab() {
  const { data, isLoading } = useQuery({
    queryKey: ["lex-disambig"],
    queryFn: getDisambiguations,
  });
  if (isLoading) return <div className="placeholder">{tr("Loading…", "加载中…")}</div>;
  return (
    <table className="projects">
      <thead>
        <tr>
          <th>{tr("Alias", "别名")}</th>
          <th>{tr("Canonical", "标准词")}</th>
          <th>{tr("Guidance", "判别指引")}</th>
        </tr>
      </thead>
      <tbody>
        {(data ?? []).map((d, i) => (
          <tr key={i}>
            <td className="mono">{d.alias}</td>
            <td>{d.canonical}</td>
            <td className="subtle">{d.guidance}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function HotwordsTab() {
  const { data, isLoading } = useQuery({ queryKey: ["lex-hotwords"], queryFn: getHotwords });
  if (isLoading) return <div className="placeholder">{tr("Loading…", "加载中…")}</div>;
  return (
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
        {(data ?? []).map((h, i) => (
          <tr key={i}>
            <td>{h.text}</td>
            <td className="mono">{h.weight}</td>
            <td className="mono subtle">{h.category}</td>
            <td className="mono subtle">{h.source}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
