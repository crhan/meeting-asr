import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  deleteLexiconTerm,
  getDisambiguations,
  getHotwords,
  getLexiconStats,
  getLexiconTerms,
  setDisambiguation,
  upsertLexiconTerm,
} from "../api/client";
import type { Disambiguation } from "../api/client";
import { tr } from "../lib/i18n";
import { confirmDialog } from "../lib/confirm";
import { promptDialog } from "../lib/prompt";

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

  const addTerm = async () => {
    const canonical = await promptDialog({
      title: tr("Add term", "新增词条"),
      message: tr("Canonical term:", "标准词："),
    });
    if (!canonical?.trim()) return;
    const aliasStr = await promptDialog({
      title: tr("Add term", "新增词条"),
      message: tr("Aliases (comma separated):", "别名（逗号分隔）："),
      defaultValue: "",
    });
    const aliases = (aliasStr ?? "")
      .split(",")
      .map((a) => a.trim())
      .filter(Boolean);
    // No category/description: the server preserves existing values on update
    // (previously a hardcoded "unknown" here silently clobbered curated categories).
    upsertMut.mutate({ canonical: canonical.trim(), aliases });
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
                    onClick={async () => {
                      if (
                        await confirmDialog({
                          message: tr(`Delete "${t.canonical}"?`, `删除「${t.canonical}」？`),
                          confirmLabel: tr("Delete", "删除"),
                          danger: true,
                        })
                      )
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
  const queryClient = useQueryClient();
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
    onSuccess: invalidate,
  });

  const add = async () => {
    const ambTitle = tr("Mark alias ambiguous", "标记歧义别名");
    const term = await promptDialog({
      title: ambTitle,
      message: tr("Term (canonical, id, or alias):", "词条（标准词/ID/别名）："),
    });
    if (!term?.trim()) return;
    const alias = await promptDialog({
      title: ambTitle,
      message: tr("Ambiguous alias:", "歧义别名："),
    });
    if (!alias?.trim()) return;
    const guidance = await promptDialog({
      title: ambTitle,
      message: tr(
        "Context guidance for the polish LLM (empty cancels):",
        "给润色 LLM 的语境判别指引（留空则取消）：",
      ),
      multiline: true,
    });
    if (!guidance?.trim()) return;
    mut.mutate({ term: term.trim(), alias: alias.trim(), guidance: guidance.trim() });
  };

  // Edit an existing row by its canonical term; empty guidance clears it back to blanket.
  const edit = async (d: Disambiguation) => {
    const guidance = await promptDialog({
      title: tr("Edit guidance", "编辑判别指引"),
      message: tr(
        `Guidance for '${d.alias}' (empty clears the ambiguity):`,
        `'${d.alias}' 的判别指引（留空则清除歧义标记）：`,
      ),
      defaultValue: d.guidance,
      multiline: true,
    });
    if (guidance == null) return;
    mut.mutate({ term: d.canonical, alias: d.alias, guidance: guidance.trim() });
  };

  if (isLoading) return <div className="placeholder">{tr("Loading…", "加载中…")}</div>;
  return (
    <div>
      <div className="row gap" style={{ marginBottom: 10 }}>
        <button className="btn" onClick={add} disabled={mut.isPending}>
          {tr("Mark alias ambiguous", "标记歧义别名")}
        </button>
      </div>
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
                <button className="btn ghost" onClick={() => edit(d)} disabled={mut.isPending}>
                  {tr("Edit", "编辑")}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
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
  );
}
