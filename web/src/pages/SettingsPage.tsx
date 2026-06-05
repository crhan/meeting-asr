import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getConfig,
  getDoctor,
  setConfig,
  unsetConfig,
  type ConfigKey,
} from "../api/client";
import { tr } from "../lib/i18n";

type Tab = "config" | "doctor";

export function SettingsPage() {
  const [tab, setTab] = useState<Tab>("config");
  return (
    <div>
      <h1>{tr("Settings", "设置")}</h1>
      <div className="row gap" style={{ margin: "10px 0 16px" }}>
        {(["config", "doctor"] as const).map((t) => (
          <button key={t} className={`chip ${tab === t ? "on" : ""}`} onClick={() => setTab(t)}>
            {t === "config" ? tr("Configuration", "配置") : tr("Diagnostics", "诊断")}
          </button>
        ))}
      </div>
      {tab === "config" ? <ConfigTab /> : <DoctorTab />}
    </div>
  );
}

function ConfigTab() {
  const queryClient = useQueryClient();
  const [reveal, setReveal] = useState(false);
  const { data, isLoading } = useQuery({
    queryKey: ["config", reveal],
    queryFn: () => getConfig(reveal),
  });
  const invalidate = () => queryClient.invalidateQueries({ queryKey: ["config"] });
  const setMut = useMutation({
    mutationFn: ({ key, value }: { key: string; value: string }) => setConfig(key, value),
    onSuccess: invalidate,
  });
  const unsetMut = useMutation({ mutationFn: (key: string) => unsetConfig(key), onSuccess: invalidate });

  const edit = (k: ConfigKey) => {
    const value = window.prompt(tr(`Set ${k.name}:`, `设置 ${k.name}：`), k.value ?? "");
    if (value != null) setMut.mutate({ key: k.name, value });
  };

  if (isLoading) return <div className="placeholder">{tr("Loading…", "加载中…")}</div>;

  return (
    <div>
      <div className="row gap" style={{ marginBottom: 10 }}>
        <span className="subtle mono">{data?.config_file}</span>
        <label className="row gap subtle" style={{ marginLeft: "auto" }}>
          <input type="checkbox" checked={reveal} onChange={(e) => setReveal(e.target.checked)} />
          {tr("Reveal secrets", "显示密钥")}
        </label>
      </div>
      <table className="projects">
        <thead>
          <tr>
            <th>{tr("Key", "键")}</th>
            <th>{tr("Value", "值")}</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {(data?.keys ?? []).map((k) => (
            <tr key={k.name}>
              <td className="mono">
                {k.name}
                {k.secret && <span className="badge" style={{ marginLeft: 6 }}>secret</span>}
              </td>
              <td className="mono">
                {k.is_set ? (k.value ?? "••••••••") : <span className="subtle">{tr("unset", "未设置")}</span>}
              </td>
              <td>
                <button className="btn ghost" onClick={() => edit(k)}>
                  {tr("Edit", "编辑")}
                </button>
                {k.is_set && (
                  <button className="icon-btn" title={tr("Unset", "清除")} onClick={() => unsetMut.mutate(k.name)}>
                    🗑
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DoctorTab() {
  const { data, isLoading, refetch, isFetching } = useQuery({
    queryKey: ["doctor"],
    queryFn: getDoctor,
  });
  if (isLoading) return <div className="placeholder">{tr("Running checks…", "检查中…")}</div>;
  return (
    <div>
      <div className="row gap" style={{ marginBottom: 10 }}>
        <span className={`badge ${data?.ok ? "state-completed" : "state-broken"}`}>
          {data?.ok ? tr("All OK", "全部正常") : tr("Issues found", "发现问题")}
        </span>
        <button className="btn ghost" onClick={() => refetch()} disabled={isFetching}>
          {tr("Re-run", "重跑")}
        </button>
      </div>
      <div className="checks">
        {(data?.checks ?? []).map((c, i) => (
          <div key={i} className="check-row">
            <span className={`status-dot status-${c.status === "ok" ? "matched" : c.status === "warn" ? "mismatch" : "conflict"}`} />
            <div className="check-body">
              <div>
                <strong>{c.name}</strong> <span className="subtle">{c.detail}</span>
              </div>
              {c.fix_prompt && <div className="subtle" style={{ fontSize: 11.5, marginTop: 2 }}>{c.fix_prompt}</div>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
