import { useState, type ReactNode } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getAuthCheck, getHealth } from "../api/client";
import { getToken, setToken } from "../lib/auth";
import { tr } from "../lib/i18n";

/**
 * Gates the app behind the bearer token on token-protected (non-loopback) binds.
 *
 * Loopback binds report `auth_required: false` and render straight through. When auth is
 * required, a probe against `/api/auth/check` decides: success renders the app, a 401
 * shows a token-entry form. The happy path needs no form -- opening the server's printed
 * `?token=` URL seeds the token automatically; the form is the fallback for a bare URL or
 * a stale/rejected token.
 */
export function AuthGate({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const health = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    staleTime: Infinity,
  });
  const authRequired = health.data?.auth_required ?? false;
  const probe = useQuery({
    queryKey: ["auth-check"],
    queryFn: getAuthCheck,
    enabled: authRequired,
    retry: false,
  });
  const [value, setValue] = useState("");

  if (health.isLoading) {
    return <div className="placeholder">…</div>;
  }
  if (!authRequired || probe.isSuccess) {
    return <>{children}</>;
  }

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const next = value.trim();
    if (!next) return;
    setToken(next);
    qc.invalidateQueries();
    void probe.refetch();
  };

  const rejected = probe.isError && getToken() !== null;

  return (
    <div className="auth-gate">
      <form className="auth-card" onSubmit={submit}>
        <h2>{tr("Authentication required", "需要鉴权")}</h2>
        <p>
          {tr(
            "This server is bound to a network address. Paste the token printed in the server console, or open the URL it printed.",
            "服务器绑定在网络地址。请粘贴控制台打印的 token，或直接打开它打印的带 token 的 URL。",
          )}
        </p>
        <input
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={tr("Paste token", "粘贴 token")}
          autoFocus
        />
        <button type="submit">{tr("Continue", "继续")}</button>
        {rejected && (
          <p className="auth-error">{tr("Token rejected.", "Token 无效。")}</p>
        )}
      </form>
    </div>
  );
}
