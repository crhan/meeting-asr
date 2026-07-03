import { NavLink, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { useEffect, useState, type ReactNode } from "react";
import { getLang, setLang, tr, type Lang } from "./lib/i18n";
import { subscribeGlobalError } from "./lib/globalError";
import { hasUnsavedEdits } from "./lib/unsavedGuard";
import { confirmDialog } from "./lib/confirm";
import { AuthGate } from "./components/AuthGate";
import { ConfirmHost } from "./components/ConfirmHost";
import { PromptHost } from "./components/PromptHost";
import { AppVersion } from "./components/AppVersion";
import { PendingCaptureBanner } from "./components/PendingCaptureBanner";
import { CapturePage } from "./pages/CapturePage";
import { CorrectionPage } from "./pages/CorrectionPage";
import { LexiconPage } from "./pages/LexiconPage";
import { ProjectsPage } from "./pages/ProjectsPage";
import { SettingsPage } from "./pages/SettingsPage";
import { SpeakerReviewPage } from "./pages/SpeakerReviewPage";
import { VoiceprintPage } from "./pages/VoiceprintPage";

/**
 * Topbar NavLink that won't silently destroy unsaved speaker-review edits: useBlocker
 * needs a data router (we stay on plain <BrowserRouter>), so guard the app's own nav
 * links instead. confirmDialog is async, so the click is always blocked first and the
 * navigation re-issued programmatically on confirm.
 */
function GuardedNavLink({ to, children }: { to: string; children: ReactNode }) {
  const navigate = useNavigate();
  return (
    <NavLink
      to={to}
      onClick={(e) => {
        // Modified/middle clicks open a new tab and leave this page's state intact.
        if (e.defaultPrevented || e.button !== 0) return;
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
        if (!hasUnsavedEdits()) return;
        e.preventDefault();
        void confirmDialog({
          message: tr(
            "Discard unsaved speaker review edits and leave this page?",
            "放弃未保存的 speaker review 改动并离开此页？",
          ),
          confirmLabel: tr("Discard", "放弃"),
          danger: true,
        }).then((ok) => {
          if (ok) navigate(to);
        });
      }}
    >
      {children}
    </NavLink>
  );
}

function LangToggle() {
  const [lang, setLangState] = useState<Lang>(getLang());
  const toggle = async () => {
    // The reload below would silently destroy unsaved speaker-review edits.
    if (
      hasUnsavedEdits() &&
      !(await confirmDialog({
        message: tr(
          "Discard unsaved speaker review edits and reload?",
          "放弃未保存的 speaker review 改动并刷新？",
        ),
        confirmLabel: tr("Discard", "放弃"),
        danger: true,
      }))
    )
      return;
    const next: Lang = lang === "zh" ? "en" : "zh";
    setLang(next);
    setLangState(next);
    // Re-render the tree: simplest correct approach for a tiny app is a full reload.
    window.location.reload();
  };
  return (
    <button className="lang-toggle" onClick={toggle}>
      {lang === "zh" ? "中文 / EN" : "EN / 中文"}
    </button>
  );
}

/** App-wide error toast fed by the QueryClient's default mutation onError (main.tsx). */
function GlobalErrorToast() {
  const [message, setMessage] = useState<string | null>(null);
  useEffect(() => subscribeGlobalError(setMessage), []);
  useEffect(() => {
    if (!message) return;
    const timer = setTimeout(() => setMessage(null), 8000);
    return () => clearTimeout(timer);
  }, [message]);
  if (!message) return null;
  return (
    <div className="toast error" onClick={() => setMessage(null)}>
      {message}
    </div>
  );
}

export function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <span className="brand">meeting-asr</span>
        <nav>
          <GuardedNavLink to="/projects">{tr("Projects", "项目")}</GuardedNavLink>
          <GuardedNavLink to="/voiceprints">{tr("Voiceprints", "声纹库")}</GuardedNavLink>
          <GuardedNavLink to="/lexicon">{tr("Lexicon", "词库")}</GuardedNavLink>
          <GuardedNavLink to="/settings">{tr("Settings", "设置")}</GuardedNavLink>
        </nav>
        <span className="spacer" />
        <AppVersion />
        <LangToggle />
      </header>
      <main className="content">
        <AuthGate>
          <PendingCaptureBanner />
          <Routes>
            <Route path="/" element={<Navigate to="/projects" replace />} />
            <Route path="/projects" element={<ProjectsPage />} />
            <Route path="/projects/:ref/speakers" element={<SpeakerReviewPage />} />
            <Route path="/projects/:ref/capture" element={<CapturePage />} />
            <Route path="/projects/:ref/corrections" element={<CorrectionPage />} />
            <Route path="/voiceprints" element={<VoiceprintPage />} />
            <Route path="/lexicon" element={<LexiconPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route
              path="*"
              element={<div className="placeholder">{tr("Not found", "未找到")}</div>}
            />
          </Routes>
        </AuthGate>
      </main>
      <ConfirmHost />
      <PromptHost />
      <GlobalErrorToast />
    </div>
  );
}
