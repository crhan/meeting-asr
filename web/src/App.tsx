import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { useEffect, useState } from "react";
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
          <NavLink to="/projects">{tr("Projects", "项目")}</NavLink>
          <NavLink to="/voiceprints">{tr("Voiceprints", "声纹库")}</NavLink>
          <NavLink to="/lexicon">{tr("Lexicon", "词库")}</NavLink>
          <NavLink to="/settings">{tr("Settings", "设置")}</NavLink>
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
