import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { useState } from "react";
import { getLang, setLang, tr, type Lang } from "./lib/i18n";
import { ProjectsPage } from "./pages/ProjectsPage";
import { SpeakerReviewPage } from "./pages/SpeakerReviewPage";

function LangToggle() {
  const [lang, setLangState] = useState<Lang>(getLang());
  const toggle = () => {
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

export function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <span className="brand">meeting-asr</span>
        <nav>
          <NavLink to="/projects">{tr("Projects", "项目")}</NavLink>
        </nav>
        <span className="spacer" />
        <LangToggle />
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<Navigate to="/projects" replace />} />
          <Route path="/projects" element={<ProjectsPage />} />
          <Route path="/projects/:ref/speakers" element={<SpeakerReviewPage />} />
          <Route
            path="*"
            element={<div className="placeholder">{tr("Not found", "未找到")}</div>}
          />
        </Routes>
      </main>
    </div>
  );
}
