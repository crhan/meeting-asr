// Minimal bilingual helper mirroring the TUI's tr(en, zh) convention.
//
// Language is chosen from localStorage, else the browser language, defaulting to zh to
// match the project's primary audience. Kept dependency-free and synchronous so it can
// be called inline at render time exactly like the TUI renderer.

export type Lang = "en" | "zh";

function detectLang(): Lang {
  let stored: string | null = null;
  try {
    stored = localStorage.getItem("masr_lang");
  } catch {
    // Storage can be disabled (private browsing / locked-down browser). Fall back to browser
    // language rather than crashing the whole SPA during module initialization.
  }
  if (stored === "en" || stored === "zh") return stored;
  return navigator.language.toLowerCase().startsWith("zh") ? "zh" : "en";
}

let current: Lang = detectLang();

export function getLang(): Lang {
  return current;
}

export function setLang(lang: Lang): void {
  current = lang;
  try {
    localStorage.setItem("masr_lang", lang);
  } catch {
    // Keep the in-memory language for this session even when persistence is unavailable.
  }
}

export function tr(en: string, zh: string): string {
  return current === "zh" ? zh : en;
}
