// Minimal bilingual helper mirroring the TUI's tr(en, zh) convention.
//
// Language is chosen from localStorage, else the browser language, defaulting to zh to
// match the project's primary audience. Kept dependency-free and synchronous so it can
// be called inline at render time exactly like the TUI renderer.

export type Lang = "en" | "zh";

function detectLang(): Lang {
  const stored = localStorage.getItem("masr_lang");
  if (stored === "en" || stored === "zh") return stored;
  return navigator.language.toLowerCase().startsWith("zh") ? "zh" : "en";
}

let current: Lang = detectLang();

export function getLang(): Lang {
  return current;
}

export function setLang(lang: Lang): void {
  current = lang;
  localStorage.setItem("masr_lang", lang);
}

export function tr(en: string, zh: string): string {
  return current === "zh" ? zh : en;
}
