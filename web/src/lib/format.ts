/** Shared duration formatting for the voiceprint surfaces. */

/** A supply figure: "42s" or "3m 20s". */
export function fmtSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

/** A position on a recording's timeline: "12:07". */
export function fmtClock(ms: number): string {
  const total = Math.round(ms / 1000);
  return `${Math.floor(total / 60)}:${(total % 60).toString().padStart(2, "0")}`;
}
