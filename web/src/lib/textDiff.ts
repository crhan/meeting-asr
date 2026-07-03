// Character-level diff for correction review. The TUI's word_diff_segments treats a
// run of consecutive CJK characters as ONE token, which paints a whole clause red for
// a single-character fix; here CJK diffs per character while Latin words/digit runs
// stay whole tokens. Zero dependencies (the artifact CSP forbids CDN imports anyway).

export interface DiffSegment {
  text: string;
  changed: boolean;
}

export interface DiffPair {
  before: DiffSegment[];
  after: DiffSegment[];
}

function tokenize(text: string): string[] {
  // Latin words / digit runs as single tokens, whitespace runs kept, everything else
  // (CJK, punctuation) per character.
  return text.match(/[A-Za-z0-9]+|\s+|[\s\S]/gu) ?? [];
}

function push(segments: DiffSegment[], text: string, changed: boolean): void {
  const last = segments[segments.length - 1];
  if (last && last.changed === changed) last.text += text;
  else segments.push({ text, changed });
}

/** Diff two sentences into per-side segment lists with changed runs marked. */
export function diffPair(before: string, after: string): DiffPair {
  const a = tokenize(before);
  const b = tokenize(after);
  // O(n*m) LCS is fine for sentences; bail out for pathological lengths.
  if (a.length * b.length > 250_000) {
    const changed = before !== after;
    return {
      before: [{ text: before, changed }],
      after: [{ text: after, changed }],
    };
  }
  const n = a.length;
  const m = b.length;
  const dp: number[][] = Array.from({ length: n + 1 }, () =>
    new Array<number>(m + 1).fill(0),
  );
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const beforeSegs: DiffSegment[] = [];
  const afterSegs: DiffSegment[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      push(beforeSegs, a[i], false);
      push(afterSegs, b[j], false);
      i += 1;
      j += 1;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      push(beforeSegs, a[i], true);
      i += 1;
    } else {
      push(afterSegs, b[j], true);
      j += 1;
    }
  }
  for (; i < n; i += 1) push(beforeSegs, a[i], true);
  for (; j < m; j += 1) push(afterSegs, b[j], true);
  return { before: beforeSegs, after: afterSegs };
}
