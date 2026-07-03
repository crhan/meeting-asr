// Shared display helpers for background-job metadata (JobsCenter + JobProgress).

import { tr } from "./i18n";

const KIND_LABEL: Record<string, [string, string]> = {
  "pipeline-run": ["Pipeline run", "管线转写"],
  "pipeline-summarize": ["Summarize", "生成纪要"],
  "correction-polish": ["Polish", "润色"],
  "voiceprint-capture": ["Voiceprint capture", "声纹采集"],
};

export function jobKindLabel(kind: string): string {
  const pair = KIND_LABEL[kind];
  return pair ? tr(pair[0], pair[1]) : kind;
}

/** Pipeline jobs carry the project_dir absolute path; show its basename. */
export function jobProjectName(projectId: string | null): string | null {
  if (!projectId) return null;
  return projectId.split("/").filter(Boolean).pop() ?? projectId;
}
