import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  artifactUrl,
  fetchArtifactText,
  getArtifacts,
  type ProjectArtifact,
} from "../api/client";
import { tr } from "../lib/i18n";
import { Modal } from "./Modal";

const ARTIFACT_LABEL: Record<string, [string, string]> = {
  meeting_summary: ["Meeting summary", "会议纪要"],
  transcript_named_corrected: ["Corrected transcript (named)", "纠错后命名转写"],
  subtitle_named_corrected: ["Corrected subtitle (named)", "纠错后命名字幕"],
  transcript_named: ["Named transcript", "命名转写"],
  subtitle_named: ["Named subtitle", "命名字幕"],
  transcript_speakers: ["Speaker transcript (anonymous)", "匿名说话人转写"],
  transcript_plain: ["Plain transcript", "纯文本转写"],
  subtitle_plain: ["Subtitle", "字幕"],
};

function artifactLabel(name: string): string {
  const pair = ARTIFACT_LABEL[name];
  return pair ? tr(pair[0], pair[1]) : name;
}

function fmtBytes(size: number): string {
  if (size >= 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  if (size >= 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${size} B`;
}

/** Final-deliverable viewer: list a project's export artifacts, preview + download. */
export function ExportsModal({ projectRef, onClose }: { projectRef: string; onClose: () => void }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["artifacts", projectRef],
    queryFn: () => getArtifacts(projectRef),
  });
  const [previewing, setPreviewing] = useState<ProjectArtifact | null>(null);
  const [previewText, setPreviewText] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  useEffect(() => {
    if (!previewing) return;
    let alive = true;
    setPreviewText(null);
    setPreviewError(null);
    fetchArtifactText(projectRef, previewing.name)
      .then((text) => alive && setPreviewText(text))
      .catch((e) => alive && setPreviewError((e as Error).message));
    return () => {
      alive = false;
    };
  }, [previewing, projectRef]);

  return (
    <Modal title={tr("Exports", "导出产物")} onClose={onClose}>
      {isLoading && <div className="placeholder">{tr("Loading…", "加载中…")}</div>}
      {error != null && <div className="error-box">{(error as Error).message}</div>}
      {data && data.artifacts.length === 0 && (
        <div className="placeholder">
          {tr(
            "No exports yet. Run the pipeline (and save the speaker review) first.",
            "还没有产物。先运行管线（并保存 speaker review）。",
          )}
        </div>
      )}
      {data && data.artifacts.length > 0 && (
        <div className="exports-list">
          {data.artifacts.map((artifact) => (
            <div key={artifact.name} className="exports-row">
              <div className="exports-row-main">
                <span>{artifactLabel(artifact.name)}</span>
                <span className="subtle mono">
                  {artifact.file_name} · {fmtBytes(artifact.size_bytes)}
                </span>
              </div>
              <div className="row gap">
                <button
                  className={`chip ${previewing?.name === artifact.name ? "on" : ""}`}
                  onClick={() =>
                    setPreviewing((prev) => (prev?.name === artifact.name ? null : artifact))
                  }
                >
                  {tr("Preview", "预览")}
                </button>
                <a
                  className="chip"
                  href={artifactUrl(projectRef, artifact.name, true)}
                  download={artifact.file_name}
                >
                  {tr("Download", "下载")}
                </a>
              </div>
            </div>
          ))}
        </div>
      )}
      {previewing && (
        <div className="exports-preview">
          {previewError && <div className="error-box">{previewError}</div>}
          {previewText == null && !previewError && (
            <div className="placeholder">{tr("Loading…", "加载中…")}</div>
          )}
          {previewText != null && <pre className="exports-preview-text">{previewText}</pre>}
        </div>
      )}
    </Modal>
  );
}
