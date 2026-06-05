import { useQuery } from "@tanstack/react-query";
import { listProjects, type ProjectSummary } from "../api/client";
import { tr } from "../lib/i18n";

function StateBadge({ project }: { project: ProjectSummary }) {
  const key = project.workflow?.state_key ?? project.status;
  const label = project.workflow?.state ?? project.status;
  return <span className={`badge state-${key}`}>{label}</span>;
}

function formatTime(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

export function ProjectsPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["projects"],
    queryFn: listProjects,
  });

  if (isLoading) {
    return <div className="placeholder">{tr("Loading projects…", "正在加载项目…")}</div>;
  }
  if (error) {
    return (
      <div className="error-box">
        {tr("Failed to load projects: ", "加载项目失败：")}
        {(error as Error).message}
      </div>
    );
  }
  const projects = data?.projects ?? [];

  return (
    <div>
      <h1>{tr("Projects", "项目")}</h1>
      <div className="subtle mono">
        {data?.projects_dir} · {projects.length} {tr("projects", "个项目")}
      </div>
      {projects.length === 0 ? (
        <div className="placeholder">{tr("No projects yet.", "暂无项目。")}</div>
      ) : (
        <table className="projects">
          <thead>
            <tr>
              <th>{tr("ID", "ID")}</th>
              <th>{tr("Title", "标题")}</th>
              <th>{tr("State", "状态")}</th>
              <th>{tr("Meeting time", "会议时间")}</th>
              <th>{tr("Outputs", "产物")}</th>
            </tr>
          </thead>
          <tbody>
            {projects.map((p) => (
              <tr key={p.project_id}>
                <td className="mono">{p.project_id}</td>
                <td>{p.title || tr("(untitled)", "（无标题）")}</td>
                <td>
                  <StateBadge project={p} />
                </td>
                <td className="mono">{formatTime(p.meeting_time)}</td>
                <td className="mono">
                  {p.workflow?.outputs.join(", ") || "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
