// Typed fetch wrapper + SSE helper for the meeting-asr web API.
//
// All calls go to same-origin /api (the FastAPI server also serves this SPA in
// production; in dev, Vite proxies /api to the server). A bearer token, if present in
// localStorage, is attached -- needed only for non-loopback binds.

export class ApiError extends Error {
  status: number;
  kind: string;
  constructor(status: number, detail: string, kind: string) {
    super(detail);
    this.status = status;
    this.kind = kind;
  }
}

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem("masr_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const res = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...(options.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    let kind = "error";
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
      kind = body.error ?? kind;
    } catch {
      // non-JSON error body; keep statusText
    }
    throw new ApiError(res.status, detail, kind);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// ---- Domain types (mirror the backend pydantic schemas) -------------------

export interface WorkflowState {
  state: string;
  state_key: string;
  next_action: string;
  next_command_short: string;
  outputs: string[];
  missing: string[];
}

export interface ProjectSummary {
  project_id: string;
  title: string;
  status: string;
  meeting_time: string | null;
  created_at: string;
  updated_at: string;
  meeting_keywords: string[];
  path: string;
  workflow: WorkflowState | null;
}

export interface ProjectListResponse {
  projects_dir: string;
  projects: ProjectSummary[];
}

export function listProjects(): Promise<ProjectListResponse> {
  return api<ProjectListResponse>("/api/projects");
}

export function getProject(ref: string): Promise<ProjectSummary> {
  return api<ProjectSummary>(`/api/projects/${encodeURIComponent(ref)}`);
}

// ---- SSE job progress ------------------------------------------------------

export interface ProgressEvent {
  type: string;
  description?: string | null;
  total?: number | null;
  completed?: number | null;
  status?: string;
  stage?: string | null;
  [key: string]: unknown;
}

/** Subscribe to a job's SSE stream; returns an unsubscribe function. */
export function subscribeToJob(
  jobId: string,
  onEvent: (event: ProgressEvent) => void,
): () => void {
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  source.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data) as ProgressEvent);
    } catch {
      // ignore malformed frames
    }
  };
  return () => source.close();
}
