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

// ---- Speaker review --------------------------------------------------------

export interface SpeakerSegment {
  sentence_id: number | null;
  begin_time_ms: number;
  end_time_ms: number;
  text: string;
  speaker_id: number | null;
  score: number | null;
  score_status: string | null;
}

export interface MatchPerson {
  person_id: number | null;
  name: string;
  score: number | null;
  person_public_id: string | null;
}

export interface SpeakerMatch {
  best_name: string | null;
  best_score: number | null;
  accepted: boolean;
  threshold: number | null;
  status: string;
  candidates: MatchPerson[];
}

export interface ReviewSpeaker {
  speaker_id: number;
  label: string;
  current_name: string;
  ignored: boolean;
  person_id: number | null;
  person_public_id: string | null;
  status: string;
  crosstalk: boolean;
  segment_count: number;
  duration_ms: number;
  match: SpeakerMatch | null;
  segments: SpeakerSegment[];
}

export interface Person {
  person_id: number;
  name: string;
  public_id: string;
}

export interface ReviewOverview {
  project_id: string;
  title: string;
  project_status: string;
  source_name: string;
  duration_ms: number;
  match_file_exists: boolean;
}

export interface SpeakerReview {
  project_id: string;
  project_dir: string;
  overview: ReviewOverview;
  speakers: ReviewSpeaker[];
  people: Person[];
  allow_correction: boolean;
}

export interface Reassignment {
  sentence_id: number | null;
  begin_time_ms: number;
  end_time_ms: number;
  original_speaker_id: number | null;
  new_speaker_id: number;
}

export interface SaveSpeakerReviewBody {
  mapping: Record<string, string>;
  person_mapping: Record<string, number>;
  person_public_mapping: Record<string, string>;
  ignored_speaker_ids: number[];
  reassignments: Reassignment[];
}

export interface SaveSpeakerReviewResult {
  mapping_path: string;
  transcript_path: string;
  srt_path: string;
  reassigned_count: number;
  deleted_sample_count: number;
  rematch_skipped_reason: string | null;
}

export function getSpeakerReview(ref: string): Promise<SpeakerReview> {
  return api<SpeakerReview>(`/api/speakers/${encodeURIComponent(ref)}`);
}

export function saveSpeakerReview(
  ref: string,
  body: SaveSpeakerReviewBody,
): Promise<SaveSpeakerReviewResult> {
  return api<SaveSpeakerReviewResult>(
    `/api/speakers/${encodeURIComponent(ref)}/save`,
    { method: "POST", body: JSON.stringify(body) },
  );
}

export function clipUrl(ref: string, beginMs: number, endMs: number): string {
  return `/api/projects/${encodeURIComponent(ref)}/clip?begin_ms=${beginMs}&end_ms=${endMs}`;
}

// ---- Voiceprint registry ---------------------------------------------------

export interface VoiceprintPerson {
  person_id: number;
  public_id: string;
  name: string;
  sample_count: number;
  project_count: number;
  embedded_sample_count: number;
  embedding_model_count: number;
  updated_at: string | null;
}

export interface VoiceprintSample {
  index: number;
  sample_id: number;
  public_id: string;
  speaker_public_id: string;
  speaker_name: string;
  project_id: string;
  begin_time_ms: number;
  end_time_ms: number;
  transcript_text: string;
  status: string;
  clip_rel_path: string;
}

export interface VoiceprintLibrary {
  store_dir: string | null;
  people: VoiceprintPerson[];
}

export interface VoiceprintSamples {
  person: VoiceprintPerson;
  samples: VoiceprintSample[];
}

export interface QualitySample {
  sample_public_id: string;
  project_id: string;
  begin_time_ms: number;
  end_time_ms: number;
  transcript_text: string;
  status: string;
  score: number | null;
  label: string;
  reason: string;
}

export interface QualityPerson {
  speaker_id: number;
  public_id: string;
  name: string;
  sample_count: number;
  active_sample_count: number;
  mean_score: number | null;
  stdev_score: number | null;
  suspicious_count: number;
  critical_count: number;
  samples: QualitySample[];
}

export interface QualityReport {
  model: string;
  sample_count: number;
  suspicious_count: number;
  critical_count: number;
  people: QualityPerson[];
}

export const getLibrary = () => api<VoiceprintLibrary>("/api/voiceprints/library");

export const getPersonSamples = (ref: string) =>
  api<VoiceprintSamples>(`/api/voiceprints/people/${encodeURIComponent(ref)}/samples`);

export const getQuality = () => api<QualityReport>("/api/voiceprints/quality");

export const sampleClipUrl = (ref: string, samplePublicId: string) =>
  `/api/voiceprints/people/${encodeURIComponent(ref)}/clips/${encodeURIComponent(samplePublicId)}`;

export const setSampleStatus = (samplePublicId: string, status: string) =>
  api<VoiceprintSample>(
    `/api/voiceprints/samples/${encodeURIComponent(samplePublicId)}/status`,
    { method: "PATCH", body: JSON.stringify({ status }) },
  );

export const deleteSample = (ref: string, index: number) =>
  api<{ deleted_sample_public_id: string }>(
    `/api/voiceprints/people/${encodeURIComponent(ref)}/samples/${index}`,
    { method: "DELETE" },
  );

export const deletePerson = (ref: string) =>
  api<{ deleted_sample_count: number }>(
    `/api/voiceprints/people/${encodeURIComponent(ref)}`,
    { method: "DELETE" },
  );

export const createPerson = (name: string) =>
  api<VoiceprintPerson>("/api/voiceprints/people", {
    method: "POST",
    body: JSON.stringify({ name }),
  });

export const renamePerson = (ref: string, name: string) =>
  api<VoiceprintPerson>(`/api/voiceprints/people/${encodeURIComponent(ref)}`, {
    method: "PATCH",
    body: JSON.stringify({ name }),
  });

export const mergePeople = (fromRef: string, intoRef: string) =>
  api<VoiceprintPerson>("/api/voiceprints/people/merge", {
    method: "POST",
    body: JSON.stringify({ from_ref: fromRef, into_ref: intoRef }),
  });

// ---- Voiceprint capture workflow -------------------------------------------

export interface CaptureClip {
  rel_path: string;
  begin_time_ms: number;
  end_time_ms: number;
  duration_seconds: number;
  text: string;
  selection_score: number;
  selection_reason: string;
  audio_score: number | null;
  audio_reason: string;
  recommended: boolean;
}

export interface CaptureSpeaker {
  speaker_id: number;
  name: string;
  person_public_id: string | null;
  clips: CaptureClip[];
}

export interface CapturePlan {
  project_ref: string;
  target_sample_count: number;
  sample_count: number;
  speakers: CaptureSpeaker[];
}

export interface ScoreChange {
  speaker_id: number;
  label: string;
  before_name: string | null;
  before_score: number | null;
  after_name: string | null;
  after_score: number | null;
  delta: number | null;
  status: string;
  is_critical: boolean;
  is_warning: boolean;
  threshold: number | null;
}

export interface HistoricalProject {
  project_id: string;
  title: string;
  improved: number;
  declined: number;
  changed_best: number;
  warning_count: number;
  critical_count: number;
  risky_changes: ScoreChange[];
}

export interface CaptureResult {
  transaction_id: string;
  captured_count: number;
  embedded_count: number;
  skipped_count: number;
  current_project_id: string;
  current_changes: ScoreChange[];
  current_improved: number;
  current_declined: number;
  current_changed_best: number;
  current_warning: number;
  current_critical: number;
  historical_project_count: number;
  historical_warning_count: number;
  historical_critical_count: number;
  historical_projects: HistoricalProject[];
}

export interface JobStatus {
  id: string;
  kind: string;
  status: string;
  error: string | null;
  result: unknown;
}

export const capturePlan = (ref: string) =>
  api<CapturePlan>(`/api/voiceprints/capture/${encodeURIComponent(ref)}/plan`, {
    method: "POST",
  });

export const captureRun = (ref: string, selectedClipRelPaths: string[]) =>
  api<{ job_id: string }>(`/api/voiceprints/capture/${encodeURIComponent(ref)}/run`, {
    method: "POST",
    body: JSON.stringify({ selected_clip_rel_paths: selectedClipRelPaths }),
  });

export const captureAccept = (txnId: string) =>
  api<{ status: string }>(
    `/api/voiceprints/capture/transactions/${encodeURIComponent(txnId)}/accept`,
    { method: "POST" },
  );

export const captureRollback = (txnId: string) =>
  api<{ status: string }>(
    `/api/voiceprints/capture/transactions/${encodeURIComponent(txnId)}/rollback`,
    { method: "POST" },
  );

export const getJob = (jobId: string) =>
  api<JobStatus>(`/api/jobs/${encodeURIComponent(jobId)}`);

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
