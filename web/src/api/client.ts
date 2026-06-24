// Typed fetch wrapper + SSE helper for the meeting-asr web API.
//
// All calls go to same-origin /api (the FastAPI server also serves this SPA in
// production; in dev, Vite proxies /api to the server). A bearer token, if present in
// localStorage, is attached -- needed only for non-loopback binds.

import { getToken, withToken } from "../lib/auth";

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
  const token = getToken();
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
      // FastAPI validation errors (422) carry `detail` as an array of objects; coercing
      // that to a string via template/`String()` yields "[object Object]". Stringify
      // non-string details so error surfaces always show something readable.
      if (typeof body.detail === "string") detail = body.detail;
      else if (body.detail != null) detail = JSON.stringify(body.detail);
      if (typeof body.error === "string") kind = body.error;
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
  review_revision: string;
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
  review_revision: string;
  mapping: Record<string, string>;
  person_mapping: Record<string, number>;
  person_public_mapping: Record<string, string>;
  new_person_names: Record<string, string>;
  ignored_speaker_ids: number[];
  reassignments: Reassignment[];
  deleted_speaker_ids: number[];
}

export interface SaveSpeakerReviewResult {
  mapping_path: string;
  transcript_path: string;
  srt_path: string;
  reassigned_count: number;
  created_person_count: number;
  deleted_speaker_count: number;
  deleted_sentence_count: number;
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
  // <audio> can't set Authorization headers, so carry the token in the query string.
  return withToken(
    `/api/projects/${encodeURIComponent(ref)}/clip?begin_ms=${beginMs}&end_ms=${endMs}`,
  );
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
  withToken(
    `/api/voiceprints/people/${encodeURIComponent(ref)}/clips/${encodeURIComponent(samplePublicId)}`,
  );

export const setSampleStatus = (samplePublicId: string, status: string) =>
  api<VoiceprintSample>(
    `/api/voiceprints/samples/${encodeURIComponent(samplePublicId)}/status`,
    { method: "PATCH", body: JSON.stringify({ status }) },
  );

// Delete by stable public id, not list position: a stale library pane could otherwise resolve
// an index to the wrong row after another tab captured/deleted a sample for this person.
export const deleteSample = (ref: string, samplePublicId: string) =>
  api<{ deleted_sample_public_id: string }>(
    `/api/voiceprints/people/${encodeURIComponent(ref)}/samples/${encodeURIComponent(samplePublicId)}`,
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

export interface SelectedCaptureClip {
  rel_path: string;
  begin_time_ms: number;
  end_time_ms: number;
  name: string;
  person_public_id: string | null;
}

export const captureRun = (ref: string, selectedClips: SelectedCaptureClip[]) =>
  api<{ job_id: string }>(`/api/voiceprints/capture/${encodeURIComponent(ref)}/run`, {
    method: "POST",
    // Send each pick's stable (begin,end) so the server can reject a drifted plan instead of
    // capturing the wrong clip under a stale index-based rel_path.
    body: JSON.stringify({ selected_clips: selectedClips }),
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

/** Token-carrying rollback URL for navigator.sendBeacon on tab close/reload, where an
 *  async fetch from an unmount handler can't reliably complete. sendBeacon issues a POST
 *  and can't set headers, so the token rides the query string like the SSE/audio paths. */
export const captureRollbackUrl = (txnId: string) =>
  withToken(
    `/api/voiceprints/capture/transactions/${encodeURIComponent(txnId)}/rollback`,
  );

export interface PendingCapture {
  transaction_id: string;
  project_id: string | null;
}

/** The capture transaction awaiting accept/rollback (one at most), or null. Powers the
 *  app-wide recovery banner so an orphaned transaction can be resolved from anywhere. */
export const getPendingCapture = () =>
  api<PendingCapture | null>("/api/voiceprints/capture/pending");

export const getJob = (jobId: string) =>
  api<JobStatus>(`/api/jobs/${encodeURIComponent(jobId)}`);

// ---- Pipeline --------------------------------------------------------------

export interface RunPipelineBody {
  input_path: string;
  title?: string | null;
  meeting_time?: string | null;
  model?: string;
  summarize?: boolean;
  polish?: boolean;
}

export const runPipeline = (body: RunPipelineBody) =>
  api<{ job_id: string }>("/api/pipeline/run", {
    method: "POST",
    body: JSON.stringify(body),
  });

export const summarizeProject = (ref: string) =>
  api<{ job_id: string }>(`/api/pipeline/summarize/${encodeURIComponent(ref)}`, {
    method: "POST",
    body: JSON.stringify({}),
  });

export interface MergePreview {
  order_source: string;
  use_corrected: boolean;
  identity_count: number;
  speaker_count: number;
  sentence_count: number;
  names: string[];
  warnings: string[];
  out_dir?: string;
  written?: string[];
}

export const mergePreview = (projectRefs: string[]) =>
  api<MergePreview>("/api/pipeline/merge-preview", {
    method: "POST",
    body: JSON.stringify({ project_refs: projectRefs }),
  });

export const mergeApply = (projectRefs: string[], outDir: string) =>
  api<MergePreview>("/api/pipeline/merge", {
    method: "POST",
    body: JSON.stringify({ project_refs: projectRefs, out_dir: outDir }),
  });

// ---- Corrections -----------------------------------------------------------

export interface CorrectionChange {
  index: number;
  sentence_id: number | null;
  begin_time_ms: number | null;
  end_time_ms: number | null;
  speaker_name: string;
  original_text: string;
  corrected_text: string;
  change_type: string;
  reason: string;
}

export interface Proposal {
  model: string;
  change_count: number;
  changes: CorrectionChange[];
  proposal_id: string;
}

export const polishProject = (ref: string) =>
  api<{ job_id: string }>(`/api/corrections/${encodeURIComponent(ref)}/polish`, {
    method: "POST",
    body: JSON.stringify({}),
  });

export const getProposal = (ref: string) =>
  api<Proposal>(`/api/corrections/${encodeURIComponent(ref)}/proposal`);

export const acceptCorrection = (
  ref: string,
  selectedIndices: number[] | null,
  proposalId: string,
) =>
  api<{ accepted: boolean; change_count: number; learned_count: number }>(
    `/api/corrections/${encodeURIComponent(ref)}/accept`,
    {
      method: "POST",
      // Echo the reviewed proposal's id so the server refuses if it changed (regenerated).
      body: JSON.stringify({ selected_indices: selectedIndices, proposal_id: proposalId }),
    },
  );

// ---- Lexicon ---------------------------------------------------------------

export interface LexiconTerm {
  term_id: number;
  public_id: string;
  canonical: string;
  category: string;
  description: string;
  status: string;
  alias_count: number;
  context_count: number;
  ambiguous_alias_count: number;
  created_at: string;
  updated_at: string;
}

export interface LexiconStats {
  active_terms: number;
  inactive_terms: number;
  aliases: number;
  contexts: number;
  hotwords: number;
  cached_vocabularies: number;
}

export interface Disambiguation {
  alias: string;
  canonical: string;
  category: string;
  guidance: string;
}

export interface Hotword {
  text: string;
  weight: number;
  category: string;
  source: string;
}

export const getLexiconTerms = (query?: string) =>
  api<{ terms: LexiconTerm[] }>(
    `/api/lexicon/terms?limit=500${query ? `&query=${encodeURIComponent(query)}` : ""}`,
  );

export const getLexiconStats = () => api<LexiconStats>("/api/lexicon/stats");

export const getDisambiguations = () =>
  api<Disambiguation[]>("/api/lexicon/disambiguations");

export const getHotwords = () => api<Hotword[]>("/api/lexicon/hotwords");

export const upsertLexiconTerm = (body: {
  canonical: string;
  category: string;
  description?: string;
  aliases?: string[];
}) =>
  api<LexiconTerm>("/api/lexicon/terms", {
    method: "POST",
    body: JSON.stringify(body),
  });

export const deleteLexiconTerm = (ref: string) =>
  api<{ deleted_public_id: string }>(
    `/api/lexicon/terms/${encodeURIComponent(ref)}`,
    { method: "DELETE" },
  );

// Mark an alias as context-ambiguous (empty guidance clears it -> response is null).
export const setDisambiguation = (body: {
  term: string;
  alias: string;
  guidance: string;
}) =>
  api<Disambiguation | null>("/api/lexicon/disambiguations", {
    method: "POST",
    body: JSON.stringify(body),
  });

// ---- Config + diagnostics --------------------------------------------------

export interface ConfigKey {
  name: string;
  env_name: string;
  secret: boolean;
  is_set: boolean;
  value: string | null;
}

export interface Config {
  config_file: string;
  keys: ConfigKey[];
}

export interface DoctorCheck {
  name: string;
  status: string;
  detail: string;
  fix_prompt: string | null;
}

export interface Doctor {
  ok: boolean;
  checks: DoctorCheck[];
}

export const getConfig = (reveal = false) =>
  api<Config>(`/api/config${reveal ? "?reveal=true" : ""}`);

export const setConfig = (key: string, value: string) =>
  api<{ key: string }>("/api/config", {
    method: "PATCH",
    body: JSON.stringify({ key, value }),
  });

export const unsetConfig = (key: string) =>
  api<{ key: string }>(`/api/config/${encodeURIComponent(key)}`, {
    method: "DELETE",
  });

export const getDoctor = () => api<Doctor>("/api/doctor");

// ---- Health + auth ---------------------------------------------------------

export interface Health {
  status: string;
  auth_required: boolean;
  /** Loopback bind: gates loopback-only affordances like revealing secret config. */
  is_local: boolean;
}

/** Unauthenticated liveness + bind metadata; tells the SPA whether a token is needed. */
export const getHealth = () => api<Health>("/api/health");

/** Token probe: resolves when the presented credential is valid, throws 401 otherwise. */
export const getAuthCheck = () => api<{ ok: boolean }>("/api/auth/check");

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
  // EventSource can't set headers, so the token rides along as a query param.
  const source = new EventSource(withToken(`/api/jobs/${jobId}/events`));
  source.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data) as ProgressEvent);
    } catch {
      // ignore malformed frames
    }
  };
  return () => source.close();
}
