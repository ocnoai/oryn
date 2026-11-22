import type { ModelSelection, SelectionExplanation } from "./model-selection.js";
import type { TaskProfile, TaskRequirements } from "./task-contracts.js";
export type ModelProvider = "huggingface";
export type InputModality = "text" | "image" | "audio" | "video" | "document";
export type OutputModality =
  "text" | "image" | "audio" | "video" | "embedding" | "structured";
export type BehavioralCapability =
  | "chat"
  | "streaming"
  | "structuredOutput"
  | "toolCalling"
  | "imageEditing"
  | "inpainting"
  | "outpainting"
  | "speechRecognition"
  | "speechSynthesis"
  | "audioOutput"
  | "lyrics"
  | "deterministicSeed"
  | "multimodalConversation";

export interface RuntimeCapabilities {
  taskContracts?: Array<{ id: string; revision: number }>;
  runtimeAvailable: boolean;
  runtimeState: "ready" | "starting" | "unavailable" | "error";
  sdkVersion: string;
  browserApiVersion?: string;
  supportedModelProviders: ModelProvider[];
  features: {
    modelSelection?: boolean;
    taskProfiles?: boolean;
    modelRequests: boolean;
    modelSubstitution: boolean;
    streaming: boolean;
    artifacts: boolean;
    artifactUrls: boolean;
    cancellation: boolean;
    queueing: boolean;
    multipleSubmittedJobs: boolean;
  };
  scheduler?: {
    generation: { globalAcrossOrigins: boolean; maxConcurrentRunning: number };
  };
}

/** Optional website-owned conveniences. All options default to disabled. */
export interface LocalAIClientOptions {
  /**
   * Show one website-owned explanation when this document is opened outside
   * Oryn Browser. This never probes the network or grants Local AI authority.
   */
  showOrynBrowserAlert?: boolean;
}

export interface ModelRef {
  files?: string[];
  provider: "huggingface";
  repository: string;
  revision?: string;
}

export interface ResolvedModelRef {
  files?: string[];
  provider: "huggingface";
  repository: string;
  revision: string;
  displayName?: string;
  publisher?: string;
  sourceUrl: string;
}

export interface ModelRequirements extends TaskRequirements {
  task?: string;
  architectures?: string[];
  parameters?: { minBillions?: number; maxBillions?: number };
  minimumContextLength?: number;
  inputModalities?: InputModality[];
  outputModalities?: OutputModality[];
  capabilities?: Partial<Record<BehavioralCapability, boolean>>;
  memory?: { maximumRecommendedBytes?: number };
}

export interface ExplicitModelRequest {
  selection?: ModelSelection;
  role: string;
  preferredModel: ModelRef;
  certifiedModels?: ModelRef[];
  requirements?: ModelRequirements;
  substitution?: { automatic?: "none" | "certified-only" | "same-checkpoint" | "compatible" };
}

export interface CapabilityModelRequest {
  selection?: never;
  role: string;
  preferredModel?: never;
  requirements: ModelRequirements & {
    task: string;
    outputModalities: OutputModality[];
  };
  substitution?: { automatic?: "compatible" };
}

export interface CheckpointModelRequest {
  role: string;
  selection: Extract<ModelSelection, {intent: 'source-checkpoint'}>;
  preferredModel?: never;
  requirements?: ModelRequirements;
  substitution?: {automatic?: 'none' | 'same-checkpoint' | 'compatible'};
}
export type ModelRequest = ExplicitModelRequest | CapabilityModelRequest | CheckpointModelRequest;

export type ModelRequestProgress =
  | { stage: "resolving" }
  | { stage: "awaiting-permission" }
  | { stage: "awaiting-model-choice" }
  | {
      stage: "downloading";
      model: ResolvedModelRef;
      downloadedBytes: number;
      totalBytes?: number;
    }
  | { stage: "loading"; model: ResolvedModelRef; progress?: number }
  | { stage: "ready"; sessionId: string };

export interface ModelRequestOptions {
  signal?: AbortSignal;
  onProgress?: (progress: ModelRequestProgress) => void;
}

export interface CompatibilityReason {
  code: string;
  status: "satisfied" | "unknown" | "failed";
  message?: string;
}

export interface ModelResolution {
  selection?: SelectionExplanation;
  preferredModel: ResolvedModelRef;
  resolvedModel: ResolvedModelRef;
  compatibility: "certified" | "compatible" | "uncertain" | "incompatible";
  executability: "executable" | "not-executable" | "unknown";
  security: "allowed" | "blocked" | "developer-mode-required";
  /** The browser, not the website, grants exact-revision trust after user consent. */
  trust: "oryn-verified" | "user-approved-unverified";
  developerCertified: boolean;
  substituted: boolean;
  overriddenByUser: boolean;
  resolutionReason:
    | "preferred-model"
    | "developer-certified-alternative"
    | "already-installed"
    | "site-preference"
    | "global-user-preference"
    | "automatic-compatible-substitution"
    | "capability-match"
    | "user-override";
  compatibilityReasons: CompatibilityReason[];
}

export interface ResolvedModelCapabilities {
  task?: string;
  architectures?: string[];
  parameterCount?: number;
  contextLength?: number;
  inputModalities?: InputModality[];
  outputModalities?: OutputModality[];
  capabilities?: Partial<Record<BehavioralCapability, boolean | "unknown">>;
}

export interface ModelSession {
  taskProfile?: TaskProfile;
  sessionId: string;
  role: string;
  origin: string;
  resolution: ModelResolution;
  capabilities: ResolvedModelCapabilities;
}

export interface ArtifactReference {
  id: string;
  mimeType: string;
  sizeBytes: number;
  name?: string;
}

export interface ArtifactMetadata extends ArtifactReference {
  createdAt: number;
  expiresAt?: number;
  url?: string;
}

export interface CreateArtifactOptions {
  name?: string;
  lifetime?: "ephemeral" | "session";
}

export interface SaveArtifactOptions {
  suggestedFilename?: string;
}

export type InferenceResult = StructuredResult | ArtifactResult | MultiResult;

export interface StructuredResult {
  kind: "structured";
  value: unknown;
}

export interface ArtifactResult {
  kind: "artifact";
  artifact: ArtifactMetadata;
}

export interface MultiResult {
  kind: "multi";
  outputs: InferenceResult[];
}

export interface ModelInvocationRequest<TInput = unknown> {
  sessionId: string;
  input: TInput;
}

export type JobState =
  | "created"
  | "queued"
  | "loading"
  | "running"
  | "streaming"
  | "saving"
  | "completed"
  | "failed"
  | "cancelled";

export interface JobQueueStatus {
  lane: "generation";
  position: number;
  jobsAhead: number;
  reason: "generation-slot" | "model-loading" | "resource-policy";
}

export interface JobReference {
  jobId: string;
  sessionId: string;
  state: "created" | "queued" | "loading" | "running";
  queue?: JobQueueStatus;
}

export interface LocalAIEventBase {
  event: string;
  timestamp: number;
  jobId: string;
  sessionId: string;
  sequence: number;
}

export type LocalAIEvent = LocalAIEventBase &
  (
    | {
        event: "job.state";
        data: {
          state: "queued" | "loading" | "running" | "streaming" | "saving";
          queue?: JobQueueStatus;
        };
      }
    | { event: "job.queue.updated"; data: JobQueueStatus }
    | { event: "job.progress"; data: { progress?: number; stage?: string } }
    | { event: "job.output.delta"; data: { channel: string; delta: unknown } }
    | { event: "job.preview"; data: { artifactId: string } }
    | { event: "job.completed"; data: { result: InferenceResult } }
    | { event: "job.failed"; data: { error: SerializedLocalAIError } }
    | {
        event: "job.cancelled";
        data: {
          reason?: "website" | "user" | "resource-policy" | "runtime-shutdown";
        };
      }
  );

export type LocalAIErrorCode =
  | "RUNTIME_NOT_AVAILABLE"
  | "RUNTIME_STARTING"
  | "RUNTIME_PROTOCOL_ERROR"
  | "RUNTIME_DEPENDENCY_UNAVAILABLE"
  | "UNAUTHORIZED_ORIGIN"
  | "UNAUTHORIZED_FRAME"
  | "PERMISSION_DENIED"
  | "USER_DENIED"
  | "MODEL_REFERENCE_INVALID"
  | "MODEL_SOURCE_INVALID"
  | "MODEL_REVISION_INVALID"
  | "MODEL_NOT_FOUND"
  | "MODEL_METADATA_INVALID"
  | "MODEL_NOT_EXECUTABLE"
  | "MODEL_SECURITY_BLOCKED"
  | "CUSTOM_CODE_REQUIRED"
  | "UNSAFE_MODEL_FORMAT"
  | "NO_EXECUTABLE_MODEL"
  | "DOWNLOAD_REQUIRED"
  | "RESOURCE_REQUIREMENTS_UNSATISFIABLE"
  | "RESOURCE_LIMIT_EXCEEDED"
  | "DISK_SPACE_INSUFFICIENT"
  | "MODEL_OUT_OF_MEMORY"
  | "MODEL_LOAD_FAILED"
  | "MODEL_NOT_LOADED"
  | "SESSION_NOT_FOUND"
  | "SESSION_ACCESS_DENIED"
  | "SESSION_RELEASED"
  | "REQUEST_ABORTED"
  | "ARTIFACT_NOT_FOUND"
  | "ARTIFACT_ACCESS_DENIED"
  | "ARTIFACT_EXPIRED"
  | "ARTIFACT_INVALID"
  | "ARTIFACT_IO_ERROR"
  | "ARTIFACT_PATH_INVALID"
  | "ARTIFACT_BINDING_INVALID"
  | "AUDIO_FORMAT_UNSUPPORTED"
  | "AUDIO_TOO_LARGE"
  | "AUDIO_DECODE_FAILED"
  | "AUDIO_PROCESSING_FAILED"
  | "JOB_NOT_FOUND"
  | "JOB_ACCESS_DENIED"
  | "JOB_INPUT_INVALID"
  | "JOB_ID_CONFLICT"
  | "JOB_QUEUE_FULL"
  | "JOB_EXECUTION_FAILED"
  | "JOB_CANCELLED"
  | "MODEL_OUTPUT_FILTERED"
  | "ASR_INFERENCE_FAILED"
  | "ASR_OUTPUT_INVALID"
  | "ALIGNMENT_LANGUAGE_UNSUPPORTED"
  | "ALIGNMENT_OUTPUT_INVALID"
  | "SONG_GENERATION_FAILED"
  | "SONG_OUTPUT_INVALID"
  | "RUNNER_NOT_AVAILABLE";

export interface SerializedLocalAIError {
  code: LocalAIErrorCode;
  message: string;
  details?: Record<string, unknown>;
}

export type BridgeResult<T> =
  { ok: true; value: T } | { ok: false; error: SerializedLocalAIError };

export interface InjectedLocalAI {
  readonly sdkVersion: string;
  getCapabilities(): Promise<RuntimeCapabilities>;
  requestModel(
    operationId: string,
    request: ModelRequest,
  ): Promise<BridgeResult<ModelSession>>;
  cancelModelRequest(operationId: string): Promise<void>;
  releaseModelSession(sessionId: string): Promise<BridgeResult<void>>;
  onModelRequestProgress(
    operationId: string,
    listener: (progress: ModelRequestProgress) => void,
  ): () => void;
  beginArtifactCreate(metadata: {
    mimeType: string;
    sizeBytes: number;
    name?: string;
    lifetime?: "ephemeral" | "session";
  }): Promise<BridgeResult<{ uploadId: string }>>;
  appendArtifactChunk(
    uploadId: string,
    chunk: ArrayBuffer,
  ): Promise<BridgeResult<void>>;
  finishArtifactCreate(
    uploadId: string,
  ): Promise<BridgeResult<ArtifactReference>>;
  abortArtifactCreate(uploadId: string): Promise<void>;
  getArtifact(artifactId: string): Promise<BridgeResult<ArtifactMetadata>>;
  getArtifactUrl(artifactId: string): Promise<BridgeResult<string>>;
  saveArtifactAs(
    artifactId: string,
    options?: SaveArtifactOptions,
  ): Promise<BridgeResult<void>>;
  releaseArtifact(artifactId: string): Promise<BridgeResult<void>>;
  invokeJob(
    request: ModelInvocationRequest,
  ): Promise<BridgeResult<JobReference>>;
  cancelJob(jobId: string): Promise<BridgeResult<void>>;
  onJobEvent(listener: (event: LocalAIEvent) => void): () => void;
}

export interface LocalAIClient {
  readonly sdkVersion: string;
  getCapabilities(): Promise<RuntimeCapabilities>;
  models: {
    request(
      request: ModelRequest,
      options?: ModelRequestOptions,
    ): Promise<ModelSession>;
    releaseSession(sessionId: string): Promise<void>;
  };
  artifacts: {
    create(
      input: Blob | File,
      options?: CreateArtifactOptions,
    ): Promise<ArtifactReference>;
    get(artifactId: string): Promise<ArtifactMetadata>;
    getUrl(artifactId: string): Promise<string>;
    readBlob(artifactId: string): Promise<Blob>;
    saveAs(artifactId: string, options?: SaveArtifactOptions): Promise<void>;
    release(artifactId: string): Promise<void>;
  };
  invoke<TInput = unknown>(
    request: ModelInvocationRequest<TInput>,
  ): Promise<JobReference>;
  cancel(jobId: string): Promise<void>;
  subscribe(listener: (event: LocalAIEvent) => void): () => void;
}
