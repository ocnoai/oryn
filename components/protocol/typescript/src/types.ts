import type { TaskProfile } from "./task-contracts.js";
export const PROTOCOL_VERSION = 1 as const;
export const MAX_MESSAGE_BYTES = 16 * 1024 * 1024;

export interface ProtocolError {
  code: string;
  category:
    | "protocol"
    | "model"
    | "session"
    | "job"
    | "artifact"
    | "resource"
    | "runtime";
  message: string;
  retryable: boolean;
  details?: Record<string, unknown>;
  diagnosticId?: string;
}

export interface HandshakeResult {
  selectedVersion: 1;
  engineVersion: string;
  engineInstanceId: string;
  maxMessageBytes: number;
}

export interface ProtocolPingResult {
  engineInstanceId: string;
  state: "ready" | "busy" | "shutting-down";
}

export interface RuntimeShutdownResult {
  accepted: true;
}

export type MemoryPressure = "normal" | "high" | "critical";

export interface RuntimePolicyUpdate {
  policyRevision: number;
  memoryBudgetBytes?: number;
  batteryMode?: "normal" | "reduce" | "disable";
  allowSpeculativePreload?: boolean;
  runtimeSelectionPolicy?: "balanced" | "performance" | "memory-saver" | "quality";
}

export interface RuntimeResourceSnapshot {
  snapshotId: string;
  timestamp: number;
  memoryPressure: MemoryPressure;
  systemMemory?: { totalBytes?: number; availableBytes?: number };
  processMemory?: {
    residentBytes?: number;
    cpuPercent?: number;
    sampleWindowMs?: number;
  };
  supervisorMemory?: { residentBytes?: number };
  workers?: Array<{
    workerId: string;
    kind: string;
    residentBytes?: number;
  }>;
  localAI: { memoryBudgetBytes?: number; estimatedResidentModelBytes: number };
  runningJob?: {
    jobId?: string;
    model?: string;
    task?: string;
    estimatedBytes?: number;
  };
  accelerator?: {
    backend: string;
    allocatedBytes?: number;
    reservedBytes?: number;
    utilizationPercent?: number;
    sampleWindowMs?: number;
  };
  throughput?: { value: number; unit: string; sampleWindowMs: number };
  generationLane: { capacity: 1; running: 0 | 1; queued: number };
}

export interface RuntimePolicyUpdateResult {
  appliedRevision: number;
  resources: RuntimeResourceSnapshot;
}

export type RuntimeResourcesUpdatedEvent =
  ProtocolEvent<RuntimeResourceSnapshot> & {
    event: "runtime.resources.updated";
  };

export type RuntimeMemoryPressureChangedEvent = ProtocolEvent<{
  previous: MemoryPressure;
  current: MemoryPressure;
  snapshotId: string;
}> & { event: "runtime.memoryPressure.changed" };

export interface ResolvedModelRef {
  files?: string[];
  provider: "huggingface";
  repository: string;
  revision: string;
}

export type InputModality = "text" | "image" | "audio" | "video" | "document";
export type OutputModality =
  "text" | "image" | "audio" | "video" | "embedding" | "structured";
export type TechnicalCapabilityValue = boolean | "unknown";

export interface ModelComposition {
  kind: "diffusers-lora-v1";
  model: { repository: string; revision: string };
  base: { repository: string; revision: string };
  adapterFile: string;
  family: "QwenImage21Pipeline" | "QwenImageEditPlusPipeline";
  scale: number;
  overrides?: Array<"scheduler/scheduler_config.json">;
  referenceImages?: { minimum: number; maximum: number };
  sampling?: { steps: number; guidanceScale: 1; negativePrompt: ""; sigmas?: number[] };
}

export interface TrustedModelMetadata {
  composition?: ModelComposition;
  license: string;
  source?: ResolvedModelRef;
  quantization?: string;
  contextLength?: number;
  chatTemplate?: "chatml" | "deepseek" | "flint" | "gemma3" | "gemma4" | "glm" | "lfm3" | "llama3" | "plain" | "spark";
  modelFile?: string;
  projectorFile?: string;
  adapterArchitecture?: string;
  inferenceDefaults?: {
    width?: number;
    height?: number;
    steps?: number;
    guidanceScale?: number;
  };
  languages?: string[];
  voices?: Array<{ id: string; language: string; name: string }>;
  conformance: { status: "passed" | "fixture-only"; platforms: string[] };
}

export interface RuntimePackSelection {
  packId: string;
  packVersion: string;
  runnerId: string;
  taskContract: string;
  compatibilityLevel:
    | "native-declarative"
    | "audited-adapter"
    | "converted-artifact";
  manifestDigest: string;
  trustedMetadata?: TrustedModelMetadata;
}

export interface RuntimeCandidateSummary {
  candidateId: string;
  foundation: { id: string; version: string };
  adapter: { id: string; version: string };
  variantId: string;
  backend: string;
  precision: string;
  quantization?: string;
  estimatedMemoryBytes: number;
}

export interface RuntimePlanSummary {
  planId: string;
  policy: "balanced" | "performance" | "memory-saver" | "quality";
  reasonCodes: string[];
  selected: RuntimeCandidateSummary;
  fallbacks: RuntimeCandidateSummary[];
  rejected: Array<{ candidateId: string; reasonCodes: string[] }>;
}

export type RuntimeOverrideResult = RuntimePlanSummary;

export interface RuntimePlanFailure {
  code: "RUNTIME_PLAN_UNAVAILABLE" | "RUNTIME_RESOURCES_INSUFFICIENT";
  rejected: Array<{ candidateId: string; reasonCodes: string[] }>;
}

export interface GenerationIntegerParameter {
  default: number;
  minimum: number;
  maximum: number;
  step: number;
  supported: boolean;
  editable: boolean;
}

export interface GenerationNumberParameter extends GenerationIntegerParameter {}

export interface ModelGenerationProfile {
  profileId: string;
  provenance: "runtime-tested" | "runtime-declared" | "adapter-derived" | "runtime-fallback";
  recommended: boolean;
  parameters: {
    width: GenerationIntegerParameter;
    height: GenerationIntegerParameter;
    steps: GenerationIntegerParameter;
    guidanceScale: GenerationNumberParameter;
  };
}

export interface HuggingFaceSourceContext {
  sourceURL: string;
  tabId: string;
  documentId: string;
}

/** Browser-private evidence; source recognition never establishes conformance. */
export interface SourceRecognition {
  state: 'recognized' | 'unknown' | 'inspection-unavailable';
  match?: 'exact' | 'structural';
  normalizer?: string;
  catalogDigest?: string;
  configDigest?: string;
  sourceDigest?: string;
  profileId?: string;
  profileVersion?: 1;
  adapterId?: 'oryn.lfm3.torch.v1';
  sourcePolicy?: 'bundled-only-no-upstream-kernels-v1';
  representation?: 'bf16-cpu' | 'fp8-unsupported' | 'nvfp4-unsupported' | 'mlx-affine4-unsupported' | 'quantization-unsupported' | 'dtype-unsupported';
  admitted?: boolean;
  reason?: 'SOURCE_SANDBOX_UNAVAILABLE';
  files?: Array<{ name: string; rawSha256: string; structuralSha256: string | null }>;
}

export interface CandidateExecutionProfile {
  admissionScope?: 'huggingface-page';
  sourceRecognition?: SourceRecognition;
  /** Private inspected offer facts, used before final Engine routing. */
  routingFacts?: {
    version: 1;
    offers: Array<{ engine: string; engineVersion: string; features: string[] }>;
  };
  runtimePreparationRequired?: boolean;
  lineageDeclarations?: string[];
  /** Exact private scope for reference evidence in the active configuration. */
  recommendationContext?: { runtime: string; hardware: string; workload: string };
  /** Trusted Browser only. Never expose this evidence context to websites. */
  recommendationEvidence?: {
    metric: 'decode-tokens-per-second' | 'ttft-ms' | 'latency-ms' | 'real-time-factor';
    value:number; phase:'warm-inference'|'cold-load'; comparisonKey:string;
    measuredAt:number; expiresAt:number; sampleCount:number;
  };
  installationRequired?: boolean;
  taskProfile?: TaskProfile;
  model: ResolvedModelRef;
  exactRevision: string;
  descriptorSchemaVersion?: string;
  capabilityRegistryDigest?: string;
  eligibility?: CandidateEligibility;
  loadDisposition?:
    | "verified-load"
    | "trust-required"
    | "authorization-required"
    | "not-admissible-on-this-device"
    | "unsupported";
  matchedCapability?: {
    foundationId: string;
    foundationVersion: string;
    familyAdapterId: string;
    familyAdapterVersion: string;
    taskAdapterId: string;
    taskAdapterVersion: string;
  };
  executable: boolean;
  runner?: string;
  runtimePack?: RuntimePackSelection;
  runtimePlan?: RuntimePlanSummary;
  runtimePlanFailure?: RuntimePlanFailure;
  executableBackends: string[];
  requiresCustomCode: boolean;
  estimatedDiskBytes?: number;
  estimatedMemoryBytes?: number;
  selectedFiles?: Array<{ name: string; sizeBytes: number }>;
  unselectedExecutableFiles?: string[];
  generationProfile?: ModelGenerationProfile;
  technicalCapabilities: {
    task?: string;
    architectures?: string[];
    parameterCount?: number;
    contextLength?: number;
    inputModalities?: InputModality[];
    outputModalities?: OutputModality[];
    capabilities?: Record<string, TechnicalCapabilityValue>;
  };
  blockingReasons?: string[];
}

export interface EligibilityAxis<TState extends string> {
  state: TState | "deferred";
  reasons: string[];
  deferredBy?: "authorization";
}

export interface CandidateEligibility {
  access: EligibilityAxis<"available" | "authorization-required" | "denied">;
  artifacts: EligibilityAxis<"safe" | "convertible" | "unsafe" | "incomplete">;
  runtime: EligibilityAxis<"supported" | "family-adapter-required" | "unsupported">;
  task: EligibilityAxis<"supported" | "task-adapter-required" | "unknown">;
  resources: EligibilityAxis<"admissible" | "not-admissible">;
  conformance: EligibilityAxis<"certified" | "family-conformant" | "untested" | "failed">;
}

export interface MediaAudioExtractV1Input {
  videoArtifactId: string;
  track: "default";
}

export interface AudioSourceSeparationV1Input {
  audioArtifactId: string;
  target: "vocals";
}

export interface ForcedAlignmentV1Input {
  audioArtifactId: string;
  transcript: string;
  language: string;
  timestamps: "word" | "segment";
}

export interface SubtitleSegmentV1 {
  startMs: number;
  endMs: number;
  text: string;
}

export interface SubtitleExportV1Input {
  format: "subrip";
  segments: SubtitleSegmentV1[];
}

export interface ModelInstallResult {
  operationId: string;
  installationId: string;
  model: ResolvedModelRef;
  logicalSizeBytes: number;
}

export interface ModelCatalogMatchResult {
  models: ResolvedModelRef[];
}

export type ModelResidencyState =
  "unloaded" | "loading" | "loaded" | "unloading" | "error";

export interface ModelResidencyResult {
  operationId: string;
  model: ResolvedModelRef;
  state: "unloaded" | "loaded" | "deferred";
  estimatedResidentBytes?: number;
}

export type ModelResidencyChangedEvent = ProtocolEvent<{
  model: ResolvedModelRef;
  state: ModelResidencyState;
  estimatedResidentBytes?: number;
  reason?:
    | "explicit-load"
    | "job-required"
    | "explicit-unload"
    | "memory-pressure"
    | "memory-budget-change"
    | "shutdown"
    | "load-failed";
}> & { event: "model.residency.changed" };

export interface EngineModelInventoryRecord {
  admissionScope?: 'huggingface-page';
  installationId: string;
  model: ResolvedModelRef;
  installationState: "installed" | "downloading" | "removing" | "error";
  residencyState: ModelResidencyState;
  logicalSizeBytes: number;
  reclaimableSizeBytes: number;
  estimatedResidentBytes?: number;
  activeSessions: number;
  runningJobs: number;
  queuedJobs: number;
  installedAt?: number;
  lastUsedAt?: number;
  runner?: string;
  runtimePack?: RuntimePackSelection;
  runtimePlan?: RuntimePlanSummary;
  executableBackends?: string[];
  technicalCapabilities?: CandidateExecutionProfile["technicalCapabilities"];
}

export interface ModelInventoryResult {
  models: EngineModelInventoryRecord[];
}

export interface ModelRemovePreviewResult {
  installationId: string;
  canRemoveNow: boolean;
  activeSessions: number;
  runningJobs: number;
  queuedJobs: number;
  reclaimableBytes: number;
  sharedBytesRetained: number;
  blockers: string[];
}

export interface ModelRemoveResult {
  operationId: string;
  installationId: string;
  state: "removed" | "deferred";
  reclaimedBytes: number;
  sharedBytesRetained: number;
}

export interface EngineSessionAttached {
  taskProfile?: TaskProfile;
  sessionId: string;
  model: ResolvedModelRef;
  runtime: {
    runner: string;
    backend: string;
    deviceClass: "cpu" | "mps" | "cuda" | "other";
  };
  technicalCapabilities: CandidateExecutionProfile["technicalCapabilities"];
  residencyState: "loaded" | "unloaded";
}

export interface SessionReleaseResult {
  released: boolean;
}

export type EngineJobState =
  | "created"
  | "queued"
  | "loading"
  | "running"
  | "streaming"
  | "saving"
  | "completed"
  | "failed"
  | "cancelled";

export interface EngineInputArtifactBinding {
  artifactId: string;
  internalPath: string;
  mimeType: string;
  sizeBytes: number;
  name?: string;
  access: "read";
}

export interface EngineOutputArtifact {
  outputId: string;
  internalPath: string;
  mimeType: string;
  sizeBytes: number;
  name?: string;
}

export interface EngineStructuredResult {
  kind: "structured";
  value: unknown;
}

export interface EngineArtifactResult {
  kind: "artifact";
  artifact: EngineOutputArtifact;
}

export interface EngineMultiResult {
  kind: "multi";
  outputs: EngineInferenceResult[];
}

export type EngineInferenceResult =
  | EngineStructuredResult
  | EngineArtifactResult
  | EngineMultiResult;

export interface JobQueueStatus {
  lane: "generation";
  position: number;
  jobsAhead: number;
  reason: "generation-slot" | "model-loading" | "resource-policy";
}

export interface EngineJobReference {
  jobId: string;
  sessionId: string;
  state: "created" | "queued" | "loading" | "running";
  queue?: JobQueueStatus;
}

export interface EngineJobStatus {
  jobId: string;
  sessionId: string;
  state: EngineJobState;
  sequence: number;
  queue?: JobQueueStatus;
  result?: unknown;
  error?: ProtocolError;
}

export interface JobEventDataBase {
  jobId: string;
  sessionId: string;
  sequence: number;
}

export type JobStateEvent = ProtocolEvent<
  JobEventDataBase & {
    state: "queued" | "loading" | "running" | "streaming" | "saving";
    queue?: JobQueueStatus;
  }
> & { event: "job.state" };

export type JobQueueUpdatedEvent = ProtocolEvent<
  JobEventDataBase & JobQueueStatus
> & {
  event: "job.queue.updated";
};

export type JobProgressEvent = ProtocolEvent<
  JobEventDataBase & { progress?: number; stage?: string }
> & { event: "job.progress" };

export type JobOutputDeltaEvent = ProtocolEvent<
  JobEventDataBase & { channel: string; delta: unknown }
> & { event: "job.output.delta" };

export type JobPreviewEvent = ProtocolEvent<
  JobEventDataBase & { output: EngineOutputArtifact }
> & { event: "job.preview" };

export type JobCompletedEvent = ProtocolEvent<
  JobEventDataBase & {
    result: EngineInferenceResult;
    execution?: {
      runner?: string;
      backend?: string;
      durationMs?: number;
      planId?: string;
      foundationId?: string;
      adapterId?: string;
      candidateId?: string;
      variantId?: string;
      precision?: string;
      quantization?: string;
      fallbacks?: Array<{
        fromCandidateId: string;
        toCandidateId: string;
        reasonCode: string;
      }>;
    };
  }
> & { event: "job.completed" };

export type JobFailedEvent = ProtocolEvent<
  JobEventDataBase & { error: ProtocolError }
> & {
  event: "job.failed";
};

export type JobCancelledEvent = ProtocolEvent<
  JobEventDataBase & {
    reason?: "website" | "user" | "resource-policy" | "runtime-shutdown";
  }
> & { event: "job.cancelled" };

export type EngineJobEvent =
  | JobStateEvent
  | JobQueueUpdatedEvent
  | JobProgressEvent
  | JobOutputDeltaEvent
  | JobPreviewEvent
  | JobCompletedEvent
  | JobFailedEvent
  | JobCancelledEvent;

export type ModelInstallProgressEvent = ProtocolEvent<{
  operationId: string;
  model: ResolvedModelRef;
  stage:
    "resolving" | "downloading" | "verifying" | "installing" | "finalizing";
  downloadedBytes: number;
  totalBytes: number;
}> & { event: "model.install.progress" };

export type ModelInstallStateEvent = ProtocolEvent<{
  operationId: string;
  model: ResolvedModelRef;
  state: "downloading" | "installed" | "cancelled" | "error";
}> & { event: "model.install.state" };

export type ModelInventoryChangedEvent = ProtocolEvent<{
  reason: "installed" | "removed" | "reconciled";
}> & { event: "model.inventory.changed" };

export type HandshakeResponse =
  | { type: "response"; id: string; ok: true; result: HandshakeResult }
  | { type: "response"; id: string; ok: false; error: ProtocolError };

export interface ProtocolRequest {
  v: 1;
  type: "request";
  id: string;
  method: string;
  params: Record<string, unknown>;
}

export interface RuntimePackInventoryRecord {
  packId: string;
  version: string;
  publisher: string;
  abi: string;
  dependencyClosure: string;
  platforms: Array<"macos-arm64" | "windows-amd64" | "linux-amd64">;
  distribution: "bundled" | "download";
  state: "available" | "staged" | "active" | "revoked";
  bundled: boolean;
  removable: boolean;
  sizeBytes: number;
  minimumSafeVersion: string;
  revoked: boolean;
  license: {
    summary: string;
    notice: string;
    access: "public" | "gated";
    acceptanceRequired: boolean;
    accepted: boolean;
  };
}

export interface RuntimePackInventoryResult {
  catalogVersion: number;
  catalogDigest: string;
  thirdPartyPublishersEnabled: false;
  automaticExecutableUpdatesEnabled: false;
  packs: RuntimePackInventoryRecord[];
}

export interface RuntimePackCleanupResult {
  removedPacks: number;
  reclaimedBytes: number;
}

export type ProtocolResponse<T = unknown> =
  | { v: 1; type: "response"; id: string; ok: true; result: T }
  | { v: 1; type: "response"; id: string; ok: false; error: ProtocolError };

export interface ProtocolEvent<T = Record<string, unknown>> {
  v: 1;
  type: "event";
  event: string;
  seq: number;
  timestamp: number;
  data: T;
}

export type RuntimeState = "ready" | "shutting-down" | "error";

export type RuntimeStateEvent = ProtocolEvent<{ state: RuntimeState }> & {
  event: "runtime.state";
};

export interface EngineRuntimeCapabilities {
  taskContracts?: Array<{ id: string; revision: number }>;
  engineVersion: string;
  engineInstanceId: string;
  protocolVersion: 1;
  providers: Array<"huggingface">;
  backends: Array<{
    id: string;
    deviceClass: "cpu" | "mps" | "cuda" | "other";
  }>;
  runners: Array<{
    id: string;
    framework: string;
  }>;
  scheduler: {
    lanes: Array<{
      id: string;
      capacity: number;
      global: boolean;
    }>;
  };
  features: {
    reviewedSourceInspection?: boolean;
    taskProfiles?: boolean;
    streaming: boolean;
    artifacts: boolean;
    modelInspection: boolean;
    memoryPressureMonitoring: boolean;
    multipleResidentModels: boolean;
  };
  limits: {
    maxMessageBytes: number;
    maxArtifactBindingsPerJob: number;
  };
}
