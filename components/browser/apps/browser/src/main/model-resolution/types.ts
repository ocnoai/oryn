import type { Decision } from '@ocno/oryn-registry';
import type { ModelSelection, SelectionExplanation } from '@ocno/oryn-protocol';
import type { TaskRequirements } from '@ocno/oryn-protocol';
import type {
  CandidateExecutionProfile,
  EngineModelInventoryRecord,
  InputModality,
  OutputModality,
  ResolvedModelRef,
} from '@ocno/oryn-protocol';

export type ModelRef = Omit<ResolvedModelRef, 'revision'> & { revision?: string };

export type BehavioralCapability =
  | 'chat'
  | 'streaming'
  | 'structuredOutput'
  | 'toolCalling'
  | 'imageEditing'
  | 'inpainting'
  | 'outpainting'
  | 'speechRecognition'
  | 'speechSynthesis'
  | 'audioOutput'
  | 'lyrics'
  | 'deterministicSeed'
  | 'multimodalConversation';

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
  substitution?: { automatic?: 'none' | 'certified-only' | 'same-checkpoint' | 'compatible' };
}

export interface CapabilityModelRequest {
  selection?: never;
  role: string;
  preferredModel?: never;
  requirements: ModelRequirements & { task: string; outputModalities: OutputModality[] };
  substitution?: { automatic?: 'compatible' };
}

export interface CheckpointModelRequest {
  role: string;
  selection: Extract<ModelSelection, { intent: 'source-checkpoint' }>;
  preferredModel?: never;
  requirements?: ModelRequirements;
  substitution?: { automatic?: 'none' | 'same-checkpoint' | 'compatible' };
}
export type ModelRequest = ExplicitModelRequest | CapabilityModelRequest | CheckpointModelRequest;

export type Compatibility = 'certified' | 'compatible' | 'uncertain' | 'incompatible';
export type Executability = 'executable' | 'not-executable' | 'unknown';
export type SecurityClassification = 'allowed' | 'blocked' | 'developer-mode-required';
export type ModelTrust = 'oryn-verified' | 'user-approved-unverified';

export interface CompatibilityReason {
  requirement: string;
  outcome: 'satisfied' | 'unknown' | 'not-satisfied';
  detail: string;
}

export interface ModelCandidateAssessment {
  recommendation?: Decision;
  performanceRank?: number;
  referenceRank?: number;
  representation?: string;
  sameCheckpoint?: boolean;
  lineage?: 'verified' | 'declared' | 'unknown' | 'conflicting';

  model: ResolvedModelRef;
  profile: CandidateExecutionProfile;
  compatibility: Compatibility;
  executability: Executability;
  security: SecurityClassification;
  developerCertified: boolean;
  installed: boolean;
  loaded: boolean;
  sources: Array<
    | 'preferred'
    | 'checkpoint'
    | 'certified'
    | 'catalog'
    | 'installed'
    | 'site-preference'
    | 'global-preference'
    | 'override'
  >;
  compatibilityReasons: CompatibilityReason[];
}

export interface ModelResolution {
  selection?: SelectionExplanation;
  preferredModel: ResolvedModelRef;
  resolvedModel: ResolvedModelRef;
  compatibility: Compatibility;
  executability: Executability;
  security: SecurityClassification;
  trust: ModelTrust;
  developerCertified: boolean;
  substituted: boolean;
  overriddenByUser: boolean;
  resolutionReason:
    | 'preferred-model'
    | 'developer-certified-alternative'
    | 'already-installed'
    | 'site-preference'
    | 'global-user-preference'
    | 'automatic-compatible-substitution'
    | 'capability-match'
    | 'user-override';
  compatibilityReasons: CompatibilityReason[];
  downloadRequired: boolean;
}

export interface ModelResolutionResult {
  resolution: ModelResolution;
  selected: ModelCandidateAssessment;
  candidates: ModelCandidateAssessment[];
}

export interface ModelInspector {
  inspectModel(
    model: ResolvedModelRef,
    requirements?: TaskRequirements,
    recommendation?: { snapshotDigest: string; engines: string[] },
  ): Promise<CandidateExecutionProfile>;
  matchModelCatalog?(
    task: string,
    outputModalities: OutputModality[],
  ): Promise<{ models: ResolvedModelRef[] }>;
}

export interface ModelResolutionContext {
  signal?: AbortSignal;
  origin: string;
  inventory: EngineModelInventoryRecord[];
  userOverride?: ModelRef;
}
