// Generated from @ocno/oryn-protocol task-contracts.ts; do not hand-edit.
export interface TaskContractRef {
  id: string;
  revision: number;
}
export type TaskFeatureState =
  "supported" | "unsupported" | "unknown" | "requires-input";
export interface TaskParameter {
  inputPointer: string;
  valueSchema: Record<string, unknown>;
  default?: unknown;
  handling: "honored" | "fixed" | "unsupported" | "requires-input";
  editable: boolean;
  affects: Array<"result" | "memory" | "latency">;
  unit?: string;
}
export interface TaskProfile {
  kind: "task-profile";
  schemaVersion: 1;
  profileId: string;
  contract: TaskContractRef;
  task: string;
  inputModalities: Array<"text" | "image" | "audio" | "video" | "document">;
  outputModalities: Array<
    "text" | "image" | "audio" | "video" | "embedding" | "structured"
  >;
  inputSchemaRef: string;
  resultSchemaRef: string;
  features: Record<string, TaskFeatureState>;
  parameters: Record<string, TaskParameter>;
  limits: Record<string, number>;
  streamingChannels: string[];
  extensions: Record<string, never>;
  provenance:
    | "runtime-tested"
    | "runtime-declared"
    | "adapter-derived"
    | "runtime-fallback";
}
export interface TaskRequirements {
  task?: string;
  contract?: TaskContractRef;
  requiredFeatures?: string[];
  parameterValues?: Record<string, unknown>;
  inputModalities?: TaskProfile["inputModalities"];
  outputModalities?: TaskProfile["outputModalities"];
}
