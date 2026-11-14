import type { ErrorObject, ValidateFunction } from "ajv";
import Ajv2020 from "ajv/dist/2020.js";
import protocolSchema from "../../schemas/v1/protocol-v1.schema.json";
import { validTaskRelations, validTaskOutputRelations } from "./task-relations.js";
import { SHARED_TASK_CONTRACTS, type SharedTaskContract, type SharedTaskInputs, type SharedTaskOutputs } from "./shared-tasks.js";

import type {
  CandidateExecutionProfile,
  EngineRuntimeCapabilities,
  EngineJobEvent,
  EngineJobReference,
  EngineJobStatus,
  EngineSessionAttached,
  ModelInstallProgressEvent,
  ModelInstallResult,
  ModelCatalogMatchResult,
  ModelInstallStateEvent,
  ModelInventoryChangedEvent,
  ModelInventoryResult,
  ModelRemovePreviewResult,
  ModelRemoveResult,
  ModelResidencyChangedEvent,
  ModelResidencyResult,
  HandshakeResponse,
  ProtocolEvent,
  ProtocolPingResult,
  ProtocolResponse,
  RuntimeMemoryPressureChangedEvent,
  RuntimePackCleanupResult,
  RuntimePackInventoryResult,
  RuntimePolicyUpdateResult,
  RuntimeOverrideResult,
  RuntimeResourceSnapshot,
  RuntimeResourcesUpdatedEvent,
  RuntimeShutdownResult,
  RuntimeStateEvent,
  SessionReleaseResult,
} from "./types.js";

const ajv = new Ajv2020({ allErrors: true, strict: true });
ajv.addSchema(protocolSchema);

function compileDefinition<T>(definition: string): ValidateFunction<T> {
  return ajv.compile<T>({
    $ref: `${protocolSchema.$id}#/$defs/${definition}`,
  });
}

const sharedTaskValidators = new Map<string, ValidateFunction>(
  SHARED_TASK_CONTRACTS.map((contract) => [contract, compileDefinition(`taskInput.${contract}`)]),
);

export function isSharedTaskContract(value: string): value is SharedTaskContract {
  return sharedTaskValidators.has(value);
}

export function validateSharedTaskInput<C extends SharedTaskContract>(
  contract: C, input: unknown,
): SharedTaskInputs[C] {
  const validate = sharedTaskValidators.get(contract);
  if (!validate || !validate(input) ||
      !validTaskRelations(contract, input as Record<string, unknown>) ||
      new TextEncoder().encode(JSON.stringify(input)).byteLength > 256 * 1024) {
    throw new ProtocolValidationError("Task input is invalid or unavailable.", validate?.errors);
  }
  return input as SharedTaskInputs[C];
}

const sharedOutputValidators = new Map<string, ValidateFunction>(
  SHARED_TASK_CONTRACTS.map((contract) => [contract, compileDefinition(`taskOutput.${contract}`)]),
);
export function validateSharedTaskOutput<C extends SharedTaskContract>(contract: C, output: unknown): SharedTaskOutputs[C] {
  const validate = sharedOutputValidators.get(contract);
  const limit = contract === "visual-document-retrieval-v1" && output !== null && typeof output === "object" && "dimensions" in output && output.dimensions === 4096
    ? 12 * 1024 * 1024 : (["visual-document-retrieval-v1", "pose-estimation-v1"].includes(contract) ? 900 : 256) * 1024;
  if (!validate || !validate(output) || new TextEncoder().encode(JSON.stringify(output)).byteLength > limit || !validTaskOutputRelations(contract, output as Record<string, unknown>)) {
    throw new ProtocolValidationError("Task output is invalid or unavailable.", validate?.errors);
  }
  return output as SharedTaskOutputs[C];
}

const handshakeResponseValidator =
  compileDefinition<HandshakeResponse>("handshakeResponse");
const protocolResponseValidator =
  compileDefinition<ProtocolResponse>("protocolResponse");
const protocolEventValidator =
  compileDefinition<ProtocolEvent>("protocolEvent");
const runtimeCapabilitiesValidator =
  compileDefinition<EngineRuntimeCapabilities>("runtimeCapabilitiesResult");
const runtimeStateEventValidator =
  compileDefinition<RuntimeStateEvent>("runtimeStateEvent");
const pingResultValidator = compileDefinition<ProtocolPingResult>("pingResult");
const runtimeShutdownResultValidator = compileDefinition<RuntimeShutdownResult>(
  "runtimeShutdownResult",
);
const runtimeResourceSnapshotValidator =
  compileDefinition<RuntimeResourceSnapshot>("runtimeResourceSnapshot");
const runtimePolicyUpdateResultValidator =
  compileDefinition<RuntimePolicyUpdateResult>("runtimePolicyUpdateResult");
const runtimeResourcesUpdatedEventValidator =
  compileDefinition<RuntimeResourcesUpdatedEvent>(
    "runtimeResourcesUpdatedEvent",
  );
const runtimeMemoryPressureChangedEventValidator =
  compileDefinition<RuntimeMemoryPressureChangedEvent>(
    "runtimeMemoryPressureChangedEvent",
  );
const runtimePackInventoryResultValidator =
  compileDefinition<RuntimePackInventoryResult>("runtimePackInventoryResult");
const runtimePackCleanupResultValidator =
  compileDefinition<RuntimePackCleanupResult>("runtimePackCleanupResult");
const runtimeOverrideResultValidator =
  compileDefinition<RuntimeOverrideResult>("runtimePlanSummary");
const candidateExecutionProfileValidator =
  compileDefinition<CandidateExecutionProfile>("candidateExecutionProfile");
const modelCatalogMatchResultValidator =
  compileDefinition<ModelCatalogMatchResult>("modelCatalogMatchResult");
const modelInstallResultValidator =
  compileDefinition<ModelInstallResult>("modelInstallResult");
const modelInventoryResultValidator = compileDefinition<ModelInventoryResult>(
  "modelInventoryResult",
);
const modelRemovePreviewResultValidator =
  compileDefinition<ModelRemovePreviewResult>("modelRemovePreviewResult");
const modelRemoveResultValidator =
  compileDefinition<ModelRemoveResult>("modelRemoveResult");
const modelResidencyResultValidator = compileDefinition<ModelResidencyResult>(
  "modelResidencyResult",
);
const modelResidencyChangedEventValidator =
  compileDefinition<ModelResidencyChangedEvent>("modelResidencyChangedEvent");
const modelInstallProgressEventValidator =
  compileDefinition<ModelInstallProgressEvent>("modelInstallProgressEvent");
const modelInstallStateEventValidator =
  compileDefinition<ModelInstallStateEvent>("modelInstallStateEvent");
const modelInventoryChangedEventValidator =
  compileDefinition<ModelInventoryChangedEvent>("modelInventoryChangedEvent");
const sessionAttachedResultValidator = compileDefinition<EngineSessionAttached>(
  "sessionAttachedResult",
);
const sessionReleaseResultValidator = compileDefinition<SessionReleaseResult>(
  "sessionReleaseResult",
);
const jobReferenceValidator =
  compileDefinition<EngineJobReference>("jobReference");
const jobStatusResultValidator =
  compileDefinition<EngineJobStatus>("jobStatusResult");
const jobEventValidators = [
  "jobStateEvent",
  "jobQueueUpdatedEvent",
  "jobProgressEvent",
  "jobOutputDeltaEvent",
  "jobPreviewEvent",
  "jobCompletedEvent",
  "jobFailedEvent",
  "jobCancelledEvent",
].map((definition) => compileDefinition<EngineJobEvent>(definition));

export class ProtocolValidationError extends Error {
  readonly errors: ErrorObject[];

  constructor(message: string, errors: ErrorObject[] | null | undefined) {
    super(message);
    this.name = "ProtocolValidationError";
    this.errors = errors ?? [];
  }
}

function assertValid<T>(
  validator: ValidateFunction<T>,
  value: unknown,
  label: string,
): asserts value is T {
  if (!validator(value)) {
    throw new ProtocolValidationError(`Invalid ${label}.`, validator.errors);
  }
}

export function assertHandshakeResponse(
  value: unknown,
): asserts value is HandshakeResponse {
  assertValid(handshakeResponseValidator, value, "protocol handshake response");
}

export function assertProtocolResponse(
  value: unknown,
): asserts value is ProtocolResponse {
  assertValid(protocolResponseValidator, value, "protocol response");
}

export function assertProtocolEvent(
  value: unknown,
): asserts value is ProtocolEvent {
  assertValid(protocolEventValidator, value, "protocol event");
}

export function assertRuntimeCapabilities(
  value: unknown,
): asserts value is EngineRuntimeCapabilities {
  assertValid(
    runtimeCapabilitiesValidator,
    value,
    "runtime capabilities result",
  );
}

export function assertRuntimeStateEvent(
  value: unknown,
): asserts value is RuntimeStateEvent {
  assertValid(runtimeStateEventValidator, value, "runtime state event");
}

export function assertProtocolPingResult(
  value: unknown,
): asserts value is ProtocolPingResult {
  assertValid(pingResultValidator, value, "protocol ping result");
}

export function assertRuntimeShutdownResult(
  value: unknown,
): asserts value is RuntimeShutdownResult {
  assertValid(runtimeShutdownResultValidator, value, "runtime shutdown result");
}

export function assertRuntimePackInventoryResult(
  value: unknown,
): asserts value is RuntimePackInventoryResult {
  assertValid(
    runtimePackInventoryResultValidator,
    value,
    "runtime pack inventory result",
  );
}

export function assertRuntimePackCleanupResult(
  value: unknown,
): asserts value is RuntimePackCleanupResult {
  assertValid(
    runtimePackCleanupResultValidator,
    value,
    "runtime pack cleanup result",
  );
}

export function assertRuntimeOverrideResult(
  value: unknown,
): asserts value is RuntimeOverrideResult {
  assertValid(runtimeOverrideResultValidator, value, "runtime override result");
}

export function assertRuntimeResourceSnapshot(
  value: unknown,
): asserts value is RuntimeResourceSnapshot {
  assertValid(
    runtimeResourceSnapshotValidator,
    value,
    "runtime resource snapshot",
  );
}

export function assertRuntimePolicyUpdateResult(
  value: unknown,
): asserts value is RuntimePolicyUpdateResult {
  assertValid(
    runtimePolicyUpdateResultValidator,
    value,
    "runtime policy update result",
  );
}

export function assertRuntimeResourcesUpdatedEvent(
  value: unknown,
): asserts value is RuntimeResourcesUpdatedEvent {
  assertValid(
    runtimeResourcesUpdatedEventValidator,
    value,
    "runtime resources updated event",
  );
}

export function assertRuntimeMemoryPressureChangedEvent(
  value: unknown,
): asserts value is RuntimeMemoryPressureChangedEvent {
  assertValid(
    runtimeMemoryPressureChangedEventValidator,
    value,
    "runtime memory pressure changed event",
  );
}

export function assertCandidateExecutionProfile(
  value: unknown,
): asserts value is CandidateExecutionProfile {
  assertValid(
    candidateExecutionProfileValidator,
    value,
    "candidate execution profile",
  );
  assertCompatibilityProjection(value);
}

function assertCompatibilityProjection(value: CandidateExecutionProfile): void {
  const awareness = [
    value.descriptorSchemaVersion,
    value.capabilityRegistryDigest,
    value.eligibility,
    value.loadDisposition,
  ];
  if (awareness.every((item) => item === undefined)) return;
  if (awareness.some((item) => item === undefined)) {
    throw new ProtocolValidationError(
      "Incomplete compatibility-plane projection.",
      [],
    );
  }
  const eligibility = value.eligibility!;
  const states = Object.values(eligibility).map((axis) => axis.state);
  const authorizationOnly =
    eligibility.access.state === "authorization-required" &&
    Object.entries(eligibility).every(
      ([name, axis]) =>
        name === "access" ||
        axis.state === "deferred" ||
        ![
          "denied",
          "failed",
          "incomplete",
          "not-admissible",
          "unsafe",
          "unknown",
          "unsupported",
        ].includes(axis.state),
    );
  const expected = authorizationOnly
    ? "authorization-required"
    : eligibility.resources.state === "not-admissible" &&
        states.every(
          (state) =>
            !["unsafe", "unsupported", "failed", "denied"].includes(state),
        )
      ? "not-admissible-on-this-device"
      : eligibility.access.state === "available" &&
          eligibility.artifacts.state === "safe" &&
          eligibility.runtime.state === "supported" &&
          eligibility.task.state === "supported" &&
          eligibility.resources.state === "admissible" &&
          ["certified", "family-conformant", "untested"].includes(
            eligibility.conformance.state,
          )
        ? eligibility.conformance.state === "certified"
          ? "verified-load"
          : "trust-required"
        : "unsupported";
  if (value.loadDisposition !== expected) {
    throw new ProtocolValidationError(
      "Inconsistent compatibility load disposition.",
      [],
    );
  }
  const executable =
    expected === "verified-load" || expected === "trust-required";
  if (value.executable !== executable) {
    throw new ProtocolValidationError(
      "Inconsistent legacy executable projection.",
      [],
    );
  }
  const flattened = Object.values(eligibility).flatMap((axis) => axis.reasons);
  if (
    value.blockingReasons &&
    flattened.some((reason) => !value.blockingReasons?.includes(reason))
  ) {
    throw new ProtocolValidationError(
      "Legacy blocking reasons discard an eligibility reason.",
      [],
    );
  }
}

export function assertModelCatalogMatchResult(
  value: unknown,
): asserts value is ModelCatalogMatchResult {
  assertValid(
    modelCatalogMatchResultValidator,
    value,
    "model catalog match result",
  );
}

export function assertModelInstallResult(
  value: unknown,
): asserts value is ModelInstallResult {
  assertValid(modelInstallResultValidator, value, "model install result");
}

export function assertModelInventoryResult(
  value: unknown,
): asserts value is ModelInventoryResult {
  assertValid(modelInventoryResultValidator, value, "model inventory result");
}

export function assertModelRemovePreviewResult(
  value: unknown,
): asserts value is ModelRemovePreviewResult {
  assertValid(
    modelRemovePreviewResultValidator,
    value,
    "model remove preview result",
  );
}

export function assertModelRemoveResult(
  value: unknown,
): asserts value is ModelRemoveResult {
  assertValid(modelRemoveResultValidator, value, "model remove result");
}

export function assertModelResidencyResult(
  value: unknown,
): asserts value is ModelResidencyResult {
  assertValid(modelResidencyResultValidator, value, "model residency result");
}

export function assertModelResidencyChangedEvent(
  value: unknown,
): asserts value is ModelResidencyChangedEvent {
  assertValid(
    modelResidencyChangedEventValidator,
    value,
    "model residency changed event",
  );
}

export function assertModelInstallProgressEvent(
  value: unknown,
): asserts value is ModelInstallProgressEvent {
  assertValid(
    modelInstallProgressEventValidator,
    value,
    "model install progress event",
  );
}

export function assertModelInstallStateEvent(
  value: unknown,
): asserts value is ModelInstallStateEvent {
  assertValid(
    modelInstallStateEventValidator,
    value,
    "model install state event",
  );
}

export function assertModelInventoryChangedEvent(
  value: unknown,
): asserts value is ModelInventoryChangedEvent {
  assertValid(
    modelInventoryChangedEventValidator,
    value,
    "model inventory changed event",
  );
}

export function assertEngineSessionAttached(
  value: unknown,
): asserts value is EngineSessionAttached {
  assertValid(sessionAttachedResultValidator, value, "session attached result");
}

export function assertEngineJobReference(
  value: unknown,
): asserts value is EngineJobReference {
  assertValid(jobReferenceValidator, value, "job reference");
}

export function assertEngineJobStatus(
  value: unknown,
): asserts value is EngineJobStatus {
  assertValid(jobStatusResultValidator, value, "job status");
}

export function assertEngineJobEvent(
  value: unknown,
): asserts value is EngineJobEvent {
  if (!jobEventValidators.some((validator) => validator(value))) {
    throw new ProtocolValidationError("Invalid job event.", []);
  }
}

export function assertSessionReleaseResult(
  value: unknown,
): asserts value is SessionReleaseResult {
  assertValid(sessionReleaseResultValidator, value, "session release result");
}

export class ProtocolEventSequence {
  private lastSequence = 0;

  accept(event: ProtocolEvent): void {
    if (event.seq <= this.lastSequence) {
      throw new ProtocolValidationError(
        "Engine event sequence did not increase.",
        [],
      );
    }
    this.lastSequence = event.seq;
  }

  reset(): void {
    this.lastSequence = 0;
  }
}
