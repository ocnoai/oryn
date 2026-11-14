import Ajv2020 from "ajv/dist/2020.js";
import registry from "../../schemas/task-contracts/registry.json";
import schema from "../../schemas/v1/protocol-v1.schema.json";
import { validTaskRelations } from "./task-relations.js";

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
export interface TaskContract {
  id: string;
  revision: number;
  task: string;
  taskAliases?: string[];
  inputModalities: string[];
  outputModalities: string[];
  inputSchema: { properties: Record<string, Record<string, unknown>> };
  resultSchema: Record<string, unknown>;
  parameterNames: string[];
  declarationRequiredParameters?: string[];
  featureDefinitions: Record<
    string,
    { requiredInput?: Record<string, unknown> }
  >;
  streamingChannels: string[];
  limitNames: string[];
  modelBacked?: boolean;
}
export const TASK_CONTRACTS = registry.contracts as unknown as Record<
  string,
  TaskContract
>;
const ajv = new Ajv2020({ strict: false, allErrors: true });
ajv.addSchema(schema);
const validators = new Map<string, ReturnType<typeof ajv.compile>>();
function matches(definition: string, value: unknown): boolean {
  let v = validators.get(definition);
  if (!v) {
    v = ajv.compile({ $ref: `${schema.$id}#/$defs/${definition}` });
    validators.set(definition, v);
  }
  return v(value) as boolean;
}
function bounded(value: unknown, maximumBytes = 262144): void {
  const encoded = JSON.stringify(value);
  if (!encoded || new TextEncoder().encode(encoded).length > maximumBytes)
    throw new Error("TASK_CONTRACT_INVALID");
  const check = (v: unknown, depth: number): void => {
    if (depth > 16 || (typeof v === "number" && !Number.isFinite(v)))
      throw new Error("TASK_CONTRACT_INVALID");
    if (v && typeof v === "object")
      for (const [k, x] of Object.entries(v)) {
        if (["__proto__", "constructor", "prototype"].includes(k))
          throw new Error("TASK_CONTRACT_INVALID");
        check(x, depth + 1);
      }
  };
  check(value, 0);
}
export function validateTaskRequirements(value: unknown): TaskRequirements {
  bounded(value);
  if (!matches("taskRequirements", value))
    throw new Error("TASK_REQUIREMENTS_INVALID");
  const r = value as TaskRequirements;
  const c = r.contract && TASK_CONTRACTS[r.contract.id];
  if (
    r.contract &&
    (!c ||
      c.revision !== r.contract.revision ||
      (r.task && ![c.task, ...(c.taskAliases ?? [])].includes(r.task)))
  )
    throw new Error("TASK_CONTRACT_UNSUPPORTED");
  if (
    (r.requiredFeatures?.length ||
      Object.keys(r.parameterValues ?? {}).length) &&
    !c
  )
    throw new Error("TASK_CONTRACT_REQUIRED");
  if (c) {
    if (
      r.requiredFeatures?.some((f) => !Object.hasOwn(c.featureDefinitions, f))
    )
      throw new Error("TASK_FEATURE_UNKNOWN");
    for (const [k, v] of Object.entries(r.parameterValues ?? {})) {
      if (
        !c.parameterNames.includes(k) ||
        !ajv.validate(c.inputSchema.properties[k]!, v)
      )
        throw new Error("TASK_PARAMETER_INVALID");
    }
    for (const f of r.requiredFeatures ?? [])
      for (const [k, v] of Object.entries(
        c.featureDefinitions[f]?.requiredInput ?? {},
      )) {
        if (
          Object.hasOwn(r.parameterValues ?? {}, k) &&
          JSON.stringify(r.parameterValues![k]) !== JSON.stringify(v)
        )
          throw new Error("TASK_REQUIREMENTS_CONTRADICTORY");
      }
  }
  return r;
}
function safeConstraint(value: unknown, depth = 0): void {
  if (!value || typeof value !== "object" || Array.isArray(value) || depth > 8)
    throw new Error("TASK_PROFILE_INVALID");
  const allowed = new Set([
    "type",
    "enum",
    "const",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "uniqueItems",
    "items",
    "allOf",
    "default",
    "pattern",
  ]);
  for (const [k, v] of Object.entries(value)) {
    if (!allowed.has(k)) throw new Error("TASK_PROFILE_INVALID");
    if (k === "allOf") {
      if (!Array.isArray(v) || v.length > 4)
        throw new Error("TASK_PROFILE_INVALID");
      v.forEach((x) => safeConstraint(x, depth + 1));
    }
    if (k === "items") safeConstraint(v, depth + 1);
    // Custom regular expressions are never accepted from providers.
    if (k === "pattern" && v !== "\\S" && v !== "^[a-z]{2,3}(?:-[A-Z]{2})?$" && v !== "^artifact_[A-Za-z0-9_-]{20,80}$")
      throw new Error("TASK_PROFILE_INVALID");
  }
}
export function validateTaskProfile(value: unknown): TaskProfile {
  bounded(value);
  if (!matches("taskProfile", value)) throw new Error("TASK_PROFILE_INVALID");
  const p = value as TaskProfile,
    c = TASK_CONTRACTS[p.contract.id];
  if (
    !c ||
    c.revision !== p.contract.revision ||
    ![c.task, ...(c.taskAliases ?? [])].includes(p.task) ||
    p.inputSchemaRef !== `taskInput.${c.id}` ||
    p.resultSchemaRef !== `taskResult.${c.id}`
  )
    throw new Error("TASK_PROFILE_INVALID");
  if (p.inputModalities.length === 0 && c.inputModalities.length > 0)
    throw new Error("TASK_PROFILE_INVALID");
  if (
    Object.keys(p.features).some(
      (k) => !Object.hasOwn(c.featureDefinitions, k),
    ) ||
    Object.keys(p.limits).some((k) => !c.limitNames.includes(k)) ||
    p.streamingChannels.some((k) => !c.streamingChannels.includes(k))
  )
    throw new Error("TASK_PROFILE_INVALID");
  for (const [k, v] of Object.entries(p.parameters)) {
    if (!c.parameterNames.includes(k) || v.inputPointer !== `/${k}`)
      throw new Error("TASK_PROFILE_INVALID");
    safeConstraint(v.valueSchema);
    const base = { ...c.inputSchema.properties[k] };
    delete base.default;
    const canonical = (value: unknown): string => {
      if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
      if (value && typeof value === "object")
        return `{${Object.entries(value)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([key, val]) => `${JSON.stringify(key)}:${canonical(val)}`)
          .join(",")}}`;
      return JSON.stringify(value);
    };
    const same = (a: unknown, b: unknown) => canonical(a) === canonical(b);
    const effective = v.valueSchema;
    const narrowed =
      same(effective, base) ||
      (Array.isArray(effective.allOf) &&
        effective.allOf.length >= 2 &&
        same(effective.allOf[0], base)) ||
      (Object.hasOwn(effective, "const") &&
        same(effective, { ...base, const: effective.const }));
    if (!narrowed) throw new Error("TASK_PROFILE_WIDENS_CONTRACT");
    if (
      Object.hasOwn(v, "default") &&
      (!ajv.validate(c.inputSchema.properties[k]!, v.default) ||
        !ajv.validate(v.valueSchema, v.default))
    )
      throw new Error("TASK_PROFILE_INVALID");
  }
  return p;
}
export function taskRequirementFailures(
  profile: TaskProfile | undefined,
  requirements: TaskRequirements,
): string[] {
  validateTaskRequirements(requirements);
  if (!profile)
    return requirements.contract ||
      requirements.requiredFeatures?.length ||
      Object.keys(requirements.parameterValues ?? {}).length
      ? ["TASK_PROFILE_UNAVAILABLE"]
      : [];
  const p = validateTaskProfile(profile),
    r = requirements,
    failures: string[] = [];
  if (
    r.contract &&
    (r.contract.id !== p.contract.id ||
      r.contract.revision !== p.contract.revision)
  )
    failures.push("TASK_CONTRACT_MISMATCH");
  if (
    r.task &&
    ![p.task, ...(TASK_CONTRACTS[p.contract.id]?.taskAliases ?? [])].includes(
      r.task,
    )
  )
    failures.push("TASK_MISMATCH");
  for (const field of ["inputModalities", "outputModalities"] as const)
    if (r[field]?.some((m) => !(p[field] as string[]).includes(m)))
      failures.push("TASK_MODALITY_MISMATCH");
  for (const k of r.requiredFeatures ?? [])
    if (p.features[k] !== "supported")
      failures.push(`TASK_FEATURE_UNSUPPORTED:${k}`);
  for (const [k, v] of Object.entries(r.parameterValues ?? {})) {
    const param = p.parameters[k];
    if (
      !param ||
      !["honored", "fixed"].includes(param.handling) ||
      !ajv.validate(
        TASK_CONTRACTS[p.contract.id]!.inputSchema.properties[k]!,
        v,
      ) ||
      !ajv.validate(param.valueSchema, v)
    )
      failures.push(`TASK_PARAMETER_UNSUPPORTED:${k}`);
  }
  return failures;
}
export function validateTaskInvocation(
  profile: TaskProfile,
  value: unknown,
  requirements: TaskRequirements = {},
): void {
  bounded(value, 16 * 1024 * 1024);
  const p = validateTaskProfile(profile),
    c = TASK_CONTRACTS[p.contract.id]!;
  if (
    taskRequirementFailures(p, requirements).length ||
    !matches(p.inputSchemaRef, value)
  )
    throw new Error("TASK_INPUT_INVALID");
  const input = value as Record<string, unknown>;
  if (!validTaskRelations(c.id, input)) throw new Error("TASK_INPUT_INVALID");
  if (c.id === "speech-synthesis-v1") {
    const voice = input.voice ?? p.parameters.voice?.default;
    const reference = Object.hasOwn(input, "audioArtifactId");
    if ((voice === "reference") !== reference || (reference && !p.inputModalities.includes("audio")))
      throw new Error("TASK_INPUT_INVALID");
    if ((input.text as string).length > (p.limits.textCharacters ?? 350))
      throw new Error("TASK_INPUT_LIMIT_EXCEEDED");
  }
  if (c.id === "time-series-forecasting-v1") {
    const horizon = input.horizon ?? p.parameters.horizon?.default ?? 16;
    const columns = input.pastCovariates ? (input.pastCovariates as unknown[][])[0]!.length : 0;
    if ((input.context as unknown[]).length > (p.limits.contextLength ?? 2048)
      || Number(horizon) > (p.limits.forecastHorizon ?? 512)
      || columns > (p.limits.covariateColumns ?? 32))
      throw new Error("TASK_INPUT_LIMIT_EXCEEDED");
  }
  if (c.id === "zero-shot-classification-v1" && (input.labels as unknown[]).length > (p.limits.candidateLabels ?? 32))
    throw new Error("TASK_INPUT_LIMIT_EXCEEDED");
  if (c.id === "embeddings-v1") {
    const texts = input.texts as string[];
    if (texts.length > (p.limits.batchItems ?? 64) || texts.some(text => [...text].length > (p.limits.textCharacters ?? 8192)))
      throw new Error("TASK_INPUT_LIMIT_EXCEEDED");
  }
  if (c.id === "table-question-answering-v1") {
    if ((input.rows as unknown[]).length > (p.limits.tableRows ?? 128) || (input.columns as unknown[]).length > (p.limits.tableColumns ?? 32))
      throw new Error("TASK_INPUT_LIMIT_EXCEEDED");
  }
  if (c.id === "tabular-classification-v1" || c.id === "tabular-regression-v1") {
    const counts: Record<string, number> = {
      trainingRows: (input.trainingRows as unknown[]).length,
      queryRows: (input.queryRows as unknown[]).length,
      featureColumns: (input.columns as unknown[]).length,
    };
    if (c.id === "tabular-classification-v1") counts.classes = new Set(input.targets as string[]).size;
    if (Object.entries(counts).some(([key, count]) => count > (p.limits[key] ?? count)))
      throw new Error("TASK_INPUT_LIMIT_EXCEEDED");
  }
  if (c.id === "multimodal-generation-v1" || c.id === "audio-question-answering-v1") {
    for (const [field, modality] of [["imageArtifactId", "image"], ["audioArtifactId", "audio"]] as const)
      if (Object.hasOwn(input, field) && !p.inputModalities.includes(modality)) throw new Error("TASK_INPUT_INVALID");
  }
  if (c.id === "multimodal-generation-v1") {
    const mode = String(input.outputMode ?? p.parameters.outputMode?.default ?? "text");
    const required = {text: ["text"], audio: ["audio"], "text-and-audio": ["text", "audio"]} as Record<string, string[]>;
    if (!required[mode]!.every(modality => (p.outputModalities as string[]).includes(modality))) throw new Error("TASK_INPUT_INVALID");
  }
  if (p.contract.id === "chat-v1" && (input.messages as unknown[]).length > Number(p.limits.maxMessages ?? 128))
    throw new Error("TASK_INPUT_LIMIT_EXCEEDED");
  if (c.id === "image-editing-v1") {
    const count = 1 + ((input.referenceImageArtifactIds as string[] | undefined)?.length ?? 0);
    if (count < (p.limits.minReferenceImages ?? 1) || count > (p.limits.maxReferenceImages ?? 1))
      throw new Error("TASK_INPUT_LIMIT_EXCEEDED");
  }
  for (const k of c.parameterNames) {
    if (!Object.hasOwn(input, k)) continue;
    const control = p.parameters[k];
    if (
      !control ||
      !["honored", "fixed"].includes(control.handling) ||
      !ajv.validate(control.valueSchema, input[k])
    )
      throw new Error("TASK_PARAMETER_UNSUPPORTED");
  }
  const effective = (k: string): unknown =>
    Object.hasOwn(input, k) ? input[k] : p.parameters[k]?.default;
  for (const [k, v] of Object.entries(requirements.parameterValues ?? {}))
    if (JSON.stringify(effective(k)) !== JSON.stringify(v))
      throw new Error("TASK_SESSION_PARAMETER_MISMATCH");
  for (const f of requirements.requiredFeatures ?? [])
    for (const [k, v] of Object.entries(
      c.featureDefinitions[f]?.requiredInput ?? {},
    ))
      if (JSON.stringify(effective(k)) !== JSON.stringify(v))
        throw new Error("TASK_FEATURE_INPUT_REQUIRED");
}

/** Strip legacy model-level requirements before sending the task negotiation envelope. */
export function taskRequirementsFrom(
  value: TaskRequirements | undefined,
): TaskRequirements | undefined {
  if (!value) return undefined;
  const result = Object.fromEntries(
    [
      "task",
      "contract",
      "requiredFeatures",
      "parameterValues",
      "inputModalities",
      "outputModalities",
    ]
      .filter((k) => (value as Record<string, unknown>)[k] !== undefined)
      .map((k) => [k, (value as Record<string, unknown>)[k]]),
  );
  return validateTaskRequirements(result);
}
