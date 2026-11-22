/** Website intent is data. Runtime, device, loader and download authority remain local. */
export interface SelectionModelRef {
  provider: "huggingface";
  repository: string;
  revision?: string;
  files?: string[];
}
export type ModelSelection =
  | {
      version: 1;
      intent: "source-checkpoint";
      sourceCheckpoint: SelectionModelRef;
    }
  | {
      version: 1;
      intent: "preferred-variant";
      sourceCheckpoint?: SelectionModelRef;
    }
  | { version: 1; intent: "exact-artifact" };
export interface SelectionExplanation {
  version: 1;
  kind: "requested-artifact" | "same-checkpoint" | "different-model";
  source:
    "local-measurements" | "active-configuration" | "deterministic-default";
  confidence: "measured" | "heuristic" | "unknown";
  catalogVersion: number;
  rulesVersion: number;
  ruleIds: string[];
  reasons: string[];
}
export function validateSelectionModelRef(value: unknown): SelectionModelRef {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("MODEL_REFERENCE_INVALID");
  const v = value as SelectionModelRef;
  if (
    Object.keys(v).some(
      (k) => !["provider", "repository", "revision", "files"].includes(k),
    ) ||
    v.provider !== "huggingface" ||
    typeof v.repository !== "string" ||
    !/^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/.test(
      v.repository,
    )
  )
    throw new Error("MODEL_REFERENCE_INVALID");
  if (
    v.revision !== undefined &&
    (typeof v.revision !== "string" ||
      !/^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$/.test(v.revision) ||
      v.revision.split("/").includes(".."))
  )
    throw new Error("MODEL_REFERENCE_INVALID");
  if (
    v.files !== undefined &&
    (!Array.isArray(v.files) ||
      v.files.length < 1 ||
      v.files.length > 256 ||
      new Set(v.files).size !== v.files.length ||
      v.files.some(
        (f) =>
          typeof f !== "string" ||
          !/^[A-Za-z0-9_. /-]{1,512}$/.test(f) ||
          f.startsWith("/") ||
          f.split("/").some((p) => p === "." || p === ".." || p === ""),
      ) ||
      !/^(?:[a-f0-9]{40}|[a-f0-9]{64})$/.test(v.revision ?? ""))
  )
    throw new Error("ARTIFACT_IDENTITY_INVALID");
  return { ...v, ...(v.files ? { files: [...v.files].sort() } : {}) };
}
export function validateModelSelection(
  value: unknown,
  preferred?: SelectionModelRef,
): ModelSelection {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("SELECTION_INVALID");
  const v = value as ModelSelection;
  if (
    v.version !== 1 ||
    !["source-checkpoint", "preferred-variant", "exact-artifact"].includes(
      v.intent,
    ) ||
    Object.keys(v).some(
      (k) => !["version", "intent", "sourceCheckpoint"].includes(k),
    )
  )
    throw new Error("SELECTION_INVALID");
  if (v.intent === "exact-artifact") {
    if ("sourceCheckpoint" in v || !preferred?.files?.length)
      throw new Error("EXACT_ARTIFACT_REQUIRED");
    validateSelectionModelRef(preferred);
  } else {
    if (v.intent === "source-checkpoint" && !v.sourceCheckpoint)
      throw new Error("CHECKPOINT_REQUIRED");
    if (v.intent === "preferred-variant" && !preferred)
      throw new Error("PREFERRED_ARTIFACT_REQUIRED");
    if (v.sourceCheckpoint) {
      validateSelectionModelRef(v.sourceCheckpoint);
      if (v.sourceCheckpoint.files) throw new Error("CHECKPOINT_FILES_INVALID");
    }
  }
  return structuredClone(v);
}
export function exactModelKey(model: {
  provider: string;
  repository: string;
  revision?: string;
  files?: string[];
}): string {
  return `${model.provider}:${model.repository}@${model.revision ?? "main"}${model.files ? ":" + JSON.stringify([...model.files].sort()) : ""}`;
}
