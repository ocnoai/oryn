import { readFile, writeFile, mkdir, copyFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const read = async (p) => JSON.parse(await readFile(resolve(root, p), "utf8"));
const wire = await read("schemas/v1/protocol-v1.schema.json");
const registry = await read("schemas/task-contracts/registry.json");
const profile = await read("schemas/task-contracts/task-profile.schema.json");
const rewrite = (value) => {
  if (Array.isArray(value)) return value.map(rewrite);
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value).map(([k, v]) => [
        k,
        k === "$ref" && v.startsWith("#/$defs/")
          ? v.replace("#/$defs/", "#/$defs/taskProfile.")
          : rewrite(v),
      ]),
    );
  return value;
};
const { $schema, $id, $defs, ...shape } = profile;
wire.$defs.taskProfile = rewrite(shape);
for (const [key, value] of Object.entries($defs))
  wire.$defs["taskProfile." + key] = rewrite(value);
wire.$defs.taskRequirements = {
  type: "object",
  additionalProperties: false,
  maxProperties: 6,
  properties: {
    task: { type: "string", minLength: 1, maxLength: 128 },
    contract: { $ref: "#/$defs/taskProfile.contract" },
    requiredFeatures: {
      type: "array",
      uniqueItems: true,
      maxItems: 64,
      items: { type: "string", pattern: "^[a-z][a-zA-Z0-9]{0,63}$" },
    },
    parameterValues: {
      type: "object",
      maxProperties: 64,
      propertyNames: { pattern: "^[a-z][a-zA-Z0-9]{0,63}$" },
    },
    inputModalities: {
      type: "array",
      maxItems: 5,
      uniqueItems: true,
      items: { enum: ["text", "image", "audio", "video", "document"] },
    },
    outputModalities: {
      type: "array",
      maxItems: 6,
      uniqueItems: true,
      items: {
        enum: ["text", "image", "audio", "video", "embedding", "structured"],
      },
    },
  },
};
for (const c of Object.values(registry.contracts)) {
  wire.$defs["taskInput." + c.id] = c.inputSchema;
  wire.$defs["taskResult." + c.id] = c.resultSchema;
  if (c.outputSchema) wire.$defs["taskOutput." + c.id] = c.outputSchema;
}
for (const name of [
  "modelInspectRequest",
  "modelInstallRequest",
  "sessionAttachRequest",
])
  wire.$defs[name].properties.params.properties.requirements = {
    $ref: "#/$defs/taskRequirements",
  };
for (const name of ["candidateExecutionProfile", "sessionAttachedResult"])
  wire.$defs[name].properties.taskProfile = { $ref: "#/$defs/taskProfile" };
wire.$defs.candidateExecutionProfile.properties.installationRequired = {
  type: "boolean",
};
wire.$defs.candidateExecutionProfile.properties.providerDiagnostics = {
  type: "array",
  maxItems: 32,
  items: {
    type: "object",
    additionalProperties: false,
    required: ["providerId", "code"],
    properties: {
      providerId: { type: "string", maxLength: 128 },
      code: { const: "PROVIDER_INSPECTION_FAILED" },
    },
  },
};
wire.$defs.runtimeCapabilitiesResult.properties.features.properties.taskProfiles =
  { type: "boolean" };
await writeFile(
  resolve(root, "schemas/v1/protocol-v1.schema.json"),
  JSON.stringify(wire, null, 2) + "\n",
);
const py = resolve(root, "python/src/oryn_protocol/schemas/task-contracts");
await mkdir(py, { recursive: true });
for (const name of ["registry.json", "task-profile.schema.json"])
  await copyFile(
    resolve(root, "schemas/task-contracts", name),
    resolve(py, name),
  );
