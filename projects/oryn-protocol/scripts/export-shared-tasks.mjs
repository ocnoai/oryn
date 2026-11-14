import { readFile, writeFile, copyFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sdk = process.argv[2];
if (!sdk) throw new Error("Pass the destination SDK repository path.");
const source = await readFile(resolve(root, "typescript/src/shared-tasks.ts"), "utf8");
const names = [...source.slice(source.indexOf("export const SHARED_TASK_CONTRACTS"), source.indexOf("] as const;")).matchAll(/"([a-z-]+-v1)"/g)].map((match) => match[1]);
const registry = JSON.parse(await readFile(resolve(root, "schemas/task-contracts/registry.json"), "utf8"));
await writeFile(resolve(sdk, "src/task-contract-revisions.ts"), "// Generated from canonical Protocol contracts; do not hand-edit.\nexport const TASK_CONTRACT_REVISIONS: Readonly<Record<string, number>> = " + JSON.stringify(Object.fromEntries(Object.values(registry.contracts).map(c => [c.id, c.revision])), null, 2) + ";\n");
const keywords = new Set(["type", "properties", "required", "additionalProperties", "items", "minItems", "maxItems", "uniqueItems", "contains", "minContains", "const", "enum", "minimum", "maximum", "exclusiveMinimum", "multipleOf", "minLength", "maxLength", "pattern", "default", "anyOf"]);
function check(schema) {
  if (Object.keys(schema).some(key => !keywords.has(key))) throw new Error("SDK schema evaluator needs a versioned keyword update.");
  for (const child of Object.values(schema.properties ?? {})) check(child);
  if (schema.items) check(schema.items);
  if (schema.contains) check(schema.contains);
  for (const child of schema.anyOf ?? []) check(child);
}
for (const id of names) check(registry.contracts[id].inputSchema);
await writeFile(resolve(sdk, "src/shared-tasks.ts"), "// Generated from oryn-protocol; do not hand-edit.\n" + source.replaceAll("EngineOutputArtifact", "ArtifactMetadata"));
await writeFile(resolve(sdk, "src/shared-task-schemas.ts"), "// Generated from canonical Protocol input schemas; do not hand-edit.\nexport const sharedTaskSchemas: Record<string, unknown> = " + JSON.stringify(Object.fromEntries(names.map((id) => [id, registry.contracts[id].inputSchema])), null, 2) + ";\n");
await copyFile(resolve(root, "typescript/src/task-relations.ts"), resolve(sdk, "src/task-relations.ts"));
await copyFile(resolve(root, "typescript/src/sprint61-relations.ts"), resolve(sdk, "src/sprint61-relations.ts"));
await copyFile(resolve(root, "fixtures/shared-tasks/inputs.json"), resolve(sdk, "src/__tests__/fixtures/shared-task-inputs.json"));
