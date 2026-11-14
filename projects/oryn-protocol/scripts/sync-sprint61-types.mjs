import { readFile, writeFile } from "node:fs/promises";
const root = new URL("../", import.meta.url);
const registry = JSON.parse(await readFile(new URL("schemas/task-contracts/registry.json", root), "utf8"));
const ids = ["unconditional-image-generation-v1", "audio-question-answering-v1", "audio-transformation-v1",
  "multimodal-generation-v1", "image-text-to-video-v1", "video-editing-v1", "tabular-classification-v1",
  "tabular-regression-v1", "time-series-forecasting-v1", "pose-estimation-v1"];
function type(s) {
  if (s.$ref === "#/$defs/engineOutputArtifact") return "EngineOutputArtifact";
  if (s.$ref) throw new Error("Unreviewed reference " + s.$ref);
  if (s.anyOf || s.oneOf) return (s.anyOf ?? s.oneOf).map(type).join(" | ");
  if (Object.hasOwn(s, "const")) return JSON.stringify(s.const);
  if (s.enum) return s.enum.map(v => JSON.stringify(v)).join(" | ");
  if (s.type === "object") return "{ " + Object.entries(s.properties).map(([k, v]) =>
    JSON.stringify(k) + (s.required.includes(k) ? "" : "?") + ": " + type(v)).join("; ") + " }";
  if (s.type === "array") return s.prefixItems ? "[" + s.prefixItems.map(type).join(", ") + "]" : "(" + type(s.items) + ")[]";
  return ({integer: "number", number: "number", string: "string", null: "null", boolean: "boolean"})[s.type] ?? (() => { throw new Error("Unknown schema"); })();
}
let source = await readFile(new URL("typescript/src/shared-tasks.ts", root), "utf8");
for (const id of ids) source = source.replace(new RegExp(`^  "${id}":.*\\n`, "gm"), "").replace(`  "${id}",\n`, "");
for (const [name, schema] of [["SharedTaskInputs", "inputSchema"], ["SharedTaskOutputs", "outputSchema"]]) {
  source = source.replace("export interface " + name + " {", "export interface " + name + " {\n" +
    ids.map(id => `  "${id}": ${type(registry.contracts[id][schema])};`).join("\n"));
}
source = source.replace("export const SHARED_TASK_CONTRACTS = [", "export const SHARED_TASK_CONTRACTS = [\n" + ids.map(id => `  "${id}",`).join("\n"));
await writeFile(new URL("typescript/src/shared-tasks.ts", root), source);
