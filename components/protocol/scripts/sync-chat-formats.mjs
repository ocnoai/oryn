// Regenerate the closed chatTemplate list from the engine's reviewed chat-format table.
// Usage: node scripts/sync-chat-formats.mjs <path/to/chat-formats.json>
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const tablePath = process.argv[2];
if (!tablePath) throw new Error("usage: sync-chat-formats.mjs <chat-formats.json>");
const table = JSON.parse(await readFile(tablePath, "utf8"));
const names = Object.keys(table.formats ?? {}).sort();
if (
  table.schemaVersion !== 1 ||
  !names.includes("plain") ||
  names.some((name) => !/^[a-z][a-z0-9-]{1,31}$/.test(name))
)
  throw new Error("chat-format table is invalid");

const schemaPath = resolve(root, "schemas/v1/protocol-v1.schema.json");
const schema = JSON.parse(await readFile(schemaPath, "utf8"));
let replaced = 0;
const visit = (node) => {
  if (Array.isArray(node)) return node.forEach(visit);
  if (!node || typeof node !== "object") return;
  for (const [key, value] of Object.entries(node)) {
    if (key === "chatTemplate" && value && Array.isArray(value.enum)) {
      value.enum = names;
      replaced += 1;
    }
    visit(value);
  }
};
visit(schema);
if (!replaced) throw new Error("schema has no chatTemplate enum");
await writeFile(schemaPath, JSON.stringify(schema, null, 2) + "\n");

const typesPath = resolve(root, "typescript/src/types.ts");
const types = await readFile(typesPath, "utf8");
const pattern = /chatTemplate\?: (?:"[a-z0-9-]+"(?: \| )?)+;/g;
if (!pattern.test(types)) throw new Error("types.ts has no chatTemplate union");
await writeFile(
  typesPath,
  types.replace(pattern, `chatTemplate?: ${names.map((name) => `"${name}"`).join(" | ")};`),
);
