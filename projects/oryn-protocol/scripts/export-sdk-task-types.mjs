import { readFile, writeFile } from "node:fs/promises";
const source = await readFile(
  new URL("../typescript/src/task-contracts.ts", import.meta.url),
  "utf8",
);
const destination = process.argv[2];
if (!destination)
  throw new Error("Pass the destination SDK task-contracts.ts path.");
await writeFile(
  destination,
  "// Generated from @ocno/oryn-protocol task-contracts.ts; do not hand-edit.\n" +
    source.slice(
      source.indexOf("export interface TaskContractRef"),
      source.indexOf("export interface TaskContract {"),
    ),
);
