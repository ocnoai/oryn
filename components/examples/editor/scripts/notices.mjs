import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { readFile, readdir, realpath, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const seen = new Set();
const notices = [];
async function packageFor(name, from) {
  const require = createRequire(join(from, "package.json"));
  for (const location of require.resolve.paths(name) ?? []) {
    try {
      const folder = await realpath(join(location, name));
      const data = JSON.parse(
        await readFile(join(folder, "package.json"), "utf8"),
      );
      if (data.name === name) return { folder, data };
    } catch {
      /* Try the next package resolution directory. */
    }
  }
  throw new Error(`Cannot find third-party notice for ${name}`);
}
async function visit(name, from) {
  const { folder, data } = await packageFor(name, from);
  const key = `${data.name}@${data.version}`;
  if (seen.has(key)) return;
  seen.add(key);
  const names = (await readdir(folder)).filter((file) =>
    /^(licen[sc]e|copying|notice)(\.|$)/i.test(file),
  );
  const text = await Promise.all(
    names.map(
      async (file) =>
        `--- ${file} ---\n${await readFile(join(folder, file), "utf8")}`,
    ),
  );
  if (name === "@ocno/oryn-sdk" && !text.length) {
    data.license = "MIT";
    text.push(
      `--- LICENSE ---\n${await readFile(join(root, "..", "LICENSE"), "utf8")}`,
    );
  }
  if (!text.length) throw new Error(`Missing license notice: ${key}`);
  notices.push(
    `${key}\nLicense: ${typeof data.license === "string" ? data.license : "See package license"}\n${text.join("\n")}`,
  );
  for (const dependency of Object.keys(data.dependencies ?? {}))
    await visit(dependency, folder);
}
const own = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
for (const name of Object.keys(own.dependencies)) await visit(name, root);
notices.sort();
await writeFile(
  join(root, "public", "THIRD_PARTY_NOTICES.txt"),
  `Editor dependency licenses\n\nThese packages form the application's production dependency tree. Some are removed by bundling. Original license notices follow.\n\n${notices.join("\n\n========================================\n\n")}\n`,
);
