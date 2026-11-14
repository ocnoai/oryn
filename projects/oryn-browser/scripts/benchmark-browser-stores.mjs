/* global process, performance, setImmediate, console */
import fs from 'node:fs';
import { syncBuiltinESMExports } from 'node:module';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import ts from 'typescript';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const baselinePath = process.argv[2];
if (!baselinePath)
  throw new Error('Usage: node scripts/benchmark-browser-stores.mjs <baseline.json>');
const baseline = JSON.parse(fs.readFileSync(baselinePath, 'utf8')).repositories[
  'projects/oryn-browser'
];
const temporary = fs.mkdtempSync(join(tmpdir(), 'oryn-store-benchmark-'));
const sourceRoot = 'apps/browser/src/main/data';
const digest = (value) => createHash('sha256').update(value).digest('hex');
const originalRename = fs.renameSync;
let writes = 0;
fs.renameSync = (...args) => {
  writes++;
  return originalRename(...args);
};
syncBuiltinESMExports();
const results = [];
try {
  for (const variant of ['baseline', 'current']) {
    const directory = join(temporary, variant);
    fs.mkdirSync(directory);
    const identities = {};
    for (const name of ['json-file-store', 'download-store', 'history-store']) {
      const path = `${sourceRoot}/${name}.ts`;
      const source =
        variant === 'baseline'
          ? execFileSync('git', ['show', `${baseline.revision}:${path}`], {
              cwd: root,
              encoding: 'utf8',
            })
          : fs.readFileSync(join(root, path), 'utf8');
      const sha256 = digest(source);
      if (variant === 'baseline' && sha256 !== baseline.file_sha256[path])
        throw new Error(
          `Cannot reproduce the captured baseline for ${name}; refusing a misleading comparison.`,
        );
      identities[name] = sha256;
      const output = ts
        .transpileModule(source, {
          compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
        })
        .outputText.replaceAll("'./json-file-store'", "'./json-file-store.mjs'");
      fs.writeFileSync(join(directory, `${name}.mjs`), output);
    }
    const { DownloadStore } = await import(
      pathToFileURL(join(directory, 'download-store.mjs')).href
    );
    const { BrowsingHistoryStore } = await import(
      pathToFileURL(join(directory, 'history-store.mjs')).href
    );
    for (const scenario of ['download-progress', 'large-history']) {
      const path = join(directory, `${scenario}.json`);
      const date = '2026-10-10T00:00:00.000Z';
      let store;
      let operation;
      if (scenario === 'download-progress') {
        fs.writeFileSync(
          path,
          JSON.stringify({
            version: 1,
            records: Array.from({ length: 1000 }, (_, index) => ({
              downloadId: `download_${index}`,
              url: 'https://fixture.test/file',
              filename: 'file',
              receivedBytes: 1000,
              totalBytes: 1000,
              state: 'completed',
              startedAt: date,
              updatedAt: date,
            })),
          }),
        );
        store = new DownloadStore(path);
        const active = store.start({
          url: 'https://fixture.test/active',
          filename: 'active',
          totalBytes: 1000,
        });
        operation = () => {
          for (let index = 1; index <= 1000; index++)
            store.update(active.downloadId, { receivedBytes: index });
          store.flush?.();
        };
      } else {
        fs.writeFileSync(
          path,
          JSON.stringify({
            version: 1,
            entries: Array.from({ length: 10000 }, (_, index) => ({
              historyId: `history_${index}`,
              url: `https://fixture.test/${index}`,
              title: 'Fixture',
              visitedAt: date,
            })),
          }),
        );
        store = new BrowsingHistoryStore(path);
        operation = () => {
          for (let index = 0; index < 20; index++)
            store.recordVisit(`https://fixture.test/new/${index}`, 'New fixture');
          store.flush?.();
        };
      }
      writes = 0;
      const start = performance.now();
      const probe = new Promise((done) => setImmediate(() => done(performance.now() - start)));
      operation();
      const operationMs = performance.now() - start;
      const eventLoopDelayMs = await probe;
      results.push({
        variant,
        scenario,
        operationMs,
        eventLoopDelayMs,
        snapshotWrites: writes,
        identities,
      });
    }
  }
  console.log(
    JSON.stringify(
      {
        formatVersion: 1,
        node: process.version,
        platform: process.platform,
        architecture: process.arch,
        note: 'Single synthetic run on this host, not a throughput or latency SLA. History includes the new durability flush cost. Bookmark baseline did not match Git HEAD and is not compared.',
        results,
      },
      null,
      2,
    ),
  );
} finally {
  fs.renameSync = originalRename;
  syncBuiltinESMExports();
  fs.rmSync(temporary, { recursive: true, force: true });
}
