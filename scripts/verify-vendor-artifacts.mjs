#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { constants } from 'node:fs';
import { lstat, open, readFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const manifest = JSON.parse(await readFile(join(root, 'vendor-artifacts.json'), 'utf8'));
if (manifest.schemaVersion !== 1 || !Array.isArray(manifest.artifacts) || !manifest.artifacts.length)
  throw new Error('Invalid vendor-artifacts.json.');

const seen = new Set();
for (const artifact of manifest.artifacts) {
  if (
    !artifact || typeof artifact.path !== 'string' || isAbsolute(artifact.path) ||
    !/^components\/(?:browser|engine)\/vendor\/[A-Za-z0-9._-]+\.(?:tgz|whl|json)$/.test(artifact.path) ||
    artifact.path.includes('..') || seen.has(artifact.path) ||
    !/^[a-f0-9]{64}$/.test(artifact.sha256) ||
    !Number.isSafeInteger(artifact.bytes) || artifact.bytes <= 0 || artifact.bytes > 100 * 1024 ** 2
  ) throw new Error('Invalid vendor artifact entry.');
  seen.add(artifact.path);
  const parts = artifact.path.split('/');
  let parent = root;
  for (const part of parts.slice(0, -1)) {
    parent = join(parent, part);
    const entry = await lstat(parent);
    if (!entry.isDirectory() || entry.isSymbolicLink())
      throw new Error(`Vendor path must contain real directories: ${artifact.path}`);
  }
  const file = await open(join(root, artifact.path), constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const before = await file.stat();
    if (!before.isFile() || before.size !== artifact.bytes)
      throw new Error(`Vendor artifact size/type mismatch: ${artifact.path}`);
    const digest = createHash('sha256');
    for await (const chunk of file.createReadStream({ autoClose: false })) digest.update(chunk);
    const after = await file.stat();
    if (after.size !== before.size || after.mtimeMs !== before.mtimeMs || digest.digest('hex') !== artifact.sha256)
      throw new Error(`Vendor artifact integrity mismatch: ${artifact.path}`);
  } finally {
    await file.close();
  }
}
console.log(`Verified ${seen.size} bundled dependency artifacts.`);
