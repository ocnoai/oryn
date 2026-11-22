import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  closeSync,
  existsSync,
  mkdtempSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
  writeSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

import { afterEach, expect, it, vi } from 'vitest';

import { ArtifactManager } from '../artifact-manager/artifact-manager';
import { ArtifactQuotaManager } from '../artifact-manager/artifact-quota-manager';
import type { EngineOutputArtifact } from '../artifact-manager/types';

const owner = {
  origin: 'https://outputs.test',
  tabId: 'tab_outputs',
  webContentsId: 42,
  frameId: 7,
  frameUrl: 'https://outputs.test/',
  documentId: 'document_outputs',
};
const roots: string[] = [];
const managers: ArtifactManager[] = [];
function setup(options: ConstructorParameters<typeof ArtifactManager>[2] = {}) {
  const root = mkdtempSync(join(tmpdir(), 'oryn-publication-'));
  roots.push(root);
  const manager = new ArtifactManager(join(root, 'registry.json'), join(root, 'objects'), {
    availableBytes: () => 100 * 1024 ** 3,
    ...options,
  });
  managers.push(manager);
  return manager;
}
function stage(
  manager: ArtifactManager,
  bytes: Buffer,
  outputId = 'output_fixture_123',
): EngineOutputArtifact {
  const directory = join(manager.storage.engineStagingRoot, 'engine_test', 'job_test');
  mkdirSync(directory, { recursive: true });
  const internalPath = join(directory, outputId);
  writeFileSync(internalPath, bytes);
  return { outputId, internalPath, sizeBytes: bytes.length, mimeType: 'application/octet-stream' };
}
function publish(manager: ArtifactManager, artifact: EngineOutputArtifact) {
  return manager.publishEngineResult(owner, 'engine_test', 'job_test', {
    kind: 'artifact',
    artifact,
  }) as Promise<{ artifact: { id: string } }>;
}
afterEach(() => {
  vi.restoreAllMocks();
  for (const manager of managers.splice(0)) manager.storage.close();
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

it('streams multiple chunks into a new inode and leaves a source writer unable to change the published copy', async () => {
  const manager = setup();
  const bytes = Buffer.alloc(2 * 1024 * 1024 + 137, 0x5a);
  const artifact = stage(manager, bytes);
  const sourceInode = statSync(artifact.internalPath).ino;
  const writer = openSync(artifact.internalPath, 'r+');
  try {
    const result = await publish(manager, artifact);
    const record = manager.registry.get(result.artifact.id)!;
    const path = manager.storage.resolveReady('published', record.storage.relativePath);
    expect(statSync(path).ino).not.toBe(sourceInode);
    expect(statSync(path).mode & 0o777).toBe(0o400);
    expect(existsSync(artifact.internalPath)).toBe(false);
    writeSync(writer, Buffer.from('changed'), 0, 7, 0);
    const hash = (value: Buffer) => createHash('sha256').update(value).digest('hex');
    expect(hash(readFileSync(path))).toBe(hash(bytes));
    expect(readdirSync(manager.storage.temporaryRoot)).toEqual([]);
    await manager.release(owner, result.artifact.id);
    expect(existsSync(path)).toBe(false);
  } finally {
    closeSync(writer);
  }
});

it('reserves additional disk space even though the source already exists', async () => {
  const manager = setup({
    quotaPolicy: { maxArtifactBytes: 100, maxTemporaryBytes: 100, freeDiskReserveBytes: 10 },
    availableBytes: () => 12,
  });
  const artifact = stage(manager, Buffer.from('four'));
  await expect(publish(manager, artifact)).rejects.toMatchObject({
    code: 'DISK_SPACE_INSUFFICIENT',
  });
  expect(manager.registry.list()).toEqual([]);
  expect(manager.hasPendingWrites()).toBe(false);
  expect(existsSync(artifact.internalPath)).toBe(false);
});

it('accounts for simultaneous reservations before disk admission', () => {
  const quota = new ArtifactQuotaManager(
    '/',
    () => [],
    { maxArtifactBytes: 100, maxTemporaryBytes: 100, freeDiskReserveBytes: 10 },
    () => 20,
  );
  quota.reserve('one', 6);
  expect(() => quota.reserve('two', 6)).toThrowError('DISK_SPACE_INSUFFICIENT');
  quota.release('one');
  expect(() => quota.reserve('two', 6)).not.toThrow();
});

it('rejects a symbolic-link directory and safely cleans only its staging link', async () => {
  const manager = setup();
  const independent = join(roots.at(-1)!, 'independent-fixture');
  mkdirSync(independent);
  const retained = join(independent, 'output_fixture_123');
  writeFileSync(retained, 'fixture');
  const job = join(manager.storage.engineStagingRoot, 'engine_test', 'job_test');
  mkdirSync(job, { recursive: true });
  symlinkSync(independent, join(job, 'linked-directory'));
  await expect(
    publish(manager, {
      outputId: 'output_fixture_123',
      internalPath: join(job, 'linked-directory', 'output_fixture_123'),
      sizeBytes: 7,
      mimeType: 'application/octet-stream',
    }),
  ).rejects.toMatchObject({ code: 'ARTIFACT_PATH_INVALID' });
  expect(readFileSync(retained, 'utf8')).toBe('fixture');
  expect(manager.registry.list()).toEqual([]);
  expect(existsSync(job)).toBe(false);
});

it('rejects special files without waiting for a writer', async () => {
  const manager = setup();
  const artifact = stage(manager, Buffer.alloc(0));
  rmSync(artifact.internalPath);
  execFileSync('python3', [
    '-I',
    '-S',
    '-B',
    '-c',
    'import os,sys; os.mkfifo(sys.argv[1])',
    artifact.internalPath,
  ]);
  await expect(publish(manager, artifact)).rejects.toMatchObject({ code: 'ARTIFACT_PATH_INVALID' });
  expect(manager.hasPendingWrites()).toBe(false);
});

it('rolls back earlier copies if a later output fails validation', async () => {
  const manager = setup();
  const first = stage(manager, Buffer.from('first'), 'output_first_123');
  const second = stage(manager, Buffer.from('second'), 'output_second_123');
  second.sizeBytes++;
  await expect(
    manager.publishEngineResult(owner, 'engine_test', 'job_test', {
      kind: 'multi',
      outputs: [first, second].map((artifact) => ({ kind: 'artifact', artifact })),
    }),
  ).rejects.toMatchObject({ code: 'ARTIFACT_INVALID' });
  expect(manager.registry.list()).toEqual([]);
  expect(manager.storage.listObjectFiles('published')).toEqual([]);
  expect(readdirSync(manager.storage.temporaryRoot)).toEqual([]);
});

it.each(['cancel', 'navigation'] as const)(
  'rejects publication after %s and releases reservations',
  async (action) => {
    const manager = setup();
    const artifact = stage(manager, Buffer.from('fixture'));
    let resume!: () => void;
    const gate = new Promise<void>((resolve) => {
      resume = resolve;
    });
    const original = manager.storage.adoptEngineOutputs.bind(manager.storage);
    vi.spyOn(manager.storage, 'adoptEngineOutputs').mockImplementation(async (outputs, signal) => {
      await gate;
      return original(outputs, signal);
    });
    const pending = publish(manager, artifact);
    const rejected = expect(pending).rejects.toMatchObject({ code: 'ARTIFACT_ACCESS_DENIED' });
    expect(manager.hasPendingWrites()).toBe(true);
    if (action === 'cancel') manager.cancelPublication('job_test');
    else await manager.invalidateDocument(owner);
    resume();
    await rejected;
    expect(manager.hasPendingWrites()).toBe(false);
    expect(manager.registry.list()).toEqual([]);
    expect(manager.storage.listObjectFiles('published')).toEqual([]);
  },
);

it('kills an active helper before removing its partial destination', async () => {
  // A cooperative stand-in pauses after creating a partial file. This tests
  // process cancellation/rollback without a filesystem substitution race.
  const root = mkdtempSync(join(tmpdir(), 'oryn-helper-cancel-'));
  roots.push(root);
  const helperPath = join(root, 'paused_helper.py');
  writeFileSync(
    helperPath,
    `import json,os,sys,time\nr=json.load(sys.stdin)\nif r['operation']=='copy':\n name=r['outputs'][0]['object']+'.adopting'\n fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600,dir_fd=4)\n os.write(fd,b'partial')\n os.close(fd)\n time.sleep(30)\n`,
  );
  const manager = setup({ artifactIo: { pythonExecutable: 'python3', helperPath } });
  const pending = publish(manager, stage(manager, Buffer.from('fixture')));
  const rejected = expect(pending).rejects.toMatchObject({ code: 'ARTIFACT_ACCESS_DENIED' });
  for (
    let count = 0;
    count < 100 && readdirSync(manager.storage.temporaryRoot).length === 0;
    count++
  )
    await delay(10);
  expect(readdirSync(manager.storage.temporaryRoot)).toHaveLength(1);
  manager.cancelPublication('job_test');
  await rejected;
  expect(readdirSync(manager.storage.temporaryRoot)).toEqual([]);
  expect(manager.registry.list()).toEqual([]);
  expect(manager.hasPendingWrites()).toBe(false);
});

it('keeps active Engine staging during live temporary-storage cleanup', () => {
  const manager = setup();
  const artifact = stage(manager, Buffer.from('active'));
  manager.cleanupTemporary();
  expect(readFileSync(artifact.internalPath, 'utf8')).toBe('active');
});

it('canonicalizes a trusted profile alias before opening Engine output', async () => {
  const root = mkdtempSync(join(tmpdir(), 'oryn-profile-alias-'));
  roots.push(root);
  mkdirSync(join(root, 'actual'));
  symlinkSync(join(root, 'actual'), join(root, 'alias'));
  const manager = new ArtifactManager(join(root, 'registry.json'), join(root, 'alias'), {
    availableBytes: () => 100 * 1024 ** 3,
  });
  managers.push(manager);
  expect(manager.storage.root).not.toContain('/alias');
  await expect(
    publish(manager, stage(manager, Buffer.from('alias fixture'))),
  ).resolves.toHaveProperty('artifact.id');
});

it('reports bounded stale-staging cleanup as incomplete without blocking new publication', async () => {
  const root = mkdtempSync(join(tmpdir(), 'oryn-stale-staging-'));
  roots.push(root);
  mkdirSync(
    join(
      root,
      'objects',
      'result',
      'staging',
      'engine_old',
      'job_old',
      ...Array.from({ length: 34 }, (_, i) => `level_${i}`),
    ),
    { recursive: true },
  );
  const warning = vi.spyOn(console, 'error').mockImplementation(() => undefined);
  const manager = new ArtifactManager(join(root, 'registry.json'), join(root, 'objects'), {
    availableBytes: () => 100 * 1024 ** 3,
  });
  managers.push(manager);
  expect(warning).toHaveBeenCalled();
  expect(manager.reconciler.reconcile().stagingCleanupComplete).toBe(false);
  await expect(publish(manager, stage(manager, Buffer.from('new output')))).resolves.toHaveProperty(
    'artifact.id',
  );
});
