import { mkdtempSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import type { BrowserCallerContext } from '../browser-caller-context';
import { ArtifactManager } from '../artifact-manager/artifact-manager';

const temporaryDirectories: string[] = [];
const managers: ArtifactManager[] = [];

function setup(options: ConstructorParameters<typeof ArtifactManager>[2] = {}): ArtifactManager {
  const root = mkdtempSync(join(tmpdir(), 'oryn-artifacts-test-'));
  temporaryDirectories.push(root);
  const manager = new ArtifactManager(join(root, 'artifacts.json'), join(root, 'objects'), {
    availableBytes: () => 100 * 1024 ** 3,
    ...options,
  });
  managers.push(manager);
  return manager;
}

function context(overrides: Partial<BrowserCallerContext> = {}): BrowserCallerContext {
  return {
    origin: 'https://site.test',
    tabId: 'tab_one',
    webContentsId: 10,
    frameId: 1,
    frameUrl: 'https://site.test/app',
    documentId: 'document_one',
    ...overrides,
  };
}

async function create(
  manager: ArtifactManager,
  owner = context(),
  bytes = new Uint8Array([1, 2, 3]),
  lifetime: 'ephemeral' | 'session' = 'session',
) {
  const { uploadId } = await manager.beginCreate(owner, {
    mimeType: 'audio/wav',
    sizeBytes: bytes.byteLength,
    name: 'sample.wav',
    lifetime,
  });
  await manager.appendChunk(owner, uploadId, bytes.slice().buffer as ArrayBuffer);
  return manager.finishCreate(owner, uploadId);
}

afterEach(() => {
  for (const manager of managers.splice(0)) manager.storage.close();
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe('ArtifactManager', () => {
  it('creates immutable opaque artifacts and enforces exact document ownership', async () => {
    const manager = setup();
    const artifact = await create(manager);

    expect(artifact.id).toMatch(/^artifact_[A-Za-z0-9_-]{20,80}$/u);
    expect(artifact).not.toHaveProperty('internalPath');
    expect(manager.get(context(), artifact.id)).toMatchObject({
      mimeType: 'audio/wav',
      sizeBytes: 3,
      name: 'sample.wav',
    });
    expect(() => manager.get(context({ origin: 'https://other.test' }), artifact.id)).toThrowError(
      expect.objectContaining({ code: 'ARTIFACT_ACCESS_DENIED' }),
    );
    expect(() => manager.get(context({ documentId: 'document_two' }), artifact.id)).toThrowError(
      expect.objectContaining({ code: 'ARTIFACT_ACCESS_DENIED' }),
    );
  });

  it('makes guessed IDs indistinguishable from inaccessible real IDs', async () => {
    const manager = setup();
    const artifact = await create(manager);
    const otherOrigin = context({ origin: 'https://other.test' });
    const errors = [artifact.id, 'artifact_abcdefghijklmnopqrstuvwxyz'].map((artifactId) => {
      try {
        manager.get(otherOrigin, artifactId);
      } catch (error) {
        return error;
      }
      throw new Error('Expected artifact lookup to fail');
    });

    expect(errors).toEqual([
      expect.objectContaining({ code: 'ARTIFACT_ACCESS_DENIED' }),
      expect.objectContaining({ code: 'ARTIFACT_ACCESS_DENIED' }),
    ]);
    expect(errors.map((error) => String(error))).toEqual([
      'ArtifactError: ARTIFACT_ACCESS_DENIED',
      'ArtifactError: ARTIFACT_ACCESS_DENIED',
    ]);
  });

  it('revokes website access immediately but defers deletion for an engine lease', async () => {
    const manager = setup();
    const artifact = await create(manager);
    const lease = manager.acquireEngineInputLeases(context(), [artifact.id]);
    const internalPath = lease.bindings[0]?.internalPath;
    expect(internalPath).toContain(join('objects', 'input'));

    await manager.release(context(), artifact.id);
    expect(() => manager.get(context(), artifact.id)).toThrowError(
      expect.objectContaining({ code: 'ARTIFACT_ACCESS_DENIED' }),
    );
    expect(manager.registry.get(artifact.id)?.state).toBe('released');
    expect(
      manager.storage.exists('input', manager.registry.get(artifact.id)!.storage.relativePath),
    ).toBe(true);

    await lease.release();
    expect(manager.registry.get(artifact.id)).toBeUndefined();
  });

  it('acquires required input leases transactionally', async () => {
    const manager = setup();
    const first = await create(manager);
    expect(() =>
      manager.acquireEngineInputLeases(context(), [
        first.id,
        'artifact_abcdefghijklmnopqrstuvwxyz',
      ]),
    ).toThrowError(expect.objectContaining({ code: 'ARTIFACT_ACCESS_DENIED' }));
    expect(manager.leases.count(first.id)).toBe(0);
  });

  it('rejects traversal and resource admission before writing bytes', async () => {
    const manager = setup({
      quotaPolicy: {
        maxArtifactBytes: 100,
        maxTemporaryBytes: 100,
        freeDiskReserveBytes: 10,
      },
      availableBytes: () => 50,
    });
    expect(() => manager.storage.resolveReady('input', '../../private')).toThrowError(
      expect.objectContaining({ code: 'ARTIFACT_PATH_INVALID' }),
    );
    await expect(
      manager.beginCreate(context(), { mimeType: 'video/mp4', sizeBytes: 50 }),
    ).rejects.toThrowError(expect.objectContaining({ code: 'DISK_SPACE_INSUFFICIENT' }));
  });

  it('reconciles creating records and partial bytes without publishing them', async () => {
    const manager = setup();
    const pending = await manager.beginCreate(context(), {
      mimeType: 'application/octet-stream',
      sizeBytes: 3,
    });
    await manager.appendChunk(context(), pending.uploadId, new Uint8Array([1]).buffer);
    expect(manager.registry.list()).toHaveLength(1);

    const restarted = new ArtifactManager(
      join(dirname(manager.storage.root), 'artifacts.json'),
      manager.storage.root,
    );
    managers.push(restarted);
    expect(restarted.registry.list()).toEqual([]);
    expect(readdirSync(restarted.storage.temporaryRoot)).toEqual([]);
    await manager.abortCreate(context(), pending.uploadId);
  });

  it('invalidates session artifacts on document replacement while preserving active input', async () => {
    const manager = setup();
    const artifact = await create(manager);
    const lease = manager.acquireEngineInputLeases(context(), [artifact.id]);
    await manager.invalidateDocument(context());
    expect(manager.registry.get(artifact.id)?.state).toBe('released');
    await lease.release();
    expect(manager.registry.get(artifact.id)).toBeUndefined();
  });

  it('expires ephemeral authorization while a running lease safely delays garbage collection', async () => {
    let now = 1_000;
    const manager = setup({ now: () => now });
    const artifact = await create(manager, context(), new Uint8Array([1, 2, 3]), 'ephemeral');
    const lease = manager.acquireEngineInputLeases(context(), [artifact.id]);
    const record = manager.registry.get(artifact.id)!;
    expect(record.metadata.expiresAt).toBe(601_000);

    now = 601_001;
    expect(() => manager.get(context(), artifact.id)).toThrowError(
      expect.objectContaining({ code: 'ARTIFACT_EXPIRED' }),
    );
    expect(manager.registry.get(artifact.id)?.state).toBe('expired');
    expect(manager.storage.exists(record.storage.root, record.storage.relativePath)).toBe(true);
    expect(await manager.cleanupReclaimable()).toBe(0);

    await lease.release();
    expect(manager.registry.get(artifact.id)).toBeUndefined();
    expect(manager.storage.exists(record.storage.root, record.storage.relativePath)).toBe(false);
  });

  it('revokes all artifacts synchronously and tolerates overlapping navigation/destruction', async () => {
    const manager = setup();
    const first = await create(manager);
    const second = await create(manager);
    const leased = await create(manager);
    const lease = manager.acquireEngineInputLeases(context(), [leased.id]);
    const pending = await manager.beginCreate(context(), {
      mimeType: 'application/octet-stream',
      sizeBytes: 3,
    });
    const invalidation = manager.invalidateDocument(context());
    for (const artifact of [first, second, leased]) {
      expect(() => manager.get(context(), artifact.id)).toThrowError(
        expect.objectContaining({ code: 'ARTIFACT_ACCESS_DENIED' }),
      );
    }
    await Promise.all([invalidation, manager.invalidateDocument(context())]);
    await expect(manager.finishCreate(context(), pending.uploadId)).rejects.toThrowError(
      expect.objectContaining({ code: 'ARTIFACT_ACCESS_DENIED' }),
    );
    expect(manager.registry.list()).toEqual([
      expect.objectContaining({ id: leased.id, state: 'released' }),
    ]);
    await lease.release();
    expect(manager.registry.list()).toEqual([]);
  });
});
