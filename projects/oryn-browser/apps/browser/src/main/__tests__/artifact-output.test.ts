import {
  linkSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { setImmediate as waitImmediate } from 'node:timers/promises';

import { afterEach, describe, expect, it } from 'vitest';

import { ArtifactManager } from '../artifact-manager/artifact-manager';
import { ArtifactProtocolHandler } from '../artifact-manager/artifact-protocol-handler';
import type { EngineOutputArtifact } from '../artifact-manager/types';
import type { BrowserCallerContext } from '../browser-caller-context';

const temporaryDirectories: string[] = [];
const managers: ArtifactManager[] = [];

function setup(options: ConstructorParameters<typeof ArtifactManager>[2] = {}): ArtifactManager {
  const root = mkdtempSync(join(tmpdir(), 'oryn-artifact-output-'));
  temporaryDirectories.push(root);
  const manager = new ArtifactManager(join(root, 'registry.json'), join(root, 'objects'), {
    availableBytes: () => 100 * 1024 ** 3,
    ...options,
  });
  managers.push(manager);
  return manager;
}

function context(overrides: Partial<BrowserCallerContext> = {}): BrowserCallerContext {
  return {
    origin: 'https://media.test',
    tabId: 'tab_media',
    webContentsId: 42,
    frameId: 7,
    frameUrl: 'https://media.test/app',
    documentId: 'document_media',
    ...overrides,
  };
}

function stage(
  manager: ArtifactManager,
  outputId: string,
  bytes: Uint8Array,
  mimeType: string,
  name?: string,
): EngineOutputArtifact {
  const directory = join(manager.storage.engineStagingRoot, 'engine_test', 'job_media');
  mkdirSync(directory, { recursive: true });
  const internalPath = join(directory, outputId);
  writeFileSync(internalPath, bytes);
  return { outputId, internalPath, mimeType, sizeBytes: bytes.byteLength, name };
}

async function read(stream: NodeJS.ReadableStream | undefined): Promise<Buffer> {
  if (!stream) return Buffer.alloc(0);
  const chunks: Buffer[] = [];
  for await (const chunk of stream) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks);
}

afterEach(() => {
  for (const manager of managers.splice(0)) manager.storage.close();
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe('artifact output publication and serving', async () => {
  it('transactionally adopts image, audio, and video outputs without exposing paths', async () => {
    const manager = setup();
    const outputs = [
      stage(
        manager,
        'output_image_123',
        new Uint8Array([0x89, 0x50, 0x4e, 0x47]),
        'image/png',
        'image.png',
      ),
      stage(
        manager,
        'output_audio_123',
        new Uint8Array([0x52, 0x49, 0x46, 0x46]),
        'audio/wav',
        'audio.wav',
      ),
      stage(
        manager,
        'output_video_123',
        new Uint8Array(64).map((_, index) => index),
        'video/mp4',
        'video.mp4',
      ),
    ];

    const result = (await manager.publishEngineResult(context(), 'engine_test', 'job_media', {
      kind: 'multi',
      outputs: outputs.map((artifact) => ({ kind: 'artifact', artifact })),
    })) as { kind: 'multi'; outputs: Array<{ artifact: { id: string; mimeType: string } }> };

    expect(result.outputs.map((output) => output.artifact.mimeType)).toEqual([
      'image/png',
      'audio/wav',
      'video/mp4',
    ]);
    expect(JSON.stringify(result)).not.toContain('internalPath');
    expect(manager.registry.list()).toHaveLength(3);
    expect(readdirSync(manager.storage.engineStagingRoot)).toEqual([]);
  });

  it('rejects output escape and disk exhaustion without a partial publication', async () => {
    const manager = setup({
      quotaPolicy: {
        maxArtifactBytes: 1024,
        maxTemporaryBytes: 1024,
        freeDiskReserveBytes: 100,
      },
      availableBytes: () => 99,
    });
    const outside = join(manager.storage.root, 'outside');
    mkdirSync(join(manager.storage.engineStagingRoot, 'engine_test', 'job_media'), {
      recursive: true,
    });
    writeFileSync(outside, new Uint8Array([1]));
    await expect(
      manager.publishEngineResult(context(), 'engine_test', 'job_media', {
        kind: 'artifact',
        artifact: {
          outputId: 'output_escape_123',
          internalPath: outside,
          mimeType: 'image/png',
          sizeBytes: 1,
        },
      }),
    ).rejects.toThrowError(expect.objectContaining({ code: 'ARTIFACT_PATH_INVALID' }));

    const descriptor = stage(manager, 'output_disk_123', new Uint8Array([1]), 'image/png');
    await expect(
      manager.publishEngineResult(context(), 'engine_test', 'job_media', {
        kind: 'artifact',
        artifact: descriptor,
      }),
    ).rejects.toThrowError(expect.objectContaining({ code: 'DISK_SPACE_INSUFFICIENT' }));
    expect(manager.registry.list()).toEqual([]);
  });

  it('rejects symlinks, hardlinks, and declared-size mismatches during adoption', async () => {
    const manager = setup();
    const jobRoot = join(manager.storage.engineStagingRoot, 'engine_test', 'job_media');
    mkdirSync(jobRoot, { recursive: true });
    const original = join(jobRoot, 'original');
    writeFileSync(original, new Uint8Array([1, 2, 3]));

    const symlink = join(jobRoot, 'output_symlink_123');
    symlinkSync(original, symlink);
    await expect(
      manager.publishEngineResult(context(), 'engine_test', 'job_media', {
        kind: 'artifact',
        artifact: {
          outputId: 'output_symlink_123',
          internalPath: symlink,
          mimeType: 'image/png',
          sizeBytes: 3,
        },
      }),
    ).rejects.toThrowError(expect.objectContaining({ code: 'ARTIFACT_PATH_INVALID' }));

    const hardlinkManager = setup();
    const hardlinkRoot = join(
      hardlinkManager.storage.engineStagingRoot,
      'engine_test',
      'job_media',
    );
    mkdirSync(hardlinkRoot, { recursive: true });
    const hardlinkOriginal = join(hardlinkRoot, 'original');
    writeFileSync(hardlinkOriginal, new Uint8Array([1, 2, 3]));
    const hardlink = join(hardlinkRoot, 'output_hardlink_123');
    linkSync(hardlinkOriginal, hardlink);
    await expect(
      hardlinkManager.publishEngineResult(context(), 'engine_test', 'job_media', {
        kind: 'artifact',
        artifact: {
          outputId: 'output_hardlink_123',
          internalPath: hardlink,
          mimeType: 'image/png',
          sizeBytes: 3,
        },
      }),
    ).rejects.toThrowError(expect.objectContaining({ code: 'ARTIFACT_PATH_INVALID' }));

    const mismatchManager = setup();
    const mismatchRoot = join(
      mismatchManager.storage.engineStagingRoot,
      'engine_test',
      'job_media',
    );
    mkdirSync(mismatchRoot, { recursive: true });
    const mismatch = join(mismatchRoot, 'output_mismatch_123');
    writeFileSync(mismatch, new Uint8Array([4, 5]));
    await expect(
      mismatchManager.publishEngineResult(context(), 'engine_test', 'job_media', {
        kind: 'artifact',
        artifact: {
          outputId: 'output_mismatch_123',
          internalPath: mismatch,
          mimeType: 'image/png',
          sizeBytes: 3,
        },
      }),
    ).rejects.toThrowError(expect.objectContaining({ code: 'ARTIFACT_INVALID' }));
    expect(mismatchManager.registry.list()).toEqual([]);
  });

  it('serves GET, HEAD, media ranges and 416 with per-request capability checks', async () => {
    const manager = setup();
    const artifact = stage(
      manager,
      'output_video_123',
      new Uint8Array(32).map((_, index) => index),
      'video/mp4',
    );
    const published = (await manager.publishEngineResult(context(), 'engine_test', 'job_media', {
      kind: 'artifact',
      artifact,
    })) as { artifact: { id: string } };
    const url = manager.getUrl(context(), published.artifact.id);
    const handler = new ArtifactProtocolHandler(manager);

    const head = await handler.serve({ url, method: 'HEAD' });
    expect(head).toMatchObject({ status: 200, headers: { 'Content-Length': '32' } });
    const partial = await handler.serve({ url, method: 'GET', range: 'bytes=8-15' });
    expect(partial).toMatchObject({
      status: 206,
      headers: { 'Content-Range': 'bytes 8-15/32', 'Content-Type': 'video/mp4' },
    });
    expect([...(await read(partial.body))]).toEqual([8, 9, 10, 11, 12, 13, 14, 15]);
    expect((await handler.serve({ url, method: 'GET', range: 'bytes=100-120' })).status).toBe(416);
    expect((await handler.serve({ url, method: 'GET', range: 'bytes=0-1,4-5' })).status).toBe(416);
    expect(
      (await handler.serve({ url, method: 'GET', range: 'bytes=0-999999999999999999999' })).status,
    ).toBe(416);
    expect((await handler.serve({ url, method: 'GET', range: 'bytes=8-7' })).status).toBe(416);
    const token = new URL(url).pathname.slice(1);
    expect(manager.capabilities.authorizeNetworkRequest(token, 42, 7)).toBe(true);
    expect(manager.capabilities.authorizeNetworkRequest(token, 99, 7)).toBe(false);
    expect(manager.capabilities.authorizeNetworkRequest(token, 42, 8)).toBe(false);
  });

  it('revokes on navigation and defers physical release until an active stream closes', async () => {
    let documentActive = true;
    const manager = setup({ hasDocumentAuthority: () => documentActive });
    const descriptor = stage(
      manager,
      'output_audio_123',
      new Uint8Array(1024 * 1024).fill(7),
      'audio/wav',
    );
    const result = (await manager.publishEngineResult(context(), 'engine_test', 'job_media', {
      kind: 'artifact',
      artifact: descriptor,
    })) as { artifact: { id: string } };
    const url = manager.getUrl(context(), result.artifact.id);
    const handler = new ArtifactProtocolHandler(manager);
    const response = await handler.serve({ url, method: 'GET' });

    await manager.release(context(), result.artifact.id);
    expect(manager.registry.get(result.artifact.id)?.state).toBe('released');
    expect((await handler.serve({ url, method: 'GET' })).status).toBe(403);
    expect((await read(response.body)).byteLength).toBe(1024 * 1024);
    await waitImmediate();
    expect(manager.registry.get(result.artifact.id)).toBeUndefined();

    const next = stage(manager, 'output_image_456', new Uint8Array([1, 2]), 'image/png');
    const nextResult = (await manager.publishEngineResult(context(), 'engine_test', 'job_media', {
      kind: 'artifact',
      artifact: next,
    })) as { artifact: { id: string } };
    const nextUrl = manager.getUrl(context(), nextResult.artifact.id);
    documentActive = false;
    expect((await handler.serve({ url: nextUrl, method: 'GET' })).status).toBe(403);
  });

  it('serves active content as attachment and removes adoption debris on restart', async () => {
    const manager = setup();
    const active = stage(
      manager,
      'output_active_123',
      new TextEncoder().encode('<script>alert(1)</script>'),
      'text/html',
      '../unsafe.html',
    );
    const result = (await manager.publishEngineResult(context(), 'engine_test', 'job_media', {
      kind: 'artifact',
      artifact: active,
    })) as { artifact: { id: string; mimeType: string; name?: string } };
    expect(result.artifact).toMatchObject({ mimeType: 'application/octet-stream' });
    const served = await new ArtifactProtocolHandler(manager).serve({
      url: manager.getUrl(context(), result.artifact.id),
      method: 'GET',
    });
    expect(served.headers['Content-Disposition']).toContain('attachment');
    await read(served.body);

    const svg = stage(
      manager,
      'output_svg_1234',
      new TextEncoder().encode('<svg onload="alert(1)"/>'),
      'image/svg+xml',
      'active.svg',
    );
    const svgResult = (await manager.publishEngineResult(context(), 'engine_test', 'job_media', {
      kind: 'artifact',
      artifact: svg,
    })) as { artifact: { id: string; mimeType: string } };
    expect(svgResult.artifact.mimeType).toBe('application/octet-stream');

    const injectedMime = stage(
      manager,
      'output_mime_123',
      new Uint8Array([1]),
      'image/png\r\nX-Injected: yes',
      'image.png',
    );
    const mimeResult = (await manager.publishEngineResult(context(), 'engine_test', 'job_media', {
      kind: 'artifact',
      artifact: injectedMime,
    })) as { artifact: { mimeType: string } };
    expect(mimeResult.artifact.mimeType).toBe('application/octet-stream');

    const orphan = join(manager.storage.engineStagingRoot, 'engine_dead', 'job_dead');
    mkdirSync(orphan, { recursive: true });
    writeFileSync(join(orphan, 'partial'), 'partial');
    writeFileSync(join(manager.storage.temporaryRoot, 'object_dead.adopting'), 'partial');
    // Use a bucket containing a registered object. A fixed random-looking bucket
    // can collide with real objects, including case-folded names on macOS.
    const orphanPublishedDirectory = dirname(
      join(
        manager.storage.publishedRoot,
        manager.registry.get(result.artifact.id)!.storage.relativePath,
      ),
    );
    const registeredFiles = readdirSync(orphanPublishedDirectory).sort();
    writeFileSync(join(orphanPublishedDirectory, 'object_orphan'), 'orphan');
    const restarted = new ArtifactManager(
      join(manager.storage.root, '..', 'registry.json'),
      manager.storage.root,
    );
    managers.push(restarted);
    expect(readdirSync(restarted.storage.engineStagingRoot)).toEqual([]);
    expect(readdirSync(restarted.storage.temporaryRoot)).toEqual([]);
    expect(readdirSync(orphanPublishedDirectory).sort()).toEqual(registeredFiles);
  });

  it('exports through a streamed trusted destination without returning the path', async () => {
    const manager = setup();
    const descriptor = stage(
      manager,
      'output_export_123',
      new Uint8Array([9, 8, 7]),
      'audio/wav',
      'song.wav',
    );
    const result = (await manager.publishEngineResult(context(), 'engine_test', 'job_media', {
      kind: 'artifact',
      artifact: descriptor,
    })) as { artifact: { id: string } };
    const destination = join(manager.storage.root, '..', 'selected-by-user.wav');

    await expect(
      manager.saveTo(context(), result.artifact.id, destination),
    ).resolves.toBeUndefined();
    expect([...readFileSync(destination)]).toEqual([9, 8, 7]);
    expect(manager.get(context(), result.artifact.id)).not.toHaveProperty('internalPath');
  });
});
