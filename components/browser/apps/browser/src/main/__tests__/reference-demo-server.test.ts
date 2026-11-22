import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { startReferenceDemoServers } from '../reference-demo-server';

describe('reference demo loopback server', () => {
  it('serves a built ordinary website and its health endpoint only from its site root', async () => {
    const root = mkdtempSync(join(tmpdir(), 'oryn-reference-demo-'));
    const dist = join(root, 'chat', 'dist');
    mkdirSync(join(dist, 'assets'), { recursive: true });
    writeFileSync(join(dist, 'index.html'), '<!doctype html><title>Oryn Local Chat</title>');
    writeFileSync(join(dist, 'assets', 'app.js'), 'globalThis.demoLoaded = true;');
    const [server] = await startReferenceDemoServers(root, [{ id: 'chat', port: 0 }]);
    expect(server).toBeDefined();
    const url = server!.url.replace('localhost', '127.0.0.1');

    try {
      const health = await fetch(`${url}__oryn_demo_health`);
      expect(await health.json()).toEqual({ id: 'chat', ready: true });
      expect(await (await fetch(url)).text()).toContain('Oryn Local Chat');
      expect(await (await fetch(`${url}assets/app.js`)).text()).toContain('demoLoaded');
      expect((await fetch(`${url}missing.js`)).status).toBe(404);
      expect((await fetch(url, { method: 'POST' })).status).toBe(405);
    } finally {
      await server!.close();
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('fails atomically when any configured demo build is missing', async () => {
    const root = mkdtempSync(join(tmpdir(), 'oryn-reference-demo-missing-'));
    await expect(
      startReferenceDemoServers(root, [
        { id: 'chat', port: 0 },
        { id: 'transcription', port: 0 },
      ]),
    ).rejects.toThrow('Reference demo build is missing');
    rmSync(root, { recursive: true, force: true });
  });
});
