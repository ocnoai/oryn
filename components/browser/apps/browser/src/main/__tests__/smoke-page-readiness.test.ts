import { EventEmitter } from 'node:events';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { WebContents } from 'electron';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ArtifactManager } from '../artifact-manager/artifact-manager';
import { BrowserCallerContextService } from '../browser-caller-context';
import { waitForSmokePage } from '../smoke-page-readiness';

class Page extends EventEmitter {
  url = '';
  loading = false;
  destroyed = false;
  id = 42;
  mainFrame = { url: '', processId: 1, routingId: 1 };
  getURL = () => this.url;
  isLoading = () => this.loading;
  isDestroyed = () => this.destroyed;

  navigate(url: string): void {
    this.loading = true;
    this.emit('did-start-navigation', {}, url, false, true);
    this.url = this.mainFrame.url = url;
    this.emit('did-navigate', {}, url);
  }

  finish(): void {
    this.emit('did-finish-load');
    this.loading = false;
    this.emit('did-stop-loading');
  }

  contents(): WebContents {
    return this as unknown as WebContents;
  }
}

afterEach(() => vi.useRealTimers());

describe('smoke document readiness', () => {
  it('captures the target owner after bootstrap and publishes without a navigation cancellation', async () => {
    const root = mkdtempSync(join(tmpdir(), 'oryn-smoke-document-'));
    const page = new Page();
    const contexts = new BrowserCallerContextService();
    contexts.register('tab_smoke', page.contents());
    const artifacts = new ArtifactManager(join(root, 'registry.json'), join(root, 'artifacts'), {
      hasDocumentAuthority: (owner) =>
        contexts.hasDocumentAuthority(owner.tabId, owner.webContentsId, owner.documentId),
    });
    contexts.on('document-invalidated', (owner) => void artifacts.invalidateDocument(owner));
    try {
      page.navigate('about:blank');
      page.finish();
      const bootstrapDocument = contexts.getDocumentId(page.contents());
      let ready = false;
      const completed = waitForSmokePage(page.contents(), {
        expectedURL: 'http://127.0.0.1:8123/',
        label: 'artifact navigation smoke document',
      }).then(() => {
        ready = true;
      });
      await Promise.resolve();
      expect(ready).toBe(false);
      page.navigate('http://127.0.0.1:8123/');
      page.finish();
      await completed;
      const documentId = contexts.getDocumentId(page.contents());
      expect(documentId).not.toBe(bootstrapDocument);
      const directory = join(artifacts.storage.engineStagingRoot, 'engine_smoke', 'job_smoke');
      mkdirSync(directory, { recursive: true });
      const output = join(directory, 'image.png');
      writeFileSync(output, 'fixture');
      await expect(
        artifacts.publishEngineResult(
          {
            origin: 'http://127.0.0.1:8123',
            frameUrl: page.getURL(),
            tabId: 'tab_smoke',
            webContentsId: page.id,
            frameId: page.mainFrame.routingId,
            documentId,
          },
          'engine_smoke',
          'job_smoke',
          {
            kind: 'artifact',
            artifact: {
              outputId: 'output_smoke_image',
              internalPath: output,
              mimeType: 'image/png',
              sizeBytes: 7,
            },
          },
        ),
      ).resolves.toHaveProperty('artifact.id');
    } finally {
      artifacts.storage.close();
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('does not resolve on bootstrap finish or an unrelated completed page', async () => {
    const page = new Page();
    let ready = false;
    const completed = waitForSmokePage(page.contents(), {
      expectedURL: 'https://target.example',
      label: 'target smoke tab',
    }).then(() => {
      ready = true;
    });
    for (const url of ['about:blank', 'https://other.example/']) {
      page.navigate(url);
      page.finish();
      await Promise.resolve();
      expect(ready).toBe(false);
    }
    page.navigate('https://target.example/');
    page.emit('did-finish-load');
    await Promise.resolve();
    expect(ready).toBe(false);
    page.loading = false;
    page.emit('did-stop-loading');
    await completed;
    expect(ready).toBe(true);
    expect(page.eventNames()).toEqual([]);
  });

  it('can wait for a non-bootstrap document without a fixed URL', async () => {
    const page = new Page();
    page.navigate('about:blank');
    page.finish();
    let ready = false;
    const completed = waitForSmokePage(page.contents(), { label: 'initial smoke document' }).then(
      () => {
        ready = true;
      },
    );
    await Promise.resolve();
    expect(ready).toBe(false);
    page.navigate('oryn://newtab/');
    page.finish();
    await completed;
  });

  it('recognizes an already completed target without leaving listeners behind', async () => {
    const page = new Page();
    page.navigate('https://target.example/');
    page.finish();
    await waitForSmokePage(page.contents(), {
      expectedURL: 'https://target.example',
      label: 'target smoke tab',
    });
    expect(page.eventNames()).toEqual([]);
  });

  it('reports the precise smoke phase on timeout', async () => {
    vi.useFakeTimers();
    const page = new Page();
    const pending = waitForSmokePage(page.contents(), {
      expectedURL: 'https://target.example/',
      label: 'third smoke tab',
      timeoutMs: 20,
    });
    const rejected = expect(pending).rejects.toThrow('Timed out waiting for third smoke tab.');
    page.navigate('about:blank');
    page.finish();
    await vi.advanceTimersByTimeAsync(20);
    await rejected;
    expect(page.eventNames()).toEqual([]);
  });

  it('ignores subframe and cancelled loads but reports a failed main-frame target', async () => {
    const page = new Page();
    const pending = waitForSmokePage(page.contents(), { label: 'second smoke tab' });
    const rejected = expect(pending).rejects.toThrow('second smoke tab failed (-105): DNS failed');
    page.emit('did-fail-load', {}, -105, 'Subframe DNS failed', '', false);
    page.emit('did-fail-load', {}, -3, 'Navigation replaced', '', true);
    page.emit('did-fail-load', {}, -105, 'DNS failed', '', true);
    await rejected;
    expect(page.eventNames()).toEqual([]);
  });

  it('lets intentional network-error smoke wait for its rendered error document', async () => {
    const page = new Page();
    const completed = waitForSmokePage(page.contents(), {
      label: 'network-error smoke document',
      ignoreLoadFailures: true,
    });
    page.emit('did-fail-load', {}, -105, 'Intentional DNS failure', '', true);
    page.navigate('oryn://network-error/fixture');
    page.finish();
    await completed;
  });
});
