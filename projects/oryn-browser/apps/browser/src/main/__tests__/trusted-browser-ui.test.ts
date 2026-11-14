import { EventEmitter } from 'node:events';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import {
  isTrustedBrowserUIRequest,
  registerTrustedBrowserUI,
  serveTrustedBrowserUI,
} from '../trusted-browser-ui';

let root: string | undefined;
afterEach(() => {
  if (root) rmSync(root, { recursive: true, force: true });
  root = undefined;
});
describe('trusted renderer route with file privileges disabled', () => {
  it('requires an explicitly registered native view and denies embedded requests', () => {
    const frame = {
      processId: 1,
      routingId: 412,
      detached: false,
      isDestroyed: () => false,
    } as Electron.WebFrameMain;
    const contents = Object.assign(new EventEmitter(), {
      id: 412,
      mainFrame: frame,
      isDestroyed: () => false,
    });
    expect(
      isTrustedBrowserUIRequest({ webContentsId: 412, resourceType: 'mainFrame', frame }),
    ).toBe(false);
    registerTrustedBrowserUI(contents as never);
    expect(
      isTrustedBrowserUIRequest({ webContentsId: 412, resourceType: 'mainFrame', frame }),
    ).toBe(true);
    expect(isTrustedBrowserUIRequest({ webContentsId: 412, resourceType: 'subFrame', frame })).toBe(
      false,
    );
    expect(isTrustedBrowserUIRequest({ webContentsId: 412, resourceType: 'script' })).toBe(false);
    expect(
      isTrustedBrowserUIRequest({
        webContentsId: 412,
        resourceType: 'script',
        frame: { ...frame, routingId: 900 } as Electron.WebFrameMain,
      }),
    ).toBe(false);
    contents.emit('destroyed');
    expect(isTrustedBrowserUIRequest({ webContentsId: 412, resourceType: 'script', frame })).toBe(
      false,
    );
  });
  it('serves only bundled index and closed asset names without exposing arbitrary files', async () => {
    root = mkdtempSync(join(tmpdir(), 'oryn-trusted-ui-'));
    mkdirSync(join(root, 'assets'));
    writeFileSync(join(root, 'index.html'), '<title>Trusted UI</title>');
    writeFileSync(join(root, 'assets', 'entry-123.js'), 'export const value=1;');
    const script = await serveTrustedBrowserUI(
      new Request('oryn://browser/assets/entry-123.js'),
      root,
    );
    expect(script.headers.get('Content-Type')).toContain('javascript');
    expect(await script.text()).toContain('export const');
    for (const path of ['/assets/%2e%2e%2fsecret', '/private.json', '/assets/subdir/entry.js'])
      expect((await serveTrustedBrowserUI(new Request(`oryn://browser${path}`), root)).status).toBe(
        404,
      );
    expect(
      (
        await serveTrustedBrowserUI(
          new Request('oryn://browser/index.html', { method: 'POST' }),
          root,
        )
      ).status,
    ).toBe(404);
    for (const url of [
      'oryn://browser:123/index.html',
      'oryn://user@browser/index.html',
      'oryn://user:password@browser/index.html',
    ])
      expect((await serveTrustedBrowserUI({ url, method: 'GET' } as Request, root)).status).toBe(
        404,
      );
  });
});
