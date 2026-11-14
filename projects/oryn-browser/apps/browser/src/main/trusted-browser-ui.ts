import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { WebContents } from 'electron';

export const TRUSTED_BROWSER_UI_URL = 'oryn://browser/index.html';
const registered = new Map<number, WebContents>();

export function registerTrustedBrowserUI(contents: WebContents): void {
  registered.set(contents.id, contents);
  contents.once('destroyed', () => registered.delete(contents.id));
}

export function isTrustedBrowserUIRequest(request: {
  webContentsId?: number;
  resourceType: string;
  frame?: Electron.WebFrameMain | null;
}): boolean {
  const contents =
    request.webContentsId === undefined ? undefined : registered.get(request.webContentsId);
  if (!contents || request.resourceType === 'subFrame' || !request.frame) return false;
  try {
    if (contents.isDestroyed() || request.frame.detached || request.frame.isDestroyed())
      return false;
    return (
      request.frame.processId === contents.mainFrame.processId &&
      request.frame.routingId === contents.mainFrame.routingId
    );
  } catch {
    return false;
  }
}

/** Only bundled index/assets are served. Native request policy also requires a registered UI view. */
export async function serveTrustedBrowserUI(request: Request, root?: string): Promise<Response> {
  if (!root || request.method !== 'GET') return new Response('Not available', { status: 404 });
  const url = new URL(request.url);
  const path = decodeURIComponent(url.pathname);
  if (
    url.protocol !== 'oryn:' ||
    url.hostname !== 'browser' ||
    url.username ||
    url.password ||
    url.port ||
    (path !== '/index.html' && !/^\/assets\/[A-Za-z0-9_.-]+$/u.test(path))
  )
    return new Response('Not available', { status: 404 });
  try {
    const bytes = await readFile(join(root, path.slice(1)));
    const extension = path.split('.').at(-1);
    const contentType =
      extension === 'html'
        ? 'text/html; charset=utf-8'
        : extension === 'js'
          ? 'text/javascript; charset=utf-8'
          : extension === 'css'
            ? 'text/css; charset=utf-8'
            : extension === 'svg'
              ? 'image/svg+xml'
              : extension === 'png'
                ? 'image/png'
                : extension === 'woff2'
                  ? 'font/woff2'
                  : 'application/octet-stream';
    return new Response(new Uint8Array(bytes), {
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'no-store',
        'X-Content-Type-Options': 'nosniff',
      },
    });
  } catch {
    return new Response('Not available', { status: 404 });
  }
}
