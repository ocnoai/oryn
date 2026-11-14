import { readFile } from 'node:fs/promises';
import { basename, join } from 'node:path';
import { serveTrustedBrowserUI } from './trusted-browser-ui';
import { newTabHTML } from './new-tab-page';

import type { Session } from 'electron';

import { ARTIFACT_SCHEME } from './artifact-manager/artifact-capability-service';
import {
  isOrynNewTabURL,
  ORYN_SCHEME,
  readOrynNetworkErrorURL,
  type OrynNetworkErrorDetails,
  readOrynModelWorkspaceReference,
} from '../shared/browser-urls';

export function installNewTabProtocol(
  targetSession: Session,
  options: {
    workspaceRendererRoot?: string;
    workspaceDevelopmentURL?: string;
    hasWorkspaceReference?: (reference: string) => boolean;
    getFavorites?: () => Array<{ title: string; url: string; thumbnail?: string }>;
    privateMode?: boolean;
  } = {},
): void {
  void targetSession.protocol.handle(ORYN_SCHEME, async (request) => {
    if (new URL(request.url).hostname === 'browser')
      return serveTrustedBrowserUI(request, options.workspaceRendererRoot);
    let body: string;
    if (isOrynNewTabURL(request.url)) {
      body = newTabHTML(options.getFavorites?.() ?? [], options.privateMode);
    } else {
      const workspaceReference = readOrynModelWorkspaceReference(request.url);
      if (workspaceReference) {
        if (!options.hasWorkspaceReference?.(workspaceReference)) {
          return new Response('Unknown model workspace', {
            status: 404,
            headers: { 'content-type': 'text/plain; charset=utf-8' },
          });
        }
        if (options.workspaceDevelopmentURL) {
          const source = new URL(
            '/src/model-workspace.tsx',
            options.workspaceDevelopmentURL,
          ).toString();
          body = workspaceDevelopmentHTML(source, options.workspaceDevelopmentURL);
        } else if (options.workspaceRendererRoot) {
          body = await readFile(
            join(options.workspaceRendererRoot, 'model-workspace.html'),
            'utf8',
          );
        } else {
          return new Response('Model workspace unavailable', { status: 503 });
        }
        return new Response(body, {
          status: 200,
          headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' },
        });
      }
      const parsed = new URL(request.url);
      if (
        parsed.hostname === 'model' &&
        parsed.pathname.startsWith('/assets/') &&
        options.workspaceRendererRoot
      ) {
        const file = basename(parsed.pathname);
        if (!/^[A-Za-z0-9_.-]+$/u.test(file)) return new Response('Not found', { status: 404 });
        try {
          const asset = await readFile(join(options.workspaceRendererRoot, 'assets', file));
          return new Response(asset, {
            status: 200,
            headers: {
              'content-type': assetContentType(file),
              'cache-control': 'public, max-age=31536000, immutable',
            },
          });
        } catch {
          return new Response('Not found', { status: 404 });
        }
      }
      const error = readOrynNetworkErrorURL(request.url);
      if (!error) {
        return new Response('Not found', {
          status: 404,
          headers: { 'content-type': 'text/plain; charset=utf-8' },
        });
      }
      body = networkErrorHTML(error);
    }

    return new Response(body, {
      status: 200,
      headers: {
        'content-type': 'text/html; charset=utf-8',
        'cache-control': 'no-store',
      },
    });
  });
}

function workspaceDevelopmentHTML(source: string, developmentURL: string): string {
  const origin = new URL(developmentURL).origin;
  return `<!doctype html><html lang="en" data-theme="system"><head><meta charset="UTF-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src ${origin}; style-src ${origin} 'unsafe-inline'; img-src ${ARTIFACT_SCHEME}: data: blob:; media-src ${ARTIFACT_SCHEME}: blob:; connect-src ${origin} ${ARTIFACT_SCHEME}:; font-src ${origin}; base-uri 'none'; form-action 'none'"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Oryn Model</title></head><body><div id="root"></div><script type="module" src="${source}"></script></body></html>`;
}

function assetContentType(file: string): string {
  if (file.endsWith('.js')) return 'text/javascript; charset=utf-8';
  if (file.endsWith('.css')) return 'text/css; charset=utf-8';
  if (file.endsWith('.svg')) return 'image/svg+xml';
  if (file.endsWith('.woff2')) return 'font/woff2';
  return 'application/octet-stream';
}

export function networkErrorHTML(error: OrynNetworkErrorDetails): string {
  const failedURL = escapeHTML(error.url);
  const description = escapeHTML(error.description.replace(/^net::/u, ''));
  const isLoopback = ['localhost', '127.0.0.1', '[::1]'].includes(new URL(error.url).hostname);
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta
      http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; form-action 'none'; base-uri 'none'"
    />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Page unavailable</title>
    <style>
      :root { color-scheme: light dark; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      * { box-sizing: border-box; }
      body {
        min-height: 100vh;
        margin: 0;
        display: grid;
        place-items: center;
        background: light-dark(#f5f6f7, #101315);
        color: light-dark(#202428, #f3f5f6);
      }
      main { width: min(620px, calc(100vw - 48px)); transform: translateY(-6vh); }
      .mark { color: #7788ff; font-size: 42px; line-height: 1; }
      h1 { margin: 20px 0 10px; font-size: clamp(28px, 5vw, 38px); letter-spacing: -0.035em; }
      p { margin: 10px 0; color: light-dark(#626a70, #aeb7bd); font-size: 16px; line-height: 1.55; }
      code { overflow-wrap: anywhere; color: inherit; }
      a {
        display: inline-block;
        margin-top: 16px;
        border-radius: 10px;
        padding: 10px 16px;
        background: #7788ff;
        color: #11152d;
        font-weight: 700;
        text-decoration: none;
      }
      .code { margin-top: 18px; font-size: 13px; }
    </style>
  </head>
  <body>
    <main>
      <div class="mark" aria-hidden="true">○</div>
      <h1>This page could not be reached</h1>
      <p><code>${failedURL}</code></p>
      ${isLoopback ? '<p>This is a local address. Make sure the local site is running, then try again.</p>' : '<p>Check the address and your network connection, then try again.</p>'}
      <a href="${failedURL}">Try again</a>
      <p class="code">${description} (${error.code})</p>
    </main>
  </body>
</html>`;
}

function escapeHTML(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}
