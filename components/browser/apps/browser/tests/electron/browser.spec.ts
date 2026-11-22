import {
  _electron as electron,
  expect,
  test,
  type ElectronApplication,
  type Page,
} from '@playwright/test';
/* eslint-disable no-empty-pattern -- Playwright inspects fixture destructuring. */
import { createServer, type Server } from 'node:http';
import { createServer as createTLSServer } from 'node:https';
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import AxeBuilder from '@axe-core/playwright';
import type { BrowserWindowState } from '../../src/shared/ipc';

let server: Server;
let base: string;
let application: ElectronApplication;
let child: ReturnType<ElectronApplication['process']>;
let chrome: Page;
let profile: string;
let tlsServer: Server;
let tlsBase: string;
let tlsDirectory: string;

test('Local AI Engines shows bundled versions and checks only the selected engine', async () => {
  await application.evaluate(() => {
    const originalFetch = globalThis.fetch;
    const state = globalThis as typeof globalThis & { engineVersionRequests: string[] };
    state.engineVersionRequests = [];
    globalThis.fetch = async (input, init) => {
      if (String(input) === 'https://pypi.org/pypi/torch/json') {
        state.engineVersionRequests.push(String(input));
        return new Response(
          JSON.stringify({ info: { name: 'torch' }, releases: { '99.0.0': [{ yanked: false }] } }),
        );
      }
      return originalFetch(input, init);
    };
  });
  await chrome.getByRole('button', { name: 'Local AI activity', exact: true }).click();
  const panel = chrome.getByRole('dialog', { name: 'Oryn Local AI', exact: true });
  await panel.getByRole('button', { name: 'Engines', exact: true }).click();
  const torch = panel.getByRole('article', { name: 'PyTorch', exact: true });
  await expect(torch).toContainText('2.15.0');
  await expect(torch).toContainText('Not checked');
  expect(
    await application.evaluate(
      () =>
        (globalThis as typeof globalThis & { engineVersionRequests: string[] })
          .engineVersionRequests,
    ),
  ).toEqual([]);
  await torch.getByRole('button', { name: 'Check version for PyTorch' }).click();
  await expect(torch).toContainText('Update available');
  await expect(torch).toContainText('99.0.0');
  await expect(panel.getByRole('article', { name: 'Transformers', exact: true })).toContainText(
    'Not checked',
  );
  await panel.getByRole('button', { name: 'Running', exact: true }).click();
  await panel.getByRole('button', { name: 'Engines', exact: true }).click();
  await expect(panel.getByRole('article', { name: 'PyTorch', exact: true })).toContainText(
    'Update available',
  );
  expect(
    await application.evaluate(
      () =>
        (globalThis as typeof globalThis & { engineVersionRequests: string[] })
          .engineVersionRequests,
    ),
  ).toHaveLength(1);
  const accessibility = await new AxeBuilder({ page: chrome })
    .setLegacyMode()
    .include('[aria-label="Engine versions"]')
    .analyze();
  expect(accessibility.violations).toEqual([]);
  await panel.screenshot({ path: '/tmp/oryn-engine-versions-panel.png' });
});

test.beforeAll(async () => {
  server = createServer((request, response) => {
    if (request.url === '/clipboard') {
      response.setHeader('Content-Type', 'text/html');
      response.end(`<!doctype html><html lang="en"><title>Clipboard fixture</title>
        <button id="copy">Copy text</button><output id="result"></output>
        <script>
          document.querySelector('#copy').addEventListener('click', async () => {
            try {
              await navigator.clipboard.writeText('Oryn clipboard regression');
              document.querySelector('#result').textContent = 'Copied';
            } catch (error) {
              document.querySelector('#result').textContent = error.message;
            }
          });
        </script></html>`);
      return;
    }
    if (request.url === '/download-ready' || request.url === '/download-risky') {
      const risky = request.url === '/download-risky';
      const body = risky
        ? 'Inert fixture: never execute this file.\n'
        : 'Safe completed download fixture.\n';
      response.writeHead(200, {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': `attachment; filename="${risky ? 's65-risky.command' : 's65-safe.txt'}"`,
        'Content-Length': Buffer.byteLength(body),
      });
      response.end(body);
      return;
    }
    if (request.url === '/download') {
      response.writeHead(200, {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': 'attachment; filename="fixture.txt"',
        'Content-Length': '100000',
      });
      let sent = 0;
      const timer = setInterval(() => {
        if (sent >= 100000) {
          clearInterval(timer);
          response.end();
        } else {
          response.write('a'.repeat(1000));
          sent += 1000;
        }
      }, 50);
      response.on('close', () => clearInterval(timer));
      return;
    }
    if (request.url === '/parent') {
      response.setHeader('Content-Type', 'text/html');
      response.end(
        '<title>Parent fixture</title><iframe title="Protected subframe" src="/subframe" width="640" height="300"></iframe>',
      );
      return;
    }
    if (request.url === '/subframe') {
      response.setHeader('Content-Type', 'text/html');
      response.end(
        `<!doctype html><html lang="en"><title>Subframe fixture</title><body><input aria-label="Draft"><button onclick="window.onbeforeunload=(event)=>{event.preventDefault();event.returnValue=''};document.body.dataset.protection='modern'">Protect modern draft</button></body></html>`,
      );
      return;
    }
    if (request.url === '/auth') {
      if (
        request.headers.authorization !==
        `Basic ${Buffer.from('fixture:secret').toString('base64')}`
      ) {
        response.writeHead(401, { 'WWW-Authenticate': 'Basic realm="Local fixture"' });
        response.end('Authentication required');
      } else
        response.end(
          '<html lang="en"><title>Authenticated fixture</title><h1>Authentication succeeded</h1></html>',
        );
      return;
    }
    if (request.url === '/popup') {
      response.setHeader('Content-Type', 'text/html');
      response.end(
        '<title>Popup fixture</title><p id="opener"></p><script>document.querySelector("#opener").textContent = window.opener ? "Has opener" : "No opener"; window.opener?.postMessage("fixture-ready", location.origin);</script>',
      );
      return;
    }
    if (request.url === '/linked') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<title>Linked fixture</title><h1>Link destination loaded</h1>');
      return;
    }
    if (request.url === '/posted') {
      let body = '';
      request.on('data', (chunk) => {
        body += chunk;
      });
      request.on('end', () => {
        response.setHeader('Content-Type', 'text/html');
        response.end(`<title>POST result</title><p>${request.method}:${body}</p>`);
      });
      return;
    }
    response.setHeader('Content-Type', 'text/html');
    response.end(
      `<!doctype html><html lang="en"><title>Reliability fixture</title><body><h1>Browser fixture</h1><input aria-label="Draft" id="draft"><button id="dirty" onclick="window.onbeforeunload=()=> 'Unsaved draft'; document.body.dataset.dirty='yes'">Protect draft</button><a href="/download">Download fixture</a><button onclick="history.pushState({}, '', '/spa');document.title='SPA visit'">SPA visit</button><button onclick="window.onbeforeunload=(event)=>{event.preventDefault();event.returnValue=''};document.body.dataset.protection='modern'">Protect modern draft</button><button onclick="window.open('/popup','fixture-child')">Open named popup</button><button onclick="window.open('/popup','_blank','noopener')">Open isolated popup</button><p id="popup-message"></p><script>addEventListener('message',event=>{if(event.origin===location.origin)document.querySelector('#popup-message').textContent=event.data});</script><form method="post" action="/posted" target="_blank"><input name="sentinel" value="original-body"><button>Submit POST</button></form></body></html>`,
    );
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Fixture server missing');
  base = `http://127.0.0.1:${address.port}`;
  tlsDirectory = await mkdtemp(join(tmpdir(), 'oryn-s60-tls-'));
  execFileSync(
    'openssl',
    [
      'req',
      '-x509',
      '-newkey',
      'rsa:2048',
      '-nodes',
      '-keyout',
      join(tlsDirectory, 'fixture.key'),
      '-out',
      join(tlsDirectory, 'fixture.crt'),
      '-days',
      '1',
      '-subj',
      '/CN=localhost',
    ],
    { stdio: 'ignore' },
  );
  tlsServer = createTLSServer(
    {
      key: await readFile(join(tlsDirectory, 'fixture.key')),
      cert: await readFile(join(tlsDirectory, 'fixture.crt')),
    },
    (_request, response) => response.end('<title>Untrusted TLS fixture</title>'),
  );
  await new Promise<void>((resolve) => tlsServer.listen(0, '127.0.0.1', resolve));
  const tlsAddress = tlsServer.address();
  if (!tlsAddress || typeof tlsAddress === 'string') throw new Error('TLS fixture server missing');
  tlsBase = `https://127.0.0.1:${tlsAddress.port}`;
});
test.afterAll(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
  tlsServer.closeAllConnections();
  await new Promise<void>((resolve) => tlsServer.close(() => resolve()));
  await rm(tlsDirectory, { recursive: true, force: true });
});
test.beforeEach(async () => {
  profile = await mkdtemp(join(tmpdir(), 'oryn-s60-ui-'));
  await mkdir(join(profile, 'browser-core'), { recursive: true });
  await writeFile(
    join(profile, 'browser-core/appearance.json'),
    JSON.stringify({ version: 1, theme: 'light', density: 'default', onboardingComplete: true }),
  );
  const env = {
    ...process.env,
    ORYN_PROFILE_ROOT: profile,
    ORYN_START_URL: `${base}/fixture`,
    ORYN_ENGINE_DIR: join(profile, 'unavailable-engine'),
  };
  delete env.ELECTRON_RUN_AS_NODE;
  application = await electron.launch({
    ...(process.env.ORYN_PACKAGED_APP
      ? {
          executablePath: resolve(process.env.ORYN_PACKAGED_APP, 'Contents/MacOS/Oryn Browser'),
          args: [],
        }
      : {
          args: [resolve('.'), '--use-mock-keychain'],
          ...(process.env.ORYN_SOURCE_ELECTRON
            ? { executablePath: process.env.ORYN_SOURCE_ELECTRON }
            : {}),
        }),
    env,
  });
  child = application.process();
  child.stderr?.on('data', (chunk) => process.stderr.write(chunk));
  await expect
    .poll(
      () =>
        application
          .context()
          .pages()
          .filter((page) => /index\.html$/.test(page.url())).length,
    )
    .toBe(1);
  chrome = application
    .context()
    .pages()
    .find((page) => /index\.html$/.test(page.url()))!;
  await expect(chrome.getByLabel('Browser controls')).toBeVisible();
  await application.evaluate(({ session }, directory) => {
    session.defaultSession.on('will-download', (_event, item) =>
      item.setSavePath(`${directory}/${item.getFilename()}`),
    );
  }, profile);
});
test.afterEach(async ({}, info) => {
  if (info.status !== info.expectedStatus && chrome && !chrome.isClosed())
    await info.attach('chrome', {
      body: await chrome.screenshot({ timeout: 3000 }).catch(() => Buffer.from('')),
      contentType: 'image/png',
    });
  // Teardown only the disposable test instance, never a user's installation.
  if (application && child.exitCode === null) {
    const exited = new Promise<void>((resolve) => child.once('exit', () => resolve()));
    child.kill('SIGKILL');
    await exited;
  }
  await rm(profile, { recursive: true, force: true });
});
async function fixturePage() {
  await expect
    .poll(() =>
      application
        .context()
        .pages()
        .some((page) => page.url().startsWith(base)),
    )
    .toBe(true);
  return application
    .context()
    .pages()
    .find((page) => page.url().startsWith(base))!;
}

async function restartDisposableApplication() {
  const closeDownloads = chrome.getByRole('button', { name: 'Close Downloads', exact: true });
  if (await closeDownloads.isVisible()) await closeDownloads.click();
  await chrome.getByRole('button', { name: 'Browser menu', exact: true }).click();
  await chrome
    .getByRole('menuitem', { name: 'Quit Oryn', exact: true })
    .click({ noWaitAfter: true });
  await expect.poll(() => child.exitCode, { timeout: 15000 }).not.toBeNull();
  expect(child.exitCode).toBe(0);
  const env = {
    ...process.env,
    ORYN_PROFILE_ROOT: profile,
    ORYN_ENGINE_DIR: join(profile, 'unavailable-engine'),
  };
  delete env.ELECTRON_RUN_AS_NODE;
  delete env.ORYN_START_URL;
  delete env.LOCAL_AI_START_URL;
  application = await electron.launch({
    ...(process.env.ORYN_PACKAGED_APP
      ? {
          executablePath: resolve(process.env.ORYN_PACKAGED_APP, 'Contents/MacOS/Oryn Browser'),
          args: [],
        }
      : {
          args: [resolve('.'), '--use-mock-keychain'],
          ...(process.env.ORYN_SOURCE_ELECTRON
            ? { executablePath: process.env.ORYN_SOURCE_ELECTRON }
            : {}),
        }),
    env,
  });
  child = application.process();
  await expect
    .poll(() =>
      application
        .context()
        .pages()
        .some((candidate) => /index\.html$/.test(candidate.url())),
    )
    .toBe(true);
  chrome = application
    .context()
    .pages()
    .find((candidate) => /index\.html$/.test(candidate.url()))!;
  await expect(chrome.getByLabel('Browser controls')).toBeVisible();
}

test('S65 verified top-level application links confirm once without navigating or opening a blank tab', async () => {
  const page = await fixturePage();
  await application.evaluate(({ app, dialog, shell }) => {
    const state = globalThis as typeof globalThis & {
      handoffCalls: string[];
      handoffPrompts: { message: string; detail: string }[];
      handoffResponse: number;
      handoffMissing: boolean;
    };
    state.handoffCalls = [];
    state.handoffPrompts = [];
    state.handoffResponse = 1;
    state.handoffMissing = false;
    app.getApplicationInfoForProtocol = async () => {
      if (state.handoffMissing) throw new Error('No fixture handler');
      return { name: 'Fixture application', path: '/fixture/application', icon: undefined! };
    };
    shell.openExternal = async (url) => {
      state.handoffCalls.push(url);
    };
    dialog.showMessageBox = (async (_window, options) => {
      state.handoffPrompts.push({ message: options.message, detail: options.detail ?? '' });
      return {
        response: options.buttons?.length === 2 ? state.handoffResponse : 0,
        checkboxChecked: false,
      };
    }) as typeof dialog.showMessageBox;
  });
  await page.evaluate(() => {
    document.body.insertAdjacentHTML(
      'beforeend',
      `
      <a id="app-mail" href="mailto:someone@example.test?subject=fixture">Mail fixture</a>
      <a id="app-call" href="tel:+421123456789">Call fixture</a>
      <a id="app-missing" href="zoommtg://zoom.us/join?confno=123">Missing fixture</a>
      <a id="app-popup" target="_blank" href="slack://channel?id=fixture">Popup fixture</a>
      <iframe title="External application child" srcdoc='<a target="_top" href="msteams://teams.test/join">Child application</a>'></iframe>
    `,
    );
  });
  const sourceURL = page.url();
  const before = await chrome.evaluate(() =>
    window.browser.getState().then((state) => state.tabs.length),
  );
  await page.locator('#app-mail').click({ noWaitAfter: true });
  await expect
    .poll(() =>
      application.evaluate(
        () => (globalThis as typeof globalThis & { handoffCalls: string[] }).handoffCalls,
      ),
    )
    .toEqual(['mailto:someone@example.test?subject=fixture']);
  expect(page.url()).toBe(sourceURL);
  expect(
    await application.evaluate(
      () =>
        (
          globalThis as typeof globalThis & {
            handoffPrompts: { message: string; detail: string }[];
          }
        ).handoffPrompts,
    ),
  ).toEqual([{ message: 'Open Fixture application?', detail: `${base} requested this link.` }]);

  await application.evaluate(() => {
    (globalThis as typeof globalThis & { handoffResponse: number }).handoffResponse = 0;
  });
  await page.locator('#app-call').click({ force: true, noWaitAfter: true });
  await expect
    .poll(() =>
      application.evaluate(
        () =>
          (globalThis as typeof globalThis & { handoffPrompts: unknown[] }).handoffPrompts.length,
      ),
    )
    .toBe(2);
  await page
    .frameLocator('iframe[title="External application child"]')
    .getByRole('link', { name: 'Child application' })
    .click({ force: true, noWaitAfter: true });
  await page.locator('#app-popup').click({ force: true, noWaitAfter: true });
  await application.evaluate(() => {
    (globalThis as typeof globalThis & { handoffMissing: boolean }).handoffMissing = true;
  });
  await page.locator('#app-missing').click({ force: true, noWaitAfter: true });
  await expect
    .poll(() =>
      application.evaluate(() =>
        (
          globalThis as typeof globalThis & { handoffPrompts: { message: string }[] }
        ).handoffPrompts.map((prompt) => prompt.message),
      ),
    )
    .toEqual([
      'Open Fixture application?',
      'Open Fixture application?',
      'No installed application can open this link.',
    ]);
  expect(
    await application.evaluate(
      () => (globalThis as typeof globalThis & { handoffCalls: string[] }).handoffCalls,
    ),
  ).toHaveLength(1);
  expect(
    await chrome.evaluate(() => window.browser.getState().then((state) => state.tabs.length)),
  ).toBe(before);
  expect(page.url()).toBe(sourceURL);
});

test('S65 Save Page As uses the keyboard control and native HTML saver, then reopens without website authority', async () => {
  const page = await fixturePage();
  const saved = join(profile, 'saved-page.html');
  const cancelled = join(profile, 'cancelled-page.html');
  await writeFile(join(profile, 'sibling-secret.txt'), 'S65_SIBLING_SECRET');
  await application.evaluate(({ dialog }, file) => {
    const state = globalThis as typeof globalThis & {
      savedPageDialogCalls: number;
      pageSaveFailures: string[];
    };
    state.savedPageDialogCalls = 0;
    state.pageSaveFailures = [];
    dialog.showMessageBox = (async (...args: unknown[]) => {
      state.pageSaveFailures.push((args.at(-1) as { message: string }).message);
      return { response: 0, checkboxChecked: false };
    }) as typeof dialog.showMessageBox;
    dialog.showSaveDialog = (async (_window, options) => {
      state.savedPageDialogCalls += 1;
      if (options.title !== 'Save Page As — HTML only') throw new Error('Unexpected save format');
      return { canceled: false, filePath: file };
    }) as typeof dialog.showSaveDialog;
  }, saved);
  await page.evaluate(() => {
    document.body.insertAdjacentHTML(
      'beforeend',
      '<p id="save-marker">S65 current document marker</p>',
    );
  });
  await application.evaluate(({ app, BrowserWindow, webContents }, url) => {
    app.focus({ steal: true });
    BrowserWindow.getAllWindows()[0]!.focus();
    webContents
      .getAllWebContents()
      .find((contents) => contents.getURL() === url)!
      .focus();
  }, page.url());
  await page.getByLabel('Draft').click();
  await application.evaluate(({ webContents }, url) => {
    const contents = webContents
      .getAllWebContents()
      .find((candidate) => candidate.getURL() === url)!;
    contents.sendInputEvent({ type: 'keyDown', keyCode: 'S', modifiers: ['meta'] });
    contents.sendInputEvent({ type: 'keyUp', keyCode: 'S', modifiers: ['meta'] });
  }, page.url());
  await expect
    .poll(async () => readFile(saved, 'utf8').catch(() => ''))
    .toContain('<h1>Browser fixture</h1>');
  // Chromium HTMLOnly stores the response HTML, excluding later DOM mutations.
  expect(await readFile(saved, 'utf8')).not.toContain('S65 current document marker');
  expect(
    await application.evaluate(
      () =>
        (globalThis as typeof globalThis & { savedPageDialogCalls: number }).savedPageDialogCalls,
    ),
  ).toBe(1);
  expect(
    await application.evaluate(
      () => (globalThis as typeof globalThis & { pageSaveFailures: string[] }).pageSaveFailures,
    ),
  ).toEqual([]);
  await application.evaluate(({ dialog }, file) => {
    const state = globalThis as typeof globalThis & { cancelledPageSaves: number };
    state.cancelledPageSaves = 0;
    dialog.showSaveDialog = (async () => {
      state.cancelledPageSaves += 1;
      return { canceled: true, filePath: file };
    }) as typeof dialog.showSaveDialog;
  }, cancelled);
  await application.evaluate(({ webContents }, url) => {
    const contents = webContents
      .getAllWebContents()
      .find((candidate) => candidate.getURL() === url)!;
    contents.sendInputEvent({ type: 'keyDown', keyCode: 'S', modifiers: ['meta'] });
    contents.sendInputEvent({ type: 'keyUp', keyCode: 'S', modifiers: ['meta'] });
  }, page.url());
  await expect
    .poll(() =>
      application.evaluate(
        () => (globalThis as typeof globalThis & { cancelledPageSaves: number }).cancelledPageSaves,
      ),
    )
    .toBe(1);
  await expect(readFile(cancelled)).rejects.toMatchObject({ code: 'ENOENT' });
  await restartDisposableApplication();
  const fileURL = pathToFileURL(saved).toString();
  await chrome.getByRole('combobox', { name: 'Address and search' }).fill(fileURL);
  await chrome.getByRole('combobox', { name: 'Address and search' }).press('Enter');
  await expect
    .poll(() =>
      application
        .context()
        .pages()
        .some((candidate) => candidate.url() === fileURL),
    )
    .toBe(true);
  const reopened = application
    .context()
    .pages()
    .find((candidate) => candidate.url() === fileURL)!;
  await expect(
    reopened.getByRole('heading', { name: 'Browser fixture', exact: true }),
  ).toBeVisible();
  await expect(reopened.locator('#save-marker')).toHaveCount(0);
  expect(
    await reopened.evaluate(async () => ({
      bridge: typeof (window as Window & { localAI?: unknown }).localAI,
      sibling: await fetch('sibling-secret.txt')
        .then((response) => response.text())
        .catch(() => 'blocked'),
    })),
  ).toEqual({ bridge: 'undefined', sibling: 'blocked' });
});

test('S65 Settings exports requested content-free failure diagnostics and Cancel writes nothing', async () => {
  const page = await fixturePage();
  const exported = join(profile, 'requested-diagnostics.json');
  const cancelled = join(profile, 'cancelled-diagnostics.json');
  await chrome.getByRole('button', { name: 'New tab', exact: true }).click();
  await expect
    .poll(() => chrome.evaluate(() => window.browser.getState().then((state) => state.tabs.length)))
    .toBe(2);
  await chrome
    .getByRole('button', { name: /^Close / })
    .last()
    .click();
  await expect
    .poll(() => chrome.evaluate(() => window.browser.getState().then((state) => state.tabs.length)))
    .toBe(1);
  await page.getByLabel('Draft').fill('S65_PRIVATE_DRAFT_SECRET');
  await page.evaluate(() => {
    document.title = 'S65_PRIVATE_TITLE_SECRET';
  });
  await application.evaluate(({ dialog }, file) => {
    dialog.showSaveDialog = (async (_window, options) => {
      if (options.title !== 'Export browser diagnostics') throw new Error('Unexpected export');
      return { canceled: false, filePath: file };
    }) as typeof dialog.showSaveDialog;
  }, exported);
  await chrome.getByRole('button', { name: 'Browser menu', exact: true }).click();
  await chrome.getByRole('menuitem', { name: 'Browser settings', exact: true }).click();
  await chrome.getByRole('button', { name: 'Export browser diagnostics', exact: true }).click();
  await expect
    .poll(async () => readFile(exported, 'utf8').catch(() => ''))
    .toContain('rendererCrashes');
  expect(JSON.parse(await readFile(exported, 'utf8')).failures.rendererCrashes).toBe(0);
  await rm(exported);
  await application.evaluate(({ webContents }, url) => {
    const contents = webContents.getAllWebContents().find((contents) => contents.getURL() === url)!;
    contents.forcefullyCrashRenderer();
  }, page.url());
  await expect(chrome.getByRole('heading', { name: 'This page stopped responding' })).toBeVisible();
  await chrome.getByRole('button', { name: 'Export browser diagnostics', exact: true }).click();
  await expect
    .poll(async () => readFile(exported, 'utf8').catch(() => ''))
    .toContain('rendererCrashes');
  const raw = await readFile(exported, 'utf8');
  const summary = JSON.parse(raw);
  expect(Object.keys(summary).sort()).toEqual(
    [
      'activeDownloads',
      'appVersion',
      'architecture',
      'build',
      'collectedAt',
      'counts',
      'engineState',
      'failures',
      'formatVersion',
      'platform',
      'versions',
    ].sort(),
  );
  expect(summary.formatVersion).toBe(2);
  expect(summary.build.compiledMainSha256).toMatch(/^[a-f0-9]{64}$/);
  expect(['source', 'release', 'local-development']).toContain(summary.build.flavor);
  expect(summary.failures.rendererCrashes).toBeGreaterThanOrEqual(1);
  expect(Object.keys(summary.failures).sort()).toEqual(['engine', 'rendererCrashes']);
  for (const forbidden of [
    'S65_PRIVATE_DRAFT_SECRET',
    'S65_PRIVATE_TITLE_SECRET',
    base,
    profile,
    'stderr',
    'stack',
    'password',
    'prompt',
  ])
    expect(raw).not.toContain(forbidden);
  await application.evaluate(({ dialog }, file) => {
    const state = globalThis as typeof globalThis & { cancelledDiagnosticSaves: number };
    state.cancelledDiagnosticSaves = 0;
    dialog.showSaveDialog = (async () => {
      state.cancelledDiagnosticSaves += 1;
      return { canceled: true, filePath: file };
    }) as typeof dialog.showSaveDialog;
  }, cancelled);
  await chrome.getByRole('button', { name: 'Export browser diagnostics', exact: true }).click();
  await expect
    .poll(() =>
      application.evaluate(
        () =>
          (globalThis as typeof globalThis & { cancelledDiagnosticSaves: number })
            .cancelledDiagnosticSaves,
      ),
    )
    .toBe(1);
  await expect(readFile(cancelled)).rejects.toMatchObject({ code: 'ENOENT' });
});

test('S65 completed downloads use the native quarantine helper, cancel risky Open, and recheck after restart', async () => {
  const page = await fixturePage();
  await application.evaluate(({ dialog, shell }) => {
    const state = globalThis as typeof globalThis & {
      downloadOpenCalls: string[];
      downloadOpenPrompts: string[];
    };
    state.downloadOpenCalls = [];
    state.downloadOpenPrompts = [];
    shell.openExternal = async (url) => {
      state.downloadOpenCalls.push(url);
    };
    shell.openPath = async (path) => {
      state.downloadOpenCalls.push(path);
      return '';
    };
    dialog.showMessageBox = (async (...args: unknown[]) => {
      const options = args.at(-1) as { title?: string; message?: string };
      state.downloadOpenPrompts.push(options.title ?? options.message ?? '');
      return { response: 0, checkboxChecked: false };
    }) as typeof dialog.showMessageBox;
  });
  await page.evaluate(() => {
    document.body.insertAdjacentHTML(
      'beforeend',
      '<a id="safe-download" href="/download-ready">Safe download</a><a id="risky-download" href="/download-risky">Risky download</a>',
    );
  });
  await page.locator('#safe-download').click();
  await page.locator('#risky-download').click();
  await chrome.getByRole('button', { name: 'Downloads', exact: true }).click();
  const safe = chrome
    .getByRole('article')
    .filter({ has: chrome.getByText('s65-safe.txt', { exact: true }) });
  const risky = chrome
    .getByRole('article')
    .filter({ has: chrome.getByText('s65-risky.command', { exact: true }) });
  await expect(safe.getByRole('button', { name: 'Open file', exact: true })).toBeEnabled();
  await expect(risky.getByRole('button', { name: 'Open file', exact: true })).toBeEnabled();
  for (const filename of ['s65-safe.txt', 's65-risky.command']) {
    expect(
      execFileSync('/usr/bin/xattr', ['-p', 'com.apple.quarantine', join(profile, filename)], {
        encoding: 'utf8',
      }),
    ).not.toBe('');
  }
  await risky.getByRole('button', { name: 'Open file', exact: true }).click();
  await expect
    .poll(() =>
      application.evaluate(
        () =>
          (globalThis as typeof globalThis & { downloadOpenPrompts: string[] }).downloadOpenPrompts,
      ),
    )
    .toEqual(['Open downloaded file?']);
  expect(
    await application.evaluate(
      () => (globalThis as typeof globalThis & { downloadOpenCalls: string[] }).downloadOpenCalls,
    ),
  ).toEqual([]);
  await restartDisposableApplication();
  await chrome.getByRole('button', { name: 'Downloads', exact: true }).click();
  await expect(
    chrome
      .getByRole('article')
      .filter({ has: chrome.getByText('s65-safe.txt', { exact: true }) })
      .getByRole('button', { name: 'Open file', exact: true }),
  ).toBeEnabled();
  await expect(
    chrome
      .getByRole('article')
      .filter({ has: chrome.getByText('s65-risky.command', { exact: true }) })
      .getByRole('button', { name: 'Open file', exact: true }),
  ).toBeEnabled();
  expect(
    execFileSync('/usr/bin/xattr', ['-p', 'com.apple.quarantine', join(profile, 's65-safe.txt')], {
      encoding: 'utf8',
    }),
  ).not.toBe('');
});

test('website Copy buttons write to the clipboard without granting clipboard reads', async () => {
  const page = await fixturePage();
  const previousClipboard = await application.evaluateHandle(async ({ clipboard, ClipboardItem }) =>
    Promise.all(
      (await clipboard.read())
        .filter((item) => item.types.length > 0)
        .map(
          async (item) =>
            new ClipboardItem(
              Object.fromEntries(
                await Promise.all(item.types.map(async (type) => [type, await item.getType(type)])),
              ),
            ),
        ),
    ),
  );
  try {
    await application.evaluate(({ dialog }) => {
      dialog.showMessageBox = async () => {
        throw new Error('Copy must not request a site permission prompt');
      };
    });
    await page.goto(`${base}/clipboard`);
    await page.getByRole('button', { name: 'Copy text' }).click();
    await expect(page.locator('#result')).toHaveText('Copied');
    expect(await application.evaluate(({ clipboard }) => clipboard.readText())).toBe(
      'Oryn clipboard regression',
    );
    expect(
      await page.evaluate(async () => ({
        write: (await navigator.permissions.query({ name: 'clipboard-write' as PermissionName }))
          .state,
        read: (await navigator.permissions.query({ name: 'clipboard-read' as PermissionName }))
          .state,
      })),
    ).toEqual({ write: 'granted', read: 'denied' });
  } finally {
    await application.evaluate(async ({ clipboard }, previous) => {
      if (previous.length) await clipboard.write(previous);
      else clipboard.clear();
    }, previousClipboard);
    await previousClipboard.dispose();
  }
});

test('B01 native legacy unload cancellation preserves the live draft, then accepts close', async () => {
  const page = await fixturePage();
  await page.getByLabel('Draft').fill('keep this draft');
  await page.getByText('Protect draft').click();
  await application.evaluate(({ dialog }) => {
    dialog.showMessageBoxSync = () => 0;
  });
  await chrome.getByRole('tab', { name: 'Reliability fixture' }).hover();
  await chrome.getByLabel('Close Reliability fixture').click({ noWaitAfter: true });
  await expect(page.getByLabel('Draft')).toHaveValue('keep this draft');
  await expect(chrome.getByRole('tab', { name: 'Reliability fixture' })).toBeVisible();
  await application.evaluate(({ dialog }) => {
    dialog.showMessageBoxSync = () => 1;
  });
  await chrome.getByRole('tab', { name: 'Reliability fixture' }).hover();
  await chrome.getByLabel('Close Reliability fixture').click({ noWaitAfter: true });
  await expect.poll(() => page.isClosed()).toBe(true);
});

test('B03 popup form retains its POST body', async () => {
  const page = await fixturePage();
  await page.getByText('Submit POST').click();
  await expect
    .poll(() =>
      application
        .context()
        .pages()
        .some((page) => page.url() === `${base}/posted`),
    )
    .toBe(true);
  const child = application
    .context()
    .pages()
    .find((page) => page.url() === `${base}/posted`)!;
  await expect(child.locator('body')).toContainText('POST:sentinel=original-body');
});

for (const gesture of [
  { name: 'middle-click', button: 'middle', modifiers: [], background: true },
  { name: 'command/control-click', button: 'left', modifiers: ['ControlOrMeta'], background: true },
  { name: 'shift-middle-click', button: 'middle', modifiers: ['Shift'], background: false },
  { name: 'target-blank click', button: 'left', modifiers: [], background: false },
] as const) {
  test(`B03 ${gesture.name} opens one link tab without a main-process exception`, async () => {
    // Capture native error dialogs in this disposable instance so a regression fails
    // with the real exception instead of leaving the automated test behind a modal.
    await application.evaluate(({ dialog }) => {
      const state = globalThis as typeof globalThis & { linkOpenErrors: string[] };
      state.linkOpenErrors = [];
      dialog.showErrorBox = (title, content) => state.linkOpenErrors.push(`${title}: ${content}`);
    });
    const page = await fixturePage();
    await page.evaluate((targetBlank) => {
      const link = document.createElement('a');
      link.href = '/linked';
      link.textContent = 'Open linked fixture';
      if (targetBlank) link.target = '_blank';
      document.body.prepend(link);
    }, gesture.name === 'target-blank click');
    await page.getByRole('link', { name: 'Open linked fixture' }).click({
      button: gesture.button,
      modifiers: [...gesture.modifiers],
    });
    await expect
      .poll(async () => ({
        errors: await application.evaluate(
          () => (globalThis as typeof globalThis & { linkOpenErrors: string[] }).linkOpenErrors,
        ),
        children: application
          .context()
          .pages()
          .filter((item) => item.url() === `${base}/linked`).length,
      }))
      .toEqual({ errors: [], children: 1 });
    expect(page.url()).toBe(`${base}/fixture`);
    await expect(chrome.getByRole('tab', { name: 'Reliability fixture' })).toHaveAttribute(
      'aria-selected',
      String(gesture.background),
    );
    const linked = application
      .context()
      .pages()
      .find((item) => item.url() === `${base}/linked`)!;
    await expect(linked.getByRole('heading')).toHaveText('Link destination loaded');
    expect(await linked.evaluate(() => document.referrer)).toBe(`${base}/fixture`);
    expect(
      await linked.evaluate(() => ({ require: typeof require, process: typeof process })),
    ).toEqual({ require: 'undefined', process: 'undefined' });
    await expect(chrome.getByRole('tab', { name: 'Linked fixture' })).toHaveAttribute(
      'aria-selected',
      String(!gesture.background),
    );
    if (gesture.background) {
      await chrome.getByRole('tab', { name: 'Linked fixture' }).click();
      await expect(chrome.getByRole('tab', { name: 'Linked fixture' })).toHaveAttribute(
        'aria-selected',
        'true',
      );
    }
  });
}

for (const foreground of [false, true]) {
  test(`tab teardown tolerates a state event before cleanup (foreground: ${foreground})`, async () => {
    const page = await fixturePage();
    await page.evaluate(() => {
      const link = document.createElement('a');
      link.href = '/linked';
      link.textContent = 'Open closing tab';
      document.body.prepend(link);
    });
    await page.getByRole('link', { name: 'Open closing tab' }).click({ button: 'middle' });
    const closingTab = chrome.getByRole('tab', { name: 'Linked fixture' });
    await expect(closingTab).toBeVisible();
    if (foreground) await closingTab.click();

    await application.evaluate(({ dialog, webContents }, origin) => {
      const state = globalThis as typeof globalThis & {
        tabTeardown: { errors: string[]; events: number };
      };
      state.tabTeardown = { errors: [], events: 0 };
      dialog.showErrorBox = (title, content) =>
        state.tabTeardown.errors.push(`${title}: ${content}`);
      const closing = webContents
        .getAllWebContents()
        .find((item) => item.getURL() === `${origin}/linked`)!;
      const survivor = webContents
        .getAllWebContents()
        .find((item) => item.getURL() === `${origin}/fixture`)!;
      // Force an ordinary state refresh while the native page is destroyed but
      // TabManager's destroyed callback has not removed its tab yet.
      closing.prependOnceListener('destroyed', () => {
        state.tabTeardown.events += 1;
        survivor.emit('media-paused');
      });
    }, base);

    await closingTab.hover();
    await chrome.getByLabel('Close Linked fixture').click({ noWaitAfter: true });
    await expect
      .poll(() =>
        application.evaluate(
          () =>
            (
              globalThis as typeof globalThis & {
                tabTeardown: { errors: string[]; events: number };
              }
            ).tabTeardown,
        ),
      )
      .toEqual({ errors: [], events: 1 });
    await expect(closingTab).toHaveCount(0);
    await expect(chrome.getByRole('tab', { name: 'Reliability fixture' })).toHaveAttribute(
      'aria-selected',
      'true',
    );
    await expect(page.getByRole('heading', { name: 'Browser fixture' })).toBeVisible();
    const session = JSON.parse(await readFile(join(profile, 'browser-core/session.json'), 'utf8'));
    expect(session.openTabs.map((tab: { url: string }) => tab.url)).toEqual([`${base}/fixture`]);
    expect(
      session.recentlyClosed.filter((tab: { url: string }) => tab.url === `${base}/linked`),
    ).toHaveLength(1);
  });
}

test('B10 same-document navigation appears in browsing history', async () => {
  const page = await fixturePage();
  await page.getByText('SPA visit', { exact: true }).click();
  await expect
    .poll(async () =>
      JSON.parse(
        await readFile(join(profile, 'browser-core/history.json'), 'utf8').catch(
          (error: NodeJS.ErrnoException) => {
            if (error.code !== 'ENOENT') throw error;
            return '{"entries":[]}';
          },
        ),
      ).entries.some((entry: { url: string }) => entry.url === `${base}/spa`),
    )
    .toBe(true);
});

test('B20 open Downloads panel follows the live transfer and cancellation', async () => {
  const page = await fixturePage();
  await page.getByText('Download fixture').click();
  await chrome.getByRole('button', { name: 'Downloads', exact: true }).click();
  await expect(chrome.getByText('fixture.txt', { exact: true })).toBeVisible();
  await expect(chrome.getByRole('progressbar')).toBeVisible();
  await chrome.getByRole('button', { name: 'Cancel download', exact: true }).click();
  await expect(chrome.getByRole('status')).toContainText('cancelled');
});

test('B02 bookmark folder dialog supports cancel, validation and save', async () => {
  const sidebar = application
    .context()
    .pages()
    .find(
      (page) => page.url().includes('surface=sidebar') && !page.url().includes('sidebar-resize'),
    )!;
  await sidebar.getByRole('button', { name: 'Favorites & Bookmarks', exact: true }).click();
  await chrome.getByRole('button', { name: 'New folder', exact: true }).click();
  await expect(chrome.getByRole('dialog', { name: 'Folder name' })).toBeVisible();
  await chrome.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(chrome.getByRole('dialog', { name: 'Folder name' })).toHaveCount(0);
  await chrome.getByRole('button', { name: 'New folder', exact: true }).click();
  await chrome.getByRole('textbox', { name: 'Folder name', exact: true }).fill('Fixture folder');
  await chrome.getByRole('button', { name: 'Save', exact: true }).click();
  await expect(chrome.getByText('Folder · Fixture folder')).toBeVisible();
  const results = await new AxeBuilder({ page: chrome })
    .setLegacyMode()
    .include('.browser-panel')
    .withTags(['wcag2a', 'wcag2aa'])
    .analyze();
  expect(results.violations).toEqual([]);
});

test('window actions stay in their owning window and a live tab can move', async () => {
  const firstPage = await fixturePage();
  await firstPage.getByLabel('Draft').fill('live across windows');
  await chrome.getByRole('button', { name: 'Browser menu', exact: true }).click();
  await chrome.getByRole('menuitem', { name: 'New window', exact: true }).click();
  await expect
    .poll(
      () =>
        application
          .context()
          .pages()
          .filter((page) => /index\.html$/.test(page.url())).length,
    )
    .toBe(2);
  const secondChrome = application
    .context()
    .pages()
    .find((page) => page !== chrome && /index\.html$/.test(page.url()))!;
  await secondChrome.getByRole('combobox', { name: 'Address and search' }).fill(`${base}/second`);
  await secondChrome.getByRole('combobox', { name: 'Address and search' }).press('Enter');
  await expect
    .poll(() =>
      application
        .context()
        .pages()
        .some((page) => page.url() === `${base}/second`),
    )
    .toBe(true);
  await expect(firstPage.getByLabel('Draft')).toHaveValue('live across windows');
  await chrome.getByRole('tab', { name: 'Reliability fixture' }).click({ button: 'right' });
  await chrome.getByRole('button', { name: 'Move tab to new window', exact: true }).click();
  await expect
    .poll(
      () =>
        application
          .context()
          .pages()
          .filter((page) => /index\.html$/.test(page.url())).length,
    )
    .toBe(3);
  await expect(firstPage.getByLabel('Draft')).toHaveValue('live across windows');
});

test('renderer crash exposes recovery and reloads the failed page', async () => {
  const page = await fixturePage();
  await application.evaluate(
    ({ webContents }, url) =>
      webContents
        .getAllWebContents()
        .find((contents) => contents.getURL() === url)!
        .forcefullyCrashRenderer(),
    page.url(),
  );
  await expect(chrome.getByRole('heading', { name: 'This page stopped responding' })).toBeVisible();
  await chrome.getByRole('button', { name: 'Reload page', exact: true }).click();
  await expect(chrome.getByRole('heading', { name: 'This page stopped responding' })).toHaveCount(
    0,
  );
  // Playwright keeps a crashed target marked crashed after native Electron reload.
  // The recovery action above is real UI; corroborate the replacement document in Chromium.
  await expect
    .poll(() =>
      application.evaluate(async ({ webContents }, url) => {
        const contents = webContents.getAllWebContents().find((item) => item.getURL() === url);
        if (!contents || contents.isLoading() || contents.isCrashed()) return false;
        return contents.executeJavaScript('Boolean(document.querySelector("#draft"))');
      }, `${base}/fixture`),
    )
    .toBe(true);
});

test('stale Site controls cannot change another document', async () => {
  await chrome.getByRole('button', { name: 'Site controls', exact: true }).click();
  const previous = await chrome.evaluate(
    async () => (await window.browser.getBrowserData()).siteContext!,
  );
  await chrome.getByRole('combobox', { name: 'Address and search' }).fill(`${base}/replacement`);
  await chrome.getByRole('combobox', { name: 'Address and search' }).press('Enter');
  await expect
    .poll(
      async () =>
        (await chrome.evaluate(async () => (await window.browser.getBrowserData()).siteContext!))
          .documentId,
    )
    .not.toBe(previous.documentId);
  const rejected = await chrome.evaluate(async (context) => {
    try {
      await window.browser.setSitePermission('camera', 'allow', context);
      return false;
    } catch {
      return true;
    }
  }, previous);
  expect(rejected).toBe(true);
  await expect(
    chrome.getByRole('dialog', { name: 'Site controls' }).getByRole('combobox').first(),
  ).toHaveValue('ask');
});

test('B01 modern unload handler keeps its live draft after a refused close', async () => {
  const page = await fixturePage();
  await page.getByLabel('Draft').fill('modern draft');
  await page.getByText('Protect modern draft', { exact: true }).click();
  await application.evaluate(({ dialog }) => {
    dialog.showMessageBoxSync = () => 0;
  });
  await chrome.getByRole('tab', { name: 'Reliability fixture' }).hover();
  await chrome.getByLabel('Close Reliability fixture').click({ noWaitAfter: true });
  await expect(page.getByLabel('Draft')).toHaveValue('modern draft');
});

test('B03 named popups retain messaging, while noopener has no opener', async () => {
  const page = await fixturePage();
  await page.getByText('Open named popup', { exact: true }).click();
  await expect(page.locator('#popup-message')).toHaveText('fixture-ready');
  const first = application
    .context()
    .pages()
    .find((item) => item.url() === `${base}/popup`)!;
  await expect(first.locator('#opener')).toHaveText('Has opener');
  await chrome.getByRole('tab', { name: 'Reliability fixture' }).click();
  await page.getByText('Open named popup', { exact: true }).click();
  await expect
    .poll(
      () =>
        application
          .context()
          .pages()
          .filter((item) => item.url() === `${base}/popup`).length,
    )
    .toBe(1);
  await chrome.getByRole('tab', { name: 'Reliability fixture' }).click();
  await page.getByText('Open isolated popup', { exact: true }).click();
  await expect
    .poll(
      () =>
        application
          .context()
          .pages()
          .filter((item) => item.url() === `${base}/popup`).length,
    )
    .toBe(2);
  const isolated = application
    .context()
    .pages()
    .find((item) => item !== first && item.url() === `${base}/popup`)!;
  await expect(isolated.locator('#opener')).toHaveText('No opener');
});

test('B19 cancel and Escape leave a workspace and its tabs intact', async () => {
  await chrome.getByRole('button', { name: 'Workspaces', exact: true }).click();
  await chrome.getByRole('button', { name: '+ New', exact: true }).click();
  await chrome.getByRole('button', { name: 'Workspaces', exact: true }).click();
  await chrome.getByRole('button', { name: 'Delete', exact: true }).click();
  const dialog = chrome.getByRole('dialog', { name: 'Delete workspace?' });
  await expect(dialog).toBeVisible();
  await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(chrome.getByRole('menuitemradio', { checked: true })).not.toContainText('Main');
  await chrome.getByRole('button', { name: 'Delete', exact: true }).click();
  await dialog.press('Escape');
  await expect(dialog).toHaveCount(0);
  await chrome.getByRole('button', { name: 'Workspaces', exact: true }).click();
  await expect(chrome.getByRole('menuitemradio', { checked: true })).not.toContainText('Main');
});

test('HTTP authentication is collected in trusted chrome and succeeds for the requesting page', async () => {
  await chrome.getByRole('combobox', { name: 'Address and search' }).fill(`${base}/auth`);
  await chrome.getByRole('combobox', { name: 'Address and search' }).press('Enter');
  const dialog = chrome.getByRole('dialog');
  await expect(dialog).toContainText('127.0.0.1');
  await dialog.getByLabel('Username', { exact: true }).fill('fixture');
  await dialog.getByLabel('Password', { exact: true }).fill('secret');
  await dialog.getByRole('button', { name: 'Sign in', exact: true }).click();
  const page = await fixturePage();
  await expect(page.getByRole('heading', { name: 'Authentication succeeded' })).toBeVisible();
});

for (const configuration of [
  { width: 960, height: 700, density: 'compact', theme: 'light', zoom: 1, reduced: false },
  { width: 1280, height: 900, density: 'default', theme: 'dark', zoom: 1, reduced: true },
  { width: 1440, height: 1000, density: 'comfortable', theme: 'light', zoom: 2, reduced: false },
] as const)
  test(`layout and accessibility ${configuration.density} ${configuration.theme} ${configuration.zoom * 100}%`, async ({}, info) => {
    await application.evaluate(
      ({ BrowserWindow }, { width, height }) =>
        BrowserWindow.getAllWindows()[0]!.setSize(width, height),
      configuration,
    );
    await chrome.evaluate(
      ({ theme, density }) => window.browser.updateAppearance({ theme, density }),
      configuration,
    );
    await chrome.emulateMedia({
      reducedMotion: configuration.reduced ? 'reduce' : 'no-preference',
    });
    if (configuration.zoom !== 1)
      await application.evaluate(
        ({ webContents }, zoom) =>
          webContents
            .getAllWebContents()
            .find((item) => /index\.html$/.test(item.getURL()))!
            .setZoomFactor(zoom),
        configuration.zoom,
      );
    await chrome.getByRole('button', { name: 'Downloads', exact: true }).click();
    await expect(chrome.getByRole('heading', { name: 'Downloads', exact: true })).toBeVisible();
    const panel = chrome.locator('.browser-panel');
    await chrome.locator('.ui-sheet').evaluate(async (element) => {
      await Promise.allSettled(element.getAnimations().map((animation) => animation.finished));
    });
    const bounds = await panel.boundingBox();
    const viewport = await chrome.evaluate(() => ({ width: innerWidth, height: innerHeight }));
    expect(bounds).not.toBeNull();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(viewport.width + 1);
    expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(viewport.height + 1);
    const scan = await new AxeBuilder({ page: chrome })
      .setLegacyMode()
      .include('.browser-panel')
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze();
    expect(scan.violations).toEqual([]);
    const pixels = await application.evaluate(async ({ webContents }) => {
      const contents = webContents
        .getAllWebContents()
        .find((item) => /index\.html$/.test(item.getURL()))!;
      return (await contents.capturePage()).toPNG().toString('base64');
    });
    await info.attach('layout', { body: Buffer.from(pixels, 'base64'), contentType: 'image/png' });
    await info.attach('bounds', {
      body: JSON.stringify({ configuration, bounds, viewport }),
      contentType: 'application/json',
    });
  });

test('B14 attachment navigation retains the unchanged document identity', async () => {
  const page = await fixturePage();
  const before = await chrome.evaluate(
    async () => (await window.browser.getBrowserData()).siteContext!,
  );
  await page.getByLabel('Draft').fill('unchanged document');
  await page.getByText('Download fixture', { exact: true }).click();
  await chrome.getByRole('button', { name: 'Downloads', exact: true }).click();
  await expect(chrome.getByText('fixture.txt', { exact: true })).toBeVisible();
  const after = await chrome.evaluate(
    async () => (await window.browser.getBrowserData()).siteContext!,
  );
  expect(after).toEqual(before);
  await expect(page.getByLabel('Draft')).toHaveValue('unchanged document');
  await chrome.getByRole('button', { name: 'Cancel download', exact: true }).click();
});

test('B19 rejected close keeps the workspace; moving its tabs preserves the document', async () => {
  await chrome.getByRole('button', { name: 'Workspaces', exact: true }).click();
  await chrome.getByRole('button', { name: '+ New', exact: true }).click();
  // Workspace creation crosses IPC and loads a new document before it can navigate.
  await expect
    .poll(async () => {
      const state = await chrome.evaluate(() => window.browser.getState());
      const active = state.tabs.find((tab) => tab.tabId === state.activeTabId);
      return active?.url === 'oryn://new-tab/' && !active.isLoading;
    })
    .toBe(true);
  await expect(chrome.getByRole('combobox', { name: 'Address and search' })).toHaveValue('');
  await chrome
    .getByRole('combobox', { name: 'Address and search' })
    .fill(`${base}/workspace-draft`);
  await chrome.getByRole('combobox', { name: 'Address and search' }).press('Enter');
  await expect
    .poll(() =>
      application
        .context()
        .pages()
        .some((item) => item.url() === `${base}/workspace-draft`),
    )
    .toBe(true);
  const page = application
    .context()
    .pages()
    .find((item) => item.url() === `${base}/workspace-draft`)!;
  await page.getByLabel('Draft').fill('preserve this workspace draft');
  await page.getByText('Protect draft', { exact: true }).click();
  const before = await chrome.evaluate(
    async () => (await window.browser.getBrowserData()).siteContext!,
  );
  await application.evaluate(({ dialog }) => {
    dialog.showMessageBoxSync = () => 0;
  });
  await chrome.getByRole('button', { name: 'Workspaces', exact: true }).click();
  await chrome.getByRole('button', { name: 'Delete', exact: true }).click();
  await chrome
    .getByRole('button', { name: 'Close tabs and delete', exact: true })
    .click({ noWaitAfter: true });
  await expect(page.getByLabel('Draft')).toHaveValue('preserve this workspace draft');
  await chrome.getByRole('button', { name: 'Workspaces', exact: true }).click();
  await expect(chrome.getByRole('menuitemradio', { checked: true })).not.toContainText('Main');
  await chrome.getByRole('button', { name: 'Delete', exact: true }).click();
  await chrome.getByRole('button', { name: 'Move tabs to Main and delete', exact: true }).click();
  await expect(page.getByLabel('Draft')).toHaveValue('preserve this workspace draft');
  expect(
    await chrome.evaluate(async () => (await window.browser.getBrowserData()).siteContext!),
  ).toEqual(before);
});

test('closing repeated new windows releases their native views and keeps the first window usable', async () => {
  const baseline = await application.evaluate(
    ({ webContents }) => webContents.getAllWebContents().length,
  );
  for (let cycle = 0; cycle < 3; cycle++) {
    await chrome.getByRole('button', { name: 'Browser menu', exact: true }).click();
    await chrome.getByRole('menuitem', { name: 'New window', exact: true }).click();
    await expect
      .poll(
        () =>
          application
            .context()
            .pages()
            .filter((page) => /index\.html$/.test(page.url())).length,
      )
      .toBe(2);
    const other = application
      .context()
      .pages()
      .find((page) => page !== chrome && /index\.html$/.test(page.url()))!;
    await other.getByRole('button', { name: 'Browser menu', exact: true }).click();
    await other
      .getByRole('menuitem', { name: 'Close window', exact: true })
      .click({ noWaitAfter: true });
    await expect.poll(() => other.isClosed()).toBe(true);
    await expect
      .poll(() => application.evaluate(({ webContents }) => webContents.getAllWebContents().length))
      .toBe(baseline);
  }
  await chrome.getByRole('combobox', { name: 'Address and search' }).fill(`${base}/still-usable`);
  await chrome.getByRole('combobox', { name: 'Address and search' }).press('Enter');
  await expect
    .poll(() =>
      application
        .context()
        .pages()
        .some((page) => page.url() === `${base}/still-usable`),
    )
    .toBe(true);
});

test('normal Quit finishes and restart restores URLs with fresh document authority and no form draft', async () => {
  const page = await fixturePage();
  await page.getByLabel('Draft').fill('must not silently persist');
  const before = await chrome.evaluate(
    async () => (await window.browser.getBrowserData()).siteContext!,
  );
  await chrome.getByRole('button', { name: 'Browser menu', exact: true }).click();
  await chrome
    .getByRole('menuitem', { name: 'Quit Oryn', exact: true })
    .click({ noWaitAfter: true });
  await expect.poll(() => child.exitCode, { timeout: 15000 }).not.toBeNull();
  expect(child.exitCode).toBe(0);
  const env = {
    ...process.env,
    ORYN_PROFILE_ROOT: profile,
    ORYN_ENGINE_DIR: join(profile, 'unavailable-engine'),
  };
  delete env.ELECTRON_RUN_AS_NODE;
  delete env.ORYN_START_URL;
  delete env.LOCAL_AI_START_URL;
  application = await electron.launch({
    ...(process.env.ORYN_PACKAGED_APP
      ? {
          executablePath: resolve(process.env.ORYN_PACKAGED_APP, 'Contents/MacOS/Oryn Browser'),
          args: [],
        }
      : {
          args: [resolve('.'), '--use-mock-keychain'],
          ...(process.env.ORYN_SOURCE_ELECTRON
            ? { executablePath: process.env.ORYN_SOURCE_ELECTRON }
            : {}),
        }),
    env,
  });
  child = application.process();
  await expect
    .poll(() =>
      application
        .context()
        .pages()
        .some((item) => /index\.html$/.test(item.url())),
    )
    .toBe(true);
  chrome = application
    .context()
    .pages()
    .find((item) => /index\.html$/.test(item.url()))!;
  const restored = await fixturePage();
  await expect(restored.getByLabel('Draft')).toHaveValue('');
  await expect
    .poll(
      async () =>
        (await chrome.evaluate(async () => (await window.browser.getBrowserData()).siteContext!))
          ?.documentId,
    )
    .not.toBe(before.documentId);
});

test('Search Tabs traps keyboard focus and Escape returns it to the invoking control', async () => {
  const open = chrome.getByRole('button', { name: 'Search tabs', exact: true });
  await open.click();
  const search = chrome.getByRole('textbox', { name: 'Search tabs', exact: true });
  await expect(search).toBeFocused();
  await search.press('Shift+Tab');
  expect(
    await chrome
      .getByRole('dialog', { name: 'Search tabs', exact: true })
      .evaluate((dialog) => dialog.contains(document.activeElement)),
  ).toBe(true);
  await chrome.keyboard.press('Tab');
  await expect(search).toBeFocused();
  await search.press('Escape');
  await expect(chrome.getByRole('dialog', { name: 'Search tabs', exact: true })).toHaveCount(0);
  await expect(open).toBeFocused();
});

test('B01 eligible subframe unload cancellation preserves its draft', async ({}, info) => {
  await chrome.getByRole('combobox', { name: 'Address and search' }).fill(`${base}/parent`);
  await chrome.getByRole('combobox', { name: 'Address and search' }).press('Enter');
  const page = await fixturePage();
  await page.waitForURL(`${base}/parent`);
  await page.waitForLoadState('load');
  await page.bringToFront();
  const frame = page.frameLocator('iframe');
  await info.attach('subframe-before-input', {
    body: await page.screenshot(),
    contentType: 'image/png',
  });
  await frame.getByLabel('Draft').fill('subframe draft');
  const protect = frame.getByRole('button', { name: 'Protect modern draft', exact: true });
  await protect.click();
  await expect(frame.locator('body')).toHaveAttribute('data-protection', 'modern');
  await application.evaluate(({ dialog }) => {
    dialog.showMessageBoxSync = () => 0;
  });
  await chrome.getByRole('tab', { name: 'Parent fixture' }).hover();
  await chrome.getByLabel('Close Parent fixture').click({ noWaitAfter: true });
  await expect(frame.getByLabel('Draft')).toHaveValue('subframe draft');
});

test('Untrusted TLS is blocked with a readable Chromium certificate error', async () => {
  await chrome.getByRole('combobox', { name: 'Address and search' }).fill(tlsBase);
  await chrome.getByRole('combobox', { name: 'Address and search' }).press('Enter');
  await expect
    .poll(() =>
      application
        .context()
        .pages()
        .some((page) => page.url().startsWith('oryn://network-error')),
    )
    .toBe(true);
  const errorPage = application
    .context()
    .pages()
    .find((page) => page.url().startsWith('oryn://network-error'))!;
  await expect(errorPage.locator('body')).toContainText('ERR_CERT_AUTHORITY_INVALID');
  await expect(errorPage.getByText('Untrusted TLS fixture', { exact: true })).toHaveCount(0);
});

// Resolve the receiving surface using Electron's actual native view bounds and stacking order.
// Clicking a Playwright Page directly would bypass an invisible view covering the website.
async function clickCompositedPage(page: Page, selector: string) {
  const box = await page.locator(selector).boundingBox();
  if (!box) throw new Error(`Missing website target: ${selector}`);
  return application.evaluate(
    ({ BrowserWindow }, { pageURL, x, y }) => {
      const window = BrowserWindow.getAllWindows()[0]!;
      const views = window.contentView.children.filter(
        (view): view is Electron.WebContentsView => 'webContents' in view,
      );
      const website = views.find((view) => view.webContents.getURL() === pageURL)!;
      const bounds = website.getBounds();
      const windowX = bounds.x + x;
      const windowY = bounds.y + y;
      const receiver = views.toReversed().find((view) => {
        const area = view.getBounds();
        return (
          view.getVisible() &&
          windowX >= area.x &&
          windowX < area.x + area.width &&
          windowY >= area.y &&
          windowY < area.y + area.height
        );
      })!;
      const area = receiver.getBounds();
      receiver.webContents.focus();
      for (const type of ['mouseDown', 'mouseUp'] as const) {
        receiver.webContents.sendInputEvent({
          type,
          x: Math.round(windowX - area.x),
          y: Math.round(windowY - area.y),
          button: 'left',
          clickCount: 1,
        });
      }
      return receiver.webContents.getURL();
    },
    { pageURL: page.url(), x: box.x + box.width / 2, y: box.y + box.height / 2 },
  );
}

test('informational model popups keep the website clickable while remaining open', async () => {
  const page = await fixturePage();
  await page.evaluate(() => {
    const button = document.createElement('button');
    button.id = 'popup-click-target';
    button.textContent = 'Website clicks: 0';
    button.style.cssText = 'position:fixed;left:20px;top:20px;z-index:9999';
    let clicks = 0;
    button.onclick = () => {
      button.textContent = `Website clicks: ${++clicks}`;
    };
    document.body.append(button);
  });
  const passive = application
    .context()
    .pages()
    .find((item) => item.url().includes('surface=passive-overlay'))!;
  const original = await chrome.evaluate(() => window.browser.getState());
  let clicks = 0;
  for (const status of [
    'unsupported',
    'blocked',
    'resource-error',
    'failed',
    'downloading',
    'access-required',
  ] as const) {
    const state = {
      ...original,
      huggingFacePageAction: {
        status,
        repository: 'fixture/model',
        sourceURL: 'https://huggingface.co/fixture/model',
        selectedFiles: [],
        message: `Model ${status}`,
        memoryBytes: 50e9,
        availableMemoryBytes: 20e9,
        blockingReasons: ['MODEL_MEMORY_INSUFFICIENT'],
      },
    };
    // Keep the simulated provider snapshot authoritative across independent native
    // connection updates, as the real provider is. Navigation still clears it.
    await application.evaluate(({ webContents }, value) => {
      const chrome = webContents
        .getAllWebContents()
        .find((contents) => /index\.html$/.test(contents.getURL()))!;
      const passive = webContents
        .getAllWebContents()
        .find((contents) => contents.getURL().includes('surface=passive-overlay'))!;
      const fixture = globalThis as typeof globalThis & {
        modelSnapshotFixture?: { current: BrowserWindowState; restore(): void };
      };
      if (!fixture.modelSnapshotFixture) {
        const originals = [chrome, passive].map((contents) => [contents, contents.send] as const);
        const holder = {
          current: value,
          restore: () => {
            for (const [contents, send] of originals) contents.send = send;
          },
        };
        fixture.modelSnapshotFixture = holder;
        for (const [contents, send] of originals)
          contents.send = (channel: string, ...args: unknown[]) => {
            const next = args[0] as BrowserWindowState | undefined;
            if (
              channel === 'browser:state-changed' &&
              next &&
              next.activeTabId === holder.current.activeTabId &&
              next.tabs.find((tab) => tab.tabId === next.activeTabId)?.url ===
                holder.current.tabs.find((tab) => tab.tabId === holder.current.activeTabId)?.url
            )
              args[0] = { ...next, huggingFacePageAction: holder.current.huggingFacePageAction };
            send.call(contents, channel, ...args);
          };
      }
      fixture.modelSnapshotFixture.current = value;
      chrome.send('browser:state-changed', value);
      passive.send('hugging-face-page-action:details', value);
    }, state);
    const card = passive.getByRole('dialog', { name: 'Oryn model support details' });
    await expect(card).toBeVisible();
    await expect(card).toContainText(`Model ${status}`);
    await expect
      .poll(async () => {
        return application.evaluate(({ BrowserWindow }) => {
          const window = BrowserWindow.getAllWindows()[0]!;
          const views = window.contentView.children.filter(
            (view): view is Electron.WebContentsView => 'webContents' in view,
          );
          const chrome = views.find((view) => /index\.html$/.test(view.webContents.getURL()))!;
          const passive = views.find((view) =>
            view.webContents.getURL().includes('surface=passive-overlay'),
          )!;
          return chrome.getBounds().height < passive.getBounds().y && passive.getVisible();
        });
      })
      .toBe(true);
    expect(await clickCompositedPage(page, '#popup-click-target')).toBe(page.url());
    await expect(page.locator('#popup-click-target')).toHaveText(`Website clicks: ${++clicks}`);
    await expect(card).toBeVisible();
  }

  // Long access instructions must scroll when the native surface is clamped to a small window.
  await application.evaluate(({ BrowserWindow }) =>
    BrowserWindow.getAllWindows()[0]!.setSize(920, 620),
  );
  await passive.getByRole('button', { name: 'Save token and check' }).scrollIntoViewIfNeeded();
  await expect(passive.getByRole('button', { name: 'Save token and check' })).toBeInViewport();
  await passive.getByRole('button', { name: 'Close details' }).click();
  await expect(passive.getByRole('dialog', { name: 'Oryn model support details' })).toHaveCount(0);

  await application.evaluate(() => {
    const fixture = globalThis as typeof globalThis & {
      modelSnapshotFixture?: { restore(): void };
    };
    fixture.modelSnapshotFixture?.restore();
    delete fixture.modelSnapshotFixture;
  });

  // Exercise the real toolbar -> trusted IPC -> passive surface path with a local HTTPS fixture.
  await application.evaluate(({ webContents }) => {
    const website = webContents
      .getAllWebContents()
      .find((contents) => contents.getURL().includes('/fixture'))!;
    website.session.protocol.handle(
      'https',
      () =>
        new Response('<title>Model fixture</title><button>Website</button>', {
          headers: { 'Content-Type': 'text/html' },
        }),
    );
  });
  await chrome.evaluate(() => window.browser.navigate('https://huggingface.co/fixture/model'));
  // The development fixture has no Engine; the packaged Engine reports missing host access.
  await expect(chrome.locator('.hf-page-action')).toHaveClass(/hf-(failed|access-required)/);
  await chrome.locator('.hf-page-action').click();
  await expect(passive.getByRole('dialog', { name: 'Oryn model support details' })).toBeVisible();
  await chrome.evaluate((url) => window.browser.navigate(url), `${base}/fixture`);
  await expect(passive.getByRole('dialog', { name: 'Oryn model support details' })).toHaveCount(0);
});

test('Allow Model keeps its modal backdrop and native website capture', async () => {
  const page = await fixturePage();
  await application.evaluate(({ webContents }) => {
    const chrome = webContents
      .getAllWebContents()
      .find((contents) => /index\.html$/.test(contents.getURL()))!;
    chrome.send('local-ai:permission-prompt', {
      promptId: 'popup-regression',
      origin: 'https://fixture.example',
      role: 'chat',
      requestedRepository: 'fixture/model',
      resolvedModel: {
        provider: 'huggingface',
        repository: 'fixture/model',
        revision: 'a'.repeat(40),
      },
      resolvedRepository: 'fixture/model',
      resolutionReason: 'Requested model',
      compatibility: 'compatible',
      developerCertified: true,
      substituted: false,
      provider: 'huggingface',
      exactRevision: 'a'.repeat(40),
      installed: true,
      candidates: [
        {
          model: { provider: 'huggingface', repository: 'fixture/model', revision: 'a'.repeat(40) },
          compatibility: 'compatible',
          developerCertified: true,
          installed: true,
          executable: true,
          securityAllowed: true,
          trustRequired: false,
          blockingReasons: [],
        },
      ],
    });
  });
  const dialog = chrome.getByRole('dialog', { name: 'Local AI permission' });
  await expect(dialog).toBeVisible();
  await expect(dialog).toHaveAttribute('aria-modal', 'true');
  await expect
    .poll(() =>
      application.evaluate(({ BrowserWindow }) => {
        const window = BrowserWindow.getAllWindows()[0]!;
        const chrome = window.contentView.children.find(
          (view): view is Electron.WebContentsView =>
            'webContents' in view &&
            /index\.html$/.test((view as Electron.WebContentsView).webContents.getURL()),
        )!;
        return chrome.getBounds().height === window.getContentSize()[1];
      }),
    )
    .toBe(true);
  expect(await clickCompositedPage(page, '#draft')).toBe(chrome.url());
  await expect(dialog).toBeVisible();
});

test('trusted menu compositor does not wait for animation or resize callbacks', async () => {
  await chrome.evaluate(async () => {
    await new Promise<void>((done) => requestAnimationFrame(() => done()));
    window.requestAnimationFrame = () => 0;
    window.cancelAnimationFrame = () => undefined;
    window.ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
    document.querySelector<HTMLButtonElement>('[aria-label="Browser menu"]')!.click();
  });
  await expect(chrome.locator('.browser-menu')).toBeVisible();
  await expect
    .poll(() =>
      application.evaluate(({ BrowserWindow }) => {
        const window = BrowserWindow.getAllWindows()[0]!;
        const view = window.contentView.children.find(
          (child): child is Electron.WebContentsView =>
            'webContents' in child &&
            /index\.html$/.test((child as Electron.WebContentsView).webContents.getURL()),
        )!;
        return view.getBounds().height === window.getContentSize()[1];
      }),
    )
    .toBe(true);
  await chrome.evaluate(() => {
    document.querySelector<HTMLButtonElement>('[aria-label="Browser menu"]')!.click();
  });
  await expect(chrome.locator('.browser-menu')).toHaveCount(0);
  await expect
    .poll(() =>
      application.evaluate(({ BrowserWindow }) => {
        const window = BrowserWindow.getAllWindows()[0]!;
        const view = window.contentView.children.find(
          (child): child is Electron.WebContentsView =>
            'webContents' in child &&
            /index\.html$/.test((child as Electron.WebContentsView).webContents.getURL()),
        )!;
        return view.getBounds().height < window.getContentSize()[1];
      }),
    )
    .toBe(true);
});

test('Find in page keeps its bounded surface and website input', async () => {
  const page = await fixturePage();
  const passive = application
    .context()
    .pages()
    .find((item) => item.url().includes('surface=passive-overlay'))!;
  await chrome.evaluate(() => window.browser.showFindInPage());
  const input = passive.getByRole('textbox', { name: 'Find in page' });
  await expect(input).toBeFocused();
  await input.fill('Browser');
  await expect(passive.getByRole('status')).toHaveText('1 of 1');
  await expect(passive.getByRole('search')).toBeInViewport();
  expect(await clickCompositedPage(page, '#draft')).toBe(page.url());
  await page.keyboard.type('Typing with Find open');
  await expect(page.locator('#draft')).toHaveValue('Typing with Find open');
  await expect(input).toBeVisible();
  await passive.getByRole('button', { name: 'Close find' }).click();
  await expect(input).toHaveCount(0);
});

test('narrow island tabs activate on the first click and never close by accident', async () => {
  const ids: string[] = await chrome.evaluate(async (url) => {
    const opened: string[] = [];
    for (let i = 0; i < 13; i += 1) opened.push(await window.browser.newTab(url));
    return opened;
  }, `${base}/linked`);
  const all: string[] = await chrome.evaluate(async () =>
    (await window.browser.getState()).tabs.map((tab) => tab.tabId),
  );
  expect(all.length).toBe(14);
  await chrome.evaluate(
    (tabIds) => window.browser.updateBrowserExperience({ type: 'island-create', tabIds }),
    all,
  );
  await application.evaluate(({ BrowserWindow }) =>
    BrowserWindow.getAllWindows()[0]!.setSize(900, 640),
  );
  const tabs = chrome.locator('.tab-island .tab');
  await expect(tabs).toHaveCount(14);
  // The newest tab is active; pick an earlier, inactive one.
  const target = tabs.nth(2);
  await expect(target).not.toHaveClass(/active/);
  await expect.poll(async () => (await target.boundingBox())?.width ?? 999).toBeLessThanOrEqual(70);
  const close = target.locator('.tab-close');
  const tabCount = () => chrome.evaluate(async () => (await window.browser.getState()).tabs.length);

  await target.hover();
  await expect(close).toHaveCSS('opacity', '1');
  await expect(close).toHaveCSS('visibility', 'visible');
  await expect(target.locator('.tab-favicon-image, .tab-favicon-fallback').first()).toHaveCSS(
    'opacity',
    '1',
  );

  const box = (await target.boundingBox())!;
  const centre = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
  const corner = (await close.boundingBox())!;
  expect(corner.x).toBeGreaterThan(centre.x);
  expect(corner.y).toBeLessThan(box.y);
  expect(corner.y + corner.height / 2).toBeLessThan(centre.y);
  expect(corner.x + corner.width).toBeGreaterThan(box.x + box.width);
  await chrome.mouse.click(centre.x, centre.y);
  await expect(target).toHaveClass(/active/);
  // An immediate second click (a double-click) must not land on Close.
  await chrome.mouse.click(centre.x, centre.y);
  expect(await tabCount()).toBe(14);

  await chrome.waitForTimeout(450);
  await expect(close).toHaveCSS('visibility', 'visible');
  await chrome.mouse.click(centre.x, centre.y);
  expect(await tabCount()).toBe(14);
  // The part of the badge outside the tab stays visible and clickable.
  await chrome.mouse.move(corner.x + corner.width / 2, box.y - 1);
  await expect.poll(() => close.evaluate((element) => element.matches(':hover'))).toBe(true);
  await chrome.mouse.click(corner.x + corner.width / 2, box.y - 1);
  await expect.poll(tabCount).toBe(13);
  expect(ids.length).toBe(13);

  // A background tab can also be closed without selecting it first.
  const activeId = await chrome.evaluate(async () => (await window.browser.getState()).activeTabId);
  const background = tabs.filter({ hasNot: chrome.locator('[aria-selected="true"]') }).first();
  await background.hover();
  await background.locator('.tab-close').click();
  await expect.poll(tabCount).toBe(12);
  expect(await chrome.evaluate(async () => (await window.browser.getState()).activeTabId)).toBe(
    activeId,
  );

  // Wide island tabs keep their favicon and show Close at the trailing edge.
  await application.evaluate(({ BrowserWindow }) =>
    BrowserWindow.getAllWindows()[0]!.setSize(1400, 800),
  );
  const remaining: string[] = await chrome.evaluate(async () =>
    (await window.browser.getState()).tabs.map((tab) => tab.tabId),
  );
  for (const tabId of remaining.slice(3))
    await chrome.evaluate((id) => window.browser.closeTab(id), tabId);
  await expect(tabs).toHaveCount(3);
  const wide = tabs.nth(0);
  await expect.poll(async () => (await wide.boundingBox())?.width ?? 0).toBeGreaterThan(70);
  await wide.hover();
  await chrome.waitForTimeout(400);
  const wideClose = wide.locator('.tab-close');
  await expect(wideClose).toHaveCSS('visibility', 'visible');
  await expect(wide.locator('.tab-favicon-image, .tab-favicon-fallback').first()).toHaveCSS(
    'opacity',
    '1',
  );
  const tabBox = (await wide.boundingBox())!;
  const closeBox = (await wideClose.boundingBox())!;
  expect(closeBox.x).toBeGreaterThan(tabBox.x + tabBox.width / 2);
});
