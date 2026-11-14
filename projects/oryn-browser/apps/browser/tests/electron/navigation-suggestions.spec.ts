import {
  _electron as electron,
  expect,
  test,
  type ElectronApplication,
  type Locator,
  type Page,
} from '@playwright/test';
import { createServer, type Server } from 'node:http';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

let server: Server;
let base: string;
let profile: string;
let application: ElectronApplication;
let chrome: Page;
let website: Page;

test.beforeAll(async () => {
  server = createServer((request, response) => {
    const path = new URL(request.url ?? '/', 'http://fixture.local').pathname;
    const title =
      path === '/history'
        ? 'Nebula Archive'
        : path === '/bookmark'
          ? 'Cobalt Guide'
          : path === '/search'
            ? 'Local search result'
            : 'Navigation start';
    response.setHeader('Content-Type', 'text/html');
    response.end(`<!doctype html><html lang="en"><title>${title}</title>
      <label>Draft <input aria-label="Draft"></label></html>`);
  });
  await new Promise<void>((done) => server.listen(0, '127.0.0.1', done));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Fixture server missing');
  base = `http://127.0.0.1:${address.port}`;
});

test.afterAll(async () => {
  server.closeAllConnections();
  await new Promise<void>((done) => server.close(() => done()));
});

test.beforeEach(async () => {
  profile = await mkdtemp(join(tmpdir(), 'oryn-navigation-suggestions-'));
  await mkdir(join(profile, 'browser-core'), { recursive: true });
  await writeFile(
    join(profile, 'browser-core/appearance.json'),
    JSON.stringify({ version: 1, theme: 'light', density: 'default', onboardingComplete: true }),
  );
  await writeFile(
    join(profile, 'browser-core/history.json'),
    JSON.stringify({
      version: 1,
      entries: [
        {
          historyId: 'history_navigation_fixture',
          url: `${base}/history`,
          title: 'Nebula Archive',
          visitedAt: new Date().toISOString(),
        },
      ],
    }),
  );
  const env = {
    ...process.env,
    ORYN_PROFILE_ROOT: profile,
    ORYN_START_URL: `${base}/start`,
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
  await expect
    .poll(() =>
      application
        .context()
        .pages()
        .some((page) => /index\.html$/.test(page.url())),
    )
    .toBe(true);
  chrome = application
    .context()
    .pages()
    .find((page) => /index\.html$/.test(page.url()))!;
  await expect(chrome.getByLabel('Browser controls')).toBeVisible();
  await expect
    .poll(() =>
      application
        .context()
        .pages()
        .some((page) => page.url() === `${base}/start`),
    )
    .toBe(true);
  website = application
    .context()
    .pages()
    .find((page) => page.url() === `${base}/start`)!;
  await website.waitForLoadState('load');
  await expect(chrome.getByRole('combobox', { name: 'Address and search' })).toHaveValue(
    `${base}/start`,
  );
});

// eslint-disable-next-line no-empty-pattern -- Playwright inspects fixture destructuring.
test.afterEach(async ({}, info) => {
  if (info.status !== info.expectedStatus && chrome && !chrome.isClosed()) {
    await info.attach('navigation-suggestions', {
      body: await chrome.screenshot({ timeout: 3000 }).catch(() => Buffer.from('')),
      contentType: 'image/png',
    });
  }
  if (application) {
    const child = application.process();
    if (child.exitCode === null) {
      const exited = new Promise<void>((done) => child.once('exit', () => done()));
      child.kill('SIGKILL');
      await exited;
    }
  }
  await rm(profile, { recursive: true, force: true });
});

async function nativeGeometry() {
  return application.evaluate(({ BrowserWindow }, pageURL) => {
    const window = BrowserWindow.getAllWindows()[0]!;
    const views = window.contentView.children.filter(
      (view): view is Electron.WebContentsView => 'webContents' in view,
    );
    const trusted = views.find((view) => /index\.html$/.test(view.webContents.getURL()))!;
    const website = views.find((view) => view.webContents.getURL() === pageURL)!;
    return {
      trusted: trusted.getBounds(),
      website: website.getBounds(),
      windowHeight: window.getContentSize()[1],
    };
  }, website.url());
}

// Resolve the native receiver so pointer coverage cannot pass by targeting a
// Playwright Page directly through another composited WebContentsView.
async function clickCompositedOption(option: Locator) {
  const box = await option.boundingBox();
  if (!box) throw new Error('Suggestion option has no bounds');
  return application.evaluate(
    ({ BrowserWindow }, { x, y }) => {
      const window = BrowserWindow.getAllWindows()[0]!;
      const views = window.contentView.children.filter(
        (view): view is Electron.WebContentsView => 'webContents' in view,
      );
      const trusted = views.find((view) => /index\.html$/.test(view.webContents.getURL()))!;
      const chromeBounds = trusted.getBounds();
      const windowX = chromeBounds.x + x;
      const windowY = chromeBounds.y + y;
      const receiver = views.toReversed().find((view) => {
        const bounds = view.getBounds();
        return (
          view.getVisible() &&
          windowX >= bounds.x &&
          windowX < bounds.x + bounds.width &&
          windowY >= bounds.y &&
          windowY < bounds.y + bounds.height
        );
      })!;
      const bounds = receiver.getBounds();
      receiver.webContents.focus();
      for (const type of ['mouseDown', 'mouseUp'] as const) {
        receiver.webContents.sendInputEvent({
          type,
          x: Math.round(windowX - bounds.x),
          y: Math.round(windowY - bounds.y),
          button: 'left',
          clickCount: 1,
        });
      }
      return receiver.webContents.getURL();
    },
    { x: box.x + box.width / 2, y: box.y + box.height / 2 },
  );
}

test('local history titles suggest by default and keyboard selection overlays the unchanged website', async () => {
  await chrome.getByRole('button', { name: 'Browser menu', exact: true }).click();
  await chrome.getByRole('menuitem', { name: 'History', exact: true }).click();
  const history = chrome.getByRole('dialog', { name: 'Browsing History', exact: true });
  await expect(history.getByText('Nebula Archive', { exact: true })).toBeVisible();
  await history
    .getByRole('textbox', { name: 'Search history', exact: true })
    .fill('unrelated-query');
  await expect(history.getByText('Nebula Archive', { exact: true })).toHaveCount(0);
  await history.getByRole('button', { name: 'Close Browsing History', exact: true }).click();
  const before = await nativeGeometry();
  const address = chrome.getByRole('combobox', { name: 'Address and search' });
  await address.fill('Nebula');
  const list = chrome.getByRole('listbox', { name: 'Address suggestions' });
  await expect(list).toBeVisible();
  const option = list.getByRole('option').filter({ hasText: 'Nebula Archive' });
  await expect(option).toHaveCount(1);
  await expect(option).toContainText(`${base}/history`);
  await expect(address).toHaveAttribute('aria-expanded', 'true');
  await expect(address).not.toHaveAttribute('aria-activedescendant', /.+/);
  await expect.poll(async () => (await nativeGeometry()).trusted.height).toBe(before.windowHeight);
  expect((await nativeGeometry()).website).toEqual(before.website);
  const box = await option.boundingBox();
  expect(box).not.toBeNull();
  expect(box!.y + box!.height).toBeLessThanOrEqual((await nativeGeometry()).trusted.height);
  await chrome.screenshot({
    path: '/tmp/oryn-address-suggestions.png',
    clip: {
      x: 0,
      y: 0,
      width: await chrome.evaluate(() => window.innerWidth),
      height: Math.ceil(box!.y + box!.height + 20),
    },
  });

  await address.press('ArrowDown');
  await expect(option).toHaveAttribute('aria-selected', 'true');
  await address.press('Enter');
  await website.waitForURL(`${base}/history`);
  await expect(list).toHaveCount(0);
  await expect
    .poll(async () => (await nativeGeometry()).trusted.height)
    .toBe(before.trusted.height);
});

test('bookmark suggestions support pointer selection, Escape and submitting unselected search text', async () => {
  await chrome.evaluate(async (url) => {
    await window.browser.addBookmark('Cobalt Guide', url);
  }, `${base}/bookmark`);
  // Refresh from persisted browser data, as the Settings controls do after updates.
  await chrome.reload();
  await expect(chrome.getByLabel('Browser controls')).toBeVisible();
  const address = chrome.getByRole('combobox', { name: 'Address and search' });
  const list = chrome.getByRole('listbox', { name: 'Address suggestions' });
  await address.fill('Cobalt');
  const option = list.getByRole('option').filter({ hasText: 'Cobalt Guide' });
  await expect(option).toBeVisible();
  expect(await clickCompositedOption(option)).toBe(chrome.url());
  await website.waitForURL(`${base}/bookmark`);
  await expect(list).toHaveCount(0);

  await address.fill('Nebula');
  await expect(list).toBeVisible();
  await address.press('Escape');
  await expect(list).toHaveCount(0);
  await expect(address).toHaveValue('Nebula');
  await expect
    .poll(async () => (await nativeGeometry()).trusted.height)
    .toBeLessThan((await nativeGeometry()).windowHeight!);

  await application.evaluate(({ session }, redirectURL) => {
    const state = globalThis as typeof globalThis & { suggestionSearchURL?: string };
    session.defaultSession.webRequest.onBeforeRequest(
      { urls: ['https://www.google.com/search*'] },
      (details, callback) => {
        state.suggestionSearchURL = details.url;
        callback({ redirectURL });
      },
    );
  }, `${base}/search`);
  await address.fill('Cobalt');
  await expect(list).toBeVisible();
  await address.press('Enter');
  await website.waitForURL(`${base}/search`);
  expect(
    await application.evaluate(
      () =>
        (globalThis as typeof globalThis & { suggestionSearchURL?: string }).suggestionSearchURL,
    ),
  ).toBe('https://www.google.com/search?q=Cobalt');
});

test('an explicit disabled preference survives reload and suppresses local matches', async () => {
  await chrome.evaluate(() => window.browser.updateAppearance({ localSuggestions: false }));
  await chrome.reload();
  await expect(chrome.getByLabel('Browser controls')).toBeVisible();
  expect(
    await chrome.evaluate(
      async () => (await window.browser.getBrowserData()).appearance.localSuggestions,
    ),
  ).toBe(false);
  const before = await nativeGeometry();
  const address = chrome.getByRole('combobox', { name: 'Address and search' });
  await address.fill('Nebula');
  await expect(address).toHaveAttribute('aria-expanded', 'false');
  await expect(chrome.getByRole('listbox', { name: 'Address suggestions' })).toHaveCount(0);
  expect((await nativeGeometry()).trusted.height).toBe(before.trusted.height);
  expect((await nativeGeometry()).website).toEqual(before.website);
  await address.fill(`${base}/history`);
  await address.press('Enter');
  await website.waitForURL(`${base}/history`);
});
