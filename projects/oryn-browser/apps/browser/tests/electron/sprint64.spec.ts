import {
  _electron as electron,
  expect,
  test,
  type ElectronApplication,
  type Page,
} from '@playwright/test';
/* eslint-disable no-empty-pattern -- Playwright inspects fixture destructuring. */
import { createHash, randomUUID } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { createServer, type Server } from 'node:http';
import {
  chmod,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  stat,
  symlink,
  writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { pathToFileURL } from 'node:url';

let server: Server;
let base: string;
let application: ElectronApplication;
let child: ReturnType<ElectronApplication['process']>;
let chrome: Page;
let profile: string;

test.beforeAll(async () => {
  server = createServer((request, response) => {
    const url = new URL(request.url ?? '/', 'http://fixture.test');
    if (url.pathname === '/embedded-parent') {
      const embedded = new URL(base);
      embedded.hostname = 'localhost';
      response.writeHead(200, {
        'Content-Type': 'text/html',
        'Cache-Control': 'no-store',
        'Permissions-Policy': `fullscreen=(self "${embedded.origin}"), geolocation=(self "${embedded.origin}")`,
      });
      response.end(`<!doctype html><html lang="en"><title>Embedded permission parent</title><body><h1>Embedded permission parent</h1>
        <iframe title="Embedded permission fixture" src="${embedded.origin}/embedded-child" allow="fullscreen; geolocation" allowfullscreen style="width:600px;height:400px"></iframe></body></html>`);
      return;
    }
    if (url.pathname === '/embedded-child') {
      response.writeHead(200, { 'Content-Type': 'text/html', 'Cache-Control': 'no-store' });
      response.end(`<!doctype html><html lang="en"><title>Embedded permission fixture</title><body><h1>Embedded permission fixture</h1>
        <button id="enter-fullscreen">Enter embedded fullscreen</button><button id="exit-fullscreen">Exit embedded fullscreen</button><output id="fullscreen-state">Idle</output>
        <button id="location">Request fixture location</button><output id="location-state">Not requested</output>
        <script>
        window.fixtureClicks = [];
        document.addEventListener('click', event => window.fixtureClicks.push({ target: event.target.id, trusted: event.isTrusted, x: event.clientX, y: event.clientY }), true);
        document.querySelector('#enter-fullscreen').onclick = async () => { try { await document.documentElement.requestFullscreen(); document.querySelector('#fullscreen-state').textContent = 'Entered'; } catch(error) { document.querySelector('#fullscreen-state').textContent = error.name; } };
        document.querySelector('#exit-fullscreen').onclick = async () => { await document.exitFullscreen(); document.querySelector('#fullscreen-state').textContent = 'Exited'; };
        document.addEventListener('fullscreenchange', () => { if (!document.fullscreenElement) document.querySelector('#fullscreen-state').textContent = 'Exited'; });
        document.querySelector('#location').onclick = () => {
          document.querySelector('#location-state').textContent = 'Pending';
          navigator.geolocation.getCurrentPosition(position => document.querySelector('#location-state').textContent = position.coords.latitude + ',' + position.coords.longitude,
            error => document.querySelector('#location-state').textContent = 'Denied:' + error.code, { timeout: 5000 });
        };
        </script></body></html>`);
      return;
    }
    if (url.pathname === '/capture') {
      response.writeHead(200, { 'Content-Type': 'text/html', 'Cache-Control': 'no-store' });
      response.end(`<!doctype html><html lang="en"><title>Fake capture fixture</title><body>
        <h1>Fake capture fixture</h1><button id="capture">Request fake camera and microphone</button><output id="capture-state">Not requested</output>
        <script>
        document.querySelector('#capture').onclick = async () => {
          try {
            window.fixtureStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
            document.querySelector('#capture-state').textContent = window.fixtureStream.getTracks().map(track => track.kind + ':' + track.readyState).sort().join(', ');
          } catch(error) { document.querySelector('#capture-state').textContent = error.name; }
        };
        window.addEventListener('beforeunload', event => { event.preventDefault(); event.returnValue = 'Fixture unsaved work'; });
        </script></body></html>`);
      return;
    }
    if (url.pathname === '/slow-private-file') {
      const token = url.searchParams.get('token') ?? 'private';
      response.writeHead(200, {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': 'attachment; filename="private-canceled.bin"',
        'Content-Length': 64 * 1024 * 1024,
        'Cache-Control': 'no-store',
      });
      response.write(token);
      const timer = setInterval(() => response.write(token.repeat(1024)), 50);
      response.on('close', () => clearInterval(timer));
      return;
    }
    if (url.pathname === '/saved-file') {
      const content = url.searchParams.get('token') ?? 'saved';
      response.writeHead(200, {
        'Content-Type': 'text/plain',
        'Content-Disposition': 'attachment; filename="private-saved.txt"',
        'Content-Length': Buffer.byteLength(content),
      });
      response.end(content);
      return;
    }
    if (url.pathname === '/cache') {
      response.writeHead(200, { 'Content-Type': 'text/plain', 'Cache-Control': 'max-age=3600' });
      response.end(url.searchParams.get('token') ?? 'cache');
      return;
    }
    if (url.pathname === '/sw.js') {
      response.writeHead(200, {
        'Content-Type': 'application/javascript',
        'Service-Worker-Allowed': '/',
      });
      response.end('self.addEventListener("fetch",()=>{});');
      return;
    }
    response.writeHead(200, { 'Content-Type': 'text/html', 'Cache-Control': 'no-store' });
    response.end(`<!doctype html><html lang="en"><title>Sprint 64 fixture</title><body><h1>Sprint 64 fixture</h1>
      <input aria-label="Private draft"><button id="store">Store private fixture data</button><output id="stored"></output>
      <a id="save">Save fixture file</a>
      <script>
      const token = new URL(location.href).searchParams.get('token') || 'ordinary';
      document.querySelector('#save').href = '/saved-file?token=' + encodeURIComponent(token);
      document.querySelector('#store').onclick = async () => {
        document.querySelector('input').value = token;
        document.cookie = 'privateToken=' + token + '; path=/; max-age=3600';
        localStorage.setItem('privateToken', token); sessionStorage.setItem('privateToken', token);
        await new Promise((resolve, reject) => {
          const request = indexedDB.open('sprint64-private-fixture', 1);
          request.onupgradeneeded = () => request.result.createObjectStore('fixture');
          request.onerror = () => reject(request.error);
          request.onsuccess = () => { const db = request.result; const tx = db.transaction('fixture','readwrite'); tx.objectStore('fixture').put(token,'token'); tx.oncomplete = () => { db.close(); resolve(); }; tx.onerror = () => reject(tx.error); };
        });
        const cache = await caches.open('sprint64-private-fixture');
        await cache.put('/cache?token=' + encodeURIComponent(token), await fetch('/cache?token=' + encodeURIComponent(token)));
        await navigator.serviceWorker.register('/sw.js?token=' + encodeURIComponent(token));
        document.querySelector('#stored').textContent = 'Stored';
      };
      </script></body></html>`);
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

async function launch(additionalArguments: string[] = [], entry = resolve('.')): Promise<void> {
  const env: Record<string, string> = {
    PATH: process.env.PATH ?? '/usr/bin:/bin',
    TMPDIR: process.env.TMPDIR ?? tmpdir(),
    LANG: 'en_US.UTF-8',
    ORYN_PROFILE_ROOT: profile,
    ORYN_START_URL: `${base}/ordinary`,
    ORYN_ENGINE_DIR: join(profile, 'unavailable-engine'),
  };
  application = await electron.launch({
    timeout: 15_000,
    ...(process.env.ORYN_PACKAGED_APP
      ? { executablePath: resolve(process.env.ORYN_PACKAGED_APP, 'Contents/MacOS/Oryn Browser') }
      : process.env.ORYN_SOURCE_ELECTRON
        ? { executablePath: process.env.ORYN_SOURCE_ELECTRON }
        : {}),
    args: [
      ...(process.env.ORYN_PACKAGED_APP ? [] : [entry]),
      '--disable-background-networking',
      '--use-mock-keychain',
      ...additionalArguments,
    ],
    env,
  });
  child = application.process();
  child.stderr?.on('data', (chunk) => process.stderr.write(chunk));
  await application.context().route('**/*', async (route) => {
    const url = new URL(route.request().url());
    if (
      !['http:', 'https:'].includes(url.protocol) ||
      ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)
    )
      await route.continue();
    else await route.abort('blockedbyclient');
  });
  await expect.poll(() => mainChromePages().length).toBe(1);
  chrome = mainChromePages()[0]!;
  await expect(chrome.getByLabel('Browser controls')).toBeVisible();
}

test.beforeEach(async () => {
  profile = await mkdtemp(join(tmpdir(), 'oryn-s64-ui-'));
  await mkdir(join(profile, 'browser-core'), { recursive: true });
  await writeFile(
    join(profile, 'browser-core', 'appearance.json'),
    JSON.stringify({ version: 1, theme: 'light', density: 'default', onboardingComplete: true }),
  );
  await launch(
    test.info().title.startsWith('S64-07') ? ['--use-fake-device-for-media-stream'] : [],
  );
});

async function killOwnedInstance(): Promise<void> {
  if (child?.exitCode === null && child.signalCode === null) {
    const exited = new Promise<void>((done) => child.once('exit', () => done()));
    child.kill('SIGKILL');
    await exited;
  }
}
test.afterEach(async ({}, info) => {
  if (info.status !== info.expectedStatus && chrome && !chrome.isClosed()) {
    await info.attach('sprint64-chrome', {
      body: await chrome.screenshot({ timeout: 3000 }).catch(() => Buffer.from('')),
      contentType: 'image/png',
    });
  }
  await killOwnedInstance();
  await rm(profile, { recursive: true, force: true });
});

function mainChromePages(): Page[] {
  return application
    .context()
    .pages()
    .filter((page) => /index\.html$/.test(page.url()));
}
async function website(url: string): Promise<Page> {
  await expect
    .poll(() =>
      application
        .context()
        .pages()
        .some((page) => page.url() === url),
    )
    .toBe(true);
  return application
    .context()
    .pages()
    .find((page) => page.url() === url)!;
}
async function navigate(owner: Page, url: string): Promise<Page> {
  const address = owner.getByRole('combobox', { name: 'Address and search' });
  await address.fill(url);
  await address.press('Enter');
  return website(url);
}
async function menu(owner: Page, name: string): Promise<void> {
  await owner.getByRole('button', { name: 'Browser menu', exact: true }).click();
  if (name === 'Close window') {
    // A successful window action must close the owning target. Observe that native outcome
    // alongside the input acknowledgement, which can end when its target is destroyed.
    const closed = owner.waitForEvent('close');
    const clicked = owner
      .getByRole('menuitem', { name, exact: true })
      .click()
      .catch((error) => {
        if (!owner.isClosed()) throw error;
      });
    await Promise.race([closed, clicked]);
    await expect.poll(() => owner.isClosed()).toBe(true);
  } else await owner.getByRole('menuitem', { name, exact: true }).click();
}
async function newPrivate(owner = chrome): Promise<Page> {
  const previous = new Set(mainChromePages());
  await menu(owner, 'New private window');
  await expect.poll(() => mainChromePages().length).toBe(previous.size + 1);
  const next = mainChromePages().find((page) => !previous.has(page))!;
  await expect(next.getByRole('button', { name: 'Private browsing', exact: true })).toBeVisible();
  return next;
}
async function journal(): Promise<{ version: 1; ids: string[] }> {
  return JSON.parse(await readFile(join(profile, 'browser-core', 'private-cleanup.json'), 'utf8'));
}
async function filesContainingToken(directory: string, token: string): Promise<string[]> {
  const found: string[] = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isSymbolicLink()) continue;
    if (entry.isDirectory()) found.push(...(await filesContainingToken(path, token)));
    else if (entry.isFile() && (await readFile(path)).includes(Buffer.from(token)))
      found.push(path);
  }
  return found;
}

test('S64-04 Open file control preserves the local-file boundary and shows Local file connection state', async ({}, info) => {
  const directory = join(profile, 'user-documents');
  await mkdir(directory);
  const file = join(directory, 'local-fixture.html');
  await writeFile(join(directory, 'sibling.txt'), 'SIBLING_SECRET');
  await writeFile(join(directory, 'sibling.html'), '<title>Sibling</title><p>SIBLING_SECRET</p>');
  await writeFile(
    file,
    `<!doctype html><html lang="en"><title>Local HTML fixture</title><body><h1>Local HTML fixture</h1><iframe id="sibling" src="sibling.html"></iframe><script>
    globalThis.boundary = { bridge: typeof globalThis.localAI, fetch: 'pending', xhr: 'pending', iframe: 'pending' };
    fetch('sibling.txt').then(r => r.text()).then(t => boundary.fetch = t.includes('SIBLING_SECRET') ? 'read' : 'blocked').catch(() => boundary.fetch = 'blocked');
    const xhr = new XMLHttpRequest(); xhr.onload = () => boundary.xhr = xhr.responseText.includes('SIBLING_SECRET') ? 'read' : 'blocked'; xhr.onerror = () => boundary.xhr = 'blocked'; xhr.open('GET','sibling.txt'); xhr.send();
    document.querySelector('#sibling').onload = () => { try { boundary.iframe = document.querySelector('#sibling').contentDocument.body.textContent.includes('SIBLING_SECRET') ? 'read' : 'blocked'; } catch { boundary.iframe = 'blocked'; } };
  </script></body></html>`,
  );
  await chrome.screenshot({ path: info.outputPath('normal-chrome.png') });
  await application.evaluate(({ dialog }, selected) => {
    dialog.showOpenDialog = (async () => ({
      canceled: false,
      filePaths: [selected],
    })) as typeof dialog.showOpenDialog;
  }, file);
  await menu(chrome, 'Open file…');
  const page = await website(pathToFileURL(file).toString());
  await expect(page.getByRole('heading', { name: 'Local HTML fixture' })).toBeVisible();
  await expect
    .poll(() =>
      page.evaluate(
        () => (globalThis as typeof globalThis & { boundary: Record<string, string> }).boundary,
      ),
    )
    .toEqual({ bridge: 'undefined', fetch: 'blocked', xhr: 'blocked', iframe: 'blocked' });
  await chrome.getByRole('button', { name: 'Site controls', exact: true }).click();
  await expect(
    chrome.getByRole('heading', { name: 'Connection: Local file', exact: true }),
  ).toBeVisible();
  await chrome.screenshot({ path: info.outputPath('local-file-site-controls.png') });
});

test('S64-10 private controls isolate two sessions and preserve explicit saves on normal cleanup', async ({}, info) => {
  const normal = await website(`${base}/ordinary`);
  await normal.evaluate(() => {
    document.cookie = 'normalToken=normal-sentinel; path=/; max-age=3600';
  });
  const first = await newPrivate();
  const firstToken = `private-first-${randomUUID()}`;
  const firstURL = `${base}/private?token=${firstToken}`;
  const firstPage = await navigate(first, firstURL);
  expect(await firstPage.evaluate(() => document.cookie)).toBe('');
  await firstPage.getByRole('button', { name: 'Store private fixture data' }).click();
  await expect(firstPage.locator('#stored')).toHaveText('Stored');
  const localAI = await firstPage.evaluate(async () => ({
    capabilities: await window.localAI.getCapabilities(),
    requested: await window.localAI.requestModel('private-fixture-operation', { role: 'chat' }),
    artifact: await window.localAI.getArtifact('private-artifact'),
  }));
  expect(localAI.capabilities).toMatchObject({
    runtimeAvailable: false,
    unavailableReason: 'PRIVATE_CONTEXT',
  });
  expect(localAI.requested).toMatchObject({
    ok: false,
    error: { code: 'RUNTIME_NOT_AVAILABLE', details: { reason: 'PRIVATE_CONTEXT' } },
  });
  expect(localAI.artifact).toMatchObject({ ok: false, error: { code: 'RUNTIME_NOT_AVAILABLE' } });

  const second = await newPrivate();
  const secondToken = `private-second-${randomUUID()}`;
  const secondPage = await navigate(second, `${base}/private?token=${secondToken}`);
  expect(await secondPage.evaluate(() => document.cookie)).toBe('');
  expect(
    await secondPage.evaluate(async () => ({
      local: localStorage.getItem('privateToken'),
      databases: await indexedDB.databases(),
      caches: await caches.keys(),
      workers: (await navigator.serviceWorker.getRegistrations()).length,
    })),
  ).toEqual({ local: null, databases: [], caches: [], workers: 0 });
  await secondPage.getByRole('button', { name: 'Store private fixture data' }).click();
  await expect(secondPage.locator('#stored')).toHaveText('Stored');
  expect(await firstPage.evaluate(() => document.cookie)).toContain(firstToken);
  expect(await firstPage.evaluate(() => document.cookie)).not.toContain(secondToken);
  expect(await normal.evaluate(() => document.cookie)).toBe('normalToken=normal-sentinel');

  const ids = (await journal()).ids;
  expect(ids).toHaveLength(2);
  await writeFile(join(profile, 'private-temp', ids[0]!, 'controlled-temp-fixture'), firstToken);
  const activeResidue = await filesContainingToken(profile, firstToken);
  expect(activeResidue.every((path) => path.startsWith(join(profile, 'private-temp') + '/'))).toBe(
    true,
  );
  const saveDirectory = join(profile, 'explicit-saves');
  await mkdir(saveDirectory);
  const savedFile = join(saveDirectory, 'private-saved.txt');
  await application.evaluate(
    ({ webContents }, { url, path }) => {
      const contents = webContents.getAllWebContents().find((item) => item.getURL() === url)!;
      contents.session.once('will-download', (_event, item) => item.setSavePath(path));
    },
    { url: firstURL, path: savedFile },
  );
  await firstPage.getByRole('link', { name: 'Save fixture file' }).click();
  await expect.poll(() => readFile(savedFile, 'utf8').catch(() => '')).toBe(firstToken);
  await first.getByRole('button', { name: 'Bookmark this page', exact: true }).click();
  await expect
    .poll(async () =>
      (await readFile(join(profile, 'browser-core', 'bookmarks.json'), 'utf8')).includes(
        firstToken,
      ),
    )
    .toBe(true);
  await first.screenshot({ path: info.outputPath('private-chrome.png') });
  await menu(first, 'Close window');
  await expect.poll(() => first.isClosed()).toBe(true);
  await expect.poll(async () => (await journal()).ids).toEqual([ids[1]]);
  expect(await readFile(savedFile, 'utf8')).toBe(firstToken);
  expect(await secondPage.evaluate(() => document.cookie)).toContain(secondToken);
  const residue = await filesContainingToken(profile, firstToken);
  expect(
    residue.filter(
      (path) => path !== savedFile && path !== join(profile, 'browser-core', 'bookmarks.json'),
    ),
  ).toEqual([]);
  const reopened = await newPrivate();
  const reopenedPage = await navigate(reopened, `${base}/private-reopened`);
  expect(await reopenedPage.evaluate(() => document.cookie)).toBe('');
  expect(await normal.evaluate(() => document.cookie)).toBe('normalToken=normal-sentinel');
});

test('S64-10 crash recovery retries journal cleanup without restoring private browsing or deleting explicit saves', async () => {
  const token = `private-crash-${randomUUID()}`;
  const privateChrome = await newPrivate();
  const privatePage = await navigate(privateChrome, `${base}/private?token=${token}`);
  await privatePage.getByRole('button', { name: 'Store private fixture data' }).click();
  await expect(privatePage.locator('#stored')).toHaveText('Stored');
  const state = await journal();
  expect(state.ids).toHaveLength(1);
  expect(JSON.stringify(state)).not.toContain(token);
  const temporary = join(profile, 'private-temp', state.ids[0]!, 'controlled-temp-fixture');
  await writeFile(temporary, token);
  const savedFile = join(profile, 'explicit-save.txt');
  await writeFile(savedFile, token);
  await privateChrome.getByRole('button', { name: 'Bookmark this page', exact: true }).click();
  await expect
    .poll(async () =>
      (await readFile(join(profile, 'browser-core', 'bookmarks.json'), 'utf8')).includes(token),
    )
    .toBe(true);
  await killOwnedInstance();
  expect(await readFile(temporary, 'utf8')).toBe(token);
  await launch();
  await expect.poll(async () => (await journal()).ids).toEqual([]);
  await expect
    .poll(() =>
      readFile(temporary, 'utf8').then(
        () => true,
        () => false,
      ),
    )
    .toBe(false);
  expect(mainChromePages()).toHaveLength(1);
  expect(await chrome.getByRole('button', { name: 'Private browsing', exact: true }).count()).toBe(
    0,
  );
  expect(
    application
      .context()
      .pages()
      .some((page) => page.url().includes(token)),
  ).toBe(false);
  expect(await readFile(savedFile, 'utf8')).toBe(token);
  const residue = await filesContainingToken(profile, token);
  expect(
    residue.filter(
      (path) => path !== savedFile && path !== join(profile, 'browser-core', 'bookmarks.json'),
    ),
  ).toEqual([]);
});

test('S64-08/09 Settings previews, cancels and confirms copied Chrome and selective Safari imports', async ({}, info) => {
  const fixtureHome = join(profile, 'browser-import-fixture-home');
  const source = join(fixtureHome, 'Library', 'Application Support', 'Google', 'Chrome');
  const selectedProfile = join(source, 'Default');
  await mkdir(selectedProfile, { recursive: true });
  await writeFile(
    join(source, 'Local State'),
    JSON.stringify({ profile: { info_cache: { Default: { name: 'Fixture Profile' } } } }),
  );
  await writeFile(
    join(selectedProfile, 'Bookmarks'),
    JSON.stringify({
      roots: {
        bookmark_bar: {
          id: '1',
          type: 'folder',
          name: 'Fixture bookmarks',
          children: [
            {
              id: '2',
              type: 'url',
              name: 'Imported Chrome bookmark',
              url: 'https://bookmark.s64.test/',
            },
            { id: '3', type: 'url', name: 'Excluded local bookmark', url: 'file:///excluded.html' },
          ],
        },
      },
    }),
  );
  await writeFile(
    join(selectedProfile, 'Preferences'),
    JSON.stringify({
      default_search_provider_data: {
        template_url_data: { url: 'https://www.bing.com/search?q={searchTerms}' },
      },
    }),
  );
  const database = new DatabaseSync(join(selectedProfile, 'History'));
  try {
    database.exec(
      'PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0; CREATE TABLE urls (id INTEGER PRIMARY KEY, url TEXT, title TEXT); CREATE TABLE visits (id INTEGER PRIMARY KEY, url INTEGER, visit_time INTEGER)',
    );
    database
      .prepare('INSERT INTO urls VALUES (?, ?, ?)')
      .run(1, 'https://history.s64.test/', 'Imported Chrome history');
    database
      .prepare('INSERT INTO urls VALUES (?, ?, ?)')
      .run(2, 'file:///excluded-history.html', 'Excluded local history');
    const timestamp =
      11_644_473_600_000_000n + BigInt(Date.parse('2026-10-30T10:00:00.000Z')) * 1000n;
    const visit = database.prepare('INSERT INTO visits VALUES (?, ?, ?)');
    visit.run(1, 1, timestamp);
    visit.run(2, 1, timestamp);
    visit.run(3, 1, timestamp + 1000n);
    visit.run(4, 2, timestamp);
    const originals = [
      join(source, 'Local State'),
      ...['Bookmarks', 'Preferences', 'History', 'History-wal', 'History-shm'].map((name) =>
        join(selectedProfile, name),
      ),
    ];
    for (const path of originals) await chmod(path, 0o400);
    const hashes = async () =>
      Promise.all(
        originals.map(async (path) =>
          createHash('sha256')
            .update(await readFile(path))
            .digest('hex'),
        ),
      );
    const originalHashes = await hashes();

    // Native main-process fixture changes only this spawned process's browser discovery home.
    // No host environment, production import service, parser or trusted UI call is replaced.
    const importedHome = await application.evaluate((_electron, home) => {
      const os = process.getBuiltinModule('node:os');
      os.homedir = () => home;
      process.getBuiltinModule('node:module').syncBuiltinESMExports();
      return os.homedir();
    }, fixtureHome);
    expect(importedHome).toBe(fixtureHome);
    const store = (name: string) =>
      readFile(join(profile, 'browser-core', name), 'utf8').catch(
        (error: NodeJS.ErrnoException) => {
          if (error.code === 'ENOENT') return undefined;
          throw error;
        },
      );
    await expect
      .poll(async () => {
        const value = await store('history.json');
        return value
          ? JSON.parse(value).entries.some(
              (entry: { url: string; title: string }) =>
                entry.url === `${base}/ordinary` && entry.title === 'Sprint 64 fixture',
            )
          : false;
      })
      .toBe(true);
    const before = await Promise.all(
      ['bookmarks.json', 'history.json', 'appearance.json'].map(store),
    );
    await menu(chrome, 'Browser settings');
    const settings = chrome.getByRole('region', { name: 'Import browser data', exact: true });
    await settings
      .getByLabel('Browser and profile')
      .selectOption({ label: 'Google Chrome — Fixture Profile' });
    await expect(settings.getByRole('checkbox', { name: 'Bookmarks', exact: true })).toBeChecked();
    await expect(
      settings.getByRole('checkbox', { name: 'Browsing history', exact: true }),
    ).toBeChecked();
    await settings
      .getByRole('checkbox', { name: 'Supported default search engine', exact: true })
      .check();
    await settings.getByRole('button', { name: 'Preview import', exact: true }).click();
    const preview = settings.getByRole('region', { name: 'Import preview', exact: true });
    await expect(preview).toContainText('1 bookmarks, 2 folders and 2 history entries.');
    await expect(preview).toContainText('Skipped 1: duplicate history entry.');
    await expect(preview).toContainText('Import starts only when you confirm.');
    await preview.screenshot({ path: info.outputPath('browser-import-preview.png') });
    expect(
      await Promise.all(['bookmarks.json', 'history.json', 'appearance.json'].map(store)),
    ).toEqual(before);
    expect(await hashes()).toEqual(originalHashes);
    expect(await readdir(join(profile, 'browser-core', 'import-temp'))).toEqual([]);
    await preview.getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect(preview).toHaveCount(0);
    expect(
      await Promise.all(['bookmarks.json', 'history.json', 'appearance.json'].map(store)),
    ).toEqual(before);
    await settings.getByRole('button', { name: 'Preview import', exact: true }).click();
    await expect(preview).toContainText('1 bookmarks, 2 folders and 2 history entries.');
    await preview.getByRole('button', { name: 'Confirm import', exact: true }).click();
    await expect(settings.getByRole('status')).toHaveText(
      'Imported 1 bookmarks and 2 history entries.',
    );
    const bookmarks = JSON.parse((await store('bookmarks.json'))!).nodes as {
      kind: string;
      title: string;
      url?: string;
    }[];
    expect(bookmarks.filter((entry) => entry.kind === 'bookmark')).toEqual([
      expect.objectContaining({
        title: 'Imported Chrome bookmark',
        url: 'https://bookmark.s64.test/',
      }),
    ]);
    expect(
      bookmarks.some(
        (entry) => entry.kind === 'folder' && entry.title === 'Imported from Google Chrome',
      ),
    ).toBe(true);
    const history = JSON.parse((await store('history.json'))!).entries as {
      url: string;
      visitedAt: string;
    }[];
    expect(
      history
        .filter((entry) => entry.url === 'https://history.s64.test/')
        .map((entry) => entry.visitedAt)
        .sort(),
    ).toEqual(['2026-10-30T10:00:00.000Z', '2026-10-30T10:00:00.001Z']);
    expect(history.some((entry) => entry.url === `${base}/ordinary`)).toBe(true);
    expect(JSON.parse((await store('appearance.json'))!).searchProvider).toBe('bing');
    expect(await hashes()).toEqual(originalHashes);

    const safari = join(fixtureHome, 'safari-export');
    await mkdir(safari);
    await writeFile(
      join(safari, 'Bookmarks.html'),
      '<!DOCTYPE NETSCAPE-Bookmark-file-1><DL><A HREF="https://safari-bookmark.s64.test/">Imported Safari bookmark</A></DL>',
    );
    await writeFile(
      join(safari, 'History.json'),
      JSON.stringify({
        metadata: { browser_name: 'Safari', schema_version: 1, data_type: 'history' },
        history: [
          {
            url: 'https://safari-history.s64.test/',
            title: 'Imported Safari history',
            time_usec: 1_793_354_400_000_000,
          },
        ],
      }),
    );
    const password = `excluded-password-${randomUUID()}`;
    await writeFile(join(safari, 'Passwords.csv'), `username,password\nfixture,${password}`);
    const archive = join(safari, 'Safari.zip');
    execFileSync(
      '/usr/bin/zip',
      ['-q', archive, 'Bookmarks.html', 'History.json', 'Passwords.csv'],
      { cwd: safari },
    );
    const archiveHash = createHash('sha256')
      .update(await readFile(archive))
      .digest('hex');
    await application.evaluate(({ dialog }, selected) => {
      dialog.showOpenDialog = (async () => ({
        canceled: false,
        filePaths: [selected],
      })) as typeof dialog.showOpenDialog;
    }, archive);
    await settings
      .getByLabel('Browser and profile')
      .selectOption({ label: 'Safari — Exported browsing data' });
    await settings
      .getByRole('button', { name: 'Choose Safari export and preview…', exact: true })
      .click();
    await expect(preview).toContainText('1 bookmarks, 1 folders and 1 history entries.');
    await preview.getByRole('button', { name: 'Confirm import', exact: true }).click();
    await expect(settings.getByRole('status')).toHaveText(
      'Imported 1 bookmarks and 1 history entries.',
    );
    expect((await store('bookmarks.json'))!).toContain('https://safari-bookmark.s64.test/');
    expect((await store('history.json'))!).toContain('https://safari-history.s64.test/');
    for (const value of await Promise.all(
      ['bookmarks.json', 'history.json', 'appearance.json'].map(store),
    ))
      expect(value).not.toContain(password);
    expect(
      createHash('sha256')
        .update(await readFile(archive))
        .digest('hex'),
    ).toBe(archiveHash);
    expect(await hashes()).toEqual(originalHashes);
    expect(await readdir(join(profile, 'browser-core', 'import-temp'))).toEqual([]);
  } finally {
    database.close();
  }
});

test('S64-07 fake-device capture access remains truthful until Stop actually closes its page', async ({}, info) => {
  // This source-lane fixture supplies synthetic Chromium devices only. It does not
  // qualify signed macOS hardware permissions or replace Chromium's permission handler.
  await application.evaluate(({ dialog, systemPreferences }) => {
    const originalStatus = systemPreferences.getMediaAccessStatus.bind(systemPreferences);
    systemPreferences.getMediaAccessStatus = (type) =>
      type === 'camera' || type === 'microphone' ? 'granted' : originalStatus(type);
    const dialogs: { kind: string; message: string; buttons: string[]; response: number }[] = [];
    (
      globalThis as typeof globalThis & { sprint64CaptureDialogs: typeof dialogs }
    ).sprint64CaptureDialogs = dialogs;
    let stopRequests = 0;
    let unloadRequests = 0;
    dialog.showMessageBox = (async (...args: unknown[]) => {
      const options = args.at(-1) as { title?: string; message: string; buttons?: string[] };
      let response: number;
      if (options.title === 'Site permission') response = 2;
      else if (options.message === 'Stop capture and close this page?')
        response = stopRequests++ === 0 ? 0 : 1;
      else throw new Error(`Unexpected fixture dialog: ${options.message}`);
      dialogs.push({
        kind: 'async',
        message: options.message,
        buttons: options.buttons ?? [],
        response,
      });
      return { response, checkboxChecked: false };
    }) as typeof dialog.showMessageBox;
    dialog.showMessageBoxSync = ((...args: unknown[]) => {
      const options = args.at(-1) as { message: string; buttons?: string[] };
      if (options.message !== 'Leave this page?')
        throw new Error(`Unexpected fixture sync dialog: ${options.message}`);
      const response = unloadRequests++ === 0 ? 0 : 1;
      dialogs.push({
        kind: 'sync',
        message: options.message,
        buttons: options.buttons ?? [],
        response,
      });
      return response;
    }) as typeof dialog.showMessageBoxSync;
  });
  const page = await navigate(chrome, `${base}/capture`);
  await page.getByRole('button', { name: 'Request fake camera and microphone' }).click();
  await expect(page.locator('#capture-state')).toHaveText('audio:live, video:live');
  const indicator = chrome.getByRole('button', {
    name: 'This page can use camera, microphone',
    exact: true,
  });
  await expect(indicator).toBeVisible();
  const permissionRecords = async () =>
    (
      JSON.parse(
        await readFile(join(profile, 'browser-core', 'site-permissions.json'), 'utf8'),
      ) as { records: { origin: string; permission: string; decision: string }[] }
    ).records.filter(
      (record) => record.origin === base && ['camera', 'microphone'].includes(record.permission),
    );
  expect(
    (await permissionRecords()).map((record) => [record.permission, record.decision]).sort(),
  ).toEqual([
    ['camera', 'allow'],
    ['microphone', 'allow'],
  ]);
  await indicator.click();
  const capture = chrome.getByRole('region', { name: 'Pages with capture access', exact: true });
  await expect(capture).toContainText(`${base}: camera, microphone`);
  await expect(capture).toContainText('This does not mean a device is currently recording.');
  await chrome.screenshot({ path: info.outputPath('fake-device-capture-access.png') });
  const stop = capture.getByRole('button', { name: 'Stop and close page', exact: true });

  // Canceling the explicit native Stop confirmation preserves the page and grant.
  await stop.click();
  await expect(indicator).toBeVisible();
  expect(page.isClosed()).toBe(false);
  expect(await permissionRecords()).toHaveLength(2);

  // Accept Stop, then cancel the real Chromium beforeunload prompt. Access remains
  // visible while the document lives, but both persistent grants have been revoked.
  await stop.click();
  await expect(capture).toContainText('still has access until the page closes');
  expect(page.isClosed()).toBe(false);
  expect(await permissionRecords()).toEqual([]);
  await chrome.getByRole('button', { name: 'Close Site controls', exact: true }).click();
  await page.getByRole('button', { name: 'Request fake camera and microphone' }).click();
  await expect(page.locator('#capture-state')).toHaveText('NotAllowedError');
  expect(
    await page.evaluate(() =>
      (window as typeof window & { fixtureStream: MediaStream }).fixtureStream
        .getTracks()
        .map((track) => track.readyState),
    ),
  ).toEqual(['live', 'live']);
  await indicator.click();
  await expect(capture).toContainText('still has access until the page closes');

  // A second accepted close discards the fixture's unsaved work through the native
  // page-close path. Only actual document destruction removes its indicator.
  await stop.click();
  await expect.poll(() => page.isClosed()).toBe(true);
  await expect(indicator).toHaveCount(0);
  await expect(capture).toHaveCount(0);
  expect(await permissionRecords()).toEqual([]);
  const dialogs = await application.evaluate(
    () =>
      (
        globalThis as typeof globalThis & {
          sprint64CaptureDialogs: {
            kind: string;
            message: string;
            buttons: string[];
            response: number;
          }[];
        }
      ).sprint64CaptureDialogs,
  );
  const permissionDialogs = dialogs.filter(
    (entry) => entry.kind === 'async' && entry.message.includes(' requests '),
  );
  expect(permissionDialogs).toHaveLength(1);
  expect(permissionDialogs[0]!.message).toContain(base);
  expect(permissionDialogs[0]!.message).toContain('camera');
  expect(permissionDialogs[0]!.message).toContain('microphone');
  expect(
    dialogs
      .filter((entry) => entry.message === 'Stop capture and close this page?')
      .map((entry) => entry.response),
  ).toEqual([0, 1, 1]);
  expect(
    dialogs.filter((entry) => entry.message === 'Leave this page?').map((entry) => entry.response),
  ).toEqual([0, 1]);
  await info.attach('fake-capture-native-dialogs', {
    body: JSON.stringify(dialogs, null, 2),
    contentType: 'application/json',
  });
});

test('S64-02 delegated embedded controls allow fullscreen and revoke an ordinary saved location grant', async ({}, info) => {
  const childOrigin = new URL(base);
  childOrigin.hostname = 'localhost';
  // CDP supplies coordinates only; Oryn's native permission check/request handlers
  // still decide access. No OS location provider or permission grant is replaced.
  await application.context().setGeolocation({ latitude: 12.345, longitude: 67.89 });
  await application.evaluate(
    ({ dialog, BrowserWindow }, origins) => {
      let requests = 0;
      const evidence: { message: string; detail?: string; response: number }[] = [];
      (
        globalThis as typeof globalThis & { sprint64EmbeddedDialogs: typeof evidence }
      ).sprint64EmbeddedDialogs = evidence;
      const window = BrowserWindow.getAllWindows()[0]!;
      const events: string[] = [];
      for (const event of [
        'enter-full-screen',
        'leave-full-screen',
        'enter-html-full-screen',
        'leave-html-full-screen',
      ] as const)
        window.on(event, () => events.push(event));
      (
        globalThis as typeof globalThis & { sprint64FullscreenEvents: string[] }
      ).sprint64FullscreenEvents = events;
      dialog.showMessageBox = (async (...args: unknown[]) => {
        const options = args.at(-1) as { title?: string; message: string; detail?: string };
        if (
          options.title !== 'Site permission' ||
          !options.message.includes(origins.top) ||
          !options.message.includes(origins.child) ||
          !options.message.includes('geolocation')
        )
          throw new Error(`Unexpected fixture permission dialog: ${options.message}`);
        const response = requests++ === 0 ? 2 : 0;
        evidence.push({ message: options.message, detail: options.detail, response });
        return { response, checkboxChecked: false };
      }) as typeof dialog.showMessageBox;
    },
    { top: base, child: childOrigin.origin },
  );
  const page = await navigate(chrome, `${base}/embedded-parent`);
  const embedded = page.frameLocator('iframe[title="Embedded permission fixture"]');
  await page.waitForLoadState('load');
  await application.evaluate(({ BrowserWindow, webContents }, url) => {
    const window = BrowserWindow.getAllWindows()[0]!;
    window.show();
    window.focus();
    webContents
      .getAllWebContents()
      .find((contents) => contents.getURL() === url)!
      .focus();
  }, `${base}/embedded-parent`);
  await embedded.getByRole('button', { name: 'Request fixture location', exact: true }).click();
  expect(
    await embedded
      .locator('html')
      .evaluate(() => (window as typeof window & { fixtureClicks: unknown[] }).fixtureClicks),
  ).toEqual(
    expect.arrayContaining([expect.objectContaining({ target: 'location', trusted: true })]),
  );
  await expect(embedded.locator('#location-state')).toHaveText('12.345,67.89');
  const permissions = async () =>
    (
      JSON.parse(
        await readFile(join(profile, 'browser-core', 'site-permissions.json'), 'utf8'),
      ) as {
        records: {
          origin: string;
          embeddedOrigin?: string;
          ancestorOrigins?: string[];
          permission: string;
          decision: string;
        }[];
      }
    ).records.filter(
      (record) =>
        record.embeddedOrigin === childOrigin.origin && record.permission === 'geolocation',
    );
  expect(await permissions()).toEqual([
    expect.objectContaining({
      origin: base,
      embeddedOrigin: childOrigin.origin,
      ancestorOrigins: [base],
      permission: 'geolocation',
      decision: 'allow',
    }),
  ]);
  await chrome.getByRole('button', { name: 'Site controls', exact: true }).click();
  const records = chrome.getByRole('region', { name: 'Embedded sites', exact: true });
  await expect(records).toContainText(`${childOrigin.origin}: geolocation — allow`);
  await records.screenshot({ path: info.outputPath('embedded-site-permission.png') });
  await records.getByRole('button', { name: 'Revoke', exact: true }).click();
  await expect(records).toHaveCount(0);
  expect(await permissions()).toEqual([]);
  await chrome.getByRole('button', { name: 'Close Site controls', exact: true }).click();
  await embedded.getByRole('button', { name: 'Request fixture location', exact: true }).click();
  await expect(embedded.locator('#location-state')).toHaveText('Denied:1');
  const evidence = await application.evaluate(
    () =>
      (
        globalThis as typeof globalThis & {
          sprint64EmbeddedDialogs: { message: string; detail?: string; response: number }[];
        }
      ).sprint64EmbeddedDialogs,
  );
  expect(evidence.map((entry) => entry.response)).toEqual([2, 0]);
  await info.attach('embedded-permission-native-dialogs', {
    body: JSON.stringify(evidence, null, 2),
    contentType: 'application/json',
  });
  // Native macOS fullscreen transitions finish after the DOM promise. Wait for
  // the actual native window/event before sending the trusted exit input.
  await embedded.getByRole('button', { name: 'Enter embedded fullscreen', exact: true }).click();
  await expect(embedded.locator('#fullscreen-state')).toHaveText('Entered');
  expect(
    await embedded
      .locator('html')
      .evaluate((element) => element.ownerDocument.fullscreenElement === element),
  ).toBe(true);
  const nativeFullscreen = () =>
    application.evaluate(({ BrowserWindow }) => ({
      fullscreen: BrowserWindow.getAllWindows()[0]!.isFullScreen(),
      visible: BrowserWindow.getAllWindows()[0]!.isVisible(),
      focused: BrowserWindow.getAllWindows()[0]!.isFocused(),
      events: [
        ...(globalThis as typeof globalThis & { sprint64FullscreenEvents: string[] })
          .sprint64FullscreenEvents,
      ],
    }));
  await expect
    .poll(async () => {
      const state = await nativeFullscreen();
      return state.fullscreen && state.events.includes('enter-full-screen');
    })
    .toBe(true);
  await embedded.getByRole('button', { name: 'Exit embedded fullscreen', exact: true }).click();
  await expect(embedded.locator('#fullscreen-state')).toHaveText('Exited');
  await expect
    .poll(async () => {
      const state = await nativeFullscreen();
      return !state.fullscreen && state.events.includes('leave-full-screen');
    })
    .toBe(true);
  await info.attach('embedded-fullscreen-native-events', {
    body: JSON.stringify(await nativeFullscreen(), null, 2),
    contentType: 'application/json',
  });
});

test('S64-10 canceling a private download removes unfinished bytes and retains no normal download record', async () => {
  const normal = await website(`${base}/ordinary`);
  await normal.evaluate(() => {
    document.cookie = 'ordinaryKeep=normal-sentinel; path=/';
  });
  const owner = await newPrivate();
  const token = `private-canceled-${randomUUID()}`;
  const pageURL = `${base}/private?token=${token}`;
  const page = await navigate(owner, pageURL);
  const target = join(profile, 'private-canceled.bin');
  await application.evaluate(
    ({ webContents }, selected) => {
      const contents = webContents
        .getAllWebContents()
        .find((item) => item.getURL() === selected.url)!;
      contents.session.once('will-download', (_event, item) => item.setSavePath(selected.path));
    },
    { url: pageURL, path: target },
  );
  await page.getByRole('link', { name: 'Save fixture file' }).evaluate((link, url) => {
    (link as HTMLAnchorElement).href = url;
  }, `${base}/slow-private-file?token=${token}`);
  await page.getByRole('link', { name: 'Save fixture file' }).click();
  await owner.getByRole('button', { name: 'Downloads', exact: true }).click();
  await expect(owner.getByText('private-canceled.bin', { exact: true })).toBeVisible();
  await expect(owner.getByRole('progressbar')).toBeVisible();
  await expect
    .poll(() =>
      readFile(target).then(
        (bytes) => bytes.includes(Buffer.from(token)),
        () => false,
      ),
    )
    .toBe(true);
  await owner.getByRole('button', { name: 'Cancel download', exact: true }).click();
  await expect(owner.getByRole('status')).toContainText('cancelled');
  await expect
    .poll(() =>
      stat(target).then(
        () => true,
        () => false,
      ),
    )
    .toBe(false);
  expect(await filesContainingToken(profile, token)).toEqual([]);
  expect(await normal.evaluate(() => document.cookie)).toContain('ordinaryKeep=normal-sentinel');
  await owner.getByRole('button', { name: 'Downloads', exact: true }).click();
  await menu(owner, 'Close window');
  await expect.poll(async () => (await journal()).ids).toEqual([]);
  expect(await filesContainingToken(profile, token)).toEqual([]);
});

test('S64-10 normal Quit finishes private cleanup before process exit and restores only normal data', async () => {
  const normal = await website(`${base}/ordinary`);
  await normal.evaluate(() => {
    document.cookie = 'ordinaryQuit=keep-normal; path=/; max-age=3600';
  });
  const owner = await newPrivate();
  const token = `private-normal-quit-${randomUUID()}`;
  const page = await navigate(owner, `${base}/private?token=${token}`);
  await page.getByRole('button', { name: 'Store private fixture data' }).click();
  await expect(page.locator('#stored')).toHaveText('Stored');
  const ids = (await journal()).ids;
  expect(ids).toHaveLength(1);
  const temporary = join(profile, 'private-temp', ids[0]!, 'normal-quit-fixture');
  await writeFile(temporary, token);
  const saved = join(profile, 'explicit-user-save.txt');
  await writeFile(saved, 'keep-explicit-save');
  const exited = new Promise<void>((done) => child.once('exit', () => done()));
  void application.evaluate(({ app }) => app.quit()).catch(() => undefined);
  await exited;
  expect(child.exitCode).toBe(0);
  expect((await journal()).ids).toEqual([]);
  expect(
    await stat(temporary).then(
      () => true,
      () => false,
    ),
  ).toBe(false);
  expect(await filesContainingToken(profile, token)).toEqual([]);
  expect(await readFile(saved, 'utf8')).toBe('keep-explicit-save');
  await launch();
  expect(mainChromePages()).toHaveLength(1);
  expect(await chrome.getByRole('button', { name: 'Private browsing', exact: true }).count()).toBe(
    0,
  );
  const restored = await website(`${base}/ordinary`);
  expect(await restored.evaluate(() => document.cookie)).toContain('ordinaryQuit=keep-normal');
  expect(await filesContainingToken(profile, token)).toEqual([]);
});

test('S64-10 failed startup cleanup warns, blocks new private windows and retries successfully next startup', async () => {
  test.skip(
    Boolean(process.env.ORYN_PACKAGED_APP),
    'Startup-dialog injection uses a source bootstrap; signed native startup recovery remains a separate qualification gate.',
  );
  await killOwnedInstance();
  const id = 'a'.repeat(32);
  const temporaryRoot = join(profile, 'private-temp');
  const unrelated = join(profile, 'explicit-kept-directory');
  await mkdir(temporaryRoot, { recursive: true });
  await mkdir(unrelated);
  await writeFile(join(unrelated, 'keep.txt'), 'keep-explicit-user-save');
  await symlink(unrelated, join(temporaryRoot, id), 'dir');
  await writeFile(
    join(profile, 'browser-core', 'private-cleanup.json'),
    JSON.stringify({ version: 1, ids: [id] }),
  );
  const bootstrap = join(profile, 'cleanup-ui-bootstrap');
  await mkdir(bootstrap);
  await writeFile(
    join(bootstrap, 'package.json'),
    JSON.stringify({
      name: 'oryn-cleanup-fixture',
      version: '0.1.0',
      type: 'commonjs',
      main: 'bootstrap.cjs',
    }),
  );
  await writeFile(
    join(bootstrap, 'bootstrap.cjs'),
    `
    const { dialog, app } = require('electron');
    app.setAppPath(${JSON.stringify(resolve('.'))});
    globalThis.cleanupFixtureDialogs = [];
    dialog.showMessageBox = async (...args) => {
      const options = args.at(-1);
      globalThis.cleanupFixtureDialogs.push({ message: options.message, detail: options.detail });
      return { response: 0, checkboxChecked: false };
    };
    require(${JSON.stringify(resolve('out/main/index.js'))});
  `,
  );
  await launch([], bootstrap);
  const warnings = await application.evaluate(
    () =>
      (
        globalThis as typeof globalThis & {
          cleanupFixtureDialogs: { message: string; detail: string }[];
        }
      ).cleanupFixtureDialogs,
  );
  expect(warnings).toContainEqual(
    expect.objectContaining({ message: 'Private temporary-file cleanup is pending' }),
  );
  expect(warnings.some((warning) => warning.detail.includes('retry on the next startup'))).toBe(
    true,
  );
  const blocked = await chrome.evaluate(() =>
    window.browser.newPrivateWindow().then(
      () => 'unexpected-success',
      (error) => String(error),
    ),
  );
  expect(blocked).toContain('PRIVATE_CLEANUP_UNAVAILABLE');
  expect(mainChromePages()).toHaveLength(1);
  expect((await journal()).ids).toEqual([id]);
  expect(await readFile(join(unrelated, 'keep.txt'), 'utf8')).toBe('keep-explicit-user-save');
  await killOwnedInstance();
  await rm(join(temporaryRoot, id));
  await launch([], bootstrap);
  expect((await journal()).ids).toEqual([]);
  expect(
    await application.evaluate(
      () =>
        (globalThis as typeof globalThis & { cleanupFixtureDialogs: unknown[] })
          .cleanupFixtureDialogs,
    ),
  ).toEqual([]);
  const owner = await newPrivate();
  await menu(owner, 'Close window');
  await expect.poll(async () => (await journal()).ids).toEqual([]);
  expect(await readFile(join(unrelated, 'keep.txt'), 'utf8')).toBe('keep-explicit-user-save');
});
