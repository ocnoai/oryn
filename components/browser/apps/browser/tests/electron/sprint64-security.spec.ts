import {
  _electron as electron,
  expect,
  test,
  type ElectronApplication,
  type Page,
} from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { X509Certificate } from 'node:crypto';
import { createServer, type IncomingMessage, type ServerResponse, type Server } from 'node:http';
import { createServer as createTLSServer } from 'node:https';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { TLSSocket } from 'node:tls';

/* eslint-disable no-empty-pattern -- Playwright inspects fixture destructuring. */
test.setTimeout(60_000);

let fixtures: string;
let http: Server;
let tls: Server;
let expiredTLS: Server;
let chainTLS: Server;
let httpPort: number;
let tlsPort: number;
let expiredPort: number;
let chainPort: number;
let caPEM: string;
let intermediatePEM: string;
let leaf: X509Certificate;
let expiredLeaf: X509Certificate;
let chainLeaf: X509Certificate;
let app: ElectronApplication;
let chrome: Page;
let profile: string;
const received: { method: string; url: string; body: string }[] = [];
const delivered: { secure: boolean; host: string; path: string }[] = [];
const escapeHTML = (value: string) =>
  value.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;');

function response(request: IncomingMessage, result: ServerResponse): void {
  const url = new URL(request.url ?? '/', 'http://fixture.test');
  delivered.push({
    secure: request.socket instanceof TLSSocket,
    host: request.headers.host ?? '',
    path: url.pathname,
  });
  result.setHeader('Content-Type', 'text/html; charset=utf-8');
  if (url.pathname === '/pixel') {
    result.setHeader('Content-Type', 'image/svg+xml');
    result.end(
      '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"><rect width="1" height="1"/></svg>',
    );
    return;
  }
  if (url.pathname === '/upgraded-image' || url.pathname === '/failed-image-upgrade') {
    const port = url.pathname === '/upgraded-image' ? tlsPort : httpPort;
    result.end(
      `<!doctype html><title>Mixed image fixture</title><h1>Mixed image fixture</h1><img alt="Mixed fixture image" src="http://insecure.test:${port}/pixel">`,
    );
    return;
  }
  if (url.pathname === '/http-iframe') {
    result.end(
      `<!doctype html><title>Mixed iframe fixture</title><h1>Mixed iframe fixture</h1><iframe title="Blocked insecure iframe" src="http://insecure.test:${httpPort}/frame-content"></iframe>`,
    );
    return;
  }
  if (url.pathname === '/frame-content') {
    result.end('<!doctype html><h1>Insecure frame delivered</h1>');
    return;
  }
  if (url.pathname === '/receive') {
    let body = '';
    request.on('data', (chunk) => {
      body += chunk.toString();
    });
    request.on('end', () => {
      received.push({ method: request.method ?? '', url: url.pathname + url.search, body });
      result.end(
        '<!doctype html><meta charset="utf-8"><title>Original request received</title><h1>Original request received</h1>',
      );
    });
    return;
  }
  if (url.pathname === '/mixed.js') {
    result.setHeader('Content-Type', 'text/javascript');
    result.end('window.mixedExecuted=true;');
    return;
  }
  if (url.pathname === '/mixed') {
    result.end(
      `<!doctype html><title>Mixed content fixture</title><script src="http://insecure.test:${httpPort}/mixed.js"></script><h1>Mixed content fixture</h1>`,
    );
    return;
  }
  if (url.pathname === '/embedded') {
    result.end(
      `<!doctype html><title>Embedded form fixture</title><iframe title="Embedded form" width="700" height="250" src="https://frame.test:${tlsPort}/form${escapeHTML(url.search)}"></iframe>`,
    );
    return;
  }
  if (url.pathname === '/form') {
    const method = url.searchParams.get('method') === 'POST' ? 'POST' : 'GET';
    const target = ['_self', '_blank', '_top'].includes(url.searchParams.get('target') ?? '')
      ? url.searchParams.get('target')!
      : '_self';
    result.end(`<!doctype html><meta charset="utf-8"><meta name="referrer" content="no-referrer"><title>Secure form fixture</title>
      <form action="http://insecure.test:${httpPort}/receive" method="${method}" target="${target}">
      <input name="sentinel" value="${escapeHTML(url.searchParams.get('sentinel') ?? '')}">
      <input name="repeated" value="one"><input name="repeated" value="two"><button>Submit form</button></form>
      <script>
      for (const type of ['click','submit']) document.querySelector('form').addEventListener(type, event => {
        document.documentElement.dataset[type] = JSON.stringify({trusted:event.isTrusted,active:navigator.userActivation.isActive});
      });
      </script>`);
    return;
  }
  result.end('<!doctype html><title>Security fixture ready</title><h1>Security fixture ready</h1>');
}

async function listen(server: Server): Promise<number> {
  await new Promise<void>((done, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', done);
  });
  const address = server.address();
  if (!address || typeof address === 'string')
    throw new Error('Fixture server address unavailable.');
  return address.port;
}

test.beforeAll(async () => {
  fixtures = await mkdtemp(join(tmpdir(), 'oryn-s64-test-ca-'));
  const openssl = (arguments_: string[]) =>
    execFileSync('openssl', arguments_, { cwd: fixtures, stdio: 'ignore' });
  openssl([
    'req',
    '-x509',
    '-newkey',
    'rsa:2048',
    '-nodes',
    '-keyout',
    'ca.key',
    '-out',
    'ca.crt',
    '-days',
    '2',
    '-subj',
    '/CN=Oryn ephemeral test CA',
  ]);
  openssl([
    'req',
    '-newkey',
    'rsa:2048',
    '-nodes',
    '-keyout',
    'leaf.key',
    '-out',
    'leaf.csr',
    '-subj',
    '/CN=secure.test',
  ]);
  await writeFile(
    join(fixtures, 'leaf.ext'),
    'basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=DNS:secure.test,DNS:frame.test,DNS:insecure.test\n',
  );
  openssl([
    'x509',
    '-req',
    '-in',
    'leaf.csr',
    '-CA',
    'ca.crt',
    '-CAkey',
    'ca.key',
    '-set_serial',
    '0x640001',
    '-out',
    'leaf.crt',
    '-days',
    '2',
    '-extfile',
    'leaf.ext',
  ]);
  openssl([
    'req',
    '-new',
    '-newkey',
    'rsa:2048',
    '-nodes',
    '-keyout',
    'intermediate.key',
    '-out',
    'intermediate.csr',
    '-subj',
    '/CN=Oryn ephemeral intermediate CA',
  ]);
  await writeFile(
    join(fixtures, 'intermediate.ext'),
    'basicConstraints=critical,CA:TRUE,pathlen:0\nkeyUsage=critical,keyCertSign,cRLSign\n',
  );
  openssl([
    'x509',
    '-req',
    '-in',
    'intermediate.csr',
    '-CA',
    'ca.crt',
    '-CAkey',
    'ca.key',
    '-set_serial',
    '0x640010',
    '-out',
    'intermediate.crt',
    '-days',
    '2',
    '-extfile',
    'intermediate.ext',
  ]);
  openssl([
    'x509',
    '-req',
    '-in',
    'leaf.csr',
    '-CA',
    'intermediate.crt',
    '-CAkey',
    'intermediate.key',
    '-set_serial',
    '0x640011',
    '-out',
    'chain-leaf.crt',
    '-days',
    '2',
    '-extfile',
    'leaf.ext',
  ]);
  openssl([
    'x509',
    '-req',
    '-in',
    'leaf.csr',
    '-CA',
    'ca.crt',
    '-CAkey',
    'ca.key',
    '-set_serial',
    '0x640002',
    '-out',
    'expired.crt',
    '-days',
    '0',
    '-extfile',
    'leaf.ext',
  ]);
  caPEM = await readFile(join(fixtures, 'ca.crt'), 'utf8');
  const leafPEM = await readFile(join(fixtures, 'leaf.crt'), 'utf8');
  const expiredPEM = await readFile(join(fixtures, 'expired.crt'), 'utf8');
  intermediatePEM = await readFile(join(fixtures, 'intermediate.crt'), 'utf8');
  const chainPEM = await readFile(join(fixtures, 'chain-leaf.crt'), 'utf8');
  leaf = new X509Certificate(leafPEM);
  expiredLeaf = new X509Certificate(expiredPEM);
  chainLeaf = new X509Certificate(chainPEM);
  const key = await readFile(join(fixtures, 'leaf.key'));
  http = createServer(response);
  httpPort = await listen(http);
  tls = createTLSServer({ key, cert: `${leafPEM}\n${caPEM}` }, response);
  tlsPort = await listen(tls);
  expiredTLS = createTLSServer({ key, cert: `${expiredPEM}\n${caPEM}` }, response);
  expiredPort = await listen(expiredTLS);
  chainTLS = createTLSServer({ key, cert: `${chainPEM}\n${intermediatePEM}\n${caPEM}` }, response);
  chainPort = await listen(chainTLS);
});

test.afterAll(async () => {
  for (const server of [http, tls, expiredTLS, chainTLS].filter(Boolean)) {
    server.closeAllConnections();
    await new Promise<void>((done) => server.close(() => done()));
  }
  if (fixtures) await rm(fixtures, { recursive: true, force: true });
});

test.beforeEach(async () => {
  received.length = 0;
  delivered.length = 0;
  profile = await mkdtemp(join(tmpdir(), 'oryn-s64-security-ui-'));
  await mkdir(join(profile, 'browser-core'), { recursive: true });
  await writeFile(
    join(profile, 'browser-core/appearance.json'),
    JSON.stringify({ version: 1, theme: 'light', density: 'default', onboardingComplete: true }),
  );
  const env = {
    PATH: process.env.PATH ?? '/usr/bin:/bin',
    TMPDIR: process.env.TMPDIR ?? tmpdir(),
    LANG: 'en_US.UTF-8',
    ORYN_PROFILE_ROOT: profile,
    ORYN_START_URL: `http://127.0.0.1:${httpPort}/start`,
    ORYN_ENGINE_DIR: join(profile, 'unavailable-engine'),
  };
  app = await electron.launch({
    ...(process.env.ORYN_PACKAGED_APP
      ? { executablePath: resolve(process.env.ORYN_PACKAGED_APP, 'Contents/MacOS/Oryn Browser') }
      : process.env.ORYN_SOURCE_ELECTRON
        ? { executablePath: process.env.ORYN_SOURCE_ELECTRON }
        : {}),
    args: [
      ...(!process.env.ORYN_PACKAGED_APP ? [resolve('.')] : []),
      '--use-mock-keychain',
      '--no-proxy-server',
      '--host-resolver-rules=MAP *.test 127.0.0.1, EXCLUDE localhost',
    ],
    env,
  });
  app.process().stderr?.on('data', (chunk) => process.stderr.write(chunk));
  await expect
    .poll(() =>
      app
        .context()
        .pages()
        .some((page) => /index\.html$/.test(page.url())),
    )
    .toBe(true);
  chrome = app
    .context()
    .pages()
    .find((page) => /index\.html$/.test(page.url()))!;
  await expect(chrome.getByLabel('Browser controls')).toBeVisible();
  await app.evaluate(
    ({ session, dialog, webContents, app: nativeApp }, data) => {
      const { X509Certificate: Certificate } = process.getBuiltinModule('node:crypto') as {
        X509Certificate: typeof X509Certificate;
      };
      const root = new Certificate(data.ca);
      const intermediate = new Certificate(data.intermediate);
      const state = globalThis as typeof globalThis & {
        s64Dialogs: { title: string; buttons: string[]; defaultId?: number; cancelId?: number }[];
        s64Answers: number[];
        s64SecurityEvents: unknown[];
      };
      state.s64Dialogs = [];
      state.s64Answers = [];
      state.s64SecurityEvents = [];
      const watched = new Set<number>();
      const watch = (contents: Electron.WebContents) => {
        if (watched.has(contents.id)) return;
        watched.add(contents.id);
        const contentsId = contents.id;
        contents.debugger.on('message', (_event, method, params, sessionId) => {
          if (state.s64SecurityEvents.length >= 256) return;
          if (
            method === 'Audits.issueAdded' ||
            method === 'Network.loadingFailed' ||
            method === 'Target.attachedToTarget' ||
            method === 'Target.detachedFromTarget'
          ) {
            state.s64SecurityEvents.push({ method, params, sessionId, contentsId });
          } else if (method === 'Network.requestWillBeSent') {
            state.s64SecurityEvents.push({
              method,
              contentsId,
              sessionId,
              requestId: params.requestId,
              frameId: params.frameId,
              loaderId: params.loaderId,
              type: params.type,
              url: params.request?.url,
              mixedContentType: params.request?.mixedContentType,
            });
          } else if (method === 'Page.frameNavigated') {
            state.s64SecurityEvents.push({ method, sessionId, frame: params.frame, contentsId });
          } else if (method === 'Network.responseReceived') {
            state.s64SecurityEvents.push({
              method,
              sessionId,
              contentsId,
              requestId: params.requestId,
              frameId: params.frameId,
              loaderId: params.loaderId,
              type: params.type,
              url: params.response?.url,
              securityState: params.response?.securityState,
              securityDetails: params.response?.securityDetails,
            });
          }
        });
      };
      for (const contents of webContents.getAllWebContents()) watch(contents);
      nativeApp.on('web-contents-created', (_event, contents) => watch(contents));
      // Accept only the generated fixture leaves after genuine cryptographic,
      // hostname and validity checks. This does not change the OS trust store.
      session.defaultSession.setCertificateVerifyProc((request, callback) => {
        try {
          const certificate = new Certificate(request.certificate.data);
          const now = Date.now();
          const validAt = (candidate: X509Certificate) =>
            now >= Date.parse(candidate.validFrom) && now <= Date.parse(candidate.validTo);
          const issuer = certificate.checkIssued(root)
            ? root
            : certificate.checkIssued(intermediate) &&
                intermediate.ca &&
                intermediate.checkIssued(root) &&
                intermediate.verify(root.publicKey) &&
                validAt(intermediate)
              ? intermediate
              : undefined;
          const trusted =
            issuer !== undefined &&
            root.ca &&
            root.verify(root.publicKey) &&
            validAt(root) &&
            data.fingerprints.includes(certificate.fingerprint256) &&
            certificate.checkIssued(issuer) &&
            certificate.verify(issuer.publicKey) &&
            Boolean(certificate.checkHost(request.hostname, { wildcards: false })) &&
            validAt(certificate) &&
            validAt(issuer);
          callback(trusted ? 0 : -2);
        } catch {
          callback(-2);
        }
      });
      dialog.showMessageBox = async (...arguments_: unknown[]) => {
        const options = arguments_.at(-1) as Electron.MessageBoxOptions;
        state.s64Dialogs.push({
          title: options.title ?? '',
          buttons: options.buttons ?? [],
          defaultId: options.defaultId,
          cancelId: options.cancelId,
        });
        return {
          response: state.s64Answers.shift() ?? options.cancelId ?? 0,
          checkboxChecked: false,
        };
      };
      // Test browsing uses loopback fixtures only. Prevent unrelated main-process
      // catalog/update requests from contacting any service during this test.
      globalThis.fetch = async () => {
        throw new Error('OFFLINE_SECURITY_FIXTURE');
      };
    },
    {
      ca: caPEM,
      intermediate: intermediatePEM,
      fingerprints: [leaf.fingerprint256, expiredLeaf.fingerprint256, chainLeaf.fingerprint256],
    },
  );
});

test.afterEach(async ({}, info) => {
  if (info.status !== info.expectedStatus && app) {
    const gestures = await Promise.all(
      app
        .context()
        .pages()
        .flatMap((page) => page.frames())
        .filter((frame) => /^https:\/\/(?:secure|frame)\.test:/.test(frame.url()))
        .map(async (frame) => ({
          url: frame.url(),
          gesture: await frame
            .evaluate(() => ({
              click: document.documentElement.dataset.click,
              submit: document.documentElement.dataset.submit,
              active: navigator.userActivation.isActive,
            }))
            .catch(() => undefined),
        })),
    );
    await info.attach('Actual fixture gesture delivery', {
      body: Buffer.from(JSON.stringify(gestures, null, 2)),
      contentType: 'application/json',
    });
  }
  if (info.status !== info.expectedStatus && chrome && !chrome.isClosed()) {
    const connection = await chrome
      .evaluate(async () => (await window.browser.getBrowserData()).connection)
      .catch(() => undefined);
    await info.attach('Observed connection metadata', {
      body: Buffer.from(JSON.stringify({ connection, delivered }, null, 2)),
      contentType: 'application/json',
    });
  }
  if (info.status !== info.expectedStatus && app) {
    const state = await app
      .evaluate(({ webContents }) => ({
        dialogs: (globalThis as typeof globalThis & { s64Dialogs?: unknown[] }).s64Dialogs,
        securityEvents: (globalThis as typeof globalThis & { s64SecurityEvents?: unknown[] })
          .s64SecurityEvents,
        contents: webContents.getAllWebContents().map((contents) => ({
          id: contents.id,
          type: contents.getType(),
          url: contents.getURL(),
          loading: contents.isLoading(),
        })),
      }))
      .catch((error: Error) => ({ error: error.message }));
    await info.attach('Native security fixture state', {
      body: Buffer.from(JSON.stringify({ ...state, received, delivered }, null, 2)),
      contentType: 'application/json',
    });
  }
  if (info.status !== info.expectedStatus && chrome && !chrome.isClosed())
    await info.attach('chrome', {
      body: await chrome.screenshot({ timeout: 3000 }).catch(() => Buffer.from('')),
      contentType: 'image/png',
    });
  const child = app?.process();
  if (child && child.exitCode === null) {
    const exit = new Promise<void>((done) => child.once('exit', () => done()));
    child.kill('SIGKILL');
    await exit;
  }
  if (profile) await rm(profile, { recursive: true, force: true });
});

async function navigate(url: string): Promise<Page> {
  await chrome.getByRole('combobox', { name: 'Address and search' }).fill(url);
  await chrome.getByRole('combobox', { name: 'Address and search' }).press('Enter');
  await expect
    .poll(() =>
      app
        .context()
        .pages()
        .some((page) => page.url() === url),
    )
    .toBe(true);
  return app
    .context()
    .pages()
    .find((page) => page.url() === url)!;
}

async function setAnswer(answer: number): Promise<void> {
  await app.evaluate((_electron, response) => {
    const state = globalThis as typeof globalThis & { s64Dialogs: unknown[]; s64Answers: number[] };
    state.s64Dialogs = [];
    state.s64Answers = [response];
  }, answer);
}

test('valid fixture TLS has truthful chrome and a read-only certificate viewer', async () => {
  await navigate(`https://secure.test:${tlsPort}/valid`);
  await chrome.getByRole('button', { name: 'Site controls', exact: true }).click();
  const details = chrome.getByRole('region', { name: 'Connection details' });
  await expect(
    details.getByRole('heading', { name: 'Connection: Secure', exact: true }),
  ).toBeVisible();
  await details.getByRole('button', { name: 'View certificates', exact: true }).click();
  const viewer = details.getByRole('region', { name: 'Certificate viewer' });
  await expect(viewer).toContainText(leaf.fingerprint256);
  await expect(viewer).toContainText(leaf.serialNumber);
  await expect(viewer).toContainText('Read-only certificate details');
  await expect(viewer.getByRole('button', { name: /trust|bypass|proceed/i })).toHaveCount(0);
  const screenshot = test.info().outputPath('certificate-viewer.png');
  await viewer.screenshot({ path: screenshot });
  await test
    .info()
    .attach('Read-only fixture certificate viewer', { path: screenshot, contentType: 'image/png' });
});

test('intermediate certificate chain has exact viewer metadata and exports the reviewed bytes', async () => {
  await navigate(`https://secure.test:${chainPort}/chain`);
  await chrome.getByRole('button', { name: 'Site controls', exact: true }).click();
  const details = chrome.getByRole('region', { name: 'Connection details' });
  await expect(
    details.getByRole('heading', { name: 'Connection: Secure', exact: true }),
  ).toBeVisible();
  await details.getByRole('button', { name: 'View certificates', exact: true }).click();
  const viewer = details.getByRole('region', { name: 'Certificate viewer' });
  const expectedChain = [
    chainLeaf,
    new X509Certificate(intermediatePEM),
    new X509Certificate(caPEM),
  ];
  await expect(viewer.locator('details')).toHaveCount(expectedChain.length);
  await expect(viewer).toContainText('The chain includes its root.');
  for (const [index, certificate] of expectedChain.entries()) {
    const item = viewer.locator('details').nth(index);
    if (index !== 0) await item.locator('summary').click();
    const displayed = await item
      .locator('dl > div')
      .evaluateAll((rows) =>
        Object.fromEntries(
          rows.map((row) => [
            row.querySelector('dt')?.textContent?.trim(),
            row.querySelector('dd')?.textContent?.trim(),
          ]),
        ),
      );
    expect(displayed).toMatchObject({
      subject: certificate.subject,
      issuer: certificate.issuer,
      'serial Number': certificate.serialNumber,
      'valid From': certificate.validFrom,
      'valid To': certificate.validTo,
      'key Type': certificate.publicKey.asymmetricKeyType,
      'key Size': String(certificate.publicKey.asymmetricKeyDetails?.modulusLength),
      'signature Algorithm': '1.2.840.113549.1.1.11',
      fingerprint256: certificate.fingerprint256,
      fingerprint: certificate.fingerprint,
      'key Usage': certificate.keyUsage?.join(', ') ?? '',
      ...(certificate.subjectAltName ? { 'subject Alt Name': certificate.subjectAltName } : {}),
    });
  }
  const exportPath = join(fixtures, 'reviewed-chain-export.pem');
  await app.evaluate(({ dialog }, filePath) => {
    dialog.showSaveDialog = async () => ({ canceled: false, filePath });
  }, exportPath);
  await viewer.getByRole('button', { name: 'Export certificate chain…', exact: true }).click();
  await expect.poll(async () => readFile(exportPath, 'utf8').catch(() => '')).not.toBe('');
  const exported = await readFile(exportPath, 'utf8');
  const exportedChain =
    exported.match(/-----BEGIN CERTIFICATE-----[\s\S]+?-----END CERTIFICATE-----/g) ?? [];
  expect(exportedChain.map((pem) => new X509Certificate(pem).raw)).toEqual(
    expectedChain.map((certificate) => certificate.raw),
  );
  expect(exported).not.toContain('PRIVATE KEY');
  await test.info().attach('Reviewed public certificate chain', {
    path: exportPath,
    contentType: 'application/x-pem-file',
  });
});

test('wrong-host and expired fixture certificates remain blocked', async () => {
  for (const url of [
    `https://wrong.test:${tlsPort}/wrong-host`,
    `https://secure.test:${expiredPort}/expired`,
  ]) {
    const matchesFailure = (page: Page) =>
      page.url().startsWith('oryn://network-error/') &&
      new URL(page.url()).searchParams.get('url') === url;
    await chrome.getByRole('combobox', { name: 'Address and search' }).fill(url);
    await chrome.getByRole('combobox', { name: 'Address and search' }).press('Enter');
    await expect.poll(() => app.context().pages().some(matchesFailure)).toBe(true);
    const errorPage = app.context().pages().find(matchesFailure)!;
    await expect(
      errorPage.getByRole('heading', { name: 'This page could not be reached' }),
    ).toBeVisible();
    await expect(errorPage.getByRole('button', { name: /proceed|trust|bypass/i })).toHaveCount(0);
  }
  expect(received).toEqual([]);
});

test('blocked mixed script is reported without running insecure code', async () => {
  const page = await navigate(`https://secure.test:${tlsPort}/mixed`);
  expect(await page.evaluate(() => 'mixedExecuted' in window)).toBe(false);
  expect(delivered.filter((request) => !request.secure && request.path === '/mixed.js')).toEqual(
    [],
  );
  await chrome.getByRole('button', { name: 'Site controls', exact: true }).click();
  await expect(chrome.getByRole('region', { name: 'Connection details' })).toContainText(
    'Some content on this page was blocked',
  );
  await expect(
    chrome.getByRole('heading', { name: 'Connection: Not fully secure', exact: true }),
  ).toBeVisible();
  const screenshot = test.info().outputPath('mixed-content-details.png');
  await chrome.getByRole('region', { name: 'Connection details' }).screenshot({ path: screenshot });
  await test
    .info()
    .attach('Blocked mixed-content details', { path: screenshot, contentType: 'image/png' });
});

test('automatically upgraded image loads through HTTPS and is reported as upgraded', async () => {
  const page = await navigate(`https://secure.test:${tlsPort}/upgraded-image`);
  await expect
    .poll(() =>
      page
        .getByRole('img', { name: 'Mixed fixture image' })
        .evaluate((image: HTMLImageElement) => image.complete && image.naturalWidth === 1),
    )
    .toBe(true);
  expect(delivered.filter((request) => request.path === '/pixel')).toEqual([
    { secure: true, host: `insecure.test:${tlsPort}`, path: '/pixel' },
  ]);
  await chrome.getByRole('button', { name: 'Site controls', exact: true }).click();
  const details = chrome.getByRole('region', { name: 'Connection details' });
  await expect(details).toContainText('Content was upgraded to HTTPS');
  await expect(
    details.getByRole('heading', { name: 'Connection: Not fully secure', exact: true }),
  ).toBeVisible();
  await expect(details).not.toContainText('Insecure content was reported');
  const connection = await chrome.evaluate(
    async () => (await window.browser.getBrowserData()).connection,
  );
  await test.info().attach('Successful image upgrade metadata', {
    body: Buffer.from(JSON.stringify({ connection, delivered }, null, 2)),
    contentType: 'application/json',
  });
});

test('failed image upgrade never falls back to delivering HTTP content', async () => {
  const page = await navigate(`https://secure.test:${tlsPort}/failed-image-upgrade`);
  await expect
    .poll(() =>
      page
        .getByRole('img', { name: 'Mixed fixture image' })
        .evaluate((image: HTMLImageElement) => image.complete && image.naturalWidth === 0),
    )
    .toBe(true);
  expect(delivered.filter((request) => request.path === '/pixel')).toEqual([]);
  await chrome.getByRole('button', { name: 'Site controls', exact: true }).click();
  const details = chrome.getByRole('region', { name: 'Connection details' });
  await expect(details).toContainText('Content was upgraded to HTTPS');
  await expect(
    details.getByRole('heading', { name: 'Connection: Not fully secure', exact: true }),
  ).toBeVisible();
  await expect(details).not.toContainText('Insecure content was reported');
  const connection = await chrome.evaluate(
    async () => (await window.browser.getBrowserData()).connection,
  );
  await test.info().attach('Failed image upgrade metadata', {
    body: Buffer.from(JSON.stringify({ connection, delivered }, null, 2)),
    contentType: 'application/json',
  });
});

test('HTTP iframe in HTTPS is blocked without delivery and reported by trusted chrome', async () => {
  const page = await navigate(`https://secure.test:${tlsPort}/http-iframe`);
  await expect(page.locator('iframe[title="Blocked insecure iframe"]')).toHaveCount(1);
  await chrome.getByRole('button', { name: 'Site controls', exact: true }).click();
  const details = chrome.getByRole('region', { name: 'Connection details' });
  await expect(details).toContainText('Some content on this page was blocked');
  await expect(
    details.getByRole('heading', { name: 'Connection: Not fully secure', exact: true }),
  ).toBeVisible();
  expect(delivered.filter((request) => request.path === '/frame-content')).toEqual([]);
  await expect(details).not.toContainText('Insecure content was reported');
  const connection = await chrome.evaluate(
    async () => (await window.browser.getBrowserData()).connection,
  );
  await test.info().attach('Blocked iframe metadata', {
    body: Buffer.from(JSON.stringify({ connection, delivered }, null, 2)),
    contentType: 'application/json',
  });
});

for (const scenario of [
  { method: 'GET', target: '_self', embedded: false },
  { method: 'POST', target: '_self', embedded: false },
  { method: 'GET', target: '_blank', embedded: false },
  { method: 'POST', target: '_blank', embedded: false },
  { method: 'POST', target: '_top', embedded: true },
]) {
  test(`${scenario.embedded ? 'OOPIF ' : ''}${scenario.method} ${scenario.target} denies by default and continues the exact original request once`, async () => {
    const sentinel = `${scenario.method}-${scenario.target}-${scenario.embedded}-A & B + ž`;
    const expectedData = new URLSearchParams([
      ['sentinel', sentinel],
      ['repeated', 'one'],
      ['repeated', 'two'],
    ]).toString();
    for (const answer of [0, 1]) {
      // A canceled network request can still finish its error-page lifecycle.
      // Exercise the independent allow decision in a new user-created tab so
      // that lifecycle cannot overwrite the next form fixture's navigation.
      if (answer === 1) {
        await chrome.getByRole('button', { name: 'New tab', exact: true }).click();
        await expect(chrome.getByRole('combobox', { name: 'Address and search' })).toHaveValue('');
      }
      const query = new URLSearchParams({
        method: scenario.method,
        target: scenario.target,
        sentinel,
        phase: answer ? 'allow' : 'deny',
      });
      const sourceURL = `https://secure.test:${tlsPort}/${scenario.embedded ? 'embedded' : 'form'}?${query}`;
      const source = await navigate(sourceURL);
      if (scenario.target === '_self' || scenario.embedded)
        await source.evaluate(() => {
          document.documentElement.dataset.sourceDocument = 'preserved';
        });
      if (scenario.embedded)
        await source
          .frameLocator('iframe[title="Embedded form"]')
          .locator('html')
          .evaluate((root) => {
            root.dataset.sourceDocument = 'preserved';
          });
      if (scenario.embedded) {
        await expect
          .poll(() =>
            chrome.evaluate(
              async ({ expectedURL, fingerprint }) => {
                const connection = (await window.browser.getBrowserData()).connection;
                return (
                  connection?.url === expectedURL &&
                  !!connection.tls &&
                  connection.certificates[0]?.fingerprint256 === fingerprint &&
                  ((connection.state === 'secure' && connection.issues.length === 0) ||
                    (connection.state === 'not-fully-secure' &&
                      connection.issues.length > 0 &&
                      connection.issues.every(
                        (issue) => issue.resourceType === 'Form' && issue.resolution === 'warning',
                      )))
                );
              },
              { expectedURL: sourceURL, fingerprint: leaf.fingerprint256 },
            ),
          )
          .toBe(true);
        const connection = await chrome.evaluate(
          async () => (await window.browser.getBrowserData()).connection,
        );
        await test.info().attach(`OOPIF ${answer ? 'allow' : 'deny'} current TLS and observation`, {
          body: Buffer.from(JSON.stringify(connection, null, 2)),
          contentType: 'application/json',
        });
      }
      await setAnswer(answer);
      // Electron stop() can terminate an OOPIF-initiated top navigation before
      // CDP exposes its terminal network event. Check the native completed
      // denial and both preserved documents directly for that case.
      const nativeStop = scenario.embedded && answer === 0;
      const requestEnded = nativeStop
        ? undefined
        : app.context().waitForEvent(answer ? 'requestfinished' : 'requestfailed', {
            predicate: (request) =>
              request.url().startsWith(`http://insecure.test:${httpPort}/receive`),
            timeout: 10_000,
          });
      const button = scenario.embedded
        ? source
            .frameLocator('iframe[title="Embedded form"]')
            .getByRole('button', { name: 'Submit form' })
        : source.getByRole('button', { name: 'Submit form' });
      await expect
        .poll(() =>
          app.evaluate(({ BrowserWindow, webContents }, url) => {
            const contents = webContents
              .getAllWebContents()
              .find((candidate) => candidate.getURL() === url);
            if (!contents) return false;
            BrowserWindow.fromWebContents(contents)?.focus();
            contents.focus();
            return contents.isFocused();
          }, sourceURL),
        )
        .toBe(true);
      await button.click({ noWaitAfter: true });
      if (nativeStop) {
        await expect
          .poll(() =>
            app.evaluate(({ webContents }, url) => {
              const dialogs = (globalThis as typeof globalThis & { s64Dialogs: unknown[] })
                .s64Dialogs;
              const sourceContents = webContents
                .getAllWebContents()
                .find((contents) => contents.getURL() === url);
              return (
                dialogs.length === 1 && sourceContents !== undefined && !sourceContents.isLoading()
              );
            }, sourceURL),
          )
          .toBe(true);
      }
      await requestEnded;
      if (answer === 0 && (scenario.target === '_self' || scenario.embedded)) {
        await expect(source).toHaveURL(sourceURL);
        expect(await source.evaluate(() => document.documentElement.dataset.sourceDocument)).toBe(
          'preserved',
        );
        if (scenario.embedded) {
          const frame = source.frameLocator('iframe[title="Embedded form"]');
          await expect(frame.locator('html')).toHaveAttribute('data-source-document', 'preserved');
          await expect(frame.locator('input[name="sentinel"]')).toHaveValue(sentinel);
        } else {
          await expect(source.locator('input[name="sentinel"]')).toHaveValue(sentinel);
        }
      }
      const prompts = await app.evaluate(
        () =>
          (
            globalThis as typeof globalThis & {
              s64Dialogs: { buttons: string[]; defaultId?: number; cancelId?: number }[];
            }
          ).s64Dialogs,
      );
      expect(prompts).toHaveLength(1);
      expect(prompts[0].defaultId).toBe(0);
      expect(prompts[0].cancelId).toBe(0);
      expect(received).toHaveLength(answer);
    }
    expect(received[0]).toEqual({
      method: scenario.method,
      url: scenario.method === 'GET' ? `/receive?${expectedData}` : '/receive',
      body: scenario.method === 'POST' ? expectedData : '',
    });
  });
}
