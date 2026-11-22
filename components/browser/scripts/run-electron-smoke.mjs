/* global clearTimeout, console, process, setTimeout */
import { spawn } from 'node:child_process';
import { readFile, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { extname, join, resolve, sep } from 'node:path';
import { URL } from 'node:url';

const appDirectory = resolve(import.meta.dirname, '../apps/browser');
const require = createRequire(join(appDirectory, 'package.json'));
const electron = require('electron');
const environment = { ...process.env, LOCAL_AI_SMOKE_TEST: '1' };
if (process.argv.includes('--packaged')) {
  if (!environment.ORYN_PACKAGED_APP)
    throw new Error('Set ORYN_PACKAGED_APP to the exact app bundle being qualified.');
  // Preserve the caller's test mode and fixture site across the packaged smoke sequence.
  if (
    !environment.ORYN_WORKSPACE_ACCEPTANCE_CASES &&
    !Object.entries(environment).some(([key, value]) => key.endsWith('_SMOKE') && value === '1')
  ) {
    environment.ORYN_CHAT_SMOKE = '1';
    environment.ORYN_SMOKE_SITE_ROOT ??= resolve(appDirectory, '../../../examples/chat/dist');
  }
}
delete environment.ELECTRON_RUN_AS_NODE;
const packagedApp = environment.ORYN_PACKAGED_APP
  ? resolve(environment.ORYN_PACKAGED_APP)
  : undefined;
if (packagedApp) {
  delete environment.ORYN_ENGINE_DIR;
  delete environment.ORYN_ENGINE_PYTHON;
  delete environment.PYTHONHOME;
  delete environment.PYTHONPATH;
}
const siteServer = await startSiteServer(environment.ORYN_SMOKE_SITE_ROOT);
if (siteServer) environment.ORYN_START_URL = siteServer.url;
const secondSiteServer = await startSiteServer(environment.ORYN_SMOKE_SECOND_SITE_ROOT);
if (secondSiteServer) environment.ORYN_SECOND_SITE_URL = secondSiteServer.url;
const thirdSiteServer = await startSiteServer(environment.ORYN_SMOKE_THIRD_SITE_ROOT);
if (thirdSiteServer) environment.ORYN_THIRD_SITE_URL = thirdSiteServer.url;

const executable = packagedApp ? join(packagedApp, 'Contents', 'MacOS', 'Oryn Browser') : electron;
const child = spawn(executable, packagedApp ? [] : [appDirectory], {
  cwd: packagedApp ?? appDirectory,
  env: environment,
  stdio: ['ignore', 'pipe', 'pipe'],
});

let stdoutBuffer = '';
let result;
let terminating = false;

child.stdout.on('data', (chunk) => {
  const text = chunk.toString('utf8');
  process.stdout.write(text);
  stdoutBuffer += text;
  const lines = stdoutBuffer.split('\n');
  stdoutBuffer = lines.pop() ?? '';
  for (const line of lines) {
    const marker = 'LOCAL_AI_SMOKE_RESULT ';
    const markerIndex = line.indexOf(marker);
    if (markerIndex >= 0) {
      result = JSON.parse(line.slice(markerIndex + marker.length));
    }
  }
});
child.stderr.on('data', (chunk) => process.stderr.write(chunk));

const timeout = setTimeout(
  () => {
    process.stderr.write('Electron smoke test timed out.\n');
    terminate();
  },
  environment.ORYN_WORKSPACE_ACCEPTANCE_CASES
    ? 7_200_000
    : environment.ORYN_TRANSCRIPTION_SMOKE === '1' ||
        environment.ORYN_CHAT_SMOKE === '1' ||
        environment.ORYN_SONG_SMOKE === '1' ||
        environment.ORYN_ALPHA_INTEGRATION_SMOKE === '1' ||
        environment.ORYN_MODEL_WORKSPACE_SMOKE === '1' ||
        environment.ORYN_HF_GATED_ACCESS_SMOKE === '1'
      ? 300_000
      : 60_000,
);

child.once('error', (error) => {
  clearTimeout(timeout);
  process.stderr.write(`${error.stack ?? error.message}\n`);
  process.exitCode = 1;
});

child.once('exit', () => {
  clearTimeout(timeout);
  siteServer?.server.close();
  secondSiteServer?.server.close();
  thirdSiteServer?.server.close();
  process.exitCode = result?.passed === true ? 0 : 1;
});

function terminate() {
  if (terminating) {
    return;
  }
  terminating = true;
  child.kill('SIGTERM');
  const force = setTimeout(() => child.kill('SIGKILL'), 2_000);
  force.unref();
}

async function startSiteServer(rootValue) {
  if (!rootValue) return undefined;
  const root = resolve(rootValue);
  const server = createServer(async (request, response) => {
    try {
      const pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
      if (environment.ORYN_SMOKE_TRACE_FIXTURES === '1') {
        const address = server.address();
        console.info(
          'ORYN_SMOKE_FIXTURE_REQUEST',
          JSON.stringify({
            origin:
              address && typeof address !== 'string'
                ? `http://127.0.0.1:${address.port}`
                : undefined,
            document: pathname === '/' || pathname === '/index.html',
            method: request.method,
          }),
        );
      }
      if (pathname === '/__oryn_alpha_download') {
        response.setHeader('Content-Type', 'application/octet-stream');
        response.setHeader('Content-Disposition', 'attachment; filename="oryn-alpha.txt"');
        response.end('Oryn developer alpha download test.');
        return;
      }
      const requested = pathname === '/' ? 'index.html' : decodeURIComponent(pathname.slice(1));
      let target = resolve(root, requested);
      if (target !== root && !target.startsWith(`${root}${sep}`)) throw new Error('invalid path');
      if ((await stat(target)).isDirectory()) target = join(target, 'index.html');
      response.setHeader('Content-Type', contentType(extname(target)));
      response.end(await readFile(target));
    } catch {
      response.statusCode = 404;
      response.end('Not found');
    }
  });
  await new Promise((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Smoke site failed to bind.');
  return { server, url: `http://127.0.0.1:${address.port}` };
}

function contentType(extension) {
  return (
    {
      '.html': 'text/html; charset=utf-8',
      '.js': 'text/javascript; charset=utf-8',
      '.css': 'text/css; charset=utf-8',
      '.wav': 'audio/wav',
    }[extension] ?? 'application/octet-stream'
  );
}
