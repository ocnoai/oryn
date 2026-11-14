/* global console, process */
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { URL } from 'node:url';
import { prepareSourceElectron } from './prepare-source-electron.mjs';

const appRequire = createRequire(new URL('../apps/browser/package.json', import.meta.url));
const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;
let temporary;
try {
  if (process.platform === 'darwin' && !env.ELECTRON_EXEC_PATH) {
    temporary = mkdtempSync(join(tmpdir(), 'oryn-source-dev-electron-'));
    const evidence = await prepareSourceElectron(temporary);
    env.ELECTRON_EXEC_PATH = evidence.executable;
    console.log(JSON.stringify({ sourceElectron: evidence }));
  }
  const child = spawn(
    process.execPath,
    [
      join(dirname(appRequire.resolve('electron-vite/package.json')), 'bin/electron-vite.js'),
      'dev',
      ...process.argv.slice(2),
    ],
    { cwd: new URL('../apps/browser/', import.meta.url), env, stdio: 'inherit' },
  );
  const interrupt = () => child.kill('SIGINT');
  const terminate = () => child.kill('SIGTERM');
  process.on('SIGINT', interrupt);
  process.on('SIGTERM', terminate);
  try {
    const outcome = await new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('close', (status, signal) => resolve({ status, signal }));
    });
    process.exitCode = outcome.status ?? (outcome.signal === 'SIGINT' ? 130 : 1);
  } finally {
    process.off('SIGINT', interrupt);
    process.off('SIGTERM', terminate);
  }
} finally {
  if (temporary) rmSync(temporary, { recursive: true, force: true });
}
