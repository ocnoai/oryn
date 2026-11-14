/* global console, process */
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { URL } from 'node:url';
import { prepareSourceElectron } from './prepare-source-electron.mjs';

const appRequire = createRequire(new URL('../apps/browser/package.json', import.meta.url));
const env = { ...process.env };
let temporary;
try {
  if (process.platform === 'darwin' && !env.ORYN_PACKAGED_APP && !env.ORYN_SOURCE_ELECTRON) {
    temporary = mkdtempSync(join(tmpdir(), 'oryn-source-ui-electron-'));
    const evidence = await prepareSourceElectron(temporary);
    env.ORYN_SOURCE_ELECTRON = evidence.executable;
    console.log(JSON.stringify({ sourceElectron: evidence }));
  }
  const result = spawnSync(
    process.execPath,
    [appRequire.resolve('@playwright/test/cli'), 'test', ...process.argv.slice(2)],
    { cwd: new URL('../apps/browser/', import.meta.url), env, stdio: 'inherit' },
  );
  if (result.error) throw result.error;
  process.exitCode = result.status ?? 1;
} finally {
  if (temporary) rmSync(temporary, { recursive: true, force: true });
}
