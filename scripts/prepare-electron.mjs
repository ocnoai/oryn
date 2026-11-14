#!/usr/bin/env node
// Resolve/install the pinned runtime once, before parallel tests or source launch.
import { accessSync, constants, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const appRequire = createRequire(join(root, 'projects/oryn-browser/apps/browser/package.json'));
let executable;
if (process.env.ORYN_OFFLINE === '1') {
  const packageRoot = dirname(appRequire.resolve('electron/package.json'));
  try {
    const relative = readFileSync(join(packageRoot, 'path.txt'), 'utf8').trim();
    if (!relative || isAbsolute(relative) || relative.split(/[\\/]/).includes('..'))
      throw new Error('Invalid Electron runtime path.');
    executable = join(packageRoot, 'dist', relative);
    accessSync(executable, constants.X_OK);
  } catch {
    throw new Error('Offline Electron runtime is unavailable. Run normal bootstrap once to install the pinned binary.');
  }
} else {
  // The upstream package installs its checksummed runtime on first resolution.
  // Do this serially so test workers cannot race the initial extraction.
  executable = appRequire('electron');
  accessSync(executable, constants.X_OK);
}
console.log('Pinned Electron runtime is ready.');
