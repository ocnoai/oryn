import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath, URL } from 'node:url';
import process from 'node:process';
import console from 'node:console';
import { spawnSync } from 'node:child_process';

if (process.platform !== 'darwin') {
  console.info('Native download safety is unavailable on this unqualified platform.');
  process.exit(0);
}
const appRoot = fileURLToPath(new URL('../apps/browser/', import.meta.url));
const require = createRequire(join(appRoot, 'package.json'));
const include = join(dirname(require.resolve('node-api-headers/package.json')), 'include');
const result = spawnSync(
  '/usr/bin/clang++',
  [
    '-bundle',
    '-mmacosx-version-min=14.0',
    '-undefined',
    'dynamic_lookup',
    '-I',
    include,
    '-framework',
    'Foundation',
    '-o',
    join(appRoot, 'build/helpers/download_safety.node'),
    join(appRoot, 'build/helpers/download_safety.mm'),
  ],
  { stdio: 'inherit' },
);
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
