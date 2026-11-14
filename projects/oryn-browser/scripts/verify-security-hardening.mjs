#!/usr/bin/env node
/* global console, process */

import {
  existsSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { createServer } from 'node:net';

const workspaceRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const securityRoot = join(workspaceRoot, 'apps', 'browser', 'build', 'security');
const engineProfile = join(securityRoot, 'engine.sb');
const packProfile = join(securityRoot, 'runtime-pack.sb');

const engine = readFileSync(engineProfile, 'utf8');
const pack = readFileSync(packProfile, 'utf8');
requireText(engine, '(deny default)', 'engine profile must default-deny');
requireText(engine, '(allow network-outbound)', 'engine control plane needs outbound networking');
requireText(engine, '(param "SECURITY_ROOT")', 'engine profile must authorize profile reads');
requireText(
  engine,
  '(param "PYTHON_ROOT")',
  'engine profile must authorize its exact Python closure',
);
requireText(pack, '(deny default)', 'runtime-pack profile must default-deny');
if (pack.includes('(allow network')) fail('runtime-pack profile must not allow networking');
for (const parameter of [
  'ENGINE_ROOT',
  'PACK_RUNTIME_ROOT',
  'FOUNDATION_ROOT',
  'MODEL_ROOT',
  'ARTIFACT_INPUT_ROOT',
  'ARTIFACT_OUTPUT_ROOT',
  'SCRATCH_ROOT',
  'METAL_COMPILER_CACHE',
]) {
  requireText(pack, `(param "${parameter}")`, `runtime-pack profile is missing ${parameter}`);
}

const builder = readFileSync(
  join(workspaceRoot, 'apps', 'browser', 'electron-builder.yml'),
  'utf8',
);
requireText(builder, 'from: build/security', 'packaging must include security profiles');
requireText(builder, 'from: build/helpers', 'packaging must include the Browser artifact helper');
if (!existsSync(join(workspaceRoot, 'apps', 'browser', 'build', 'helpers', 'artifact_io.py')))
  fail('Browser artifact helper is missing');
if (
  !existsSync(join(workspaceRoot, 'apps', 'browser', 'build', 'helpers', 'artifact_io_windows.py'))
)
  fail('Windows Browser artifact helper is missing');
requireText(
  builder,
  '- artifact_io_windows.py',
  'packaging must include the Windows artifact helper',
);
const lock = readFileSync(
  join(workspaceRoot, 'packaging', 'mlx-foundation-macos-arm64-py312.lock.txt'),
  'utf8',
);
requireText(lock, 'mlx==0.32.4', 'MLX must be pinned to the patched release');
requireText(lock, 'mlx-metal==0.32.4', 'MLX Metal must match the patched release');
const runtimeSpec = readFileSync(join(workspaceRoot, 'packaging', 'runtime-spec.json'), 'utf8');
requireText(runtimeSpec, '"mlx": "0.32.4"', 'runtime specification must require patched MLX');
if (process.env.ORYN_VERIFY_STAGED_RUNTIME === '1') {
  const stagedMLX = join(
    workspaceRoot,
    'packaging',
    '.stage',
    'engine',
    'runtime-foundations',
    'oryn.runtime.mlx',
    'site-packages',
  );
  if (
    !existsSync(join(stagedMLX, 'mlx-0.32.4.dist-info')) ||
    !existsSync(join(stagedMLX, 'mlx_metal-0.32.4.dist-info'))
  ) {
    fail('staged packaged runtime is missing the patched MLX 0.32.4 closure; reassemble it');
  }
}

if (process.platform === 'darwin') await verifyMacOSSandbox();
console.log('Security hardening verification passed.');

async function verifyMacOSSandbox() {
  const allowedRoot = realpathSync(mkdtempSync(join(tmpdir(), 'oryn-security-allowed-')));
  const deniedRoot = realpathSync(mkdtempSync(join(tmpdir(), 'oryn-security-denied-')));
  const deniedFile = join(deniedRoot, 'sentinel.txt');
  writeFileSync(deniedFile, 'must remain inaccessible', { mode: 0o600 });
  const empty = '/private/var/empty';
  const cacheRoot = spawnSync('/usr/bin/getconf', ['DARWIN_USER_CACHE_DIR'], { encoding: 'utf8' });
  if (cacheRoot.status !== 0) fail('Could not resolve the system compiler cache');
  const metalCache = join(realpathSync(cacheRoot.stdout.trim()), 'com.apple.metalfe');
  const sandboxArguments = [
    '-f',
    packProfile,
    '-D',
    `ENGINE_ROOT=${workspaceRoot}`,
    '-D',
    `PACK_RUNTIME_ROOT=${empty}`,
    '-D',
    `FOUNDATION_ROOT=${empty}`,
    '-D',
    `MODEL_ROOT=${empty}`,
    '-D',
    `ARTIFACT_INPUT_ROOT=${empty}`,
    '-D',
    `ARTIFACT_OUTPUT_ROOT=${allowedRoot}`,
    '-D',
    `SCRATCH_ROOT=${allowedRoot}`,
    '-D',
    `METAL_COMPILER_CACHE=${metalCache}`,
  ];
  const writeProbe = spawnSync(
    '/usr/bin/sandbox-exec',
    [...sandboxArguments, '/bin/sh', '-c', 'echo ok > "$1/write-ok"', 'sh', allowedRoot],
    { encoding: 'utf8', timeout: 15_000 },
  );
  const readProbe = spawnSync(
    '/usr/bin/sandbox-exec',
    [...sandboxArguments, '/bin/cat', deniedFile],
    { encoding: 'utf8', timeout: 15_000 },
  );
  const server = createServer((socket) => socket.destroy());
  await new Promise((resolveListen, rejectListen) => {
    server.once('error', rejectListen);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  const port = String(server.address().port);
  const networkProbe = spawnSync(
    '/usr/bin/sandbox-exec',
    [...sandboxArguments, '/usr/bin/nc', '-z', '-w', '1', '127.0.0.1', port],
    { encoding: 'utf8', timeout: 5000 },
  );
  server.close();
  let gpuProbe;
  if (process.env.ORYN_VERIFY_STAGED_RUNTIME === '1') {
    const runtime = join(workspaceRoot, 'packaging', '.stage', 'engine');
    const foundation = join(runtime, 'runtime-foundations', 'oryn.runtime.mlx', 'site-packages');
    gpuProbe = spawnSync(
      '/usr/bin/sandbox-exec',
      [
        ...sandboxArguments,
        join(runtime, 'runtime', 'bin', 'python3.12'),
        '-I',
        '-B',
        '-c',
        `import site; site.addsitedir(${JSON.stringify(foundation)}); import mlx.core as mx; assert mx.metal.is_available(); k=mx.fast.metal_kernel(name='oryn_sandbox_probe',input_names=['inp'],output_names=['out'],source='out[thread_position_in_grid.x] = inp[thread_position_in_grid.x] * 2;'); x=k(inputs=[mx.ones((32,))],output_shapes=[(32,)],output_dtypes=[mx.float32],grid=(32,1,1),threadgroup=(32,1,1)); mx.eval(x); assert x[0].sum().item()==64`,
      ],
      {
        encoding: 'utf8',
        timeout: 30_000,
        env: { ...process.env, HOME: allowedRoot, TMPDIR: allowedRoot },
      },
    );
  }
  rmSync(allowedRoot, { recursive: true, force: true });
  rmSync(deniedRoot, { recursive: true, force: true });
  if (writeProbe.status !== 0) {
    fail(
      `macOS runtime-pack sandbox write probe failed: ${(writeProbe.stderr || writeProbe.stdout).trim()}`,
    );
  }
  if (readProbe.status === 0) {
    fail('macOS runtime-pack sandbox allowed a denied host-file read');
  }
  if (networkProbe.status === 0 || networkProbe.error) {
    fail('macOS runtime-pack sandbox failed the local network denial probe');
  }
  if (gpuProbe && gpuProbe.status !== 0) {
    fail(`macOS runtime-pack sandbox Metal compute probe failed: ${gpuProbe.stderr}`);
  }
}

function requireText(value, expected, message) {
  if (!value.includes(expected)) fail(message);
}

function fail(message) {
  console.error(`Security hardening verification failed: ${message}`);
  process.exit(1);
}
