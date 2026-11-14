/* global console, process */
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';
import YAML from 'yaml';
import { browserRoot, localSigningEnvironment } from './macos-security-policy.mjs';
import { verifyMacOSSecurity } from './verify-macos-security.mjs';
import { createInspectVariant } from './create-macos-inspect-variant.mjs';

if (process.platform !== 'darwin') throw new Error('The hardened ad-hoc lane requires macOS.');
const arguments_ = process.argv.slice(2);
const inspectTest = arguments_.includes('--inspect-test');
const outputIndex = arguments_.indexOf('--output');
const inspectOutputIndex = arguments_.indexOf('--inspect-output');
const accepted = new Set(['--inspect-test', '--output', '--inspect-output']);
for (let index = 0; index < arguments_.length; index += 1) {
  if (
    (index === outputIndex + 1 && outputIndex >= 0) ||
    (index === inspectOutputIndex + 1 && inspectOutputIndex >= 0)
  )
    continue;
  if (!accepted.has(arguments_[index]))
    throw new Error(
      'Use --output <fresh directory>, optional --inspect-test or --inspect-output <fresh inspect-test directory>.',
    );
}
if (
  outputIndex >= 0 &&
  (!arguments_[outputIndex + 1] || arguments_[outputIndex + 1].startsWith('--'))
)
  throw new Error('--output requires a fresh output directory.');
if (
  inspectOutputIndex >= 0 &&
  (inspectTest ||
    !arguments_[inspectOutputIndex + 1] ||
    arguments_[inspectOutputIndex + 1].startsWith('--'))
)
  throw new Error(
    '--inspect-output requires a fresh labelled directory and a release-policy source.',
  );
const variant = inspectTest ? 'inspect-test' : 'release';
const appDirectory = join(browserRoot, 'apps/browser');
const output = resolve(
  outputIndex >= 0
    ? arguments_[outputIndex + 1]
    : join(appDirectory, `release-${inspectTest ? 'inspect-test' : 'hardened'}-${Date.now()}`),
);
if (inspectTest && !basename(output).includes('inspect-test'))
  throw new Error('The inspector test output directory must contain inspect-test in its name.');
if (existsSync(output))
  throw new Error('Use a fresh output directory; existing artifacts are never overwritten.');
mkdirSync(output, { recursive: true });
const temporary = mkdtempSync(join(tmpdir(), 'oryn-macos-local-'));
try {
  const configPath = join(temporary, 'electron-builder.yml');
  writeFileSync(
    configPath,
    YAML.stringify({
      extends: join(appDirectory, 'electron-builder.yml'),
      directories: { output },
      mac: { identity: '-', hardenedRuntime: true, forceCodeSigning: true, notarize: false },
      publish: null,
    }),
    { mode: 0o600 },
  );
  const result = spawnSync(
    'pnpm',
    [
      'exec',
      'electron-builder',
      '--config',
      configPath,
      '--mac',
      'dir',
      '--arm64',
      '--publish',
      'never',
    ],
    {
      cwd: appDirectory,
      stdio: 'inherit',
      env: localSigningEnvironment(process.env, variant),
    },
  );
  if (result.error || result.status !== 0) throw new Error('Hardened ad-hoc packaging failed.');
  const app = join(output, 'mac-arm64/Oryn Browser.app');
  const security = await verifyMacOSSecurity(app, { variant });
  const evidence = {
    schemaVersion: 1,
    app,
    variant,
    signing: 'ad-hoc',
    notarized: false,
    distributionQualified: false,
    executionQualified: false,
    nativeExecutionGate:
      'An ad-hoc signature has no Apple Team ID; hardened library validation can reject Electron before startup. A real signing identity is required for execution qualification without weakening release policy.',
    security,
    manualGates: [
      'hardened real device consent and helper necessity',
      'macOS denial and restriction UI',
      'cookie migration and Gmail session continuity',
      'exact-release smoke and native accessibility',
    ],
  };
  writeFileSync(
    join(output, 'macos-security-evidence.json'),
    `${JSON.stringify(evidence, null, 2)}\n`,
  );
  console.log(`MACOS_LOCAL_RESULT ${JSON.stringify(evidence)}`);
  if (inspectOutputIndex >= 0) {
    const inspectEvidence = await createInspectVariant(app, arguments_[inspectOutputIndex + 1]);
    console.log(`MACOS_INSPECT_VARIANT_RESULT ${JSON.stringify(inspectEvidence)}`);
  }
} finally {
  rmSync(temporary, { recursive: true, force: true });
}
