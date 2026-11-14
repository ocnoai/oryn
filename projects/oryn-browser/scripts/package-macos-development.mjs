/* global console, process */
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, rmSync, statfsSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import YAML from 'yaml';
import { browserRoot, localSigningEnvironment } from './macos-security-policy.mjs';
import { developmentAppId, developmentProductName } from './sign-macos-development.mjs';
import { verifyMacOSDevelopment } from './verify-macos-development.mjs';
import { createEngineInventory } from './macos-development-engine.mjs';

if (process.platform !== 'darwin') throw new Error('The local development lane requires macOS.');
const args = process.argv.slice(2);
if (
  args.length !== 0 &&
  (args.length !== 2 || args[0] !== '--output' || !args[1] || args[1].startsWith('--'))
)
  throw new Error('Use optional --output <fresh local-development directory>.');
const appDirectory = join(browserRoot, 'apps/browser');
const output = resolve(args[1] ?? join(appDirectory, `release-local-development-${Date.now()}`));
if (!basename(output).includes('local-development'))
  throw new Error('Development output must be explicitly labelled local-development.');
if (existsSync(output))
  throw new Error('Use a fresh output directory; never overwrite another artifact.');
const temporary = mkdtempSync(join(tmpdir(), 'oryn-development-package-'));
try {
  const inventoryPath = join(temporary, 'engine-preservation.json');
  const sourceInventory = await createEngineInventory(join(browserRoot, 'packaging/.stage/engine'));
  mkdirSync(dirname(output), { recursive: true });
  const filesystem = statfsSync(dirname(output), { bigint: true });
  const available = filesystem.bavail * filesystem.bsize;
  const required = BigInt(sourceInventory.logicalBytes) + 1536n * 1024n * 1024n;
  if (available < required)
    throw new Error(
      `Insufficient disk space: need ${(Number(required) / 1024 ** 3).toFixed(2)} GiB, available ${(Number(available) / 1024 ** 3).toFixed(2)} GiB. Free space or choose another output volume; no app was copied.`,
    );
  mkdirSync(output, { recursive: true });
  writeFileSync(inventoryPath, `${JSON.stringify(sourceInventory)}\n`, { mode: 0o600, flag: 'wx' });
  const config = join(temporary, 'electron-builder.yml');
  writeFileSync(
    config,
    YAML.stringify({
      extends: join(appDirectory, 'electron-builder.yml'),
      appId: developmentAppId,
      productName: developmentProductName,
      artifactName: 'Oryn-Browser-Local-Development-${version}-${arch}.${ext}',
      directories: { output },
      extraMetadata: { orynBuildFlavor: 'local-development', orynUpdate: { enabled: false } },
      mac: {
        identity: '-',
        hardenedRuntime: false,
        forceCodeSigning: true,
        notarize: false,
        sign: join(browserRoot, 'scripts/sign-macos-development.mjs'),
        entitlements: join(appDirectory, 'build/entitlements.mac.inherit.plist'),
        entitlementsInherit: join(appDirectory, 'build/entitlements.mac.inherit.plist'),
      },
      publish: null,
    }),
    { mode: 0o600 },
  );
  const environment = {
    ...localSigningEnvironment(process.env, 'release'),
    ORYN_MAC_SIGNING_MODE: 'local-development',
    ORYN_LOCAL_DEVELOPMENT_ENGINE_INVENTORY: inventoryPath,
  };
  const result = spawnSync(
    'pnpm',
    [
      'exec',
      'electron-builder',
      '--config',
      config,
      '--mac',
      'dir',
      '--arm64',
      '--publish',
      'never',
    ],
    { cwd: appDirectory, env: environment, stdio: 'inherit' },
  );
  if (result.error || result.status !== 0) throw new Error('Local development packaging failed.');
  const app = join(output, `mac-arm64/${developmentProductName}.app`);
  const security = await verifyMacOSDevelopment(app);
  const evidence = {
    schemaVersion: 1,
    app,
    buildFlavor: 'local-development',
    signing: 'ad-hoc',
    hardenedRuntime: false,
    notarized: false,
    distributionQualified: false,
    executionQualified: false,
    qualification:
      'Static verification passed. Launch, Engine/runtime and isolated-profile execution must be tested on this exact local app before calling it usable.',
    security,
  };
  writeFileSync(
    join(output, 'macos-development-evidence.json'),
    `${JSON.stringify(evidence, null, 2)}\n`,
  );
  console.log(`MACOS_DEVELOPMENT_RESULT ${JSON.stringify(evidence)}`);
} finally {
  rmSync(temporary, { recursive: true, force: true });
}
