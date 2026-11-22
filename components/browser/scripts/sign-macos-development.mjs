/* global process */
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { lstatSync, readFileSync, readdirSync, realpathSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { URL } from 'node:url';
import { signAsync, getCurrentFuseWire } from './macos-packaging-dependencies.mjs';
import {
  assertFusePolicy,
  browserRoot,
  isMachO,
  relativeBundlePath,
} from './macos-security-policy.mjs';
import { engineInventoryResource, verifyEnginePreservation } from './macos-development-engine.mjs';

export const developmentAppId = 'ai.ocno.oryn.browser.localdevelopment';
export const developmentProductName = 'Oryn Browser Local';
export const developmentMainExecutable = `Contents/MacOS/${developmentProductName}`;
export const developmentEnginePath = 'Contents/Resources/engine';
export const developmentEntitlementKeys = Object.freeze([
  'com.apple.security.cs.allow-jit',
  'com.apple.security.cs.allow-unsigned-executable-memory',
]);
const appRequire = createRequire(new URL('../apps/browser/package.json', import.meta.url));
const builderRequire = createRequire(appRequire.resolve('electron-builder'));
const libraryRequire = createRequire(builderRequire.resolve('app-builder-lib'));
const asar = libraryRequire('@electron/asar');

export function readDevelopmentEngineInventory(path) {
  const stat = lstatSync(path);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 1 || stat.size > 2 * 1024 * 1024)
    throw new Error('Development Engine preservation inventory must be a bounded regular file.');
  return JSON.parse(readFileSync(path, 'utf8'));
}

export function isDevelopmentEnginePath(path) {
  return path === developmentEnginePath || path.startsWith(`${developmentEnginePath}/`);
}

export function assertDevelopmentEngineCoverage(root, nativeFiles, proof) {
  const count = nativeFiles.filter((path) =>
    isDevelopmentEnginePath(relativeBundlePath(root, path)),
  ).length;
  if (count !== proof.nativeFiles)
    throw new Error('An Engine Mach-O is outside the preserved native inventory.');
}

export function readDevelopmentPlist(path) {
  const result = spawnSync('/usr/bin/plutil', ['-convert', 'json', '-o', '-', path], {
    encoding: 'utf8',
    maxBuffer: 1024 * 1024,
  });
  if (result.error || result.status !== 0)
    throw new Error(`Cannot read development bundle plist: ${path}.`);
  return JSON.parse(result.stdout);
}

export function assertDevelopmentIdentity(info, metadata) {
  if (
    info.CFBundleIdentifier !== developmentAppId ||
    info.CFBundleExecutable !== developmentProductName ||
    info.CFBundleName !== developmentProductName ||
    metadata.orynBuildFlavor !== 'local-development' ||
    metadata.orynUpdate?.enabled !== false
  ) {
    throw new Error(
      'Local development signing requires the exact separate app identity, flavor and disabled updater.',
    );
  }
}

export function readDevelopmentIdentity(app) {
  const info = readDevelopmentPlist(join(app, 'Contents/Info.plist'));
  const archive = join(app, 'Contents/Resources/app.asar');
  const entry = asar.statFile(archive, 'package.json');
  if (!Number.isSafeInteger(entry.size) || entry.size < 1 || entry.size > 64 * 1024)
    throw new Error('Invalid development package metadata size.');
  const metadata = JSON.parse(asar.extractFile(archive, 'package.json').toString('utf8'));
  assertDevelopmentIdentity(info, metadata);
  return { info, metadata };
}

export function scanDevelopmentBundle(app) {
  const root = realpathSync(app);
  const nativeFiles = [];
  const pending = [root];
  while (pending.length) {
    const path = pending.pop();
    const entry = lstatSync(path);
    if (entry.isSymbolicLink()) relativeBundlePath(root, realpathSync(path));
    else if (entry.isDirectory())
      for (const name of readdirSync(path)) pending.push(join(path, name));
    else if (entry.isFile() && isMachO(path)) nativeFiles.push(path);
  }
  if (!nativeFiles.includes(join(root, developmentMainExecutable)))
    throw new Error('The separate local development executable is missing.');
  return { root, nativeFiles: nativeFiles.sort() };
}

export function developmentSigningOptions(options) {
  const { root } = scanDevelopmentBundle(options.app);
  return {
    ...options,
    app: root,
    identity: '-',
    identityValidation: false,
    hardenedRuntime: false,
    preAutoEntitlements: false,
    preEmbedProvisioningProfile: false,
    optionsForFile(path) {
      relativeBundlePath(root, realpathSync(path));
      return {
        entitlements: join(browserRoot, 'apps/browser/build/entitlements.mac.inherit.plist'),
        hardenedRuntime: false,
        // Explicit zero clears runtime/library flags on previously signed inputs.
        signatureFlags: ['0x0'],
        timestamp: 'none',
      };
    },
    ignore(path) {
      const canonical = realpathSync(path);
      relativeBundlePath(root, canonical);
      const entry = lstatSync(path);
      return (
        isDevelopmentEnginePath(relativeBundlePath(root, canonical)) ||
        entry.isSymbolicLink() ||
        canonical !== path ||
        (entry.isFile() && !isMachO(path))
      );
    },
  };
}

export default async function signDevelopment(options) {
  if (
    process.platform !== 'darwin' ||
    process.env.ORYN_MAC_SIGNING_MODE !== 'local-development' ||
    process.env.ORYN_MAC_SECURITY_VARIANT !== 'release'
  )
    throw new Error('Use only the explicit local development packaging lane.');
  const root = realpathSync(options.app);
  const bundle = scanDevelopmentBundle(root);
  readDevelopmentIdentity(root); // Guard the real packaged identity before any signing.
  assertFusePolicy(await getCurrentFuseWire(root), 'release');
  const inventoryPath = process.env.ORYN_LOCAL_DEVELOPMENT_ENGINE_INVENTORY;
  if (!inventoryPath)
    throw new Error(
      'Use the maintained development runner with a trusted pre-copy Engine inventory.',
    );
  const inventory = readDevelopmentEngineInventory(inventoryPath);
  const engineProof = await verifyEnginePreservation(join(root, developmentEnginePath), inventory);
  assertDevelopmentEngineCoverage(root, bundle.nativeFiles, engineProof);
  writeFileSync(
    join(root, 'Contents/Resources', engineInventoryResource),
    `${JSON.stringify(inventory)}\n`,
    { flag: 'wx', mode: 0o644 },
  );
  await signAsync(developmentSigningOptions({ ...options, app: root }));
  assertFusePolicy(await getCurrentFuseWire(root), 'release');
}
