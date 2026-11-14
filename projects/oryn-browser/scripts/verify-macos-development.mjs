/* global console, process */
import { spawnSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { getCurrentFuseWire } from './macos-packaging-dependencies.mjs';
import {
  assertDocumentAssociations,
  assertFusePolicy,
  assertPrivacyDescriptions,
  isExecutableMachO,
  relativeBundlePath,
} from './macos-security-policy.mjs';
import {
  developmentAppId,
  developmentEntitlementKeys,
  developmentEnginePath,
  assertDevelopmentEngineCoverage,
  isDevelopmentEnginePath,
  readDevelopmentEngineInventory,
  readDevelopmentIdentity,
  scanDevelopmentBundle,
} from './sign-macos-development.mjs';
import { engineInventoryResource, verifyEnginePreservation } from './macos-development-engine.mjs';

function capture(command, args, options = {}) {
  const result = spawnSync(command, args, {
    encoding: 'utf8',
    maxBuffer: 4 * 1024 * 1024,
    ...options,
  });
  if (result.error || result.status !== 0)
    throw new Error(
      `${command} development verification failed: ${(result.stderr || result.error?.message || '').trim()}`,
    );
  return `${result.stdout}\n${result.stderr}`;
}

function entitlements(path) {
  const output = capture('/usr/bin/codesign', ['-d', '--entitlements', '-', '--xml', path]);
  const start = output.indexOf('<?xml');
  const end = output.indexOf('</plist>', start);
  if (start < 0 && end < 0) return {};
  if (start < 0 || end < 0) throw new Error(`Malformed development entitlements: ${path}.`);
  return JSON.parse(
    capture('/usr/bin/plutil', ['-convert', 'json', '-o', '-', '-'], {
      input: output.slice(start, end + 8),
    }),
  );
}

export function assertDevelopmentSignature(details, grants, { executable = true } = {}) {
  const flags = /flags=(0x[0-9a-f]+)/iu.exec(details);
  if (
    !flags ||
    Number.parseInt(flags[1], 16) !== 2 ||
    !/^Signature=adhoc$/mu.test(details) ||
    !/^TeamIdentifier=not set$/mu.test(details)
  )
    throw new Error(
      'Development requires ad-hoc code with no runtime/library flags or publisher identity.',
    );
  const keys = Object.keys(grants).sort();
  const expected = !executable && keys.length === 0 ? [] : [...developmentEntitlementKeys].sort();
  if (JSON.stringify(keys) !== JSON.stringify(expected) || keys.some((key) => grants[key] !== true))
    throw new Error(
      'Development permits only the exact JIT inherit entitlements; no device or library-validation exception grants.',
    );
}

export async function verifyMacOSDevelopment(app) {
  if (process.platform !== 'darwin') throw new Error('Development verification requires macOS.');
  const { root, nativeFiles } = scanDevelopmentBundle(resolve(app));
  const { info } = readDevelopmentIdentity(root);
  assertPrivacyDescriptions(info);
  assertDocumentAssociations(info);
  assertFusePolicy(await getCurrentFuseWire(root), 'release');
  if (!info.ElectronAsarIntegrity?.['Resources/app.asar']?.hash)
    throw new Error('Embedded ASAR integrity is missing.');
  capture('/usr/bin/codesign', ['--verify', '--deep', '--strict', '--architecture', 'arm64', root]);
  const inventory = readDevelopmentEngineInventory(
    join(root, 'Contents/Resources', engineInventoryResource),
  );
  const enginePreservation = await verifyEnginePreservation(
    join(root, developmentEnginePath),
    inventory,
  );
  assertDevelopmentEngineCoverage(root, nativeFiles, enginePreservation);
  assertDevelopmentSignature(
    capture('/usr/bin/codesign', ['-d', '--verbose=4', root]),
    entitlements(root),
  );
  for (const path of nativeFiles) {
    if (isDevelopmentEnginePath(relativeBundlePath(root, path))) continue;
    try {
      assertDevelopmentSignature(
        capture('/usr/bin/codesign', ['-d', '--verbose=4', path]),
        entitlements(path),
        { executable: isExecutableMachO(path) },
      );
    } catch (error) {
      throw new Error(`${relativeBundlePath(root, path)}: ${error.message}`);
    }
  }
  return {
    passed: true,
    policy: 'local-development',
    appId: developmentAppId,
    hardenedRuntime: false,
    signing: 'ad-hoc',
    deviceEntitlements: [],
    nativeFiles: nativeFiles.length,
    enginePreservation,
    fuseCount: 9,
    fusePolicy: 'release',
    distributionQualified: false,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  if (process.argv.length !== 3) throw new Error('Pass the explicit Oryn Browser Local.app path.');
  console.log(
    `MACOS_DEVELOPMENT_SECURITY_RESULT ${JSON.stringify(await verifyMacOSDevelopment(process.argv[2]))}`,
  );
}
