/* global console, process */
import { spawnSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { getCurrentFuseWire } from './macos-packaging-dependencies.mjs';
import { assertNativeMinimumOS } from './macos-minimum-os.mjs';
import {
  assertDocumentAssociations,
  assertEntitlements,
  assertFusePolicy,
  assertPrivacyDescriptions,
  captureHelperExecutables,
  isExecutableMachO,
  relativeBundlePath,
  scanBundle,
  securityVariant,
} from './macos-security-policy.mjs';

function capture(command, arguments_, options = {}) {
  const result = spawnSync(command, arguments_, {
    encoding: 'utf8',
    maxBuffer: 4 * 1024 * 1024,
    ...options,
  });
  if (result.error || result.status !== 0)
    throw new Error(
      `${command} verification failed: ${(result.stderr || result.error?.message || '').trim()}`,
    );
  return result;
}

function plist(path) {
  return JSON.parse(capture('/usr/bin/plutil', ['-convert', 'json', '-o', '-', path]).stdout);
}

function entitlements(path) {
  const result = capture('/usr/bin/codesign', ['-d', '--entitlements', ':-', path]);
  const output = `${result.stdout}\n${result.stderr}`;
  const start = output.indexOf('<?xml');
  const end = output.indexOf('</plist>', start);
  if (start < 0 && end < 0) return {};
  if (start < 0 || end < 0) throw new Error(`Malformed signed entitlements: ${path}.`);
  const xml = output.slice(start, end + '</plist>'.length);
  return JSON.parse(
    capture('/usr/bin/plutil', ['-convert', 'json', '-o', '-', '-'], { input: xml }).stdout,
  );
}

export async function verifyMacOSSecurity(app, { variant = 'release', expectedTeamId } = {}) {
  if (process.platform !== 'darwin') throw new Error('macOS security verification requires macOS.');
  securityVariant(variant);
  const { root, nativeFiles } = scanBundle(resolve(app));
  const info = plist(join(root, 'Contents/Info.plist'));
  const minimumOS = assertNativeMinimumOS(info.LSMinimumSystemVersion, nativeFiles);
  if (info.CFBundleIdentifier !== 'ai.ocno.oryn.browser')
    throw new Error('Unexpected application identifier.');
  assertPrivacyDescriptions(info);
  assertDocumentAssociations(info);
  const wire = await getCurrentFuseWire(root);
  assertFusePolicy(wire, variant);
  if (!info.ElectronAsarIntegrity?.['Resources/app.asar']?.hash)
    throw new Error('Embedded ASAR integrity is missing.');
  capture('/usr/bin/codesign', ['--verify', '--deep', '--strict', root]);
  assertEntitlements('', entitlements(root));
  for (const path of nativeFiles) {
    const relativePath = relativeBundlePath(root, path);
    const result = capture('/usr/bin/codesign', ['-d', '--verbose=4', path]);
    const details = `${result.stdout}${result.stderr}`;
    if (!/flags=.*\bruntime\b/u.test(details))
      throw new Error(`Hardened runtime is missing: ${relativePath}.`);
    if (expectedTeamId && !details.includes(`TeamIdentifier=${expectedTeamId}\n`))
      throw new Error(`Unexpected signing team: ${relativePath}.`);
    assertEntitlements(relativePath, entitlements(path), { executable: isExecutableMachO(path) });
  }
  return {
    passed: true,
    variant,
    hardenedRuntime: true,
    nativeFiles: nativeFiles.length,
    fuseCount: 9,
    helperDeviceAllowlist: [...captureHelperExecutables],
    distributionQualified: false,
    minimumOS,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const arguments_ = process.argv.slice(2);
  const app = arguments_.find((argument) => !argument.startsWith('--'));
  if (
    !app ||
    arguments_.some((argument) => argument.startsWith('--') && argument !== '--inspect-test')
  )
    throw new Error('Pass an Oryn Browser.app path and optional --inspect-test.');
  console.log(
    `MACOS_SECURITY_RESULT ${JSON.stringify(await verifyMacOSSecurity(app, { variant: arguments_.includes('--inspect-test') ? 'inspect-test' : 'release' }))}`,
  );
}
