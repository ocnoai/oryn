import { Buffer } from 'node:buffer';
import { closeSync, lstatSync, openSync, readSync, readdirSync, realpathSync } from 'node:fs';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FuseState, FuseV1Options, FuseVersion } from './macos-packaging-dependencies.mjs';

export const browserRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const mainExecutable = 'Contents/MacOS/Oryn Browser';
export const deviceEntitlements = Object.freeze([
  'com.apple.security.device.camera',
  'com.apple.security.device.audio-input',
  'com.apple.security.personal-information.location',
]);
const inheritedEntitlements = Object.freeze([
  'com.apple.security.cs.allow-jit',
  'com.apple.security.cs.allow-unsigned-executable-memory',
]);

// No helper is approved merely because Electron uses it. Add an exact executable
// here only after hardened-runtime capture tests establish its need (SECURITY §61.1).
export const captureHelperExecutables = Object.freeze([]);

export const usageDescriptions = Object.freeze({
  NSCameraUsageDescription: 'Oryn lets websites use your camera only after you allow the site.',
  NSMicrophoneUsageDescription:
    'Oryn lets websites use your microphone only after you allow the site.',
  NSAudioCaptureUsageDescription:
    'Oryn lets websites capture audio only after you allow screen or audio sharing.',
  NSLocationUsageDescription: 'Oryn lets websites use your location only after you allow the site.',
  NSLocationWhenInUseUsageDescription:
    'Oryn lets websites use your location only after you allow the site.',
  NSBluetoothAlwaysUsageDescription:
    'Oryn asks for your permission before a website can use Bluetooth devices.',
  NSBluetoothPeripheralUsageDescription:
    'Oryn asks for your permission before a website can use Bluetooth devices.',
});

export function securityVariant(value = 'release') {
  if (value !== 'release' && value !== 'inspect-test')
    throw new Error('macOS security variant must be release or inspect-test.');
  return value;
}

export function localSigningEnvironment(environment, variant) {
  securityVariant(variant);
  const sanitized = Object.fromEntries(
    Object.entries(environment).filter(
      ([key]) => !/^(?:CSC_|WIN_CSC_|APPLE_|ORYN_APPLE_)/u.test(key),
    ),
  );
  return {
    ...sanitized,
    CSC_IDENTITY_AUTO_DISCOVERY: 'false',
    ORYN_MAC_SIGNING_MODE: 'ad-hoc',
    ORYN_MAC_SECURITY_VARIANT: variant,
  };
}

export function fusePolicy(variant = 'release') {
  securityVariant(variant);
  return {
    version: FuseVersion.V1,
    strictlyRequireAllFuses: true,
    resetAdHocDarwinSignature: false,
    [FuseV1Options.RunAsNode]: false,
    [FuseV1Options.EnableCookieEncryption]: true,
    [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false,
    [FuseV1Options.EnableNodeCliInspectArguments]: variant === 'inspect-test',
    [FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: true,
    [FuseV1Options.OnlyLoadAppFromAsar]: true,
    [FuseV1Options.LoadBrowserProcessSpecificV8Snapshot]: false,
    [FuseV1Options.GrantFileProtocolExtraPrivileges]: false,
    [FuseV1Options.WasmTrapHandlers]: true,
  };
}

export function assertFusePolicy(wire, variant = 'release') {
  const expected = fusePolicy(variant);
  const indices = Object.keys(wire).filter((key) => /^\d+$/u.test(key));
  if (wire.version !== FuseVersion.V1 || indices.length !== 9)
    throw new Error('Unexpected Electron fuse schema; review every fuse before packaging.');
  for (let index = 0; index < 9; index += 1) {
    const state = expected[index] ? FuseState.ENABLE : FuseState.DISABLE;
    if (wire[index] !== state)
      throw new Error(`Electron fuse ${FuseV1Options[index]} does not match ${variant} policy.`);
  }
}

export function relativeBundlePath(app, path) {
  const result = relative(resolve(app), resolve(path));
  if (isAbsolute(result) || result === '..' || result.startsWith(`..${sep}`))
    throw new Error('Signing path escapes the application bundle.');
  return result.split(sep).join('/');
}

export function expectedEntitlements(path) {
  if (path === '' || path === mainExecutable)
    return [...inheritedEntitlements, ...deviceEntitlements];
  if (captureHelperExecutables.includes(path))
    return [...inheritedEntitlements, ...deviceEntitlements.slice(0, 2)];
  return [...inheritedEntitlements];
}

export function entitlementFile(app, path) {
  const relativePath = relativeBundlePath(app, path);
  const name =
    relativePath === '' || relativePath === mainExecutable
      ? 'entitlements.mac.plist'
      : 'entitlements.mac.inherit.plist';
  return join(browserRoot, 'apps/browser/build', name);
}

export function assertEntitlements(path, entitlements, { executable = true } = {}) {
  if (!entitlements || typeof entitlements !== 'object' || Array.isArray(entitlements))
    throw new Error(`Missing or invalid entitlements: ${path || 'main app'}.`);
  const actual = Object.keys(entitlements).sort();
  // codesign omits entitlement blobs on dylibs/framework code. They cannot
  // receive process grants; still reject any unexpected entitlement if present.
  const expected =
    !executable && path !== mainExecutable && actual.length === 0
      ? []
      : expectedEntitlements(path).sort();
  if (
    JSON.stringify(expected) !== JSON.stringify(actual) ||
    actual.some((key) => entitlements[key] !== true)
  )
    throw new Error(`Unexpected entitlements: ${path || 'main app'}.`);
}

export function assertPrivacyDescriptions(info) {
  for (const [key, description] of Object.entries(usageDescriptions)) {
    if (info[key] !== description)
      throw new Error(`Missing or incorrect Oryn privacy string: ${key}.`);
  }
}

export function assertDocumentAssociations(info) {
  const declarations = info.CFBundleDocumentTypes;
  if (!Array.isArray(declarations)) throw new Error('HTML document associations are missing.');
  for (const extension of ['html', 'htm', 'xhtml', 'shtml']) {
    const matches = declarations.filter((entry) =>
      entry.CFBundleTypeExtensions?.includes(extension),
    );
    if (
      !matches.length ||
      matches.some(
        (entry) => entry.CFBundleTypeRole !== 'Viewer' || entry.LSHandlerRank !== 'Alternate',
      )
    )
      throw new Error(`The ${extension} association must be a user-selected Alternate Viewer.`);
  }
  for (const type of ['public.html', 'public.xhtml']) {
    const matches = declarations.filter((entry) => entry.LSItemContentTypes?.includes(type));
    if (
      !matches.length ||
      matches.some(
        (entry) => entry.CFBundleTypeRole !== 'Viewer' || entry.LSHandlerRank !== 'Alternate',
      )
    )
      throw new Error(`The ${type} association must be a user-selected Alternate Viewer.`);
  }
}

const machOMagic = new Set([
  0xfeedface, 0xfeedfacf, 0xcefaedfe, 0xcffaedfe, 0xcafebabe, 0xbebafeca, 0xcafebabf, 0xbfbafeca,
]);
export function isMachO(path) {
  const descriptor = openSync(path, 'r');
  try {
    const header = Buffer.alloc(4);
    return readSync(descriptor, header, 0, 4, 0) === 4 && machOMagic.has(header.readUInt32BE());
  } finally {
    closeSync(descriptor);
  }
}

export function isExecutableMachO(path) {
  const descriptor = openSync(path, 'r');
  try {
    const header = Buffer.alloc(32);
    if (readSync(descriptor, header, 0, header.length, 0) < 16)
      throw new Error('Truncated Mach-O header.');
    const magic = header.readUInt32BE();
    const little = magic === 0xcefaedfe || magic === 0xcffaedfe;
    if ([0xfeedface, 0xfeedfacf, 0xcefaedfe, 0xcffaedfe].includes(magic))
      return (little ? header.readUInt32LE(12) : header.readUInt32BE(12)) === 2;
    const fatLittle = magic === 0xbebafeca || magic === 0xbfbafeca;
    const fat64 = magic === 0xcafebabf || magic === 0xbfbafeca;
    if (![0xcafebabe, 0xbebafeca, 0xcafebabf, 0xbfbafeca].includes(magic))
      throw new Error('Unknown Mach-O header.');
    const count = fatLittle ? header.readUInt32LE(4) : header.readUInt32BE(4);
    if (count < 1 || count > 16) throw new Error('Unbounded Mach-O architecture table.');
    const entry = Buffer.alloc(fat64 ? 32 : 20);
    for (let index = 0; index < count; index += 1) {
      if (readSync(descriptor, entry, 0, entry.length, 8 + index * entry.length) !== entry.length)
        throw new Error('Truncated Mach-O architecture table.');
      const offset = fat64
        ? Number(fatLittle ? entry.readBigUInt64LE(8) : entry.readBigUInt64BE(8))
        : fatLittle
          ? entry.readUInt32LE(8)
          : entry.readUInt32BE(8);
      if (!Number.isSafeInteger(offset) || offset < 8 + count * entry.length)
        throw new Error('Invalid Mach-O slice offset.');
      if (readSync(descriptor, header, 0, 16, offset) !== 16)
        throw new Error('Truncated Mach-O slice.');
      const sliceMagic = header.readUInt32BE();
      const sliceLittle = sliceMagic === 0xcefaedfe || sliceMagic === 0xcffaedfe;
      if (![0xfeedface, 0xfeedfacf, 0xcefaedfe, 0xcffaedfe].includes(sliceMagic))
        throw new Error('Unknown Mach-O slice header.');
      if ((sliceLittle ? header.readUInt32LE(12) : header.readUInt32BE(12)) === 2) return true;
    }
    return false;
  } finally {
    closeSync(descriptor);
  }
}

// Never traverse symlinks. Framework/Python aliases may target only this bundle.
export function scanBundle(app) {
  const root = realpathSync(app);
  const nativeFiles = [];
  const bundles = [];
  const pending = [root];
  while (pending.length) {
    const path = pending.pop();
    const info = lstatSync(path);
    if (info.isSymbolicLink()) {
      relativeBundlePath(root, realpathSync(path));
    } else if (info.isDirectory()) {
      if (path.endsWith('.app') || path.endsWith('.framework')) bundles.push(path);
      for (const name of readdirSync(path)) pending.push(join(path, name));
    } else if (info.isFile() && isMachO(path)) {
      nativeFiles.push(path);
    }
  }
  if (!nativeFiles.includes(join(root, mainExecutable)))
    throw new Error('Main Oryn Mach-O executable is missing.');
  return { root, nativeFiles: nativeFiles.sort(), bundles: bundles.sort() };
}
