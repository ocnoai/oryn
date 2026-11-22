/* global process */
import { lstatSync, realpathSync } from 'node:fs';
import { signAsync, getCurrentFuseWire } from './macos-packaging-dependencies.mjs';
import {
  assertNativeReleaseUnchanged,
  prepareNativeRelease,
  sealNativeRelease,
  verifyReleaseRuntime,
} from './macos-native-release-bindings.mjs';
import {
  assertFusePolicy,
  entitlementFile,
  isMachO,
  relativeBundlePath,
  scanBundle,
  securityVariant,
} from './macos-security-policy.mjs';

export function signingOptions(options, { adHoc = false } = {}) {
  const { root } = scanBundle(options.app);
  const priorIgnore = Array.isArray(options.ignore)
    ? options.ignore
    : [options.ignore].filter(Boolean);
  return {
    ...options,
    app: root,
    identity: adHoc ? '-' : options.identity,
    identityValidation: false,
    preAutoEntitlements: false,
    preEmbedProvisioningProfile: false,
    optionsForFile(path) {
      const canonical = realpathSync(path);
      relativeBundlePath(root, canonical);
      return {
        ...options.optionsForFile?.(path),
        entitlements: entitlementFile(root, canonical),
        hardenedRuntime: true,
        ...(adHoc ? { timestamp: 'none' } : {}),
      };
    },
    ignore(path) {
      relativeBundlePath(root, realpathSync(path));
      const info = lstatSync(path);
      if (info.isSymbolicLink() || realpathSync(path) !== path) return true;
      if (info.isFile() && !isMachO(path)) return true;
      return priorIgnore.some((ignore) =>
        typeof ignore === 'function' ? ignore(path) : new RegExp(ignore).test(path),
      );
    },
  };
}

export function enclosingAppSigningOptions(signing) {
  // osx-sign 1.3.1 drops array-form ignore filters during option validation.
  // A function is normalized correctly and must skip every descendant.
  return { ...signing, ignore: (path) => realpathSync(path) !== signing.app };
}

export async function signWithSecurityPolicy(
  options,
  { variant = 'release', mode = 'release' } = {},
) {
  securityVariant(variant);
  if (mode !== 'release' && mode !== 'ad-hoc') throw new Error('Unknown macOS signing mode.');
  if (
    mode === 'release' &&
    (!options.identity || options.identity === '-' || variant !== 'release')
  )
    throw new Error('A release requires the real signing identity and release fuse policy.');
  assertFusePolicy(await getCurrentFuseWire(options.app), variant);
  const signing = signingOptions(options, { adHoc: mode === 'ad-hoc' });
  const prepared =
    mode === 'release'
      ? prepareNativeRelease(signing.app, scanBundle(signing.app).nativeFiles)
      : undefined;
  await signAsync(signing);
  if (prepared) {
    const sealed = sealNativeRelease(prepared, signing);
    // Sign only the enclosing app now. Re-signing Engine Mach-O files would
    // invalidate the final native hashes just authorized by the binding bundle.
    await signAsync(enclosingAppSigningOptions(signing));
    assertNativeReleaseUnchanged(prepared, sealed);
    verifyReleaseRuntime(signing.app);
  }
  assertFusePolicy(await getCurrentFuseWire(options.app), variant);
}

export default async function sign(options) {
  await signWithSecurityPolicy(options, {
    variant: securityVariant(process.env.ORYN_MAC_SECURITY_VARIANT),
    mode: process.env.ORYN_MAC_SIGNING_MODE ?? 'release',
  });
}
