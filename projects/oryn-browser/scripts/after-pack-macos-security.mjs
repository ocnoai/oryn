/* global process */
import { join } from 'node:path';
import { flipFuses, getCurrentFuseWire } from './macos-packaging-dependencies.mjs';
import { assertFusePolicy, fusePolicy, securityVariant } from './macos-security-policy.mjs';

export default async function afterPack(context) {
  if (context.electronPlatformName !== 'darwin') return;
  const variant = securityVariant(process.env.ORYN_MAC_SECURITY_VARIANT);
  if (variant === 'inspect-test' && process.env.ORYN_MAC_SIGNING_MODE !== 'ad-hoc')
    throw new Error(
      'The inspect-test fuse is restricted to the explicitly labelled local test lane.',
    );
  const app = join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`);
  const before = await getCurrentFuseWire(app);
  if (
    before.version !== '1' ||
    Object.keys(before).filter((key) => /^\d+$/u.test(key)).length !== 9
  )
    throw new Error('Unreviewed Electron fuse schema; update the strict policy first.');
  await flipFuses(app, fusePolicy(variant));
  assertFusePolicy(await getCurrentFuseWire(app), variant);
}
