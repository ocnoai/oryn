/* global console, process */
import { spawnSync } from 'node:child_process';

if (process.platform !== 'darwin') throw new Error('macOS release builds must run on macOS.');
const identities = spawnSync('security', ['find-identity', '-v', '-p', 'codesigning'], {
  encoding: 'utf8',
});
const hasDeveloperIdentity = /Developer ID Application:/u.test(identities.stdout);
if (!hasDeveloperIdentity && !process.env.CSC_LINK) {
  throw new Error('A Developer ID Application identity or CSC_LINK is required.');
}
const hasApiKey = Boolean(
  process.env.APPLE_API_KEY && process.env.APPLE_API_KEY_ID && process.env.APPLE_API_ISSUER,
);
const hasAppleId = Boolean(
  process.env.APPLE_ID && process.env.APPLE_APP_SPECIFIC_PASSWORD && process.env.APPLE_TEAM_ID,
);
const profile = process.env.APPLE_KEYCHAIN_PROFILE;
if (!hasApiKey && !hasAppleId && !profile) {
  throw new Error('Apple notarization credentials are required.');
}
if (profile) {
  const profileCheck = spawnSync(
    'xcrun',
    ['notarytool', 'history', '--keychain-profile', profile],
    { stdio: 'ignore' },
  );
  if (profileCheck.status !== 0) throw new Error('APPLE_KEYCHAIN_PROFILE is not usable.');
}
console.log('Developer ID signing and notarization configuration are present.');
