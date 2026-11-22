/* global console, process */
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { verifyMacOSSecurity } from './verify-macos-security.mjs';
import { assertPackagedAppUpdateConfiguration } from './macos-app-update-config.mjs';
import {
  releaseBundleVersion,
  releaseVersion,
  validateBundleVersions,
} from './app-release-validation.mjs';

const appPath = process.argv[2];
const zipPath = process.argv[3];
const expectedVersion = process.argv[4];
const expectedBundleVersion = process.argv[5];
if (!appPath || !zipPath) {
  throw new Error('Pass the signed/notarized Oryn Browser.app and final distributed ZIP paths.');
}
if ((expectedVersion === undefined) !== (expectedBundleVersion === undefined))
  throw new Error('Expected release version and bundle version must be supplied together.');
if (expectedVersion !== undefined) {
  releaseVersion(expectedVersion);
  releaseBundleVersion(expectedBundleVersion);
}
const expectedTeamId = process.env.ORYN_APPLE_TEAM_ID;
if (!expectedTeamId || !/^[A-Z0-9]{10}$/u.test(expectedTeamId)) {
  throw new Error('ORYN_APPLE_TEAM_ID must identify the expected signing team.');
}

const identity = appPath === '-' ? undefined : await verifyApp(resolve(appPath));
const temporaryDirectory = mkdtempSync(join(tmpdir(), 'oryn-release-verify-'));
try {
  run('ditto', ['-x', '-k', resolve(zipPath), temporaryDirectory]);
  const appEntries = readdirSync(temporaryDirectory).filter((entry) => entry.endsWith('.app'));
  if (appEntries.length !== 1)
    throw new Error('The update ZIP must contain exactly one app bundle.');
  const zipIdentity = await verifyApp(join(temporaryDirectory, appEntries[0]));
  if (identity !== undefined && JSON.stringify(zipIdentity) !== JSON.stringify(identity))
    throw new Error('The distributed ZIP has a different application identity or version/build.');
} finally {
  rmSync(temporaryDirectory, { recursive: true, force: true });
}
console.log(
  `MACOS_RELEASE_RESULT ${JSON.stringify({ passed: true, signed: true, notarized: true, stapled: true, teamId: expectedTeamId, zipVerified: true })}`,
);

async function verifyApp(app) {
  await verifyMacOSSecurity(app, { variant: 'release', expectedTeamId });
  assertPackagedAppUpdateConfiguration(app, { expectedTeamId });
  run('codesign', [
    '--verify',
    '--deep',
    '--strict',
    '--verbose=2',
    '-R=anchor apple generic and identifier "ai.ocno.oryn.browser" and certificate 1[field.1.2.840.113635.100.6.2.6] exists and certificate leaf[field.1.2.840.113635.100.6.1.13] exists and certificate leaf[subject.OU] = "' +
      expectedTeamId +
      '"',
    app,
  ]);
  const details = capture('codesign', ['-d', '--verbose=4', app]);
  if (!/^Identifier=ai\.ocno\.oryn\.browser$/mu.test(details)) {
    throw new Error('The packaged app has the wrong bundle identifier.');
  }
  if (!details.includes('Authority=Developer ID Application:')) {
    throw new Error('The packaged app is not signed with Developer ID Application.');
  }
  if (!details.includes(`TeamIdentifier=${expectedTeamId}`)) {
    throw new Error('The packaged app is signed by an unexpected Apple team.');
  }
  if (!/flags=.*\bruntime\b/u.test(details)) throw new Error('Hardened runtime is required.');
  run('spctl', ['--assess', '--verbose', '--type', 'exec', app]);
  run('xcrun', ['stapler', 'validate', app]);
  if (plistValue(app, 'CFBundleIdentifier') !== 'ai.ocno.oryn.browser')
    throw new Error('The authenticated Info.plist has an unexpected bundle identifier.');
  const versions = validateBundleVersions(
    {
      version: plistValue(app, 'CFBundleShortVersionString'),
      bundleVersion: plistValue(app, 'CFBundleVersion'),
    },
    { version: expectedVersion, bundleVersion: expectedBundleVersion },
  );
  const architectures = capture('/usr/bin/lipo', [
    '-archs',
    join(app, 'Contents/MacOS/Oryn Browser'),
  ])
    .trim()
    .split(/\s+/u)
    .sort();
  if (!architectures.includes('arm64'))
    throw new Error('The signed application does not support arm64.');
  return { ...versions, applicationId: 'ai.ocno.oryn.browser', architectures };
}

function plistValue(app, name) {
  const result = spawnSync(
    '/usr/libexec/PlistBuddy',
    ['-c', `Print :${name}`, join(app, 'Contents/Info.plist')],
    { encoding: 'utf8', maxBuffer: 64 * 1024 },
  );
  if (result.status !== 0) throw new Error('Authenticated bundle fields could not be read.');
  return result.stdout.replace(/\r?\n$/u, '');
}

function capture(command, arguments_) {
  const result = spawnSync(command, arguments_, { encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`${command} verification failed.`);
  return `${result.stdout}${result.stderr}`;
}

function run(command, arguments_) {
  const result = spawnSync(command, arguments_, { stdio: 'inherit' });
  if (result.status !== 0) throw new Error(`${command} verification failed.`);
}
