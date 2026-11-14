/* global console, process, setTimeout, window */
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir, userInfo } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, URL } from 'node:url';
import {
  assertNewerRelease,
  readMetadata,
  sha512,
  validateReleaseArtifacts,
  validateBundleVersions,
} from './app-release-validation.mjs';

// Run only on the dedicated disposable macOS qualification account. Native
// Squirrel relaunch behavior is part of the test and may not preserve env vars;
// an ordinary developer account is therefore not a valid isolation boundary.
if (
  process.platform !== 'darwin' ||
  !process.env.ORYN_QUALIFICATION_ACCOUNT ||
  userInfo().username !== process.env.ORYN_QUALIFICATION_ACCOUNT
)
  throw new Error(
    'Run on the dedicated disposable macOS account named by ORYN_QUALIFICATION_ACCOUNT.',
  );
const [sourceApp, candidateDirectory, outputDirectory] = process.argv
  .slice(2)
  .map((value) => resolve(value));
if (!sourceApp || !candidateDirectory || !outputDirectory)
  throw new Error(
    'Usage: qualify-app-update.mjs <signed-A.app> <candidate-B-directory> <new-evidence-directory>',
  );
if (existsSync(outputDirectory)) throw new Error('Evidence output must be a new directory.');
mkdirSync(outputDirectory, { recursive: true });
const manifest = JSON.parse(
  readFileSync(join(candidateDirectory, 'release-manifest.json'), 'utf8'),
);
const metadata = readMetadata(
  join(candidateDirectory, `${manifest.channel === 'beta' ? 'beta' : 'latest'}-mac.yml`),
);
await validateReleaseArtifacts(candidateDirectory, metadata);
if (manifest.bundleVersion !== metadata.oryn.bundleVersion)
  throw new Error('Release manifest and metadata have different bundle versions.');
const oldVersion = plistVersion(sourceApp);
assertNewerRelease(metadata.version, oldVersion);
const oldSignature = run('codesign', ['-d', '--verbose=4', sourceApp]);
if (
  !oldSignature.includes(`TeamIdentifier=${manifest.teamId}`) ||
  !oldSignature.includes('Authority=Developer ID Application:')
)
  throw new Error('Version A must have the expected Developer ID publisher.');
run('codesign', ['--verify', '--deep', '--strict', sourceApp]);
run('spctl', ['--assess', '--type', 'exec', sourceApp]);
const zip = metadata.files.find((file) => file.url.endsWith('.zip'));
run(
  process.execPath,
  [
    fileURLToPath(new URL('./verify-macos-release.mjs', import.meta.url)),
    '-',
    join(candidateDirectory, zip.url),
    metadata.version,
    metadata.oryn.bundleVersion,
  ],
  { ...process.env, ORYN_APPLE_TEAM_ID: manifest.teamId },
);

const fixture = mkdtempSync(join(tmpdir(), 'oryn-signed-upgrade-'));
const appPath = join(fixture, 'Oryn Browser.app');
const profile = join(fixture, 'profile');
const executable = join(appPath, 'Contents/MacOS/Oryn Browser');
run('ditto', [sourceApp, appPath]);
const sentinels = [
  'browser-core/update-qualification-sentinel',
  'local-ai/models/.update-qualification-sentinel',
];
for (const relative of sentinels) {
  const file = join(profile, relative);
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, `Oryn preserved data: ${oldVersion} -> ${metadata.version}\n`);
}
const before = await Promise.all(sentinels.map((relative) => sha512(join(profile, relative))));
const require = createRequire(new URL('../apps/browser/package.json', import.meta.url));
const { _electron } = require('@playwright/test');
const environment = { ...process.env, ORYN_PROFILE_ROOT: profile };
delete environment.ELECTRON_RUN_AS_NODE;
for (const key of [
  'ORYN_START_URL',
  'LOCAL_AI_START_URL',
  'LOCAL_AI_SMOKE_TEST',
  'ORYN_SMOKE_PROFILE_ROOT',
])
  delete environment[key];
let application;
const events = [];
const record = {
  schemaVersion: 2,
  version: metadata.version,
  bundleVersion: metadata.oryn.bundleVersion,
  fromVersion: oldVersion,
  zipSha512: zip.sha512,
  teamId: manifest.teamId,
  checks: {},
};
try {
  application = await _electron.launch({
    executablePath: executable,
    env: environment,
    timeout: 120_000,
  });
  let chrome = await trustedChrome(application);
  await chrome.waitForFunction(() => Boolean(window.browser), null, { timeout: 120_000 });
  await chrome.evaluate(() =>
    window.browser.addBookmark(
      'Update qualification',
      'https://example.com/oryn-update-qualification',
    ),
  );
  await chrome.evaluate(() =>
    window.browser.updateAppUpdateSettings({ channel: 'beta', automaticChecks: false }),
  );
  const offer = await chrome.evaluate(() => window.browser.checkForAppUpdate());
  if (
    offer.candidate?.version !== metadata.version ||
    offer.candidate?.bundleVersion !== metadata.oryn.bundleVersion ||
    offer.status !== 'available'
  )
    throw new Error('Beta feed did not offer the exact qualified candidate.');
  events.push({
    phase: 'offered',
    version: offer.candidate.version,
    bundleVersion: offer.candidate.bundleVersion,
  });
  const downloaded = await chrome.evaluate(
    ({ version, approvalId }) => window.browser.downloadAppUpdate(version, false, approvalId),
    { version: metadata.version, approvalId: offer.candidate.approvalId },
  );
  if (downloaded.status !== 'ready') throw new Error('Candidate did not finish downloading.');
  events.push({ phase: 'downloaded', version: metadata.version });
  // The renderer disappears on success. Observe the signed bundle replacement
  // rather than treating that expected disconnection as an installation error.
  void chrome
    .evaluate(
      ({ version, approvalId }) => window.browser.restartToInstallAppUpdate(version, approvalId),
      { version: metadata.version, approvalId: downloaded.candidate.approvalId },
    )
    .catch(() => undefined);
  await poll(
    () =>
      plistVersion(appPath) === metadata.version &&
      plistValue(appPath, 'CFBundleVersion') === metadata.oryn.bundleVersion,
    10 * 60 * 1000,
  );
  await application.close().catch(() => undefined);
  application = undefined;
  await stopFixtureProcesses(executable);
  run('codesign', ['--verify', '--deep', '--strict', appPath]);
  run('spctl', ['--assess', '--type', 'exec', appPath]);
  const afterSignature = run('codesign', ['-d', '--verbose=4', appPath]);
  if (!afterSignature.includes(`TeamIdentifier=${manifest.teamId}`))
    throw new Error('Installed publisher mismatch.');
  validateBundleVersions(
    {
      version: plistVersion(appPath),
      bundleVersion: plistValue(appPath, 'CFBundleVersion'),
    },
    { version: metadata.version, bundleVersion: metadata.oryn.bundleVersion },
  );
  record.checks.signedUpgrade = {
    passed: true,
    evidence: 'events.json: signed bundle A replaced by exact version/build B',
  };
  const after = await Promise.all(sentinels.map((relative) => sha512(join(profile, relative))));
  if (before.some((digest, index) => digest !== after[index]))
    throw new Error('Profile/model sentinel changed.');
  record.checks.dataPreserved = {
    passed: true,
    evidence: 'events.json: Browser Core and model-directory sentinel SHA-512 checks',
  };
  application = await _electron.launch({
    executablePath: executable,
    env: environment,
    timeout: 120_000,
  });
  chrome = await trustedChrome(application);
  await chrome.waitForFunction(
    async () => (await window.browser.getState()).engineState === 'ready',
    null,
    { timeout: 120_000 },
  );
  const policy = await chrome.evaluate(() => window.browser.getAppUpdate());
  if (policy.channel !== 'beta' || policy.automaticChecks !== false)
    throw new Error('Update preferences were not preserved.');
  const data = await chrome.evaluate(() => window.browser.getBrowserData());
  if (
    !data.bookmarks.some(
      (bookmark) => bookmark.url === 'https://example.com/oryn-update-qualification',
    )
  )
    throw new Error('The saved bookmark was not preserved.');
  record.checks.dataPreserved.evidence =
    'events.json: bookmark, update preferences, and Browser Core/model-directory sentinel hashes';
  record.checks.engineHealthy = {
    passed: true,
    evidence: 'events.json: installed B Engine reached ready',
  };
  events.push({
    phase: 'installed-and-restarted',
    version: metadata.version,
    bundleVersion: metadata.oryn.bundleVersion,
    preservedDigests: after,
    engineState: 'ready',
  });
  // These are separate required exercises. Never invent a passing attestation
  // from an ordinary successful upgrade.
  for (const check of [
    'wrongPublisherRejected',
    'interruptionRecovery',
    'networkPrivacy',
    'channelIsolation',
    'securityReview',
  ])
    record.checks[check] = { passed: false, evidence: '' };
  console.log(
    'Signed A-to-B smoke passed. Complete the remaining qualification cases before stable promotion.',
  );
} finally {
  if (application) await application.close().catch(() => undefined);
  await stopFixtureProcesses(executable);
  writeFileSync(join(outputDirectory, 'events.json'), `${JSON.stringify(events, null, 2)}\n`);
  writeFileSync(
    join(outputDirectory, 'qualification.json'),
    `${JSON.stringify(record, null, 2)}\n`,
  );
  // Keep the disposable app and profile for diagnosis, never a user's installed app.
  writeFileSync(join(outputDirectory, 'fixture-path.txt'), `${fixture}\n`);
}

function run(command, args, env = process.env) {
  const result = spawnSync(command, args, { encoding: 'utf8', env });
  if (result.status !== 0) throw new Error(`${command} failed during signed qualification.`);
  return `${result.stdout}${result.stderr}`;
}
function plistVersion(app) {
  return plistValue(app, 'CFBundleShortVersionString');
}
function plistValue(app, name) {
  return run('/usr/libexec/PlistBuddy', [
    '-c',
    `Print :${name}`,
    join(app, 'Contents/Info.plist'),
  ]).replace(/\r?\n$/u, '');
}
async function poll(predicate, milliseconds) {
  const deadline = Date.now() + milliseconds;
  while (Date.now() < deadline) {
    try {
      if (await predicate()) return;
    } catch {
      /* Native replacement can temporarily move the app. */
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error('Signed update qualification timed out.');
}
async function trustedChrome(application) {
  await poll(
    () =>
      application
        .context()
        .pages()
        .some((page) => /index\.html$/u.test(page.url())),
    120_000,
  );
  return application
    .context()
    .pages()
    .find((page) => /index\.html$/u.test(page.url()));
}
async function stopFixtureProcesses(executable) {
  const processes = () =>
    run('/bin/ps', ['-ww', '-axo', 'pid=,comm='])
      .split('\n')
      .flatMap((line) => {
        const match = /^\s*(\d+)\s+(.+)$/u.exec(line);
        return match?.[2] === executable ? [Number(match[1])] : [];
      });
  for (const pid of processes()) {
    try {
      process.kill(pid, 'SIGTERM');
    } catch {
      /* Already exited. */
    }
  }
  await poll(() => processes().length === 0, 30_000);
}
