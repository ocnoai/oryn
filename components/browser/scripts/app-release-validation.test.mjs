import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import YAML from 'yaml';
import {
  assertNewerRelease,
  assertReleaseChannel,
  releaseBundleVersion,
  sha512,
  validateBundleVersions,
  validatePromotionEvidence,
  validateReleaseArtifacts,
} from './app-release-validation.mjs';
import { publishRelease } from './publish-app-update.mjs';
import {
  macOSReleaseBuilderConfiguration,
  macOSReleaseOptions,
  manualReleaseArtifacts,
} from './macos-release-config.mjs';

async function fixture(t, version = '2.0.0-beta.1') {
  const root = mkdtempSync(join(tmpdir(), 'oryn-release-test-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const directory = join(root, 'candidate');
  const target = join(root, 'host');
  mkdirSync(directory);
  mkdirSync(target);
  const files = [];
  for (const ext of ['zip', 'dmg']) {
    const url = `Oryn-Browser-${version}-arm64.${ext}`;
    writeFileSync(join(directory, url), 'synthetic test artifact');
    files.push({ url, size: 23, sha512: await sha512(join(directory, url)) });
  }
  const metadata = {
    version,
    files,
    oryn: {
      schemaVersion: 2,
      platform: 'darwin',
      arch: 'arm64',
      unpackedSize: 100,
      bundleVersion: '2.0.0.7',
    },
  };
  const sourceChannel = version.includes('-') ? 'beta' : 'stable';
  const source = join(directory, `${sourceChannel === 'beta' ? 'beta' : 'latest'}-mac.yml`);
  writeFileSync(source, YAML.stringify(metadata));
  const baseUrl = 'https://updates.example/oryn/macos/arm64/';
  const teamId = 'ABCDEFGHIJ';
  writeFileSync(
    join(directory, 'release-manifest.json'),
    JSON.stringify({
      schemaVersion: 1,
      version,
      bundleVersion: metadata.oryn.bundleVersion,
      channel: sourceChannel,
      baseUrl,
      teamId,
      signedAndNotarized: true,
      metadataSha512: await sha512(source),
    }),
  );
  const calls = [];
  const verifyHosted = async (url, file) => {
    const name = url.pathname.split('/').at(-1);
    assert.equal(await sha512(join(target, name)), file.sha512);
    calls.push(name);
  };
  const options = {
    directory,
    target,
    baseUrl,
    channel: 'beta',
    verifyHosted,
    evidence: {
      schemaVersion: 2,
      version,
      bundleVersion: metadata.oryn.bundleVersion,
      zipSha512: files[0].sha512,
      teamId,
      checks: Object.fromEntries(
        [
          'signedUpgrade',
          'dataPreserved',
          'engineHealthy',
          'wrongPublisherRejected',
          'interruptionRecovery',
          'networkPrivacy',
          'channelIsolation',
          'securityReview',
        ].map((name) => [name, { passed: true, evidence: `synthetic-fixture:${name}` }]),
      ),
    },
    verifySigned: () => undefined,
  };
  return { directory, target, metadata, calls, options, teamId };
}

test('release channels enforce stable versions and monotonically advance beta/stable', () => {
  assert.throws(() => assertReleaseChannel('2.0.0-beta.1', 'stable'));
  assert.throws(() => assertNewerRelease('2.0.0-beta.1', '2.0.0'));
  assert.throws(() => assertNewerRelease('2.0.0', '2.0.0'));
  assertNewerRelease('2.0.0-beta.10', '2.0.0-beta.9');
  assertNewerRelease('2.0.0', '2.0.0-beta.10');
});

test('schema v2 rejects historical offers, unknown fields and invalid bundle versions', async (t) => {
  const f = await fixture(t);
  await validateReleaseArtifacts(f.directory, f.metadata);
  for (const bundleVersion of [
    '',
    '1 2',
    '1\t2',
    '1\n',
    'é',
    '\u007f',
    '1'.repeat(65),
    7,
    undefined,
  ]) {
    await assert.rejects(
      validateReleaseArtifacts(f.directory, {
        ...f.metadata,
        oryn: { ...f.metadata.oryn, bundleVersion },
      }),
      /Bundle version/,
    );
  }
  const historical = { ...f.metadata.oryn };
  delete historical.bundleVersion;
  await assert.rejects(
    validateReleaseArtifacts(f.directory, {
      ...f.metadata,
      oryn: { ...historical, schemaVersion: 1 },
    }),
    /metadata schema/,
  );
  await assert.rejects(
    validateReleaseArtifacts(f.directory, {
      ...f.metadata,
      oryn: { ...f.metadata.oryn, applicationId: 'other.app' },
    }),
    /metadata schema/,
  );
  assert.equal(releaseBundleVersion('1'.repeat(64)), '1'.repeat(64));
});

test('authenticated version fields must both match the advertised version and build exactly', () => {
  const expected = { version: '2.0.0', bundleVersion: '2.0.0.7' };
  assert.deepEqual(validateBundleVersions(expected, expected), expected);
  for (const actual of [
    { ...expected, version: '3.0.0' },
    { ...expected, bundleVersion: '2.0.0.8' },
    { ...expected, bundleVersion: '2.0.0.7 ' },
  ])
    assert.throws(() => validateBundleVersions(actual, expected));
});

test('qualification records bind exact bundle version and reject historical version-only evidence', async (t) => {
  const f = await fixture(t);
  validatePromotionEvidence(f.options.evidence, f.metadata, f.teamId);
  for (const evidence of [
    { ...f.options.evidence, schemaVersion: 1 },
    { ...f.options.evidence, bundleVersion: undefined },
    { ...f.options.evidence, bundleVersion: '2.0.0.8' },
  ])
    assert.throws(
      () => validatePromotionEvidence(evidence, f.metadata, f.teamId),
      /Qualification evidence/,
    );
});

test('beta publishes verified immutable artifacts first and leaves stable byte-for-byte intact', async (t) => {
  const f = await fixture(t);
  const stable = join(f.target, 'latest-mac.yml');
  writeFileSync(stable, 'stable sentinel');
  const verify = f.options.verifyHosted;
  f.options.verifyHosted = async (url, file) => {
    if (!url.pathname.endsWith('.yml'))
      assert.equal(existsSync(join(f.target, 'beta-mac.yml')), false);
    await verify(url, file);
  };
  await publishRelease(f.options);
  assert.equal(readFileSync(stable, 'utf8'), 'stable sentinel');
  assert.deepEqual(f.calls, [...f.metadata.files.map((file) => file.url), 'beta-mac.yml']);
});

test('mismatched bytes, occupied immutable filenames and failed hosted checks cannot publish metadata', async (t) => {
  const f = await fixture(t);
  writeFileSync(join(f.directory, f.metadata.files[0].url), 'wrong');
  await assert.rejects(validateReleaseArtifacts(f.directory, f.metadata), /size mismatch/);
  writeFileSync(join(f.directory, f.metadata.files[0].url), 'synthetic test artifact');
  writeFileSync(join(f.target, f.metadata.files[0].url), 'other signed bytes');
  await assert.rejects(publishRelease(f.options), /immutable/);
  rmSync(join(f.target, f.metadata.files[0].url));
  f.options.verifyHosted = async () => {
    throw new Error('CDN serves a redirect');
  };
  await assert.rejects(publishRelease(f.options), /redirect/);
  assert.equal(existsSync(join(f.target, 'beta-mac.yml')), false);
});

test('stable promotion requires evidence bound to the exact final stable ZIP', async (t) => {
  const f = await fixture(t, '2.0.0');
  await publishRelease(f.options); // The final stable bytes are first offered on beta.
  f.options.channel = 'stable';
  f.options.evidence = undefined;
  await assert.rejects(publishRelease(f.options), /Qualification evidence/);
  const checks = Object.fromEntries(
    [
      'signedUpgrade',
      'dataPreserved',
      'engineHealthy',
      'wrongPublisherRejected',
      'interruptionRecovery',
      'networkPrivacy',
      'channelIsolation',
      'securityReview',
    ].map((name) => [name, { passed: true, evidence: `test-fixture:${name}` }]),
  );
  f.options.evidence = {
    schemaVersion: 2,
    version: '2.0.0',
    bundleVersion: f.metadata.oryn.bundleVersion,
    zipSha512: f.metadata.files[0].sha512,
    teamId: f.teamId,
    checks,
  };
  await publishRelease(f.options);
  assert.equal(
    readFileSync(join(f.target, 'beta-mac.yml'), 'utf8'),
    readFileSync(join(f.target, 'latest-mac.yml'), 'utf8'),
  );
});

test('publication refuses a busy release lock and a version downgrade', async (t) => {
  const f = await fixture(t);
  mkdirSync(join(f.target, '.oryn-publication-lock'));
  await assert.rejects(publishRelease(f.options), /EEXIST/);
  rmSync(join(f.target, '.oryn-publication-lock'), { recursive: true });
  writeFileSync(join(f.target, 'beta-mac.yml'), YAML.stringify({ version: '3.0.0-beta.1' }));
  await assert.rejects(publishRelease(f.options), /advance/);
});

test('manual releases need a signing team but no host and embed no updater configuration', () => {
  const options = macOSReleaseOptions('0.1.0', {
    ORYN_RELEASE_UPDATES: '0',
    ORYN_APPLE_TEAM_ID: 'ABCDEFGHIJ',
    // A shell's stale update settings cannot enable an updater in manual mode.
    ORYN_UPDATE_BASE_URL: 'https://old.example/updates/',
    ORYN_UPDATE_CHANNEL: 'stable',
  });
  const config = macOSReleaseBuilderConfiguration('/app', '/candidate', options);
  assert.equal(options.baseUrl, null);
  assert.equal(options.channel, null);
  assert.deepEqual(config.extraMetadata.orynUpdate, { enabled: false });
  assert.equal(config.publish, null);
  assert.equal(config.extends, '/app/electron-builder.yml');
  assert.throws(
    () => macOSReleaseOptions('0.1.0', { ORYN_RELEASE_UPDATES: '0' }),
    /ORYN_APPLE_TEAM_ID/,
  );
  assert.throws(
    () =>
      macOSReleaseOptions('invalid', {
        ORYN_RELEASE_UPDATES: '0',
        ORYN_APPLE_TEAM_ID: 'ABCDEFGHIJ',
      }),
    /Release version/,
  );
});

test('updater-enabled releases retain host, channel and staged rollout requirements by default', () => {
  const environment = {
    ORYN_APPLE_TEAM_ID: 'ABCDEFGHIJ',
    ORYN_UPDATE_BASE_URL: 'https://updates.example/oryn',
    ORYN_UPDATE_CHANNEL: 'stable',
  };
  const options = macOSReleaseOptions('0.1.0', environment);
  const config = macOSReleaseBuilderConfiguration('/app', '/candidate', options);
  assert.deepEqual(config.extraMetadata.orynUpdate, {
    enabled: true,
    baseUrl: 'https://updates.example/oryn/',
    expectedTeamId: 'ABCDEFGHIJ',
    stagedRolloutQualified: false,
    channel: 'stable',
  });
  assert.deepEqual(config.publish, {
    provider: 'generic',
    url: 'https://updates.example/oryn/',
    channel: 'latest',
  });
  assert.throws(
    () => macOSReleaseOptions('0.1.0', { ...environment, ORYN_UPDATE_BASE_URL: undefined }),
    /Invalid URL/,
  );
  assert.throws(
    () => macOSReleaseOptions('0.1.0', { ...environment, ORYN_UPDATE_CHANNEL: 'beta' }),
    /beta build needs/,
  );
  assert.throws(
    () => macOSReleaseOptions('0.1.0', { ...environment, ORYN_UPDATE_ROLLOUT_PERCENTAGE: '10' }),
    /Percentage rollout requires/,
  );
});

test('manual mode rejects rollout claims and misspelled mode values', () => {
  const environment = { ORYN_RELEASE_UPDATES: '0', ORYN_APPLE_TEAM_ID: 'ABCDEFGHIJ' };
  for (const additional of [
    { ORYN_STAGED_ROLLOUT_QUALIFIED: '1' },
    { ORYN_UPDATE_ROLLOUT_PERCENTAGE: '100' },
  ]) {
    assert.throws(
      () => macOSReleaseOptions('0.1.0', { ...environment, ...additional }),
      /manual release/,
    );
  }
  assert.throws(
    () => macOSReleaseOptions('0.1.0', { ...environment, ORYN_RELEASE_UPDATES: 'false' }),
    /ORYN_RELEASE_UPDATES must be 0 or 1/,
  );
});

test('manual artifact records validate the final DMG and ZIP bytes without feed metadata', async (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'oryn-manual-release-test-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const version = '0.1.0';
  for (const extension of ['dmg', 'zip']) {
    writeFileSync(
      join(directory, `Oryn-Browser-${version}-arm64.${extension}`),
      `final ${extension} bytes`,
    );
  }
  const metadata = await manualReleaseArtifacts(directory, version, 100);
  await validateReleaseArtifacts(directory, metadata, { manual: true });
  await assert.rejects(validateReleaseArtifacts(directory, metadata), /metadata schema/);
  assert.equal(existsSync(join(directory, 'latest-mac.yml')), false);
  assert.equal(existsSync(join(directory, 'beta-mac.yml')), false);
  writeFileSync(join(directory, `Oryn-Browser-${version}-arm64.zip`), 'wrong zip bytes');
  await assert.rejects(
    validateReleaseArtifacts(directory, metadata, { manual: true }),
    /SHA-512 mismatch/,
  );
});

test('manual release manifests cannot be published to an update feed', async (t) => {
  const f = await fixture(t);
  const manifestPath = join(f.directory, 'release-manifest.json');
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  writeFileSync(
    manifestPath,
    JSON.stringify({
      ...manifest,
      baseUrl: null,
      channel: null,
      updatesEnabled: false,
      distribution: 'manual',
    }),
  );
  await assert.rejects(publishRelease(f.options), /Invalid release manifest/);
  assert.equal(f.calls.length, 0);
  assert.equal(existsSync(join(f.target, 'beta-mac.yml')), false);
});
