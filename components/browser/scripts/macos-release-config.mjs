import { statSync } from 'node:fs';
import { join } from 'node:path';
import {
  assertReleaseChannel,
  releaseBaseUrl,
  releaseVersion,
  sha512,
} from './app-release-validation.mjs';

export function macOSReleaseOptions(version, environment) {
  const updateMode = environment.ORYN_RELEASE_UPDATES;
  if (updateMode !== undefined && updateMode !== '0' && updateMode !== '1') {
    throw new Error('ORYN_RELEASE_UPDATES must be 0 or 1.');
  }
  const updatesEnabled = updateMode !== '0';
  const teamId = environment.ORYN_APPLE_TEAM_ID;
  if (!teamId || !/^[A-Z0-9]{10}$/u.test(teamId)) {
    throw new Error('ORYN_APPLE_TEAM_ID must be the expected 10-character signing team ID.');
  }
  const stagedRolloutQualified = environment.ORYN_STAGED_ROLLOUT_QUALIFIED === '1';
  const percentage = parseRolloutPercentage(environment.ORYN_UPDATE_ROLLOUT_PERCENTAGE);
  if (!updatesEnabled) {
    releaseVersion(version);
    if (stagedRolloutQualified || percentage !== undefined) {
      throw new Error('A manual release cannot declare update rollout settings.');
    }
    return {
      updatesEnabled: false,
      teamId,
      baseUrl: null,
      channel: null,
      stagedRolloutQualified: false,
      percentage: undefined,
    };
  }

  const baseUrl = releaseBaseUrl(environment.ORYN_UPDATE_BASE_URL);
  const channel = environment.ORYN_UPDATE_CHANNEL;
  if (channel !== 'stable' && channel !== 'beta') {
    throw new Error('ORYN_UPDATE_CHANNEL must be stable or beta.');
  }
  assertReleaseChannel(version, channel);
  if (channel === 'beta' && !version.includes('-beta.')) {
    throw new Error(
      'A beta build needs an X.Y.Z-beta.N version. Build a stable candidate as stable, then qualify its exact bytes on the beta feed before promotion.',
    );
  }
  if (channel === 'beta' && percentage !== undefined) {
    throw new Error('Beta metadata must not contain a rollout percentage.');
  }
  if (!stagedRolloutQualified && percentage !== undefined) {
    throw new Error('Percentage rollout requires ORYN_STAGED_ROLLOUT_QUALIFIED=1.');
  }
  return { updatesEnabled, teamId, baseUrl, channel, stagedRolloutQualified, percentage };
}

export function macOSReleaseBuilderConfiguration(appDirectory, releaseDirectory, options) {
  const { updatesEnabled, baseUrl, teamId, stagedRolloutQualified, channel } = options;
  return {
    extends: join(appDirectory, 'electron-builder.yml'),
    directories: { output: releaseDirectory },
    generateUpdatesFilesForAllChannels: false,
    extraMetadata: {
      orynUpdate: updatesEnabled
        ? { enabled: true, baseUrl, expectedTeamId: teamId, stagedRolloutQualified, channel }
        : { enabled: false },
    },
    publish: updatesEnabled
      ? { provider: 'generic', url: baseUrl, channel: channel === 'beta' ? 'beta' : 'latest' }
      : null,
  };
}

export async function manualReleaseArtifacts(directory, version, unpackedSize) {
  const files = [];
  for (const extension of ['dmg', 'zip']) {
    const url = `Oryn-Browser-${version}-arm64.${extension}`;
    const path = join(directory, url);
    files.push({ url, size: statSync(path).size, sha512: await sha512(path) });
  }
  return {
    version,
    files,
    oryn: { schemaVersion: 1, platform: 'darwin', arch: 'arm64', unpackedSize },
  };
}

function parseRolloutPercentage(value) {
  if (value === undefined || value === '') return undefined;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0 || parsed > 100) {
    throw new Error('ORYN_UPDATE_ROLLOUT_PERCENTAGE must be an integer from 0 through 100.');
  }
  return parsed;
}
