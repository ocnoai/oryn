import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { browserApplicationName, readBrowserBuildFlavor } from '../browser-build-flavor';
import { readAppUpdateBuildConfiguration } from '../app-updates/app-update-build-config';

const roots: string[] = [];

function metadataRoot(metadata: unknown): string {
  const root = mkdtempSync(join(tmpdir(), 'oryn-build-flavor-'));
  roots.push(root);
  writeFileSync(join(root, 'package.json'), JSON.stringify(metadata));
  return root;
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe('packaged Browser build identity', () => {
  it('preserves the release identity for existing packages without a flavor marker', () => {
    const flavor = readBrowserBuildFlavor(metadataRoot({ name: '@ocno/oryn-browser' }));
    expect(flavor).toBe('release');
    expect(browserApplicationName(flavor)).toBe('Oryn Browser');
  });

  it('gives a development package its own visible application name', () => {
    const flavor = readBrowserBuildFlavor(metadataRoot({ orynBuildFlavor: 'local-development' }));
    expect(browserApplicationName(flavor)).toBe('Oryn Browser Local');
  });

  it('rejects unknown build markers instead of silently using the release profile', () => {
    for (const marker of ['development', 'release', null, 1]) {
      expect(() => readBrowserBuildFlavor(metadataRoot({ orynBuildFlavor: marker }))).toThrow(
        'Unsupported Browser build flavor.',
      );
    }
  });

  it('keeps development updates disabled even if release update metadata is included', () => {
    const metadata = {
      version: '0.1.0',
      orynUpdate: {
        enabled: true,
        baseUrl: 'https://updates.example/oryn/',
        expectedTeamId: 'A123456789',
        stagedRolloutQualified: false,
      },
    };
    expect(
      readAppUpdateBuildConfiguration({ packaged: true, appPath: metadataRoot(metadata) }),
    ).toMatchObject({ enabled: true, expectedTeamId: 'A123456789' });
    expect(
      readAppUpdateBuildConfiguration({
        packaged: true,
        appPath: metadataRoot({ ...metadata, orynBuildFlavor: 'local-development' }),
      }),
    ).toEqual({ enabled: false });
  });
});
