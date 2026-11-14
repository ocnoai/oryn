import { describe, expect, it, vi } from 'vitest';
import type { UpdateInfo } from 'electron-updater';
import { EventEmitter } from 'node:events';
import type { AppUpdater } from 'electron-updater';

import { validateUpdateBaseUrl } from '../app-updates/app-update-build-config';
import {
  candidateFromUpdateInfo,
  ElectronUpdaterAdapter,
  isAllowedUpdateRequest,
  sanitizeUpdateRequestHeaders,
  validateUpdateResponse,
} from '../app-updates/electron-updater-adapter';

const base = new URL('https://updates.example/oryn/macos/arm64/');

function adapterFixture() {
  const info = {
    version: '2.0.0',
    files: [
      {
        url: 'Oryn-Browser-2.0.0-arm64.zip',
        sha512: Buffer.alloc(64, 1).toString('base64'),
        size: 123,
      },
    ],
    oryn: {
      schemaVersion: 2,
      bundleVersion: '2.0.0',
      platform: 'darwin',
      arch: 'arm64',
      unpackedSize: 456,
    },
  };
  const updater = Object.assign(new EventEmitter(), {
    allowDowngrade: false,
    isUserWithinRollout: async () => true,
    netSession: {
      webRequest: {
        onBeforeRequest: vi.fn(),
        onBeforeSendHeaders: vi.fn(),
        onHeadersReceived: vi.fn(),
      },
    },
    checkForUpdates: vi.fn(async () => ({
      isUpdateAvailable: true,
      updateInfo: info,
    })),
    downloadUpdate: vi.fn(async () => [] as string[]),
  });
  const adapter = new ElectronUpdaterAdapter(
    base.toString(),
    {
      stagedRolloutQualified: false,
      expectedAppId: 'ai.ocno.oryn.browser',
      expectedTeamId: 'ABCDEFGHIJ',
      currentVersion: '1.0.0',
      currentSystemVersion: '26.6.2',
      verificationRoot: '/private/tmp/oryn-unused-verification',
    },
    updater as unknown as AppUpdater,
  );
  return { adapter, updater, info };
}

describe('application update network boundary', () => {
  it.each([false, true])(
    'captures async native-server failures/cancel races without throwing from the downloaded event',
    async (cancel) => {
      const { adapter, updater } = adapterFixture();
      adapter.configure('stable');
      await adapter.check();
      let finish!: () => void;
      updater.downloadUpdate.mockImplementationOnce(async () => {
        await new Promise<void>((resolve) => {
          finish = resolve;
        });
        expect(() =>
          updater.emit('update-downloaded', {
            downloadedFile: '/fixture/update.zip',
          }),
        ).not.toThrow();
        return ['/fixture/update.zip'];
      });
      const pending = adapter.download(() => undefined);
      if (cancel) adapter.cancelDownload();
      finish();
      await expect(pending).rejects.toThrow(
        cancel ? 'APP_UPDATE_CANCELLED' : 'APP_UPDATE_NATIVE_HANDOFF_UNAVAILABLE',
      );
      expect(updater.listenerCount('update-downloaded')).toBe(0);
      await expect(adapter.install()).rejects.toThrow('APP_UPDATE_PAYLOAD_UNVERIFIED');
      adapter.dispose();
    },
  );
  it('rejects a stale adapter result after a channel change, before it can authorize a ZIP', async () => {
    const { adapter, updater, info } = adapterFixture();
    let finish!: (value: { isUpdateAvailable: boolean; updateInfo: typeof info }) => void;
    updater.checkForUpdates.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    adapter.configure('beta');
    const pending = adapter.check();
    adapter.configure('stable');
    finish({ isUpdateAvailable: true, updateInfo: info });
    await expect(pending).rejects.toThrow('APP_UPDATE_STALE_CHECK');
    await expect(adapter.download(() => undefined)).rejects.toThrow('APP_UPDATE_CANDIDATE_MISSING');
    expect(updater.downloadUpdate).not.toHaveBeenCalled();
    adapter.dispose();
  });

  it('treats a cached download completion after token cancellation as cancelled', async () => {
    const { adapter, updater } = adapterFixture();
    let finish!: (value: string[]) => void;
    updater.downloadUpdate.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    adapter.configure('stable');
    await adapter.check();
    const progress = vi.fn();
    const pending = adapter.download(progress);
    adapter.cancelDownload();
    updater.emit('download-progress', { transferred: 123, total: 123 });
    finish([]);
    await expect(pending).rejects.toThrow('APP_UPDATE_CANCELLED');
    expect(progress).not.toHaveBeenCalled();
    expect(updater.listenerCount('download-progress')).toBe(0);
    adapter.dispose();
  });

  it('disables implicit download/install/logging and resets downgrade after every channel assignment', async () => {
    const updater = Object.assign(new EventEmitter(), {
      logger: console,
      autoDownload: true,
      autoInstallOnAppQuit: true,
      autoRunAppAfterInstall: false,
      disableWebInstaller: false,
      disableDifferentialDownload: false,
      requestHeaders: {},
      allowDowngrade: false,
      allowPrerelease: false,
      isUserWithinRollout: async () => true,
      netSession: {
        webRequest: {
          onBeforeRequest: () => undefined,
          onBeforeSendHeaders: () => undefined,
          onHeadersReceived: () => undefined,
        },
      },
      checkForUpdates: async () => null,
      downloadUpdate: async () => [],
      quitAndInstall: () => undefined,
    }) as unknown as AppUpdater & { channel?: string };
    Object.defineProperty(updater, 'channel', {
      set(value: string) {
        Object.defineProperty(this, 'channel', {
          value,
          writable: true,
          configurable: true,
        });
        this.allowDowngrade = true;
      },
      configurable: true,
    });
    const adapter = new ElectronUpdaterAdapter(
      base.toString(),
      {
        stagedRolloutQualified: false,
        expectedAppId: 'ai.ocno.oryn.browser',
        expectedTeamId: 'ABCDEFGHIJ',
        currentVersion: '1.0.0',
        currentSystemVersion: '26.6.2',
        verificationRoot: '/private/tmp/oryn-unused-verification',
      },
      updater,
    );
    adapter.configure('beta');
    expect(updater).toMatchObject({
      logger: null,
      autoDownload: false,
      autoInstallOnAppQuit: false,
      disableDifferentialDownload: true,
      channel: 'beta',
      allowDowngrade: false,
      allowPrerelease: true,
    });
    expect(
      await updater.isUserWithinRollout({
        stagingPercentage: 50,
      } as unknown as UpdateInfo),
    ).toBe(false);
  });

  it('allows only the selected metadata and explicitly authorized ZIP', () => {
    expect(
      isAllowedUpdateRequest(
        { url: `${base}latest-mac.yml?noCache=1abc`, method: 'GET' },
        base,
        'stable',
      ),
    ).toBe(true);
    expect(
      isAllowedUpdateRequest({ url: `${base}beta-mac.yml`, method: 'GET' }, base, 'stable'),
    ).toBe(false);
    expect(
      isAllowedUpdateRequest(
        { url: `${base}Oryn-Browser-2.0.0-arm64.zip`, method: 'GET' },
        base,
        'stable',
        { url: `${base}Oryn-Browser-2.0.0-arm64.zip`, size: 123 },
      ),
    ).toBe(true);
    for (const url of [
      'http://updates.example/oryn/macos/arm64/latest-mac.yml',
      'https://evil.example/latest-mac.yml',
      `${base}Oryn-Browser-2.0.0-arm64.zip?token=secret`,
      `${base}%2e%2e/secret.zip`,
    ]) {
      expect(isAllowedUpdateRequest({ url, method: 'GET' }, base, 'stable')).toBe(false);
    }
  });

  it('removes linkable and credential headers before every request', () => {
    const result = sanitizeUpdateRequestHeaders({
      'X-User-Staging-Id': 'unique-value',
      Cookie: 'private=1',
      Authorization: 'Bearer secret',
      Referer: 'https://private.example/',
      Accept: '*/*',
    });
    expect(result).toEqual({ Accept: '*/*', 'Accept-Encoding': 'identity' });
  });

  it('rejects redirects, encoded responses, and dishonest payload lengths', () => {
    expect(
      validateUpdateResponse({
        statusCode: 302,
        responseHeaders: { location: ['https://evil.example/'] },
        expectedSize: 100,
        payload: true,
      }).allowed,
    ).toBe(false);
    expect(
      validateUpdateResponse({
        statusCode: 200,
        responseHeaders: { 'content-length': ['99'] },
        expectedSize: 100,
        payload: true,
      }).allowed,
    ).toBe(false);
    expect(
      validateUpdateResponse({
        statusCode: 200,
        responseHeaders: {
          'content-length': ['100'],
          'content-encoding': ['gzip'],
        },
        expectedSize: 100,
        payload: true,
      }).allowed,
    ).toBe(false);
  });

  it('requires one same-origin ZIP plus the Oryn footprint declaration', () => {
    const info = {
      version: '2.0.0',
      files: [
        {
          url: 'Oryn-Browser-2.0.0-arm64.zip',
          sha512: Buffer.alloc(64, 1).toString('base64'),
          size: 123,
        },
      ],
      releaseDate: '2026-09-24T00:00:00.000Z',
      oryn: {
        schemaVersion: 2,
        bundleVersion: '2.0.0',
        platform: 'darwin',
        arch: 'arm64',
        unpackedSize: 456,
      },
    } as unknown as UpdateInfo;
    expect(candidateFromUpdateInfo(info, base)).toMatchObject({
      version: '2.0.0',
      sizeBytes: 123,
      unpackedSizeBytes: 456,
    });
    expect(() => candidateFromUpdateInfo({ ...info, oryn: undefined } as never, base)).toThrow(
      'APP_UPDATE_FOOTPRINT_METADATA_INVALID',
    );
  });

  it('rejects legacy metadata, extra authority fields and ambiguous build strings before download', () => {
    const { info } = adapterFixture();
    for (const oryn of [
      { ...info.oryn, schemaVersion: 1 },
      { ...info.oryn, bundleVersion: '2 00' },
      { ...info.oryn, unpackedSize: 33 * 1024 ** 3 },
      { ...info.oryn, bundleVersion: 'a'.repeat(65) },
      { ...info.oryn, expectedTeamId: 'ATTACKER00' },
    ])
      expect(() =>
        candidateFromUpdateInfo({ ...info, oryn } as unknown as UpdateInfo, base),
      ).toThrow('APP_UPDATE_FOOTPRINT_METADATA_INVALID');
  });

  it('accepts only production HTTPS feed roots', () => {
    expect(validateUpdateBaseUrl(base.toString())).toBe(base.toString());
    expect(() => validateUpdateBaseUrl('http://updates.example/')).toThrow();
    expect(() => validateUpdateBaseUrl('https://user:secret@updates.example/')).toThrow();
    expect(() => validateUpdateBaseUrl('https://localhost/releases/')).toThrow();
  });
});
