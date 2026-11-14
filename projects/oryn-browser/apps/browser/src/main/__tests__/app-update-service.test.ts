import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it, vi } from 'vitest';

import type {
  AppUpdateAdapter,
  AppUpdateCandidate,
  AppUpdateProgress,
} from '../app-updates/app-update-adapter';
import { AppUpdateService, isStrictlyNewerVersion } from '../app-updates/app-update-service';
import { AppUpdateSettingsStore } from '../app-updates/app-update-settings-store';

const candidate: AppUpdateCandidate = {
  version: '2.0.0',
  bundleVersion: '2.0.0',
  releaseName: 'Oryn 2',
  releaseNotes: '<script>plain text only</script>',
  sizeBytes: 100,
  unpackedSizeBytes: 200,
};

class FakeAdapter implements AppUpdateAdapter {
  configured: string[] = [];
  checks = 0;
  downloads = 0;
  cancelled = 0;
  installed = 0;
  nextCandidate: AppUpdateCandidate | undefined = candidate;

  configure(channel: 'stable' | 'beta') {
    this.configured.push(channel);
  }
  async check() {
    this.checks += 1;
    return this.nextCandidate;
  }
  async download(onProgress: (progress: AppUpdateProgress) => void) {
    this.downloads += 1;
    onProgress({ transferredBytes: 50, totalBytes: 100 });
  }
  cancelDownload() {
    this.cancelled += 1;
  }
  async install() {
    this.installed += 1;
  }
  dispose() {}
}

const temporaryDirectories: string[] = [];
afterEach(() => {
  vi.useRealTimers();
  for (const directory of temporaryDirectories.splice(0))
    rmSync(directory, { recursive: true, force: true });
});

function fixture(options?: { disk?: number; volumes?: number[]; installAccepted?: boolean }) {
  const adapter = new FakeAdapter();
  const directory = mkdtempSync(join(tmpdir(), 'oryn-app-updates-'));
  temporaryDirectories.push(directory);
  const settings = new AppUpdateSettingsStore(join(directory, 'app-updates.json'));
  const prepareInstall = vi.fn(async () => ({
    accepted: options?.installAccepted ?? true,
    reason: 'Protected work is active.',
  }));
  const recoverInstall = vi.fn(async () => undefined);
  const service = new AppUpdateService({
    enabled: true,
    currentVersion: '1.0.0',
    settings,
    adapter,
    availableDiskBytes: () => options?.volumes ?? [options?.disk ?? 4 * 1024 ** 3],
    prepareInstall,
    recoverInstall,
  });
  return { adapter, service, settings, prepareInstall, recoverInstall };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

describe('AppUpdateService', () => {
  it('requires an exact ephemeral offer approval for download and restart, including same-version build changes', async () => {
    const { service, adapter } = fixture();
    await service.check();
    const first = service.snapshot().candidate!.approvalId!;
    expect(() => service.download(candidate.version)).toThrow('STALE_UPDATE_APPROVAL');
    adapter.nextCandidate = { ...candidate, bundleVersion: '200-new' };
    await service.check();
    const second = service.snapshot().candidate!.approvalId!;
    expect(second).not.toBe(first);
    expect(() => service.download(candidate.version, false, first)).toThrow(
      'STALE_UPDATE_APPROVAL',
    );
    expect(adapter.downloads).toBe(0);
    await service.download(candidate.version, false, second);
    expect(service.snapshot().candidate!.approvalId).toBe(second);
    expect(() => service.restart(candidate.version, first)).toThrow('STALE_UPDATE_APPROVAL');
    expect(adapter.installed).toBe(0);
    await service.restart(candidate.version, second);
    expect(adapter.installed).toBe(1);
  });
  it('enforces monotonic semantic versions, including beta-to-stable transitions', () => {
    expect(isStrictlyNewerVersion('2.0.0-beta.2', '2.0.0-beta.1')).toBe(true);
    expect(isStrictlyNewerVersion('2.0.0', '2.0.0-beta.2')).toBe(true);
    expect(isStrictlyNewerVersion('1.9.0', '2.0.0-beta.2')).toBe(false);
    expect(isStrictlyNewerVersion('2.0.0-beta.1', '2.0.0')).toBe(false);
  });
  it('checks without downloading and requires explicit approval for the exact version', async () => {
    const { adapter, service } = fixture();
    await service.check();
    expect(service.snapshot()).toMatchObject({
      status: 'available',
      candidate: { version: '2.0.0' },
    });
    expect(adapter.downloads).toBe(0);
    expect(() => service.download('1.9.0')).toThrow('STALE_UPDATE_CANDIDATE');
    await service.download('2.0.0', false, service.snapshot().candidate?.approvalId);
    expect(adapter.downloads).toBe(1);
    expect(service.snapshot()).toMatchObject({
      status: 'ready',
      progress: { percent: 100 },
    });
  });

  it('persists an exact channel/version skip and offers a newer release', async () => {
    const { adapter, service, settings } = fixture();
    await service.check();
    service.skip('2.0.0');
    expect(settings.read().skipped).toEqual({
      channel: 'stable',
      version: '2.0.0',
    });
    await service.check(false);
    expect(service.snapshot().status).toBe('skipped');
    adapter.nextCandidate = { ...candidate, version: '2.0.1' };
    await service.check(false);
    expect(service.snapshot().status).toBe('available');
  });

  it('persists Remind me later without authorizing a payload', async () => {
    const { adapter, service, settings } = fixture();
    await service.check();
    service.defer('2.0.0');
    expect(adapter.downloads).toBe(0);
    expect(settings.read().deferred).toMatchObject({
      channel: 'stable',
      version: '2.0.0',
    });
    await service.check(false);
    expect(service.snapshot().status).toBe('deferred');
  });

  it('blocks a payload before contact when conservative disk admission fails', async () => {
    const { adapter, service } = fixture({ disk: 1024 });
    await service.check();
    await service.download('2.0.0', false, service.snapshot().candidate?.approvalId);
    expect(adapter.downloads).toBe(0);
    expect(service.snapshot()).toMatchObject({
      status: 'error',
      error: { code: 'UPDATE_DISK_SPACE_LOW', availableDiskBytes: 1024 },
    });
  });

  it('never installs when restart admission is rejected', async () => {
    const { adapter, service } = fixture({ installAccepted: false });
    await service.check();
    await service.download('2.0.0', false, service.snapshot().candidate?.approvalId);
    await service.restart('2.0.0', service.snapshot().candidate?.approvalId);
    expect(adapter.installed).toBe(0);
    expect(service.snapshot()).toMatchObject({
      status: 'ready',
      restartBlockingReason: 'Protected work is active.',
    });
  });

  it('clears skips and candidates when changing channel without enabling downgrades itself', async () => {
    const { adapter, service } = fixture();
    service.start();
    await service.check();
    service.skip('2.0.0');
    const state = service.updateSettings({ channel: 'beta' });
    expect(adapter.configured).toEqual(['stable', 'beta']);
    expect(state).toMatchObject({
      channel: 'beta',
      status: 'idle',
      candidate: undefined,
    });
    service.stop();
  });

  it('drains an old beta check and checks stable without publishing the beta result', async () => {
    const { adapter, service } = fixture();
    const beta = deferred<AppUpdateCandidate | undefined>();
    const check = vi
      .spyOn(adapter, 'check')
      .mockImplementationOnce(() => beta.promise)
      .mockResolvedValue(candidate);
    service.updateSettings({ channel: 'beta' });
    const oldCheck = service.check();
    service.updateSettings({ channel: 'stable' });
    const states: string[] = [];
    service.on('changed', (state) => {
      if (state.candidate) states.push(state.candidate.version);
    });
    const stableCheck = service.check();
    expect(check).toHaveBeenCalledTimes(1);
    beta.resolve({ ...candidate, version: '2.0.0-beta.1' });
    await Promise.all([oldCheck, stableCheck]);
    expect(check).toHaveBeenCalledTimes(2);
    expect(states).toEqual(['2.0.0']);
    expect(service.snapshot()).toMatchObject({
      channel: 'stable',
      status: 'available',
    });
  });

  it('rejects a prerelease accidentally advertised in the stable feed', async () => {
    const { adapter, service } = fixture();
    adapter.nextCandidate = { ...candidate, version: '2.0.0-beta.1' };
    await service.check();
    expect(service.snapshot()).toMatchObject({
      status: 'idle',
      candidate: undefined,
    });
  });

  it('preserves a downloaded candidate across manual, wake and scheduled checks', async () => {
    vi.useFakeTimers();
    const { adapter, service } = fixture();
    service.start();
    await service.check();
    await service.download(candidate.version, false, service.snapshot().candidate?.approvalId);
    await service.check(false);
    await service.check(true);
    await vi.advanceTimersByTimeAsync(24 * 60 * 60 * 1000);
    expect(adapter.checks).toBe(1);
    expect(service.snapshot()).toMatchObject({
      status: 'ready',
      candidate: { version: candidate.version },
    });
    service.stop();
  });

  it('keeps cancellation authoritative when completion wins, then permits a deliberate retry', async () => {
    const { adapter, service } = fixture();
    const transfer = deferred<void>();
    vi.spyOn(adapter, 'download').mockImplementationOnce(() => transfer.promise);
    await service.check();
    const pending = service.download(
      candidate.version,
      false,
      service.snapshot().candidate?.approvalId,
    );
    service.cancelDownload();
    expect(() => service.updateSettings({ channel: 'beta' })).toThrow(
      'UPDATE_CHANNEL_CHANGE_DURING_DOWNLOAD',
    );
    transfer.resolve();
    await pending;
    expect(service.snapshot().status).toBe('cancelled');
    await expect(
      service.restart(candidate.version, service.snapshot().candidate?.approvalId),
    ).rejects.toThrow('UPDATE_NOT_READY');
    await service.download(candidate.version, false, service.snapshot().candidate?.approvalId);
    expect(service.snapshot().status).toBe('ready');
  });

  it.each([[0, 4 * 1024 ** 3], [NaN, 4 * 1024 ** 3], [-1, 4 * 1024 ** 3], []])(
    'rejects all disk admission when any required volume is full or unknown: %j',
    async (...volumes) => {
      const { adapter, service } = fixture({ volumes });
      await service.check();
      await service.download(candidate.version, false, service.snapshot().candidate?.approvalId);
      expect(adapter.downloads).toBe(0);
      expect(service.snapshot().error?.code).toBe('UPDATE_DISK_SPACE_LOW');
    },
  );

  it('rechecks every disk before restart', async () => {
    const volumes = [4 * 1024 ** 3, 4 * 1024 ** 3];
    const { adapter, service, prepareInstall } = fixture({ volumes });
    await service.check();
    await service.download(candidate.version, false, service.snapshot().candidate?.approvalId);
    volumes[0] = 0;
    await service.restart(candidate.version, service.snapshot().candidate?.approvalId);
    expect(prepareInstall).not.toHaveBeenCalled();
    expect(adapter.installed).toBe(0);
  });

  it('serializes restart and blocks channel/check changes throughout preparation', async () => {
    const { adapter, service, prepareInstall } = fixture();
    await service.check();
    await service.download(candidate.version, false, service.snapshot().candidate?.approvalId);
    const preparation = deferred<{ accepted: boolean; reason: string }>();
    prepareInstall.mockImplementationOnce(() => preparation.promise);
    const first = service.restart(candidate.version, service.snapshot().candidate?.approvalId);
    expect(service.restart(candidate.version, service.snapshot().candidate?.approvalId)).toBe(
      first,
    );
    expect(() => service.updateSettings({ channel: 'beta' })).toThrow('UPDATE_INSTALL_IN_PROGRESS');
    await service.check();
    expect(adapter.checks).toBe(1);
    preparation.resolve({ accepted: true, reason: '' });
    await first;
    expect(adapter.installed).toBe(1);
  });

  it('recovers after native installation fails and allows another metadata check', async () => {
    const { adapter, service, recoverInstall } = fixture();
    vi.spyOn(adapter, 'install').mockRejectedValue(new Error('private native details'));
    await service.check();
    await service.download(candidate.version, false, service.snapshot().candidate?.approvalId);
    await service.restart(candidate.version, service.snapshot().candidate?.approvalId);
    expect(recoverInstall).toHaveBeenCalledOnce();
    expect(service.snapshot()).toMatchObject({
      status: 'error',
      candidate: undefined,
      error: { code: 'UPDATE_INSTALL_FAILED' },
    });
    expect(JSON.stringify(service.snapshot())).not.toContain('private native details');
    await service.check();
    expect(service.snapshot().status).toBe('available');
  });

  it('recovers a partial shutdown and reports a recovery failure truthfully', async () => {
    const { service, prepareInstall, recoverInstall } = fixture();
    prepareInstall.mockRejectedValue(new Error('shutdown failed'));
    recoverInstall.mockRejectedValue(new Error('engine failed'));
    await service.check();
    await service.download(candidate.version, false, service.snapshot().candidate?.approvalId);
    await service.restart(candidate.version, service.snapshot().candidate?.approvalId);
    expect(service.snapshot().error?.code).toBe('UPDATE_RECOVERY_FAILED');
  });

  it('seeds beta only on the first launch and preserves later user preferences', () => {
    const { settings } = fixture();
    settings.initializeChannel('beta');
    expect(settings.read().channel).toBe('beta');
    settings.update({ channel: 'stable' });
    settings.initializeChannel('beta');
    expect(settings.read().channel).toBe('stable');
  });
});
