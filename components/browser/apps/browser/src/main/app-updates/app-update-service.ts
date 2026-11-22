import { EventEmitter } from 'node:events';
import { randomUUID } from 'node:crypto';

import type { AppUpdateSnapshot } from '../../shared/ipc';
import type { AppUpdateAdapter, AppUpdateCandidate, AppUpdateProgress } from './app-update-adapter';
import type {
  AppUpdateChannel,
  AppUpdateSettingsData,
  AppUpdateSettingsStore,
} from './app-update-settings-store';

const CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000;
const CHECK_JITTER_MS = 5 * 60 * 1000;
const INITIAL_CHECK_DELAY_MS = 5_000;
const REMIND_LATER_MS = 24 * 60 * 60 * 1000;
const RETRY_DELAYS_MS = [60_000, 5 * 60_000, 15 * 60_000] as const;
const INSTALL_SPACE_RESERVE_BYTES = 1024 * 1024 * 1024;
type ApprovedCandidate = AppUpdateCandidate & { approvalId: string };

interface AppUpdateServiceOptions {
  enabled: boolean;
  currentVersion: string;
  settings: AppUpdateSettingsStore;
  adapter?: AppUpdateAdapter;
  availableDiskBytes: () => number[];
  prepareInstall: () => Promise<{ accepted: boolean; reason?: string }>;
  recoverInstall: () => Promise<void>;
  now?: () => number;
  random?: () => number;
  setTimer?: (callback: () => void, delayMs: number) => ReturnType<typeof setTimeout>;
  clearTimer?: (timer: ReturnType<typeof setTimeout>) => void;
}

export class AppUpdateService extends EventEmitter {
  private readonly now: () => number;
  private readonly random: () => number;
  private readonly setTimer: NonNullable<AppUpdateServiceOptions['setTimer']>;
  private readonly clearTimer: NonNullable<AppUpdateServiceOptions['clearTimer']>;
  private state: AppUpdateSnapshot;
  private candidate: ApprovedCandidate | undefined;
  private checkPromise: Promise<AppUpdateSnapshot> | undefined;
  private downloadPromise: Promise<AppUpdateSnapshot> | undefined;
  private restartPromise: Promise<AppUpdateSnapshot> | undefined;
  private generation = 0;
  private checkGeneration = 0;
  private downloadCancelled = false;
  private stopped = false;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private started = false;
  private consecutiveFailures = 0;

  constructor(private readonly options: AppUpdateServiceOptions) {
    super();
    this.now = options.now ?? Date.now;
    this.random = options.random ?? Math.random;
    this.setTimer = options.setTimer ?? setTimeout;
    this.clearTimer = options.clearTimer ?? clearTimeout;
    const settings = options.settings.read();
    this.state = {
      status: options.enabled ? 'idle' : 'unavailable',
      enabled: options.enabled,
      currentVersion: options.currentVersion,
      channel: settings.channel,
      automaticChecks: settings.automaticChecks,
      lastCheckAt: settings.lastCheckAt,
      unavailableReason: options.enabled
        ? undefined
        : 'Updates are unavailable in this development or unsigned build. Install the first signed Oryn release manually to enable updates.',
    };
  }

  snapshot(): AppUpdateSnapshot {
    return structuredClone(this.state);
  }

  start(): void {
    if (this.started || this.stopped || !this.options.enabled || !this.options.adapter) return;
    this.started = true;
    this.options.adapter.configure(this.state.channel);
    if (this.state.automaticChecks) this.schedule(INITIAL_CHECK_DELAY_MS);
  }

  stop(): void {
    this.stopped = true;
    this.generation += 1;
    this.downloadCancelled = true;
    this.started = false;
    if (this.timer) this.clearTimer(this.timer);
    this.timer = undefined;
    this.options.adapter?.dispose();
  }

  check(manual = true): Promise<AppUpdateSnapshot> {
    if (this.stopped || !this.options.enabled || !this.options.adapter)
      return Promise.resolve(this.snapshot());
    if (
      this.downloadPromise ||
      this.restartPromise ||
      ['downloading', 'ready', 'installing'].includes(this.state.status)
    )
      return Promise.resolve(this.snapshot());
    if (this.checkPromise) {
      // The library coalesces checks. Drain the old channel's request before
      // starting a fresh one, and never reuse its result for a new channel.
      return this.checkGeneration === this.generation
        ? this.checkPromise
        : this.checkPromise.then(() => this.check(manual));
    }
    const generation = this.generation;
    this.checkGeneration = generation;
    this.clearScheduledCheck();
    this.candidate = undefined;
    this.publish({
      status: 'checking',
      candidate: undefined,
      progress: undefined,
      error: undefined,
      restartBlockingReason: undefined,
    });
    this.checkPromise = this.options.adapter
      .check()
      .then((candidate) => {
        if (generation !== this.generation || this.stopped) return this.snapshot();
        this.consecutiveFailures = 0;
        const checkedAt = this.now();
        const settings = this.options.settings.update({
          lastCheckAt: checkedAt,
        });
        this.candidate = candidate ? normalizeCandidate(candidate) : undefined;
        if (this.candidate && !this.candidate.bundleVersion)
          throw new Error('APP_UPDATE_FOOTPRINT_METADATA_INVALID');
        if (
          this.candidate &&
          (!isStrictlyNewerVersion(this.candidate.version, this.state.currentVersion) ||
            (this.state.channel === 'stable' &&
              Boolean(parseSemver(this.candidate.version)?.prerelease.length)))
        ) {
          this.candidate = undefined;
        }
        if (!this.candidate) {
          this.publish({
            status: 'idle',
            candidate: undefined,
            progress: undefined,
            error: undefined,
            lastCheckAt: checkedAt,
          });
          return this.snapshot();
        }
        if (!this.candidate.sizeBytes || !this.candidate.unpackedSizeBytes) {
          this.publish({
            status: 'error',
            candidate: publicCandidate(this.candidate),
            lastCheckAt: checkedAt,
            error: {
              code: 'UPDATE_SIZE_UNAVAILABLE',
              message: 'This release does not advertise a verified download size.',
            },
          });
          return this.snapshot();
        }
        const skipped = isSkipped(settings, this.candidate.version);
        const deferred = isDeferred(settings, this.candidate.version, checkedAt);
        this.publish({
          status: skipped ? 'skipped' : deferred ? 'deferred' : 'available',
          candidate: publicCandidate(this.candidate),
          progress: undefined,
          error: undefined,
          lastCheckAt: checkedAt,
        });
        return this.snapshot();
      })
      .catch((error: unknown) => {
        if (generation !== this.generation || this.stopped) return this.snapshot();
        this.consecutiveFailures += 1;
        this.publish({
          status: 'error',
          error: publicError(error, 'UPDATE_CHECK_FAILED', 'Oryn could not check for updates.'),
        });
        return this.snapshot();
      })
      .finally(() => {
        this.checkPromise = undefined;
        if (this.started && this.state.automaticChecks) {
          this.schedule(generation === this.generation ? this.nextCheckDelay() : 0);
        }
      });
    if (manual) this.emit('manual-check');
    return this.checkPromise;
  }

  download(version: string, allowSkipped = false, approvalId?: string): Promise<AppUpdateSnapshot> {
    this.requireApproval(version, approvalId);
    if (this.downloadPromise) return this.downloadPromise;
    this.assertNotInstalling();
    if (this.state.status === 'ready') return Promise.resolve(this.snapshot());
    const adapter = this.requireAdapter();
    const candidate = this.requireCandidate(version);
    if (!candidate.sizeBytes || !candidate.unpackedSizeBytes) {
      throw new Error('UPDATE_SIZE_UNAVAILABLE');
    }
    const settings = this.options.settings.read();
    if (isSkipped(settings, version) && !allowSkipped) throw new Error('UPDATE_VERSION_SKIPPED');
    if (allowSkipped && isSkipped(settings, version)) {
      this.options.settings.update({ skipped: undefined });
    }
    if (isDeferred(settings, version, this.now())) {
      this.options.settings.update({ deferred: undefined });
    }
    const availableVolumes = this.availableVolumeBytes();
    const availableDiskBytes = availableVolumes.length ? Math.min(...availableVolumes) : 0;
    const requiredDiskBytes = requiredFreeBytes(candidate);
    if (!Number.isSafeInteger(requiredDiskBytes) || availableDiskBytes < requiredDiskBytes) {
      this.publish({
        status: 'error',
        candidate: publicCandidate(candidate),
        error: {
          code: 'UPDATE_DISK_SPACE_LOW',
          message: 'There is not enough free disk space to safely download and stage this update.',
          requiredDiskBytes,
          availableDiskBytes,
        },
      });
      return Promise.resolve(this.snapshot());
    }
    this.clearScheduledCheck();
    this.downloadCancelled = false;
    const generation = this.generation;
    this.publish({
      status: 'downloading',
      candidate: publicCandidate(candidate),
      progress: {
        transferredBytes: 0,
        totalBytes: candidate.sizeBytes,
        percent: 0,
      },
      error: undefined,
    });
    this.downloadPromise = adapter
      .download((progress) => {
        if (!this.downloadCancelled && generation === this.generation)
          this.updateProgress(progress, candidate);
      })
      .then(() => {
        if (generation !== this.generation || this.stopped) return this.snapshot();
        if (this.downloadCancelled) {
          this.publish({
            status: 'cancelled',
            progress: undefined,
            error: undefined,
          });
          return this.snapshot();
        }
        this.publish({
          status: 'ready',
          progress: {
            transferredBytes: candidate.sizeBytes!,
            totalBytes: candidate.sizeBytes!,
            percent: 100,
          },
        });
        return this.snapshot();
      })
      .catch((error: unknown) => {
        if (generation !== this.generation || this.stopped) return this.snapshot();
        const cancelled = this.downloadCancelled || isCancellation(error);
        this.publish({
          status: cancelled ? 'cancelled' : 'error',
          progress: undefined,
          error: cancelled
            ? undefined
            : publicError(error, 'UPDATE_DOWNLOAD_FAILED', 'The update download failed.'),
        });
        return this.snapshot();
      })
      .finally(() => {
        this.downloadPromise = undefined;
        if (this.started && this.state.automaticChecks && this.state.status !== 'ready')
          this.schedule(this.nextCheckDelay());
      });
    return this.downloadPromise;
  }

  cancelDownload(): AppUpdateSnapshot {
    if (this.state.status === 'downloading') {
      this.downloadCancelled = true;
      this.options.adapter?.cancelDownload();
    }
    return this.snapshot();
  }

  defer(version: string): AppUpdateSnapshot {
    this.assertNotTransferring();
    this.requireCandidate(version);
    if (this.state.status === 'ready') return this.snapshot();
    this.options.settings.update({
      deferred: {
        channel: this.state.channel,
        version,
        until: this.now() + REMIND_LATER_MS,
      },
    });
    this.publish({ status: 'deferred', error: undefined });
    return this.snapshot();
  }

  skip(version: string): AppUpdateSnapshot {
    this.assertNotTransferring();
    this.requireCandidate(version);
    this.options.settings.update({
      skipped: { channel: this.state.channel, version },
      deferred: undefined,
    });
    this.publish({ status: 'skipped', error: undefined });
    return this.snapshot();
  }

  clearSkip(): AppUpdateSnapshot {
    this.assertNotTransferring();
    this.options.settings.update({ skipped: undefined });
    if (this.candidate && this.state.status === 'skipped')
      this.publish({ status: 'available', error: undefined });
    return this.snapshot();
  }

  updateSettings(value: {
    automaticChecks?: boolean;
    channel?: AppUpdateChannel;
  }): AppUpdateSnapshot {
    this.assertNotInstalling();
    if (
      (value.automaticChecks !== undefined && typeof value.automaticChecks !== 'boolean') ||
      (value.channel !== undefined && value.channel !== 'stable' && value.channel !== 'beta')
    ) {
      throw new Error('INVALID_APP_UPDATE_SETTINGS');
    }
    const previous = this.options.settings.read();
    const channelChanged = value.channel !== undefined && value.channel !== previous.channel;
    if (channelChanged && (this.downloadPromise || this.state.status === 'downloading')) {
      throw new Error('UPDATE_CHANNEL_CHANGE_DURING_DOWNLOAD');
    }
    const next = this.options.settings.update({
      ...value,
      ...(channelChanged ? { skipped: undefined, deferred: undefined } : {}),
    });
    if (channelChanged) {
      this.generation += 1;
      this.candidate = undefined;
      this.options.adapter?.configure(next.channel);
    }
    this.publish({
      status: channelChanged ? 'idle' : this.state.status,
      channel: next.channel,
      automaticChecks: next.automaticChecks,
      ...(channelChanged ? { candidate: undefined, progress: undefined, error: undefined } : {}),
    });
    this.clearScheduledCheck();
    if (
      this.started &&
      next.automaticChecks &&
      !['ready', 'downloading'].includes(this.state.status)
    )
      this.schedule(channelChanged ? 0 : CHECK_INTERVAL_MS);
    return this.snapshot();
  }

  restart(version: string, approvalId?: string): Promise<AppUpdateSnapshot> {
    this.requireApproval(version, approvalId);
    if (this.restartPromise) return this.restartPromise;
    this.restartPromise = this.performRestart(version).finally(() => {
      this.restartPromise = undefined;
    });
    return this.restartPromise;
  }

  private async performRestart(version: string): Promise<AppUpdateSnapshot> {
    const candidate = this.requireCandidate(version);
    if (this.state.status !== 'ready') throw new Error('UPDATE_NOT_READY');
    const required = requiredFreeBytes(candidate);
    const available = this.availableVolumeBytes();
    if (!Number.isSafeInteger(required) || !available.length || Math.min(...available) < required) {
      this.publish({
        restartBlockingReason: 'There is not enough free disk space to safely install this update.',
      });
      return this.snapshot();
    }
    this.clearScheduledCheck();
    this.publish({
      status: 'installing',
      restartBlockingReason: undefined,
      error: undefined,
    });
    try {
      const result = await this.options.prepareInstall();
      if (!result.accepted) {
        this.publish({
          status: 'ready',
          restartBlockingReason: boundedText(
            result.reason ?? 'Current work must finish first.',
            512,
          ),
        });
        return this.snapshot();
      }
      await this.requireAdapter().install();
    } catch {
      let recovered = true;
      try {
        await this.options.recoverInstall();
      } catch {
        recovered = false;
      }
      this.candidate = undefined;
      this.publish({
        status: 'error',
        candidate: undefined,
        progress: undefined,
        error: {
          code: recovered ? 'UPDATE_INSTALL_FAILED' : 'UPDATE_RECOVERY_FAILED',
          message: recovered
            ? 'The update could not be installed. Oryn is still usable. Restart Oryn before trying another update, or use the signed manual installer.'
            : 'The update could not be installed and Local AI could not restart. Restart Oryn to restore Local AI.',
        },
      });
    }
    return this.snapshot();
  }

  private requireAdapter(): AppUpdateAdapter {
    if (this.stopped || !this.options.enabled || !this.options.adapter)
      throw new Error('UPDATES_UNAVAILABLE');
    return this.options.adapter;
  }

  private availableVolumeBytes(): number[] {
    try {
      const volumes = this.options.availableDiskBytes();
      return volumes.every((value) => Number.isSafeInteger(value) && value >= 0) ? volumes : [];
    } catch {
      return [];
    }
  }

  private assertNotInstalling(): void {
    if (this.restartPromise || this.state.status === 'installing')
      throw new Error('UPDATE_INSTALL_IN_PROGRESS');
  }

  private assertNotTransferring(): void {
    this.assertNotInstalling();
    if (this.downloadPromise || this.state.status === 'downloading')
      throw new Error('UPDATE_DOWNLOAD_IN_PROGRESS');
  }

  private requireCandidate(version: string): ApprovedCandidate {
    if (!this.candidate || version !== this.candidate.version)
      throw new Error('STALE_UPDATE_CANDIDATE');
    return this.candidate;
  }

  private requireApproval(version: string, approvalId: string | undefined): void {
    this.requireCandidate(version);
    if (!approvalId || approvalId !== this.candidate?.approvalId)
      throw new Error('STALE_UPDATE_APPROVAL');
  }

  private updateProgress(progress: AppUpdateProgress, candidate: AppUpdateCandidate): void {
    if (this.state.status !== 'downloading') return;
    const totalBytes = positiveInteger(progress.totalBytes) ?? candidate.sizeBytes ?? 0;
    const transferredBytes = Math.min(totalBytes, positiveInteger(progress.transferredBytes) ?? 0);
    this.publish({
      progress: {
        transferredBytes,
        totalBytes,
        percent: totalBytes > 0 ? Math.min(100, (transferredBytes / totalBytes) * 100) : 0,
        bytesPerSecond: positiveInteger(progress.bytesPerSecond),
      },
    });
  }

  private publish(value: Partial<AppUpdateSnapshot>): void {
    this.state = { ...this.state, ...value };
    this.emit('changed', this.snapshot());
  }

  private clearScheduledCheck(): void {
    if (this.timer) this.clearTimer(this.timer);
    this.timer = undefined;
  }

  private schedule(delayMs: number): void {
    this.clearScheduledCheck();
    this.timer = this.setTimer(
      () => {
        this.timer = undefined;
        void this.check(false);
      },
      Math.max(0, delayMs),
    );
    this.timer.unref?.();
  }

  private nextCheckDelay(): number {
    if (this.consecutiveFailures > 0) {
      return RETRY_DELAYS_MS[Math.min(this.consecutiveFailures - 1, RETRY_DELAYS_MS.length - 1)]!;
    }
    return CHECK_INTERVAL_MS + Math.round((this.random() * 2 - 1) * CHECK_JITTER_MS);
  }
}

function normalizeCandidate(candidate: AppUpdateCandidate): ApprovedCandidate {
  return {
    version: boundedText(candidate.version, 128),
    bundleVersion:
      typeof candidate.bundleVersion === 'string' &&
      /^[\x21-\x7e]{1,64}$/u.test(candidate.bundleVersion)
        ? candidate.bundleVersion
        : undefined,
    approvalId: randomUUID(),
    releaseName: boundedOptionalText(candidate.releaseName, 256),
    releaseNotes: boundedOptionalText(candidate.releaseNotes, 8_000),
    releaseDate: boundedOptionalText(candidate.releaseDate, 64),
    sizeBytes: positiveInteger(candidate.sizeBytes),
    unpackedSizeBytes: positiveInteger(candidate.unpackedSizeBytes),
  };
}

function publicCandidate(
  candidate: ApprovedCandidate,
): NonNullable<AppUpdateSnapshot['candidate']> {
  return {
    version: candidate.version,
    bundleVersion: candidate.bundleVersion,
    approvalId: candidate.approvalId,
    releaseName: candidate.releaseName,
    releaseNotes: candidate.releaseNotes,
    releaseDate: candidate.releaseDate,
    sizeBytes: candidate.sizeBytes!,
    unpackedSizeBytes: candidate.unpackedSizeBytes!,
    requiredFreeBytes: requiredFreeBytes(candidate),
  };
}

function requiredFreeBytes(candidate: AppUpdateCandidate): number {
  return (
    (candidate.sizeBytes ?? 0) * 3 +
    (candidate.unpackedSizeBytes ?? 0) * 3 +
    INSTALL_SPACE_RESERVE_BYTES
  );
}

function isSkipped(settings: AppUpdateSettingsData, version: string): boolean {
  return settings.skipped?.channel === settings.channel && settings.skipped.version === version;
}

function isDeferred(settings: AppUpdateSettingsData, version: string, now: number): boolean {
  return (
    settings.deferred?.channel === settings.channel &&
    settings.deferred.version === version &&
    settings.deferred.until > now
  );
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : undefined;
}

function boundedOptionalText(value: unknown, maximum: number): string | undefined {
  return typeof value === 'string' && value.trim() ? boundedText(value, maximum) : undefined;
}

function boundedText(value: string, maximum: number): string {
  return [...value]
    .filter((character) => {
      const code = character.codePointAt(0) ?? 0;
      return code === 9 || code === 10 || code === 13 || (code >= 32 && code !== 127);
    })
    .join('')
    .slice(0, maximum);
}

function publicError(
  error: unknown,
  code: string,
  fallback: string,
): NonNullable<AppUpdateSnapshot['error']> {
  const verificationErrors: Record<string, string> = {
    APP_UPDATE_ARCHIVE_INVALID: 'The downloaded update archive is invalid.',
    APP_UPDATE_SIGNATURE_INVALID: 'The downloaded update has an invalid application signature.',
    APP_UPDATE_NOTARIZATION_INVALID: 'The downloaded update did not pass macOS trust checks.',
    APP_UPDATE_IDENTITY_MISMATCH: 'The downloaded update is for a different application.',
    APP_UPDATE_VERSION_MISMATCH: 'The downloaded update does not match the release you approved.',
    APP_UPDATE_ARCH_MISMATCH: 'The downloaded update is incompatible with this Mac.',
    APP_UPDATE_OS_UNSUPPORTED: 'This update requires a newer version of macOS.',
    APP_UPDATE_PAYLOAD_CHANGED: 'The downloaded update changed. Check for updates again.',
    APP_UPDATE_VERIFICATION_UNAVAILABLE:
      'Oryn could not verify this update. Use the signed manual installer.',
    APP_UPDATE_NATIVE_HANDOFF_UNAVAILABLE:
      'Oryn could not safely stage this update. Use the signed manual installer.',
    APP_UPDATE_DISK_SPACE_LOW: 'There is not enough free disk space to verify this update.',
  };
  if (error instanceof Error && Object.hasOwn(verificationErrors, error.message))
    return { code: error.message, message: verificationErrors[error.message]! };
  return { code, message: boundedText(fallback, 512) };
}

function isCancellation(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.name === 'CancellationError' || /cancel(?:led|ed)/iu.test(error.message))
  );
}

export function isStrictlyNewerVersion(candidate: string, current: string): boolean {
  const left = parseSemver(candidate);
  const right = parseSemver(current);
  if (!left || !right) return false;
  for (let index = 0; index < 3; index += 1) {
    if (left.core[index]! !== right.core[index]!) return left.core[index]! > right.core[index]!;
  }
  if (!left.prerelease.length && right.prerelease.length) return true;
  if (left.prerelease.length && !right.prerelease.length) return false;
  for (
    let index = 0;
    index < Math.max(left.prerelease.length, right.prerelease.length);
    index += 1
  ) {
    const leftPart = left.prerelease[index];
    const rightPart = right.prerelease[index];
    if (leftPart === undefined) return false;
    if (rightPart === undefined) return true;
    if (leftPart === rightPart) continue;
    const leftNumber = /^\d+$/u.test(leftPart) ? Number(leftPart) : undefined;
    const rightNumber = /^\d+$/u.test(rightPart) ? Number(rightPart) : undefined;
    if (leftNumber !== undefined && rightNumber !== undefined) return leftNumber > rightNumber;
    if (leftNumber !== undefined) return false;
    if (rightNumber !== undefined) return true;
    return leftPart > rightPart;
  }
  return false;
}

function parseSemver(
  value: string,
): { core: [number, number, number]; prerelease: string[] } | undefined {
  const match =
    /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$/u.exec(
      value,
    );
  if (!match) return undefined;
  const core = [Number(match[1]), Number(match[2]), Number(match[3])] as [number, number, number];
  if (core.some((part) => !Number.isSafeInteger(part))) return undefined;
  const prerelease = match[4]?.split('.') ?? [];
  if (
    prerelease.some(
      (part) => !part || (/^\d+$/u.test(part) && part.length > 1 && part.startsWith('0')),
    )
  ) {
    return undefined;
  }
  return { core, prerelease };
}
