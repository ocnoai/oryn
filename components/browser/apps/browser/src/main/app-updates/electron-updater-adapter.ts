import electronUpdater, {
  CancellationToken,
  type AppUpdater,
  type ProgressInfo,
  type UpdateInfo,
} from 'electron-updater';
import { autoUpdater as nativeUpdater } from 'electron';
import { Server } from 'node:http';
import { monitorMacInstall } from './mac-install-handoff';
import { VerifiedMacHandoff } from './verified-mac-handoff';
import {
  validBundleVersion,
  validUpdatePayloadSizes,
  verifyUpdateArchive,
  type NativeVerificationPolicy,
  type VerifiedUpdateArchive,
} from './verified-update-archive';

import type { AppUpdateAdapter, AppUpdateCandidate, AppUpdateProgress } from './app-update-adapter';
import type { AppUpdateChannel } from './app-update-settings-store';

const NON_UNIQUE_STAGING_HEADER = 'oryn-no-staging-id';
const MAX_METADATA_BYTES = 256 * 1024;

interface AdapterOptions extends NativeVerificationPolicy {
  stagedRolloutQualified: boolean;
}

interface VerificationDependencies {
  verifyArchive: typeof verifyUpdateArchive;
  captureHandoff: (updater: AppUpdater, isCurrent: () => boolean) => VerifiedMacHandoff;
  nativeUpdater: typeof nativeUpdater;
}

interface AuthorizedPayload {
  url: string;
  size: number;
  sha512?: string;
}

interface InternalCandidate extends AppUpdateCandidate {
  payload: AuthorizedPayload;
}

export class ElectronUpdaterAdapter implements AppUpdateAdapter {
  private cancellationToken: CancellationToken | undefined;
  private progressListener: ((progress: ProgressInfo) => void) | undefined;
  private disposed = false;
  private generation = 0;
  private installFailed = false;
  private boundaryInstalled = false;
  private channel: AppUpdateChannel = 'stable';
  private candidate: InternalCandidate | undefined;
  private authorizedPayload: AuthorizedPayload | undefined;
  private verification: AbortController | undefined;
  private handoff: VerifiedMacHandoff | undefined;
  private readonly nativeRolloutPredicate: (info: UpdateInfo) => boolean | Promise<boolean>;

  constructor(
    private readonly baseUrl: string,
    private readonly options: AdapterOptions,
    private readonly updater: AppUpdater = electronUpdater.autoUpdater,
    private readonly verificationDependencies: VerificationDependencies = {
      verifyArchive: verifyUpdateArchive,
      captureHandoff: (value, current) =>
        new VerifiedMacHandoff(value as AppUpdater & { server?: Server }, current),
      nativeUpdater,
    },
  ) {
    this.nativeRolloutPredicate = this.updater.isUserWithinRollout.bind(this.updater);
    this.updater.logger = null;
    this.updater.autoDownload = false;
    this.updater.autoInstallOnAppQuit = false;
    this.updater.autoRunAppAfterInstall = true;
    this.updater.disableWebInstaller = true;
    this.updater.disableDifferentialDownload = true;
    this.updater.requestHeaders = {
      'x-user-staging-id': NON_UNIQUE_STAGING_HEADER,
      'Accept-Encoding': 'identity',
    };
    this.updater.isUserWithinRollout = (info) => this.isWithinRollout(info);
  }

  configure(channel: AppUpdateChannel): void {
    this.assertActive();
    if (!this.boundaryInstalled) this.installNetworkBoundary();
    this.channel = channel;
    this.generation += 1;
    this.revokeVerifiedPayload();
    this.candidate = undefined;
    this.authorizedPayload = undefined;
    this.updater.channel = channel === 'beta' ? 'beta' : 'latest';
    // electron-updater sets allowDowngrade=true whenever channel is assigned.
    this.updater.allowDowngrade = false;
    this.updater.allowPrerelease = channel === 'beta';
    this.updater.autoDownload = false;
    this.updater.autoInstallOnAppQuit = false;
    this.updater.disableDifferentialDownload = true;
  }

  async check(): Promise<AppUpdateCandidate | undefined> {
    this.assertSafePolicy();
    const generation = ++this.generation;
    this.revokeVerifiedPayload();
    this.candidate = undefined;
    this.authorizedPayload = undefined;
    const result = await this.updater.checkForUpdates();
    if (this.disposed || generation !== this.generation) throw new Error('APP_UPDATE_STALE_CHECK');
    if (!result?.isUpdateAvailable) {
      this.candidate = undefined;
      return undefined;
    }
    this.candidate = candidateFromUpdateInfo(result.updateInfo, new URL(this.baseUrl));
    return publicCandidate(this.candidate);
  }

  async download(onProgress: (progress: AppUpdateProgress) => void): Promise<void> {
    this.assertSafePolicy();
    if (!this.candidate) throw new Error('APP_UPDATE_CANDIDATE_MISSING');
    const candidate = this.candidate;
    const generation = this.generation;
    this.revokeVerifiedPayload();
    this.authorizedPayload = candidate.payload;
    const token = new CancellationToken();
    const verification = new AbortController();
    this.verification = verification;
    const isCurrent = () =>
      !this.disposed &&
      generation === this.generation &&
      this.candidate === candidate &&
      !token.cancelled &&
      !verification.signal.aborted;
    this.cancellationToken = token;
    let downloadedPath: string | undefined;
    let handoffError: Error | undefined;
    let verified: VerifiedUpdateArchive | undefined;
    const downloaded = (event: unknown) => {
      // This is emitted from MacUpdater's asynchronous server.listen callback.
      // Never throw across EventEmitter into that callback / Browser main.
      try {
        if (!isCurrent()) throw new Error('APP_UPDATE_CANCELLED');
        if (!isRecord(event) || typeof event.downloadedFile !== 'string' || downloadedPath)
          throw new Error('APP_UPDATE_NATIVE_HANDOFF_UNAVAILABLE');
        downloadedPath = event.downloadedFile;
        this.handoff = this.verificationDependencies.captureHandoff(this.updater, isCurrent);
      } catch {
        handoffError = new Error(
          isCurrent() ? 'APP_UPDATE_NATIVE_HANDOFF_UNAVAILABLE' : 'APP_UPDATE_CANCELLED',
        );
        try {
          this.revokeVerifiedPayload();
        } catch {
          /* Fail closed below even if cleanup failed. */
        }
        try {
          const server = (this.updater as AppUpdater & { server?: Server }).server;
          if (server instanceof Server) {
            server.removeAllListeners('request');
            server.on('request', (_request, response) => {
              response.statusCode = 503;
              response.end();
            });
            server.closeAllConnections();
            server.close();
          }
        } catch {
          /* No event callback may escape into server.listen. */
        }
      }
    };
    this.updater.on('update-downloaded', downloaded);
    this.removeProgressListener();
    this.progressListener = (progress) => {
      if (token.cancelled || this.disposed) return;
      onProgress({
        transferredBytes: progress.transferred,
        totalBytes: progress.total,
        bytesPerSecond: progress.bytesPerSecond,
      });
    };
    this.updater.on('download-progress', this.progressListener);
    try {
      const paths = await this.updater.downloadUpdate(token);
      // Cache hits and a request that has already completed may resolve after
      // cancellation. They must not become an authorized install.
      if (handoffError) throw handoffError;
      if (!isCurrent()) throw new Error('APP_UPDATE_CANCELLED');
      if (!downloadedPath || paths.length !== 1 || paths[0] !== downloadedPath || !this.handoff)
        throw new Error('APP_UPDATE_NATIVE_HANDOFF_UNAVAILABLE');
      verified = await this.verificationDependencies.verifyArchive(
        downloadedPath,
        {
          version: candidate.version,
          bundleVersion: candidate.bundleVersion!,
          size: candidate.payload.size,
          sha512: candidate.payload.sha512!,
          unpackedSize: candidate.unpackedSizeBytes!,
        },
        this.options,
        verification.signal,
      );
      if (!isCurrent()) {
        throw new Error('APP_UPDATE_CANCELLED');
      }
      this.handoff.attach(verified);
      verified = undefined; // Handoff now owns the retained read descriptor.
    } catch (error) {
      verified?.close();
      this.revokeVerifiedPayload();
      throw error;
    } finally {
      this.authorizedPayload = undefined;
      this.removeProgressListener();
      this.cancellationToken = undefined;
      this.updater.removeListener('update-downloaded', downloaded);
    }
  }

  cancelDownload(): void {
    this.authorizedPayload = undefined;
    this.cancellationToken?.cancel();
    this.revokeVerifiedPayload();
  }

  async install(): Promise<void> {
    this.assertSafePolicy();
    if (this.installFailed) throw new Error('APP_UPDATE_RESTART_REQUIRED');
    if (!this.handoff) throw new Error('APP_UPDATE_PAYLOAD_UNVERIFIED');
    try {
      await this.handoff.revalidate();
      await monitorMacInstall(this.updater, this.verificationDependencies.nativeUpdater, () =>
        this.updater.quitAndInstall(false, true),
      );
    } catch (error) {
      // Native staging has no cancellation API. A fresh process is required
      // before another handoff, even though ordinary browsing can recover now.
      this.installFailed = true;
      this.revokeVerifiedPayload();
      throw error;
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.generation += 1;
    this.cancelDownload();
    this.removeProgressListener();
    const webRequest = this.updater.netSession.webRequest;
    // The isolated session lives until process exit. Keep it closed while any
    // cancelled requests drain instead of removing the boundary underneath them.
    webRequest.onBeforeRequest({ urls: ['*://*/*'] }, (_details, callback) =>
      callback({ cancel: true }),
    );
    webRequest.onHeadersReceived({ urls: ['*://*/*'] }, (_details, callback) =>
      callback({ cancel: true }),
    );
  }

  private async isWithinRollout(info: UpdateInfo): Promise<boolean> {
    const percentage = info.stagingPercentage;
    if (percentage === undefined) return true;
    if (this.channel === 'beta' || !this.options.stagedRolloutQualified) return false;
    if (!Number.isInteger(percentage) || percentage < 0 || percentage > 100) return false;
    if (percentage === 0) return false;
    if (percentage === 100) return true;
    return this.nativeRolloutPredicate(info);
  }

  private installNetworkBoundary(): void {
    const allowedBase = new URL(this.baseUrl);
    const webRequest = this.updater.netSession.webRequest;
    webRequest.onBeforeRequest({ urls: ['*://*/*'] }, (details, callback) => {
      callback({
        cancel: !isAllowedUpdateRequest(
          { url: details.url, method: details.method },
          allowedBase,
          this.channel,
          this.authorizedPayload,
        ),
      });
    });
    webRequest.onBeforeSendHeaders({ urls: ['*://*/*'] }, (details, callback) => {
      callback({
        requestHeaders: sanitizeUpdateRequestHeaders(details.requestHeaders),
      });
    });
    webRequest.onHeadersReceived({ urls: ['*://*/*'] }, (details, callback) => {
      const payload = this.authorizedPayload?.url === canonicalUrl(details.url);
      const result = validateUpdateResponse({
        statusCode: details.statusCode,
        responseHeaders: details.responseHeaders,
        expectedSize: payload ? this.authorizedPayload!.size : MAX_METADATA_BYTES,
        payload,
      });
      callback(result.allowed ? { responseHeaders: result.responseHeaders } : { cancel: true });
    });
    this.boundaryInstalled = true;
  }

  private removeProgressListener(): void {
    if (!this.progressListener) return;
    this.updater.removeListener('download-progress', this.progressListener);
    this.progressListener = undefined;
  }

  private revokeVerifiedPayload(): void {
    this.verification?.abort();
    this.verification = undefined;
    this.handoff?.close();
    this.handoff = undefined;
  }

  private assertSafePolicy(): void {
    this.assertActive();
    if (this.updater.allowDowngrade) throw new Error('APP_UPDATE_DOWNGRADE_POLICY_VIOLATION');
    if (this.updater.autoDownload || this.updater.autoInstallOnAppQuit) {
      throw new Error('APP_UPDATE_CONSENT_POLICY_VIOLATION');
    }
  }

  private assertActive(): void {
    if (this.disposed) throw new Error('APP_UPDATE_ADAPTER_DISPOSED');
  }
}

export function isAllowedUpdateRequest(
  request: { url: string; method: string },
  allowedBase: URL,
  channel: AppUpdateChannel,
  payload?: AuthorizedPayload,
): boolean {
  if (request.method !== 'GET') return false;
  try {
    const candidate = new URL(request.url);
    if (!isSafeUpdateUrl(candidate, allowedBase)) return false;
    const metadata = new URL(`${channel === 'beta' ? 'beta' : 'latest'}-mac.yml`, allowedBase);
    if (candidate.pathname === metadata.pathname) {
      if (!candidate.search) return true;
      return (
        candidate.searchParams.size === 1 &&
        /^[0-9a-v]+$/u.test(candidate.searchParams.get('noCache') ?? '')
      );
    }
    return Boolean(
      payload &&
      canonicalUrl(candidate.toString()) === payload.url &&
      !candidate.search &&
      candidate.pathname.toLowerCase().endsWith('.zip'),
    );
  } catch {
    return false;
  }
}

export function isAllowedUpdateUrl(value: string, allowedBase: URL): boolean {
  try {
    return isSafeUpdateUrl(new URL(value), allowedBase);
  } catch {
    return false;
  }
}

export function sanitizeUpdateRequestHeaders(
  headers: Record<string, string>,
): Record<string, string> {
  const blocked = new Set([
    'x-user-staging-id',
    'cookie',
    'authorization',
    'proxy-authorization',
    'referer',
    'origin',
  ]);
  const sanitized = Object.fromEntries(
    Object.entries(headers).filter(([name]) => !blocked.has(name.toLowerCase())),
  );
  sanitized['Accept-Encoding'] = 'identity';
  return sanitized;
}

export function validateUpdateResponse(input: {
  statusCode: number;
  responseHeaders?: Record<string, string[]>;
  expectedSize: number;
  payload: boolean;
}): { allowed: boolean; responseHeaders?: Record<string, string[]> } {
  if (input.statusCode !== 200 || !input.responseHeaders) return { allowed: false };
  const lengthValues = headerValues(input.responseHeaders, 'content-length');
  const encodingValues = headerValues(input.responseHeaders, 'content-encoding');
  if (
    lengthValues.length !== 1 ||
    encodingValues.some((value) => value.toLowerCase() !== 'identity')
  ) {
    return { allowed: false };
  }
  const length = Number(lengthValues[0]);
  if (!Number.isSafeInteger(length) || length <= 0) return { allowed: false };
  if (
    (input.payload && length !== input.expectedSize) ||
    (!input.payload && length > input.expectedSize)
  ) {
    return { allowed: false };
  }
  const responseHeaders = Object.fromEntries(
    Object.entries(input.responseHeaders).filter(([name]) => name.toLowerCase() !== 'set-cookie'),
  );
  return { allowed: true, responseHeaders };
}

export function candidateFromUpdateInfo(info: UpdateInfo, allowedBase: URL): InternalCandidate {
  if (
    typeof info.version !== 'string' ||
    info.version.length > 128 ||
    !Array.isArray(info.files) ||
    info.files.length < 1 ||
    info.files.length > 16 ||
    info.files.some(
      (file) => !isRecord(file) || typeof file.url !== 'string' || file.url.length > 2048,
    )
  )
    throw new Error('APP_UPDATE_ZIP_METADATA_INVALID');
  const zipFiles = info.files.filter((file) => safePathname(file.url).endsWith('.zip'));
  if (zipFiles.length !== 1) throw new Error('APP_UPDATE_ZIP_COUNT_INVALID');
  const zip = zipFiles[0]!;
  const payloadUrl = canonicalUrl(new URL(zip.url, allowedBase).toString());
  if (!isAllowedUpdateUrl(payloadUrl, allowedBase) || new URL(payloadUrl).search) {
    throw new Error('APP_UPDATE_ZIP_URL_INVALID');
  }
  if (
    !positiveInteger(zip.size) ||
    typeof zip.sha512 !== 'string' ||
    Buffer.from(zip.sha512, 'base64').length !== 64 ||
    Buffer.from(zip.sha512, 'base64').toString('base64') !== zip.sha512
  ) {
    throw new Error('APP_UPDATE_ZIP_METADATA_INVALID');
  }
  const oryn = isRecord(info) ? info.oryn : undefined;
  if (
    !isRecord(oryn) ||
    oryn.schemaVersion !== 2 ||
    Object.keys(oryn).sort().join(',') !==
      'arch,bundleVersion,platform,schemaVersion,unpackedSize' ||
    !validBundleVersion(oryn.bundleVersion) ||
    oryn.platform !== 'darwin' ||
    oryn.arch !== 'arm64' ||
    !positiveInteger(oryn.unpackedSize) ||
    !validUpdatePayloadSizes(zip.size, oryn.unpackedSize)
  ) {
    throw new Error('APP_UPDATE_FOOTPRINT_METADATA_INVALID');
  }
  return {
    version: info.version,
    bundleVersion: oryn.bundleVersion,
    releaseName: typeof info.releaseName === 'string' ? info.releaseName : undefined,
    releaseNotes: normalizeReleaseNotes(info.releaseNotes),
    releaseDate: info.releaseDate,
    sizeBytes: zip.size,
    unpackedSizeBytes: oryn.unpackedSize,
    payload: { url: payloadUrl, size: zip.size, sha512: zip.sha512 },
  };
}

function publicCandidate(candidate: InternalCandidate): AppUpdateCandidate {
  return {
    version: candidate.version,
    bundleVersion: candidate.bundleVersion,
    releaseName: candidate.releaseName,
    releaseNotes: candidate.releaseNotes,
    releaseDate: candidate.releaseDate,
    sizeBytes: candidate.sizeBytes,
    unpackedSizeBytes: candidate.unpackedSizeBytes,
  };
}

function isSafeUpdateUrl(candidate: URL, allowedBase: URL): boolean {
  const raw = candidate.pathname.toLowerCase();
  return (
    candidate.protocol === 'https:' &&
    !candidate.username &&
    !candidate.password &&
    !candidate.hash &&
    candidate.origin === allowedBase.origin &&
    candidate.pathname.startsWith(allowedBase.pathname) &&
    !/%(?:2e|2f|5c)/u.test(raw)
  );
}

function canonicalUrl(value: string): string {
  const url = new URL(value);
  url.hash = '';
  return url.toString();
}

function headerValues(headers: Record<string, string[]>, name: string): string[] {
  return Object.entries(headers)
    .filter(([key]) => key.toLowerCase() === name)
    .flatMap(([, values]) => values)
    .flatMap((value) => value.split(',').map((part) => part.trim()))
    .filter(Boolean);
}

function normalizeReleaseNotes(value: UpdateInfo['releaseNotes']): string | undefined {
  if (typeof value === 'string') return value.slice(0, 8_000);
  if (!Array.isArray(value)) return undefined;
  if (value.length > 16) throw new Error('APP_UPDATE_NOTES_METADATA_INVALID');
  return value
    .map((item) =>
      typeof item.version === 'string' &&
      item.version.length <= 128 &&
      typeof item.note === 'string'
        ? `${item.version}: ${item.note.slice(0, 8_000)}`
        : '',
    )
    .filter(Boolean)
    .join('\n')
    .slice(0, 8_000);
}

function safePathname(value: string): string {
  try {
    return new URL(value, 'https://updates.invalid/').pathname.toLowerCase();
  } catch {
    return '';
  }
}

function positiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
