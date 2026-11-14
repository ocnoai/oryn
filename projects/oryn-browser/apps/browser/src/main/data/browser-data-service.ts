import { join, isAbsolute } from 'node:path';
import { JsonFileStore, isRecord } from './json-file-store';
import { SiteZoomStore } from './site-zoom-store';
import {
  dialog,
  shell,
  systemPreferences,
  type Session,
  type DownloadItem,
  type WebContents,
} from 'electron';

import { BookmarkStore } from './bookmark-store';
import { DownloadStore, type DownloadState } from './download-store';
import { BrowsingHistoryStore } from './history-store';
import { ProfileDataPaths } from './profile-data-paths';
import { BrowserSessionStore } from './session-store';
import { isSupportedSitePermission, SitePermissionStore } from './site-permission-store';
import { AIActivityStore } from './ai-activity-store';
import { OrynSettingsStore } from '../settings/oryn-settings-store';
import { BrowserAppearanceStore } from './appearance-store';
import { WindowPlacementStore } from './window-placement-store';
import { ModelWorkspaceStore } from '../model-workspace/model-workspace-store';
import { BrowserExperienceStore } from './browser-experience-store';
import { AppUpdateSettingsStore } from '../app-updates/app-update-settings-store';
import {
  resolvePermissionFrame,
  permitsEmbeddedPermission,
  type PermissionFrameContext,
} from './permission-frame-policy';
import { CaptureAccessRegistry } from './capture-access-registry';
import { DownloadSafety, DownloadSafetyError } from '../download-safety';

export class BrowserDataService {
  readonly downloadPreferences: JsonFileStore<{ directory?: string }>;
  readonly zoom: SiteZoomStore;
  readonly paths: ProfileDataPaths;
  readonly history: BrowsingHistoryStore;
  readonly bookmarks: BookmarkStore;
  readonly downloads: DownloadStore;
  readonly sessions: BrowserSessionStore;
  readonly sitePermissions: SitePermissionStore;
  readonly aiActivity: AIActivityStore;
  readonly orynSettings: OrynSettingsStore;
  readonly appearance: BrowserAppearanceStore;
  readonly windowPlacement: WindowPlacementStore;
  readonly modelWorkspaces: ModelWorkspaceStore;
  readonly experience: BrowserExperienceStore;
  readonly appUpdates: AppUpdateSettingsStore;
  private readonly pagePermissions = new Map<WebContents, Set<string>>();
  private readonly captureAccess = new CaptureAccessRegistry();
  private readonly stoppedCapture = new WeakSet<WebContents>();
  private readonly permissionEpochs = new WeakMap<WebContents, number>();
  private readonly watchedPermissionContents = new WeakSet<WebContents>();
  private readonly permissionPrompts = new WeakSet<WebContents>();
  private readonly liveDownloads = new Map<string, DownloadItem>();
  private readonly downloadSessions = new Map<string, Session>();
  private readonly sessionPartitions = new WeakMap<Session, string>();
  private readonly installedSessions = new WeakSet<Session>();
  private readonly downloadChecks = new Map<string, Promise<void>>();
  private readonly openingDownloads = new Set<string>();
  private downloadCheckQueue: Promise<void> = Promise.resolve();

  constructor(
    profileRoot: string,
    private readonly downloadSafety = new DownloadSafety(),
  ) {
    this.paths = new ProfileDataPaths(profileRoot);
    this.downloadPreferences = new JsonFileStore(
      join(this.paths.browserCoreRoot, 'download-preferences.json'),
      () => ({}),
      (value): value is { directory?: string } =>
        isRecord(value) &&
        (value.directory === undefined ||
          (typeof value.directory === 'string' && isAbsolute(value.directory))),
    );
    this.zoom = new SiteZoomStore(join(this.paths.browserCoreRoot, 'site-zoom.json'));
    this.history = new BrowsingHistoryStore(this.paths.history);
    this.bookmarks = new BookmarkStore(this.paths.bookmarks);
    this.downloads = new DownloadStore(this.paths.downloads);
    this.sessions = new BrowserSessionStore(this.paths.session);
    this.sitePermissions = new SitePermissionStore(this.paths.sitePermissions);
    this.aiActivity = new AIActivityStore(this.paths.aiActivity);
    this.orynSettings = new OrynSettingsStore(this.paths.localAISettings);
    this.appearance = new BrowserAppearanceStore(this.paths.appearance);
    this.windowPlacement = new WindowPlacementStore(this.paths.windowPlacement);
    this.modelWorkspaces = new ModelWorkspaceStore(this.paths.modelWorkspaces);
    this.experience = new BrowserExperienceStore(this.paths.browserExperience);
    this.appUpdates = new AppUpdateSettingsStore(this.paths.appUpdates);
    for (const record of this.downloads.list())
      if (record.state === 'completed')
        void this.checkDownloadSafety(record.downloadId).catch(() => undefined);
  }

  installChromiumSessionHandlers(session: Session, partition = 'default'): void {
    if (this.installedSessions.has(session)) {
      return;
    }
    this.installedSessions.add(session);
    this.sessionPartitions.set(session, partition);
    this.downloadSessions.set(partition, session);
    // macOS 15+ owns source selection and its capture indicator. Older platforms deny;
    // never silently choose the first display or persist a display-capture grant.
    session.setDisplayMediaRequestHandler?.((_request, callback) => callback({}), {
      useSystemPicker: true,
    });

    session.setPermissionRequestHandler((contents, permission, callback, details) => {
      this.watchPermissionContents(contents);
      const context = resolvePermissionFrame(contents, session, details);
      const requested =
        permission === 'media'
          ? 'mediaTypes' in details
            ? [
                ...new Set(
                  details.mediaTypes?.map((type) => (type === 'video' ? 'camera' : 'microphone')) ??
                    [],
                ),
              ]
            : []
          : [permission];
      if (
        !context ||
        !requested.length ||
        !isSupportedSitePermission(permission) ||
        ['popups', 'sound'].includes(permission) ||
        requested.some((key) => !permitsEmbeddedPermission(context, key))
      ) {
        callback(false);
        return;
      }
      if (requested.some(isCapturePermission) && this.stoppedCapture.has(contents)) {
        callback(false);
        return;
      }
      if (this.osMediaDenied(requested)) {
        callback(false);
        this.explainMediaDenial(requested);
        return;
      }
      const saved = requested.map((key) => this.savedPermission(context, key, partition));
      if (saved.includes('deny')) {
        callback(false);
        return;
      }
      const accept = () => {
        for (const key of requested) this.captureAccess.record(contents, partition, context, key);
        callback(true);
      };
      // The system picker owns source consent. Its grant is never persisted. Tracking
      // access here is conservative because macOS may bypass the display handler.
      if (permission === 'display-capture' && !context.embedded) {
        accept();
        return;
      }
      if (
        requested.every((key) => this.hasPagePermission(contents, context, key)) ||
        saved.every((decision) => decision === 'allow') ||
        ['clipboard-sanitized-write', 'fullscreen', 'pointerLock'].includes(permission)
      ) {
        accept();
        return;
      }
      if (this.permissionPrompts.has(contents)) {
        callback(false);
        return;
      }
      this.permissionPrompts.add(contents);
      const epoch = this.permissionEpochs.get(contents) ?? 0;
      let current = true;
      const invalidate = () => {
        current = false;
      };
      contents.once('did-start-navigation', invalidate);
      contents.once('destroyed', invalidate);
      void dialog
        .showMessageBox({
          type: 'question',
          title: 'Site permission',
          message: context.embedded
            ? `${context.origin} embedded in ${context.topOrigin} requests ${requested.join(' and ')}`
            : `${context.origin} requests ${requested.join(' and ')}`,
          detail: context.embedded
            ? 'This permission applies only to this embedded site in this page and session.'
            : partition === 'default'
              ? 'Allow for this request, remember for this site, or block.'
              : 'This permission applies only to this isolated browser session.',
          buttons:
            permission === 'display-capture'
              ? ['Block', 'Allow for this page']
              : ['Block', 'Allow for this page', 'Always allow'],
          defaultId: 0,
          cancelId: 0,
          noLink: true,
        })
        .then(
          ({ response }) => {
            const latest = resolvePermissionFrame(contents, session, details);
            const valid =
              current &&
              latest?.identity === context.identity &&
              (this.permissionEpochs.get(contents) ?? 0) === epoch &&
              !(requested.some(isCapturePermission) && this.stoppedCapture.has(contents));
            if (!valid || ![1, 2].includes(response)) {
              callback(false);
              return;
            }
            if (response === 2 && permission !== 'display-capture') {
              for (const key of requested) {
                if (context.embedded)
                  this.sitePermissions.setEmbedded(
                    context.topOrigin,
                    context.origin,
                    key,
                    'allow',
                    context.ancestorOrigins,
                    partition,
                  );
                else this.sitePermissions.set(context.origin, key, 'allow', partition);
              }
            } else {
              let grants = this.pagePermissions.get(contents);
              if (!grants) {
                grants = new Set();
                this.pagePermissions.set(contents, grants);
              }
              for (const key of requested)
                grants.add(this.pagePermissionKey(contents, context, key));
            }
            accept();
          },
          () => callback(false),
        )
        .finally(() => {
          this.permissionPrompts.delete(contents);
          contents.removeListener('did-start-navigation', invalidate);
          contents.removeListener('destroyed', invalidate);
        });
    });
    session.setPermissionCheckHandler((contents, permission, requestingOrigin, details) => {
      if (contents) this.watchPermissionContents(contents);
      const context = resolvePermissionFrame(contents, session, details, requestingOrigin);
      const key =
        permission === 'media'
          ? details.mediaType === 'video'
            ? 'camera'
            : details.mediaType === 'audio'
              ? 'microphone'
              : undefined
          : permission;
      if (
        !context ||
        !contents ||
        !key ||
        !isSupportedSitePermission(key) ||
        ['popups', 'sound', 'display-capture'].includes(permission) ||
        !permitsEmbeddedPermission(context, key) ||
        (isCapturePermission(key) && this.stoppedCapture.has(contents)) ||
        this.osMediaDenied([key])
      )
        return false;
      const saved = this.savedPermission(context, key, partition);
      const granted =
        saved !== 'deny' &&
        (['clipboard-sanitized-write', 'fullscreen', 'pointerLock'].includes(key) ||
          this.hasPagePermission(contents, context, key) ||
          saved === 'allow');
      if (granted) this.captureAccess.record(contents, partition, context, key);
      return granted;
    });

    session.on('will-download', (_event, item) => {
      const directory = this.downloadPreferences.read().directory;
      if (directory) session.setDownloadPath(directory);
      const record = this.downloads.start({
        partition,
        url: item.getURL(),
        filename: item.getFilename(),
        savePath: item.getSavePath() || undefined,
        totalBytes: item.getTotalBytes(),
      });
      this.liveDownloads.set(record.downloadId, item);
      item.on('updated', (_downloadEvent, state) => {
        this.downloads.update(record.downloadId, {
          receivedBytes: item.getReceivedBytes(),
          totalBytes: item.getTotalBytes(),
          savePath: item.getSavePath() || undefined,
          state: state === 'interrupted' || item.isPaused() ? 'paused' : 'progressing',
        });
      });
      item.once('done', (_downloadEvent, state) => {
        this.liveDownloads.delete(record.downloadId);
        this.downloads.update(record.downloadId, {
          receivedBytes: item.getReceivedBytes(),
          totalBytes: item.getTotalBytes(),
          savePath: item.getSavePath() || undefined,
          state: state as DownloadState,
          openState: 'pending',
          openError: undefined,
          fileIdentity: undefined,
        });
        if (state === 'completed')
          void this.checkDownloadSafety(record.downloadId).catch(() => undefined);
      });
    });
  }

  private watchPermissionContents(contents: WebContents): void {
    if (this.watchedPermissionContents.has(contents)) return;
    this.watchedPermissionContents.add(contents);
    const invalidate = () => {
      this.permissionEpochs.set(contents, (this.permissionEpochs.get(contents) ?? 0) + 1);
      this.pagePermissions.delete(contents);
    };
    contents.on('did-frame-navigate', invalidate);
    contents.on('did-navigate', invalidate);
    contents.once('destroyed', invalidate);
  }

  private pagePermissionKey(
    contents: WebContents,
    context: PermissionFrameContext,
    permission: string,
  ): string {
    return `${context.topOrigin}|${context.origin}:${permission}|${this.permissionEpochs.get(contents) ?? 0}|${context.identity}`;
  }

  private hasPagePermission(
    contents: WebContents,
    context: PermissionFrameContext,
    permission: string,
  ): boolean {
    return (
      this.pagePermissions
        .get(contents)
        ?.has(this.pagePermissionKey(contents, context, permission)) === true
    );
  }

  private savedPermission(context: PermissionFrameContext, permission: string, partition: string) {
    return context.embedded
      ? this.sitePermissions.getEmbedded(
          context.topOrigin,
          context.origin,
          permission,
          context.ancestorOrigins,
          partition,
        )
      : this.sitePermissions.get(context.origin, permission, partition);
  }

  private osMediaDenied(permissions: string[]): boolean {
    if (process.platform !== 'darwin') return false;
    return permissions.some((permission) => {
      if (permission !== 'camera' && permission !== 'microphone') return false;
      const status = systemPreferences?.getMediaAccessStatus?.(permission);
      return status === 'denied' || status === 'restricted';
    });
  }

  private explainMediaDenial(permissions: string[]): void {
    const permission = permissions.find((key) => key === 'camera' || key === 'microphone');
    if (!permission) return;
    void dialog
      .showMessageBox({
        type: 'info',
        title: 'macOS privacy permission',
        message: `macOS is blocking ${permission} access for Oryn.`,
        detail: `Allow Oryn in System Settings → Privacy & Security → ${permission === 'camera' ? 'Camera' : 'Microphone'}, then try again.`,
        buttons: ['Cancel', 'Open System Settings'],
        defaultId: 0,
        cancelId: 0,
        noLink: true,
      })
      .then(({ response }) => {
        if (response === 1)
          void shell.openExternal(
            `x-apple.systempreferences:com.apple.preference.security?Privacy_${permission === 'camera' ? 'Camera' : 'Microphone'}`,
          );
      })
      .catch(() => undefined);
  }

  subscribeCaptureAccess(listener: () => void): () => void {
    return this.captureAccess.subscribe(listener);
  }
  captureSnapshot() {
    return this.captureAccess.snapshot();
  }

  requestStopCapture(webContentsId: number): WebContents | undefined {
    const contents = this.captureAccess.markStopping(webContentsId);
    if (!contents) return undefined;
    this.stoppedCapture.add(contents);
    this.pagePermissions.delete(contents);
    this.permissionEpochs.set(contents, (this.permissionEpochs.get(contents) ?? 0) + 1);
    const partition = this.partitionFor(contents.session);
    for (const record of this.captureSnapshot().filter(
      (entry) => entry.webContentsId === webContentsId,
    )) {
      for (const permission of ['camera', 'microphone', 'display-capture']) {
        if (record.origin === record.topOrigin)
          this.sitePermissions.clear(record.origin, partition, permission);
        else
          this.sitePermissions.clearEmbedded(
            record.topOrigin,
            record.origin,
            permission,
            partition,
          );
      }
    }
    // Root owns normal close/reload. A canceled before-unload must leave this indicator.
    // This WebContents cannot reacquire until explicitly reset after actual replacement.
    return contents;
  }

  allowFreshCaptureConsent(contents: WebContents): void {
    this.stoppedCapture.delete(contents);
  }

  listEmbeddedPermissions(topOrigin: string, partition = 'default') {
    return this.sitePermissions.listEmbedded(new URL(topOrigin).origin, partition);
  }

  revokeEmbeddedPermission(
    topOrigin: string,
    embeddedOrigin: string,
    permission: string,
    partition = 'default',
  ): void {
    this.sitePermissions.clearEmbedded(
      new URL(topOrigin).origin,
      new URL(embeddedOrigin).origin,
      permission,
      partition,
    );
    this.clearSitePagePermissions(embeddedOrigin, partition, permission);
  }

  clearPagePermission(contents: WebContents, permission: string): void {
    const prefix = `${safeOrigin(contents.getURL())}|${safeOrigin(contents.getURL())}:${permission}|`;
    const grants = this.pagePermissions.get(contents);
    if (grants) for (const key of grants) if (key.startsWith(prefix)) grants.delete(key);
  }

  clearSitePagePermissions(origin: string, partition: string, permission: string): void {
    const prefix = `${new URL(origin).origin}:${permission}|`;
    for (const [contents, grants] of this.pagePermissions) {
      if (!contents.isDestroyed() && this.sessionPartitions.get(contents.session) === partition) {
        for (const key of grants)
          if (key.slice(key.indexOf('|') + 1).startsWith(prefix)) grants.delete(key);
      }
    }
  }

  partitionFor(session: Session): string {
    const partition = this.sessionPartitions.get(session);
    if (!partition) throw new Error('SITE_PARTITION_UNAVAILABLE');
    return partition;
  }

  hasActiveDownloads(): boolean {
    return this.liveDownloads.size > 0;
  }

  private checkDownloadSafety(id: string, freshReview = false): Promise<void> {
    const existing = this.downloadChecks.get(id);
    if (existing) return existing;
    const record = this.downloads.list().find((entry) => entry.downloadId === id);
    if (!record || record.state !== 'completed') return Promise.resolve();
    this.downloads.update(id, { openState: 'pending', openError: undefined });
    const operation = this.downloadCheckQueue
      .then(async () => {
        try {
          const identity = await this.downloadSafety.check(
            freshReview ? { ...record, fileIdentity: undefined } : record,
          );
          const current = this.downloads.list().find((entry) => entry.downloadId === id);
          if (current?.state === 'completed' && current.savePath === record.savePath)
            this.downloads.update(id, {
              openState: 'ready',
              openError: undefined,
              fileIdentity: identity,
            });
        } catch (error) {
          if (this.downloads.list().some((entry) => entry.downloadId === id))
            this.downloads.update(id, {
              openState: 'error',
              openError: error instanceof DownloadSafetyError ? error.reason : 'quarantine',
              fileIdentity: undefined,
            });
        }
      })
      .finally(() => this.downloadChecks.delete(id));
    this.downloadChecks.set(id, operation);
    this.downloadCheckQueue = operation.catch(() => undefined);
    return operation;
  }

  async controlDownload(id: string, action: string): Promise<void> {
    const item = this.liveDownloads.get(id);
    const record = this.downloads.list().find((entry) => entry.downloadId === id);
    if (!record) throw new Error('DOWNLOAD_NOT_FOUND');
    switch (action) {
      case 'pause':
        if (!item) throw new Error('DOWNLOAD_NOT_ACTIVE');
        item.pause();
        this.downloads.update(id, { state: 'paused' });
        break;
      case 'resume':
        if (!item || !item.canResume()) throw new Error('DOWNLOAD_CANNOT_RESUME');
        item.resume();
        this.downloads.update(id, { state: 'progressing' });
        break;
      case 'cancel':
        if (!item) throw new Error('DOWNLOAD_NOT_ACTIVE');
        item.cancel();
        break;
      case 'retry':
        if (item || !['http:', 'https:'].includes(new URL(record.url).protocol))
          throw new Error('DOWNLOAD_CANNOT_RETRY');
        {
          const target = this.downloadSessions.get(record.partition ?? 'default');
          if (!target)
            throw new Error('Open the original sidebar app before retrying this download.');
          target.downloadURL(record.url);
        }
        break;
      case 'remove':
        if (item) throw new Error('DOWNLOAD_STILL_ACTIVE');
        this.downloads.remove(id);
        break;
      case 'reveal':
        if (!record.savePath) throw new Error('DOWNLOAD_FILE_UNAVAILABLE');
        shell.showItemInFolder(record.savePath);
        break;
      case 'open':
        if (record.state !== 'completed' || !record.savePath)
          throw new Error('DOWNLOAD_FILE_UNAVAILABLE');
        if (record.openState !== 'ready' || !record.fileIdentity)
          throw new Error('Check the downloaded file before opening it.');
        if (this.openingDownloads.has(id)) throw new Error('This file is already being reviewed.');
        this.openingDownloads.add(id);
        this.downloads.update(id, { openState: 'pending', openError: undefined });
        try {
          await this.downloadSafety.open(record, () => {
            const current = this.downloads.list().find((entry) => entry.downloadId === id);
            return (
              current?.state === 'completed' &&
              current.savePath === record.savePath &&
              JSON.stringify(current.fileIdentity) === JSON.stringify(record.fileIdentity)
            );
          });
          if (this.downloads.list().some((entry) => entry.downloadId === id))
            this.downloads.update(id, { openState: 'ready', openError: undefined });
        } catch (error) {
          if (this.downloads.list().some((entry) => entry.downloadId === id))
            this.downloads.update(id, {
              openState: 'error',
              openError: error instanceof DownloadSafetyError ? error.reason : 'open-failed',
            });
          throw new Error(
            'The downloaded file could not be opened. Check its status and try again.',
          );
        } finally {
          this.openingDownloads.delete(id);
        }
        break;
      case 'check':
        if (record.state !== 'completed') throw new Error('DOWNLOAD_FILE_UNAVAILABLE');
        if (this.openingDownloads.has(id)) throw new Error('This file is already being reviewed.');
        await this.checkDownloadSafety(id, true);
        break;
      default:
        throw new Error('INVALID_DOWNLOAD_ACTION');
    }
  }

  async clearSiteData(session: Session, origin: string): Promise<void> {
    const parsed = new URL(origin);
    await session.clearStorageData({ origin: parsed.origin });
    const cookies = await session.cookies.get({ domain: parsed.hostname });
    await Promise.all(
      cookies.map((cookie) =>
        session.cookies.remove(
          `${cookie.secure ? 'https' : 'http'}://${parsed.hostname}${cookie.path}`,
          cookie.name,
        ),
      ),
    );
    this.sitePermissions.clear(parsed.origin, this.partitionFor(session));
    for (const [contents, grants] of this.pagePermissions) {
      if (!contents.isDestroyed() && contents.session === session) {
        for (const key of grants)
          if (
            key.startsWith(`${parsed.origin}|`) ||
            key.slice(key.indexOf('|') + 1).startsWith(`${parsed.origin}:`)
          )
            grants.delete(key);
      }
    }
  }
}

function safeOrigin(value: string): string | undefined {
  try {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) ? url.origin : undefined;
  } catch {
    return undefined;
  }
}

function isCapturePermission(permission: string): boolean {
  return ['camera', 'microphone', 'display-capture'].includes(permission);
}
