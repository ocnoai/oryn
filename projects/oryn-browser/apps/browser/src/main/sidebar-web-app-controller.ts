import { requestPageClose } from './page-close';
import { installBrowserRequestGate } from './insecure-form-guard';
import { EventEmitter } from 'node:events';
import { boundedObservationReady } from './observation-readiness';

import { app, session, WebContentsView, type BrowserWindow } from 'electron';

import type { BrowserDataService } from './data/browser-data-service';
import {
  isExternalApplicationNavigation,
  type ExternalApplicationHandoff,
} from './external-application-handoff';

export class SidebarWebAppController extends EventEmitter {
  private readonly views = new Map<string, WebContentsView>();
  private activeAppId: string | null = null;
  private readonly observationReady = new WeakMap<WebContentsView, Promise<void>>();

  constructor(
    private readonly window: BrowserWindow,
    private readonly browserData: BrowserDataService,
    private readonly openExternalInTab: (url: string) => void,
    private readonly onContentsCreated?: (contents: Electron.WebContents) => void | Promise<void>,
    private readonly externalApplications?: ExternalApplicationHandoff,
  ) {
    super();
    this.activeAppId = browserData.experience.read().sidebar.activeAppId;
  }

  open(appId?: string): void {
    const next = appId ?? null;
    if (this.activeAppId) this.views.get(this.activeAppId)?.setVisible(false);
    this.activeAppId = next;
    this.browserData.experience.updateSidebar({ activeAppId: next });
    if (!next) {
      this.emit('changed');
      return;
    }
    const config = this.browserData.experience
      .read()
      .sidebar.apps.find((app) => app.appId === next);
    if (!config || !config.enabled) throw new Error('SIDEBAR_APP_NOT_FOUND');
    let view = this.views.get(next);
    if (!view) {
      view = this.createView(config);
      this.views.set(next, view);
      this.window.contentView.addChildView(view);
      const created = view;
      const load = () => {
        if (
          this.window.isDestroyed() ||
          this.views.get(next) !== created ||
          created.webContents.isDestroyed()
        )
          return;
        void created.webContents.loadURL(config.url).catch(() => undefined);
      };
      const ready = this.observationReady.get(created);
      if (ready && !created.webContents.getURL())
        void created.webContents.loadURL('about:blank').catch(() => undefined);
      if (ready) void ready.then(load);
      else load();
    }
    view.webContents.setAudioMuted(config.muted);
    view.setVisible(true);
    this.emit('changed');
  }

  setMuted(appId: string, muted: boolean): void {
    this.browserData.experience.updateSidebarApp(appId, { muted });
    this.views.get(appId)?.webContents.setAudioMuted(muted);
    this.emit('changed');
  }

  async clearData(appId: string): Promise<void> {
    const config = this.browserData.experience
      .read()
      .sidebar.apps.find((app) => app.appId === appId);
    if (!config) throw new Error('SIDEBAR_APP_NOT_FOUND');
    const existing = this.views.get(appId);
    const isolatedSession =
      existing?.webContents.session ?? session.fromPartition(`persist:oryn-sidebar-${appId}`);
    if (existing) {
      if (!(await requestPageClose(existing.webContents, this.window))) return;
      this.views.delete(appId);
      this.window.contentView.removeChildView(existing);
    }
    if (this.activeAppId === appId) {
      this.activeAppId = null;
      this.browserData.experience.updateSidebar({ activeAppId: null });
    }
    await Promise.all([isolatedSession.clearStorageData(), isolatedSession.clearCache()]);
    this.browserData.sitePermissions.clear(undefined, `persist:oryn-sidebar-${appId}`);
    this.emit('changed');
  }

  setPanelVisible(visible: boolean): void {
    if (this.activeAppId) this.views.get(this.activeAppId)?.setVisible(visible);
  }

  layout(bounds: { x: number; y: number; width: number; height: number }): void {
    if (!this.activeAppId) return;
    const view = this.views.get(this.activeAppId);
    view?.setBounds(bounds);
    if (view) this.window.contentView.addChildView(view);
  }

  raise(): void {
    if (!this.activeAppId) return;
    const view = this.views.get(this.activeAppId);
    if (view) this.window.contentView.addChildView(view);
  }

  hasOpenPages(): boolean {
    return this.views.size > 0;
  }

  async requestClose(): Promise<boolean> {
    for (const [id, view] of this.views) {
      if (!(await requestPageClose(view.webContents, this.window))) return false;
      this.views.delete(id);
    }
    this.activeAppId = null;
    return true;
  }

  destroy(): void {
    for (const view of this.views.values()) {
      const contents = view.webContents;
      if (contents && !contents.isDestroyed()) contents.close();
    }
    this.views.clear();
  }

  private createView(config: { appId: string; allowedHosts: string[] }): WebContentsView {
    const partition = `persist:oryn-sidebar-${config.appId}`;
    const isolatedSession = session.fromPartition(partition, { cache: true });
    installBrowserRequestGate(isolatedSession);
    isolatedSession.setUserAgent(
      chromiumCompatibleSidebarUserAgent(app.userAgentFallback, app.getName()),
    );
    this.browserData.installChromiumSessionHandlers(isolatedSession, partition);
    const view = new WebContentsView({
      webPreferences: {
        partition,
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
        webSecurity: true,
        spellcheck: true,
      },
    });
    view.setBackgroundColor('#101315');
    const unregisterExternal = this.externalApplications?.register(
      view.webContents,
      () => this.views.get(config.appId) === view,
    );
    const ready = boundedObservationReady(this.onContentsCreated?.(view.webContents));
    if (ready) this.observationReady.set(view, ready);
    view.webContents.once('destroyed', () => {
      unregisterExternal?.();
      if (this.views.get(config.appId) !== view) return;
      this.views.delete(config.appId);
      if (this.activeAppId === config.appId) {
        this.activeAppId = null;
        this.browserData.experience.updateSidebar({ activeAppId: null });
      }
      this.emit('changed');
    });
    const isAllowed = (url: string) => isAllowedSidebarNavigation(url, config.allowedHosts);
    const openExternal = (value: string) => {
      try {
        const url = new URL(value);
        if (
          value.length <= 8192 &&
          ['http:', 'https:'].includes(url.protocol) &&
          !url.username &&
          !url.password
        )
          this.openExternalInTab(url.toString());
      } catch {
        /* Unsupported destinations stay inside the blocked navigation flow. */
      }
    };
    view.webContents.setWindowOpenHandler(({ url }) => {
      // Popup details expose no initiating frame; non-web popups stay denied.
      openExternal(url);
      return { action: 'deny' };
    });
    view.webContents.on('will-frame-navigate', (details) => {
      if (!isExternalApplicationNavigation(details.url)) return;
      details.preventDefault();
      void this.externalApplications?.requestNavigation(view.webContents, details);
    });
    view.webContents.on('will-navigate', (event, url) => {
      if (!isAllowed(url)) {
        event.preventDefault();
        openExternal(url);
      }
    });
    view.webContents.on('will-redirect', (event, url) => {
      if (!isAllowed(url)) event.preventDefault();
    });
    view.webContents.on('will-attach-webview', (event) => event.preventDefault());
    return view;
  }
}

export function chromiumCompatibleSidebarUserAgent(
  userAgent: string,
  applicationName: string,
): string {
  const applicationToken = applicationName.replaceAll(/\s+/gu, '');
  const withoutApplication = applicationToken
    ? userAgent.replace(new RegExp(`\\s${escapeRegExp(applicationToken)}\\/[^\\s]+`, 'gu'), '')
    : userAgent;
  return withoutApplication
    .replace(/\sElectron\/[^\s]+/gu, '')
    .replaceAll(/\s+/gu, ' ')
    .trim();
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&');
}

export function isAllowedSidebarNavigation(url: string, allowedHosts: string[]): boolean {
  try {
    const parsed = new URL(url);
    return (
      parsed.protocol === 'https:' &&
      allowedHosts.some((host) => parsed.hostname === host || parsed.hostname.endsWith(`.${host}`))
    );
  } catch {
    return false;
  }
}
