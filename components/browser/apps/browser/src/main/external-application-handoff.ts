import type { WebContents, WebFrameMain } from 'electron';

const APPLICATION_PROTOCOLS = new Set(['mailto:', 'tel:', 'zoommtg:', 'slack:', 'msteams:']);
const BROWSER_PROTOCOLS = new Set([
  'http:',
  'https:',
  'about:',
  'oryn:',
  'file:',
  'blob:',
  'data:',
  'javascript:',
]);

export interface ExternalApplicationHandler {
  name: string;
  path: string;
}

export interface ExternalApplicationHandoffDependencies {
  resolveHandler(url: string): Promise<ExternalApplicationHandler | null>;
  confirm(request: { origin: string; application: string; signal: AbortSignal }): Promise<boolean>;
  notify(message: 'missing-handler' | 'open-failed'): Promise<void>;
  openExternal(url: string): Promise<void>;
}

interface Source {
  eligible: () => boolean;
  generation: number;
  dispose: () => void;
}

interface SourceDocument {
  source: Source;
  frame: WebFrameMain;
  frameToken: string;
  url: string;
  origin: string;
  partition: Electron.Session;
  generation: number;
}

export type ExternalApplicationHandoffResult =
  'opened' | 'cancelled' | 'denied' | 'duplicate' | 'stale' | 'missing-handler' | 'open-failed';

/** Non-browser schemes are always stopped before considering an OS handoff. */
export function isExternalApplicationNavigation(value: string): boolean {
  const protocol = /^([a-z][a-z\d+.-]*:)/iu.exec(value)?.[1]?.toLowerCase();
  return Boolean(protocol && !BROWSER_PROTOCOLS.has(protocol));
}

export function externalApplicationURL(value: string): string | undefined {
  if (value.length > 8192 || /[\p{Cc}\s]/u.test(value)) return undefined;
  try {
    const url = new URL(value);
    if (!APPLICATION_PROTOCOLS.has(url.protocol) || url.username || url.password) return undefined;
    if (!value.slice(value.indexOf(':') + 1)) return undefined;
    return url.href;
  } catch {
    return undefined;
  }
}

/** The OS-facing capability is private to managed Browser Core views, never a page IPC API. */
export class ExternalApplicationHandoff {
  private readonly sources = new Map<WebContents, Source>();
  private pending?: { contents: WebContents; controller: AbortController };
  private readonly recent = new WeakMap<
    WebContents,
    { generation: number; url: string; until: number }
  >();

  constructor(
    private readonly dependencies: ExternalApplicationHandoffDependencies,
    private readonly enabled = true,
  ) {}

  register(contents: WebContents, eligible: () => boolean): () => void {
    this.sources.get(contents)?.dispose();
    const source: Source = { eligible, generation: 0, dispose: () => {} };
    const invalidate = () => {
      source.generation += 1;
      this.recent.delete(contents);
      if (this.pending?.contents === contents) this.pending.controller.abort();
    };
    const navigating = (
      _event: Electron.Event,
      _url: string,
      _sameDocument: boolean,
      mainFrame: boolean,
    ) => {
      if (mainFrame) invalidate();
    };
    const destroy = () => source.dispose();
    const inPage = (_event: Electron.Event, _url: string, mainFrame: boolean) => {
      if (mainFrame) invalidate();
    };
    contents.on('did-start-navigation', navigating);
    contents.on('did-navigate-in-page', inPage);
    contents.on('did-navigate', invalidate);
    contents.on('render-process-gone', invalidate);
    contents.once('destroyed', destroy);
    source.dispose = () => {
      if (this.sources.get(contents) !== source) return;
      invalidate();
      this.sources.delete(contents);
      contents.removeListener('did-start-navigation', navigating);
      contents.removeListener('did-navigate-in-page', inPage);
      contents.removeListener('did-navigate', invalidate);
      contents.removeListener('render-process-gone', invalidate);
      contents.removeListener('destroyed', destroy);
    };
    this.sources.set(contents, source);
    return source.dispose;
  }

  requestNavigation(
    contents: WebContents,
    details: Electron.WebContentsWillFrameNavigateEventParams,
  ): Promise<ExternalApplicationHandoffResult> {
    // The target may be mainFrame while an iframe is the initiator (target=_top).
    try {
      if (contents.isDestroyed() || !details.isMainFrame) return Promise.resolve('denied');
      return this.request(contents, details.url, details.initiator);
    } catch {
      return Promise.resolve('denied');
    }
  }

  requestMenu(
    contents: WebContents,
    url: string,
    frame: WebFrameMain | null,
  ): Promise<ExternalApplicationHandoffResult> {
    return this.request(contents, url, frame);
  }

  canRequestMenu(contents: WebContents, url: string, frame: WebFrameMain | null): boolean {
    return Boolean(externalApplicationURL(url) && this.capture(contents, frame));
  }

  private capture(contents: WebContents, initiator?: WebFrameMain | null): SourceDocument | null {
    try {
      const source = this.sources.get(contents);
      if (!this.enabled || !source?.eligible() || contents.isDestroyed()) return null;
      const frame = contents.mainFrame;
      if (!initiator || initiator !== frame || frame.detached || frame.parent) return null;
      const url = new URL(frame.url);
      if (
        !['http:', 'https:'].includes(url.protocol) ||
        url.username ||
        url.password ||
        contents.getURL() !== frame.url ||
        !frame.frameToken
      )
        return null;
      return {
        source,
        frame,
        frameToken: frame.frameToken,
        url: frame.url,
        origin: url.origin,
        partition: contents.session,
        generation: source.generation,
      };
    } catch {
      return null;
    }
  }

  private current(contents: WebContents, document: SourceDocument): boolean {
    const current = this.capture(contents, document.frame);
    return Boolean(
      current &&
      current.source === document.source &&
      current.generation === document.generation &&
      current.frameToken === document.frameToken &&
      current.url === document.url &&
      current.partition === document.partition,
    );
  }

  private async request(
    contents: WebContents,
    value: string,
    initiator?: WebFrameMain | null,
  ): Promise<ExternalApplicationHandoffResult> {
    const url = externalApplicationURL(value);
    const document = this.capture(contents, initiator);
    if (!url || !document) return 'denied';
    const recent = this.recent.get(contents);
    if (
      this.pending ||
      (recent?.generation === document.generation &&
        recent.url === url &&
        recent.until > Date.now())
    )
      return 'duplicate';
    const controller = new AbortController();
    const pending = { contents, controller };
    this.pending = pending;
    this.recent.set(contents, { generation: document.generation, url, until: Date.now() + 1000 });
    try {
      const handler = await this.dependencies.resolveHandler(url);
      if (!this.current(contents, document) || controller.signal.aborted) return 'stale';
      if (!handler?.name || !handler.path) {
        await this.dependencies.notify('missing-handler');
        return 'missing-handler';
      }
      const approved = await this.dependencies.confirm({
        origin: document.origin,
        application: handler.name,
        signal: controller.signal,
      });
      if (!this.current(contents, document) || controller.signal.aborted) return 'stale';
      if (!approved) return 'cancelled';
      const currentHandler = await this.dependencies.resolveHandler(url);
      if (
        !this.current(contents, document) ||
        controller.signal.aborted ||
        currentHandler?.name !== handler.name ||
        currentHandler.path !== handler.path
      )
        return 'stale';
      await this.dependencies.openExternal(url);
      return 'opened';
    } catch {
      if (!this.current(contents, document) || controller.signal.aborted) return 'stale';
      await this.dependencies.notify('open-failed').catch(() => {});
      return 'open-failed';
    } finally {
      if (this.pending === pending) this.pending = undefined;
    }
  }
}
