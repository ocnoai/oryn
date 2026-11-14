import { BrowserWindow, dialog, webContents, type Session } from 'electron';
import { isTrustedBrowserUIRequest } from './trusted-browser-ui';

export interface NavigationRequest {
  url: string;
  resourceType: string;
  initiatorOrigin?: string;
  webContentsId?: number;
  webContents?: Electron.WebContents;
  frame?: Electron.WebFrameMain | null;
}
export type RequestDecision = (response: { cancel: boolean }) => void;
type ArtifactAuthorizer = (request: NavigationRequest) => boolean;
const installed = new WeakMap<Session, { authorizeArtifact?: ArtifactAuthorizer }>();
const popupSources = new WeakMap<
  Electron.WebContents,
  { secure: boolean; current: () => boolean }
>();
const declined = new WeakMap<Electron.WebContents, Map<string, number>>();
const loopback = (host: string) =>
  ['localhost', '[::1]'].includes(host) || /^127(?:\.\d{1,3}){3}$/u.test(host);

/** Browser-created fallback popups lose initiatorOrigin; preserve their native opener provenance. */
export function registerPopupNavigationSource(
  child: Electron.WebContents,
  opener: Electron.WebContents,
): void {
  let secure = true;
  try {
    const frames = opener.mainFrame.framesInSubtree;
    secure =
      opener.getURL().startsWith('https:') ||
      frames.length > 2048 ||
      frames.some((frame) => frame.origin.startsWith('https:'));
  } catch {
    /* Missing source identity requires conservative confirmation. */
  }
  let current = true;
  const invalidate = () => {
    current = false;
  };
  const release = () => {
    current = false;
    clearTimeout(timer);
    popupSources.delete(child);
    opener.removeListener('did-start-navigation', invalidate);
    opener.removeListener('destroyed', invalidate);
    child.removeListener('did-navigate', committed);
    child.removeListener('destroyed', release);
  };
  const committed = (_event: unknown, url: string) => {
    if (url !== 'about:blank') release();
  };
  const timer = setTimeout(() => {
    current = false;
    opener.removeListener('did-start-navigation', invalidate);
    opener.removeListener('destroyed', invalidate);
    // Keep a deny tombstone until this unresolved child commits or is destroyed.
  }, 30_000);
  timer.unref();
  popupSources.set(child, { secure, current: () => current });
  opener.once('did-start-navigation', invalidate);
  opener.once('destroyed', invalidate);
  child.on('did-navigate', committed);
  child.once('destroyed', release);
}

export function consumeInsecureNavigationCancellation(
  contents: Electron.WebContents,
  url: string,
  code: number,
): boolean {
  if (code !== -20) return false;
  const entries = declined.get(contents);
  const expires = entries?.get(url);
  if (expires === undefined) return false;
  entries!.delete(url);
  return expires > Date.now();
}

/** Pauses the original native request. It never reads, copies or reconstructs upload bodies. */
export async function guardInsecureNavigation(
  request: NavigationRequest,
  callback: RequestDecision,
): Promise<void> {
  let target: URL;
  try {
    target = new URL(request.url);
  } catch {
    callback({ cancel: true });
    return;
  }
  if (
    target.protocol !== 'http:' ||
    loopback(target.hostname) ||
    !['mainFrame', 'subFrame'].includes(request.resourceType)
  ) {
    callback({ cancel: false });
    return;
  }
  const contents =
    request.webContents ??
    (request.webContentsId === undefined ? undefined : webContents.fromId(request.webContentsId));
  const popup = contents ? popupSources.get(contents) : undefined;
  let completed = false;
  const finish = (cancel: boolean, preserveSource = false) => {
    if (completed) return;
    completed = true;
    if (cancel && contents && !contents.isDestroyed()) {
      let entries = declined.get(contents);
      if (!entries) {
        entries = new Map();
        declined.set(contents, entries);
      }
      if (entries.size >= 16) entries.delete(entries.keys().next().value!);
      entries.set(target.href, Date.now() + 30_000);
    }
    if (preserveSource && contents && !contents.isDestroyed()) {
      try {
        contents.stop();
      } catch {
        /* Still deny if the native target disappeared. */
      }
    }
    callback({ cancel });
  };
  if (popup && !popup.current()) {
    finish(true);
    return;
  }
  let source = request.initiatorOrigin;
  let topURL: string | undefined;
  let frameURL: string | undefined;
  try {
    topURL = contents && !contents.isDestroyed() ? contents.getURL() : undefined;
    frameURL = request.frame?.url;
  } catch {
    callback({ cancel: true });
    return;
  }
  source = [source, frameURL, topURL].find((value) => value?.startsWith('https:'));
  if (!source && !popup?.secure) {
    finish(false);
    return;
  }
  if (!contents || contents.isDestroyed()) {
    finish(true);
    return;
  }
  let current = true;
  const invalidate = () => {
    current = false;
  };
  contents.on('did-start-navigation', invalidate);
  contents.once('destroyed', invalidate);
  try {
    const owner = BrowserWindow.fromWebContents(contents);
    const options: Electron.MessageBoxOptions = {
      type: 'warning',
      title: 'Insecure connection',
      message: 'Continue over an insecure connection?',
      detail: `Any form data sent to ${target.origin} can be read or changed on the network.`,
      buttons: ['Go back', 'Send anyway'],
      defaultId: 0,
      cancelId: 0,
      noLink: true,
    };
    const answer = owner
      ? await dialog.showMessageBox(owner, options)
      : await dialog.showMessageBox(options);
    let unchanged = false;
    try {
      unchanged =
        !contents.isDestroyed() && contents.getURL() === topURL && request.frame?.url === frameURL;
    } catch {
      /* Detached frames cannot authorize a queued request. */
    }
    const reviewedContextCurrent = current && unchanged && (popup === undefined || popup.current());
    finish(
      answer.response !== 1 || !reviewedContextCurrent,
      answer.response !== 1 && reviewedContextCurrent,
    );
  } catch {
    finish(true);
  } finally {
    contents.removeListener('did-start-navigation', invalidate);
    contents.removeListener('destroyed', invalidate);
  }
}

/** Exactly one listener owns these sessions' request boundary, including artifact capabilities. */
export function installBrowserRequestGate(
  target: Session,
  authorizeArtifact?: ArtifactAuthorizer,
): void {
  const existing = installed.get(target);
  if (existing) {
    if (authorizeArtifact) existing.authorizeArtifact = authorizeArtifact;
    return;
  }
  const policy = { authorizeArtifact };
  installed.set(target, policy);
  target.webRequest.onBeforeRequest(
    { urls: ['http://*/*', 'oryn-artifact://*/*', 'oryn://*/*'] },
    (request, callback) => {
      const url = new URL(request.url);
      if (url.protocol === 'oryn:' && url.hostname === 'browser') {
        callback({
          cancel:
            Boolean(url.username || url.password || url.port) ||
            !isTrustedBrowserUIRequest(request),
        });
        return;
      }
      if (request.url.startsWith('oryn-artifact:')) {
        callback({ cancel: policy.authorizeArtifact?.(request) !== true });
        return;
      }
      void guardInsecureNavigation(request, callback);
    },
  );
}
