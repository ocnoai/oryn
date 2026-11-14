import type { Session, WebContents, WebFrameMain } from 'electron';

export interface PermissionFrameContext {
  topOrigin: string;
  origin: string;
  embedded: boolean;
  isMainFrame: boolean;
  ancestorOrigins: string[];
  frameKey: string;
  /** Captured browser identities; evaluated again after asynchronous consent. */
  identity: string;
}

export interface PermissionFrameDetails {
  requestingUrl?: string;
  isMainFrame: boolean;
  embeddingOrigin?: string;
}

export function webOrigin(value: string): string | undefined {
  try {
    const url = new URL(value);
    return ['https:', 'http:'].includes(url.protocol) && url.origin !== 'null'
      ? url.origin
      : undefined;
  } catch {
    return undefined;
  }
}

/** URL matching is accepted only when Chromium's exact URL identifies one live frame. */
export function resolvePermissionFrame(
  contents: WebContents | null,
  session: Session,
  details: PermissionFrameDetails,
  requestingOrigin?: string,
): PermissionFrameContext | undefined {
  if (!contents || contents.isDestroyed() || contents.session !== session) return undefined;
  const topOrigin = webOrigin(contents.getURL());
  const origin = webOrigin(details.requestingUrl ?? '');
  if (!topOrigin || !origin || (requestingOrigin && webOrigin(requestingOrigin) !== origin))
    return undefined;
  try {
    const main = contents.mainFrame;
    if (!main || main.origin !== topOrigin) return undefined;
    if (!details.isMainFrame && main.framesInSubtree.length > 2048) return undefined;
    const candidates = details.isMainFrame
      ? [main]
      : main.framesInSubtree.filter(
          (frame) => frame !== main && !frame.detached && frame.url === details.requestingUrl,
        );
    if (candidates.length !== 1) return undefined;
    const frame = candidates[0]!;
    if (frame.detached || frame.isDestroyed() || frame.origin !== origin) return undefined;
    if (details.isMainFrame && (origin !== topOrigin || frame.url !== details.requestingUrl))
      return undefined;
    const chain: WebFrameMain[] = [];
    let cursor: WebFrameMain | null = frame;
    while (cursor && chain.length < 32) {
      if (cursor.detached || cursor.isDestroyed() || !webOrigin(cursor.origin)) return undefined;
      chain.push(cursor);
      if (cursor === main) break;
      cursor = cursor.parent;
    }
    if (chain.at(-1) !== main) return undefined;
    const ancestorOrigins = chain
      .slice(1)
      .reverse()
      .map((ancestor) => ancestor.origin);
    // Electron 45 checks report the top-level embedding origin, including a main
    // frame's own origin and nested cross-origin descendants. Actual immediate
    // ancestry is independently derived above, never inferred from this field.
    if (details.embeddingOrigin && webOrigin(details.embeddingOrigin) !== topOrigin)
      return undefined;
    return {
      topOrigin,
      origin,
      embedded: !details.isMainFrame && chain.some((ancestor) => ancestor.origin !== topOrigin),
      isMainFrame: details.isMainFrame,
      ancestorOrigins,
      frameKey: `${frame.frameTreeNodeId}:${frame.frameToken}`,
      identity: JSON.stringify(
        chain.map((ancestor) => [ancestor.frameTreeNodeId, ancestor.frameToken, ancestor.origin]),
      ),
    };
  } catch {
    return undefined;
  }
}

/** Populated only by gate-first actual Electron fixtures; unknown permissions stay denied. */
// Electron 45.1.0 / Chromium 156: three-origin gate-first fixture, 4 November 2026.
// Camera/microphone reached the handler without delegation and remain denied.
// Clipboard/pointer-lock lack focused qualification; speaker selection is absent.
export const QUALIFIED_EMBEDDED_PERMISSIONS: ReadonlySet<string> = new Set([
  'fullscreen',
  'geolocation',
  'display-capture',
]);

export function permitsEmbeddedPermission(
  context: PermissionFrameContext,
  permission: string,
): boolean {
  if (context.isMainFrame) return true;
  return QUALIFIED_EMBEDDED_PERMISSIONS.has(permission);
}
