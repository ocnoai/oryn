import { describe, expect, it } from 'vitest';
import type { Session, WebContents, WebFrameMain } from 'electron';
import { resolvePermissionFrame } from '../data/permission-frame-policy';

function fixture() {
  const session = {} as Session;
  const main = frame('https://top.example/page', 1);
  const child = frame('https://embed.example/widget', 2, main);
  Object.assign(main, { framesInSubtree: [main, child] });
  const contents = {
    session,
    mainFrame: main,
    getURL: () => main.url,
    isDestroyed: () => false,
  } as WebContents;
  return { session, main, child, contents };
}
function frame(url: string, id: number, parent: WebFrameMain | null = null): WebFrameMain {
  return {
    url,
    origin: new URL(url).origin,
    frameTreeNodeId: id,
    frameToken: `token-${id}`,
    parent,
    detached: false,
    isDestroyed: () => false,
  } as WebFrameMain;
}
describe('browser-derived permission frame identity', () => {
  it('denies a main-frame request when its native mainFrame is unavailable', () => {
    const { session, main, contents } = fixture();
    Object.assign(contents, { mainFrame: undefined });
    expect(
      resolvePermissionFrame(contents, session, { requestingUrl: main.url, isMainFrame: true }),
    ).toBeUndefined();
  });
  it('binds the exact frame and complete ordered ancestor chain', () => {
    const { session, main, child, contents } = fixture();
    const nested = frame('https://leaf.example/request', 3, child);
    Object.assign(main, { framesInSubtree: [main, child, nested] });
    const details = {
      requestingUrl: nested.url,
      isMainFrame: false,
      embeddingOrigin: main.origin,
    };
    const context = resolvePermissionFrame(contents, session, details)!;
    expect(context.ancestorOrigins).toEqual([main.origin, child.origin]);
    const before = context.identity;
    Object.assign(child, { frameToken: 'replacement-document' });
    expect(resolvePermissionFrame(contents, session, details)?.identity).not.toBe(before);
  });

  it('checks native main/nested embeddingOrigin against the verified top origin', () => {
    const { session, main, child, contents } = fixture();
    expect(
      resolvePermissionFrame(contents, session, {
        requestingUrl: main.url,
        isMainFrame: true,
        embeddingOrigin: main.origin + '/',
      }),
    ).toBeDefined();
    const nested = frame('https://leaf.example/widget', 3, child);
    Object.assign(main, { framesInSubtree: [main, child, nested] });
    expect(
      resolvePermissionFrame(contents, session, {
        requestingUrl: nested.url,
        isMainFrame: false,
        embeddingOrigin: main.origin,
      })?.ancestorOrigins,
    ).toEqual([main.origin, child.origin]);
    expect(
      resolvePermissionFrame(contents, session, {
        requestingUrl: nested.url,
        isMainFrame: false,
        embeddingOrigin: child.origin,
      }),
    ).toBeUndefined();
  });
  it('keeps document identity through same-document path and fragment changes', () => {
    const { session, main, child, contents } = fixture();
    const before = resolvePermissionFrame(contents, session, {
      requestingUrl: child.url,
      isMainFrame: false,
    })!;
    Object.assign(main, { url: 'https://top.example/updated#main' });
    Object.assign(child, { url: 'https://embed.example/updated#child' });
    const after = resolvePermissionFrame(contents, session, {
      requestingUrl: child.url,
      isMainFrame: false,
    })!;
    expect(after.identity).toBe(before.identity);
    expect(after.frameKey).toBe(before.frameKey);
  });
  it('denies duplicate URL ambiguity rather than selecting a sibling', () => {
    const { session, main, child, contents } = fixture();
    Object.assign(main, { framesInSubtree: [main, child, frame(child.url, 3, main)] });
    expect(
      resolvePermissionFrame(contents, session, { requestingUrl: child.url, isMainFrame: false }),
    ).toBeUndefined();
  });
  it('never treats an origin returning below a foreign ancestor as top-level', () => {
    const { session, main, child, contents } = fixture();
    const nested = frame(main.url, 3, child);
    Object.assign(main, { framesInSubtree: [main, child, nested] });
    expect(
      resolvePermissionFrame(contents, session, { requestingUrl: nested.url, isMainFrame: false })
        ?.embedded,
    ).toBe(true);
  });
  it('shares ordinary top-origin treatment only across an entirely same-origin chain', () => {
    const { session, main, child, contents } = fixture();
    Object.assign(child, { url: 'https://top.example/frame', origin: main.origin });
    expect(
      resolvePermissionFrame(contents, session, { requestingUrl: child.url, isMainFrame: false })
        ?.embedded,
    ).toBe(false);
  });
  it.each(['opaque', 'detached', 'foreign embedding', 'wrong origin', 'local file'])(
    'denies %s identity',
    (scenario) => {
      const { session, child, contents } = fixture();
      const details = {
        requestingUrl: child.url,
        isMainFrame: false,
        embeddingOrigin: 'https://top.example',
      };
      if (scenario === 'opaque') Object.assign(child, { origin: 'null' });
      if (scenario === 'detached') Object.assign(child, { detached: true });
      if (scenario === 'foreign embedding') details.embeddingOrigin = 'https://attacker.example';
      if (scenario === 'local file')
        Object.assign(child, { url: 'file:///tmp/secret', origin: 'null' });
      expect(
        resolvePermissionFrame(
          contents,
          session,
          details,
          scenario === 'wrong origin' ? 'https://attacker.example' : undefined,
        ),
      ).toBeUndefined();
    },
  );
});
