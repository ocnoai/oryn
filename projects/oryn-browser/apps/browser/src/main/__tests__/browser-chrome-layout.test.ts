import { describe, expect, it } from 'vitest';

import {
  resolveBrowserChromeLayout,
  type BrowserChromeSurfaceState,
} from '../../shared/chrome-layout';

const baseState: BrowserChromeSurfaceState = {
  contentHeight: 112,
  overlayBottom: 112,
  captureWindow: false,
  sidebarInteractive: false,
};

describe('trusted chrome composition layout', () => {
  it('does not change website geometry for the three-dot menu', () => {
    const closed = resolveBrowserChromeLayout(baseState);
    const open = resolveBrowserChromeLayout({ ...baseState, overlayBottom: 414 });

    expect(open.contentHeight).toBe(closed.contentHeight);
    expect(open.overlayHeight).toBe(414);
    expect(open.coverWindow).toBe(false);
  });

  it('overlays browser panels without reflowing the website', () => {
    const layout = resolveBrowserChromeLayout({
      ...baseState,
      contentHeight: 256,
      overlayBottom: 644,
    });

    expect(layout.contentHeight).toBe(256);
    expect(layout.overlayHeight).toBe(644);
  });

  it('reserves website space only for persistent chrome', () => {
    expect(resolveBrowserChromeLayout({ ...baseState, contentHeight: 328 }).contentHeight).toBe(
      328,
    );
  });

  it('can capture the full window without changing website geometry', () => {
    expect(resolveBrowserChromeLayout({ ...baseState, captureWindow: true })).toMatchObject({
      contentHeight: 112,
      overlayHeight: 112,
      coverWindow: true,
      sidebarInteractive: false,
    });
  });

  it('keeps the sidebar interactive for switchable sheets', () => {
    expect(
      resolveBrowserChromeLayout({
        ...baseState,
        captureWindow: true,
        sidebarInteractive: true,
      }),
    ).toMatchObject({ coverWindow: true, sidebarInteractive: true });
  });
});
