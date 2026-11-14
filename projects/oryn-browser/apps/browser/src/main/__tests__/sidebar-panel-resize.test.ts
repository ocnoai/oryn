import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  SIDEBAR_PANEL_MAX_WIDTH,
  SIDEBAR_PANEL_MIN_WIDTH,
  clampSidebarPanelWidth,
} from '../../shared/ipc';

const sidebar = readFileSync(
  join(import.meta.dirname, '../../renderer/src/BrowserSidebar.tsx'),
  'utf8',
);
const browserWindow = readFileSync(join(import.meta.dirname, '../browser-window.ts'), 'utf8');

describe('sidebar panel resize', () => {
  it('clamps both drag previews and persisted widths to the supported range', () => {
    expect(clampSidebarPanelWidth(120)).toBe(SIDEBAR_PANEL_MIN_WIDTH);
    expect(clampSidebarPanelWidth(480.4)).toBe(480);
    expect(clampSidebarPanelWidth(900)).toBe(SIDEBAR_PANEL_MAX_WIDTH);
  });

  it('uses a draggable accessible separator instead of narrow/wide buttons', () => {
    expect(sidebar).toContain('role="separator"');
    expect(sidebar).toContain('onPointerDown');
    expect(sidebar).toContain('onPointerMove');
    expect(sidebar).toContain("event.key !== 'ArrowLeft'");
    expect(sidebar).not.toContain('aria-label="Narrower sidebar panel"');
    expect(sidebar).not.toContain('aria-label="Wider sidebar panel"');
  });

  it('composites the resize surface above untrusted sidebar content', () => {
    expect(browserWindow).toContain("withSurface(trustedUIBaseURL, 'sidebar-resize')");
    expect(browserWindow).toContain('this.sidebarApps.raise();');
    expect(browserWindow).toContain('addChildView(this.sidebarResizeView)');
    expect(browserWindow.indexOf('this.sidebarApps.raise();')).toBeLessThan(
      browserWindow.lastIndexOf('addChildView(this.sidebarResizeView)'),
    );
  });
});
