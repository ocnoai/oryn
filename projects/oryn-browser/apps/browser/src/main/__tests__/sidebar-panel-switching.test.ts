import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

const browserWindow = readFileSync(join(import.meta.dirname, '../browser-window.ts'), 'utf8');
const browserChrome = readFileSync(
  join(import.meta.dirname, '../../renderer/src/BrowserChrome.tsx'),
  'utf8',
);
const ipcHandlers = readFileSync(join(import.meta.dirname, '../ipc-handlers.ts'), 'utf8');

describe('sidebar panel switching', () => {
  it('keeps the rail interactive while a switchable trusted sheet covers the page', () => {
    expect(browserWindow).toContain(
      'this.chromeAllowsSidebarInteraction = layout.sidebarInteractive',
    );
    expect(browserWindow).toContain(
      'sidebar.visible && (!this.chromeCoversWindow || this.chromeAllowsSidebarInteraction)',
    );
    expect(ipcHandlers).toContain("typeof candidate.sidebarInteractive === 'boolean'");
  });

  it('closes an active web app before opening a browser tool', () => {
    expect(browserWindow).toContain(
      'if (experience.read().sidebar.activeAppId) this.sidebarApps.open();',
    );
  });

  it('closes trusted tool sheets before opening a sidebar web app', () => {
    expect(browserWindow).toContain('this.trustedWebContents.send(IPC.browserOpenPanel, null);');
  });

  it('switches different tool sheets and toggles the selected sheet closed', () => {
    expect(browserChrome).toContain(
      'nextPanel !== null && current === nextPanel ? null : nextPanel',
    );
    expect(browserChrome).toContain('setTabSearchOpen(false);');
  });
});
