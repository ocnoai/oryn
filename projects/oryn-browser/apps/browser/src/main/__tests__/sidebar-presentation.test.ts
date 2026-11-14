import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { nextWorkspaceName, sidebarToolTooltip } from '../../renderer/src/sidebar-presentation';

const sidebar = readFileSync(
  join(import.meta.dirname, '../../renderer/src/BrowserSidebar.tsx'),
  'utf8',
);
const chrome = readFileSync(
  join(import.meta.dirname, '../../renderer/src/BrowserChrome.tsx'),
  'utf8',
);
const styles = readFileSync(join(import.meta.dirname, '../../renderer/src/styles.css'), 'utf8');

describe('sidebar workspace controls and tooltips', () => {
  it('creates the next unique workspace name without relying on unsupported window.prompt', () => {
    expect(nextWorkspaceName([{ name: 'Main' }])).toBe('Workspace 2');
    expect(
      nextWorkspaceName([{ name: 'Main' }, { name: 'Workspace 2' }, { name: 'Workspace 3' }]),
    ).toBe('Workspace 4');
    expect(sidebar).not.toContain("window.prompt('Workspace name'");
    expect(chrome).not.toContain("window.prompt('Workspace name'");
  });

  it('defines explanatory tooltips for every built-in sidebar tool', () => {
    for (const toolId of ['tab-search', 'bookmarks', 'history', 'downloads', 'settings'] as const) {
      expect(sidebarToolTooltip(toolId)).toContain(' — ');
    }
    expect(sidebar).not.toContain("toolId: 'resources'");
    expect(sidebar).toContain('function SidebarIconButton');
    expect(sidebar).toContain('aria-label={label} title={tooltip}');
  });

  it('uses a uniform active workspace border without a left inset stripe', () => {
    expect(styles).not.toContain('box-shadow: inset 3px 0 0 var(--color-brand-primary)');
  });
});
