import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

import { describe, expect, it } from 'vitest';

import { SidebarAppIcon } from '../../renderer/src/components/ui/SidebarAppIcon';

describe('sidebar app icons', () => {
  it.each([
    ['whatsapp', '#25d366'],
    ['messenger', 'messenger-logo-gradient'],
    ['instagram', 'instagram-logo-gradient'],
  ])('renders the bundled %s logo', (appId, marker) => {
    const output = renderToStaticMarkup(createElement(SidebarAppIcon, { appId, name: appId }));

    expect(output).toContain('class="sidebar-app-icon"');
    expect(output).toContain('aria-hidden="true"');
    expect(output).toContain(marker);
  });

  it('keeps a text fallback for future declarative sidebar apps', () => {
    const output = renderToStaticMarkup(
      createElement(SidebarAppIcon, { appId: 'future-service', name: 'Future' }),
    );

    expect(output).toContain('class="sidebar-app-fallback"');
    expect(output).toContain('>F</span>');
  });
});
