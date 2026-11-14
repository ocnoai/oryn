import { describe, expect, it } from 'vitest';

import {
  chromiumCompatibleSidebarUserAgent,
  isAllowedSidebarNavigation,
} from '../sidebar-web-app-controller';

describe('sidebar web-app navigation policy', () => {
  const hosts = ['web.whatsapp.com', 'whatsapp.com'];

  it('admits only HTTPS navigation within the configured app host family', () => {
    expect(isAllowedSidebarNavigation('https://web.whatsapp.com/', hosts)).toBe(true);
    expect(isAllowedSidebarNavigation('https://static.whatsapp.com/image', hosts)).toBe(true);
    expect(isAllowedSidebarNavigation('http://web.whatsapp.com/', hosts)).toBe(false);
    expect(isAllowedSidebarNavigation('https://whatsapp.com.attacker.example/', hosts)).toBe(false);
    expect(isAllowedSidebarNavigation('javascript:alert(1)', hosts)).toBe(false);
  });

  it('uses the embedded Chromium identity without Electron or Oryn product tokens', () => {
    const compatible = chromiumCompatibleSidebarUserAgent(
      'Mozilla/5.0 AppleWebKit/537.36 OrynBrowser/0.1.0 Chrome/150.0.7871.224 Electron/43.4.0 Safari/537.36',
      'Oryn Browser',
    );

    expect(compatible).toBe('Mozilla/5.0 AppleWebKit/537.36 Chrome/150.0.7871.224 Safari/537.36');
  });
});
