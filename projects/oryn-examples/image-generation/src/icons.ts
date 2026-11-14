import { createIcons, ShieldCheck, Sparkles } from 'lucide';

/**
 * Semantic example-app icon map. HTML refers to Oryn names through
 * data-oryn-icon; only this module knows which vendor provides each glyph.
 */
const activeIconPack = {
  Generate: Sparkles,
  LocalPrivacy: ShieldCheck,
};

export function initializeIcons(): void {
  createIcons({
    nameAttr: 'data-oryn-icon',
    icons: activeIconPack,
    attrs: {
      'aria-hidden': 'true',
      focusable: 'false',
    },
  });
}
