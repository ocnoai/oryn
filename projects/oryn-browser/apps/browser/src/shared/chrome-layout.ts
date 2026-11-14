import type { BrowserChromeLayout } from './ipc';

export interface BrowserChromeSurfaceState {
  contentHeight: number;
  overlayBottom: number;
  captureWindow: boolean;
  sidebarInteractive: boolean;
}

/**
 * Keeps persistent browser chrome separate from transient trusted overlays.
 * Website geometry follows contentHeight only; overlayHeight controls the
 * independently composited trusted view above it.
 */
export function resolveBrowserChromeLayout(state: BrowserChromeSurfaceState): BrowserChromeLayout {
  const contentHeight = Math.ceil(state.contentHeight);
  const overlayHeight = Math.max(contentHeight, Math.ceil(state.overlayBottom));

  return {
    contentHeight,
    overlayHeight,
    coverWindow: state.captureWindow,
    sidebarInteractive: state.sidebarInteractive,
  };
}
