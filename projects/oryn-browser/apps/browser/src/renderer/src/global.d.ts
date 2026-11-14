import type { ModelWorkspaceAPI, TrustedBrowserAPI } from '../../shared/ipc';

declare global {
  interface Window {
    readonly browser: TrustedBrowserAPI;
    readonly orynWorkspace: ModelWorkspaceAPI;
  }
}

export {};
