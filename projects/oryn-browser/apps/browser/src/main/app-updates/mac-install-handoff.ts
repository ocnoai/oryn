import type { EventEmitter } from 'node:events';

// Observe the maintained updater's native handoff; do not implement another
// downloader or installer. Pinned MacUpdater 6.8.9 adds its quit callback
// synchronously. Remove only those added callbacks on failure so a late native
// completion cannot quit an application that has already recovered.
export function monitorMacInstall(
  updater: EventEmitter,
  nativeUpdater: EventEmitter,
  begin: () => void,
  timeoutMs = 10 * 60 * 1000,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = new Set(nativeUpdater.rawListeners('update-downloaded'));
    let owned: ReturnType<EventEmitter['rawListeners']> = [];
    let failed = false;
    let settled = false;
    const removeQuitCallbacks = () => {
      for (const listener of owned) nativeUpdater.removeListener('update-downloaded', listener);
    };
    const cleanup = () => {
      clearTimeout(timer);
      nativeUpdater.removeListener('before-quit-for-update', success);
      nativeUpdater.removeListener('error', failure);
      updater.removeListener('error', failure);
    };
    const success = () => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve();
    };
    const failure = () => {
      if (settled) return;
      settled = true;
      failed = true;
      cleanup();
      removeQuitCallbacks();
      reject(new Error('APP_UPDATE_NATIVE_INSTALL_FAILED'));
    };
    const timer = setTimeout(failure, timeoutMs);
    timer.unref?.();
    nativeUpdater.once('before-quit-for-update', success);
    nativeUpdater.on('error', failure);
    updater.on('error', failure);
    try {
      begin();
    } catch {
      failure();
    } finally {
      owned = nativeUpdater
        .rawListeners('update-downloaded')
        .filter((listener) => !existing.has(listener));
      if (failed) removeQuitCallbacks();
    }
  });
}
