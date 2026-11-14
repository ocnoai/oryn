import { dialog, type BrowserWindow, type WebContents } from 'electron';

const pending = new WeakMap<WebContents, Promise<boolean>>();
/** The native Chromium close path runs modern, legacy and eligible subframe unload handlers. */
export function requestPageClose(contents: WebContents, window: BrowserWindow): Promise<boolean> {
  if (contents.isDestroyed()) return Promise.resolve(true);
  const existing = pending.get(contents);
  if (existing) return existing;
  const title = contents.getTitle() || contents.getURL();
  const result = new Promise<boolean>((resolve) => {
    let settled = false;
    let timer: ReturnType<typeof setTimeout>;
    const finish = (accepted: boolean) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      contents.removeListener('will-prevent-unload', blocked);
      contents.removeListener('destroyed', destroyed);
      resolve(accepted);
    };
    const blocked = (event: Electron.Event) => {
      const discard =
        dialog.showMessageBoxSync(window, {
          type: 'question',
          buttons: ['Keep page open', 'Discard changes and close'],
          defaultId: 0,
          cancelId: 0,
          message: 'Leave this page?',
          detail: `${title} may have unsaved changes.`,
          noLink: true,
        }) === 1;
      if (discard) event.preventDefault();
      else finish(false);
    };
    const destroyed = () => finish(true);
    const wait = () => {
      timer = setTimeout(() => {
        if (contents.isDestroyed()) {
          finish(true);
          return;
        }
        const force =
          dialog.showMessageBoxSync(window, {
            type: 'warning',
            buttons: ['Keep waiting', 'Force close'],
            defaultId: 0,
            cancelId: 0,
            message: 'This page is not responding to the close request.',
            detail: 'Force closing can lose unsaved changes.',
          }) === 1;
        if (force) contents.close();
        else wait();
      }, 10_000);
    };
    wait();
    contents.on('will-prevent-unload', blocked);
    contents.once('destroyed', destroyed);
    try {
      contents.close({ waitForBeforeUnload: true });
    } catch {
      finish(false);
    }
  }).finally(() => {
    pending.delete(contents);
  });
  pending.set(contents, result);
  return result;
}
