import type { WebContents } from 'electron';

export interface SmokePageReadiness {
  expectedURL?: string;
  label: string;
  timeoutMs?: number;
  ignoreLoadFailures?: boolean;
}

/** Smoke callers must own the requested document, never its observation bootstrap. */
export function waitForSmokePage(
  contents: Pick<WebContents, 'getURL' | 'isLoading' | 'isDestroyed' | 'on' | 'removeListener'>,
  { expectedURL, label, timeoutMs = 20_000, ignoreLoadFailures = false }: SmokePageReadiness,
): Promise<void> {
  const target = expectedURL ? new URL(expectedURL).href : undefined;
  return new Promise((resolve, reject) => {
    let finished = false;
    const timer = setTimeout(() => finish(new Error(`Timed out waiting for ${label}.`)), timeoutMs);
    const finish = (error?: Error) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      contents.removeListener('did-finish-load', check);
      contents.removeListener('did-stop-loading', check);
      contents.removeListener('did-fail-load', failed);
      contents.removeListener('destroyed', destroyed);
      if (error) reject(error);
      else resolve();
    };
    const check = () => {
      if (contents.isDestroyed()) return destroyed();
      const current = contents.getURL();
      // A fresh tab first loads about:blank while its observer attaches. Its
      // did-finish-load is unrelated to the document the smoke intends to own.
      const intendedDocument = target ? current === target : current && current !== 'about:blank';
      if (intendedDocument && !contents.isLoading()) finish();
    };
    const failed = (
      _event: Electron.Event,
      code: number,
      description: string,
      _url: string,
      isMainFrame: boolean,
    ) => {
      if (ignoreLoadFailures || code === -3 || !isMainFrame) return;
      finish(new Error(`${label} failed (${code}): ${description}`));
    };
    const destroyed = () => finish(new Error(`The page closed while waiting for ${label}.`));
    contents.on('did-finish-load', check);
    contents.on('did-stop-loading', check);
    contents.on('did-fail-load', failed);
    contents.on('destroyed', destroyed);
    // Check after subscribing so a completed or concurrently finishing page
    // cannot leave a waiter behind without a future event to wake it.
    check();
  });
}
