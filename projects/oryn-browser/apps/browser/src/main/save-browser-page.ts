import type { WebContents } from 'electron';
import { browserDocumentAction } from './browser-document-action';

// MHTML is deliberately not offered without a packaged save/restart/reopen boundary qualification.
export const BROWSER_PAGE_SAVE_FORMAT = {
  label: 'HTML only',
  extension: 'html',
  type: 'HTMLOnly',
} as const;

export function suggestedPageFilename(title: string): string {
  const name = title
    .replace(/[\p{Cc}/\\:*?"<>|]/gu, ' ')
    .replace(/\s+/gu, ' ')
    .replace(/^\.+|\.+$/gu, '')
    .trim()
    .slice(0, 120);
  return `${name || 'Saved page'}.html`;
}

export async function saveBrowserPage(
  contents: WebContents,
  options: {
    owned: () => boolean;
    choosePath(defaultFilename: string): Promise<string | undefined>;
    failed(): Promise<void>;
  },
): Promise<'saved' | 'cancelled' | 'stale' | 'unavailable' | 'failed'> {
  let document: ReturnType<typeof browserDocumentAction>;
  try {
    if (contents.isDestroyed() || !options.owned() || !/^(https?:|file:)/u.test(contents.getURL()))
      return 'unavailable';
    document = browserDocumentAction(contents, options.owned);
  } catch {
    return 'unavailable';
  }
  try {
    const path = await options.choosePath(suggestedPageFilename(contents.getTitle()));
    if (!path) return 'cancelled';
    if (!document.current()) return 'stale';
    await contents.savePage(path, BROWSER_PAGE_SAVE_FORMAT.type);
    return document.current() ? 'saved' : 'stale';
  } catch {
    if (!document.current()) return 'stale';
    await options.failed().catch(() => {});
    return 'failed';
  } finally {
    document.dispose();
  }
}
