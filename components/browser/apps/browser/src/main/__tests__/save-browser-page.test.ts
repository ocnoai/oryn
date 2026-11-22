import { EventEmitter } from 'node:events';
import type { WebContents } from 'electron';
import { describe, expect, it, vi } from 'vitest';
import {
  BROWSER_PAGE_SAVE_FORMAT,
  saveBrowserPage,
  suggestedPageFilename,
} from '../save-browser-page';

function fixture() {
  const frame = { detached: false };
  const contents = Object.assign(new EventEmitter(), {
    mainFrame: frame,
    isDestroyed: () => false,
    getURL: () => 'https://website.test/article',
    getTitle: () => 'An article',
    savePage: vi.fn().mockResolvedValue(undefined),
  }) as unknown as WebContents;
  const options = {
    owned: () => true,
    choosePath: vi.fn().mockResolvedValue('/chosen/article.html'),
    failed: vi.fn().mockResolvedValue(undefined),
  };
  return { contents, options };
}

describe('native Save Page As', () => {
  it('offers HTML only until MHTML has real packaged round-trip qualification', async () => {
    expect(BROWSER_PAGE_SAVE_FORMAT).toEqual({
      label: 'HTML only',
      extension: 'html',
      type: 'HTMLOnly',
    });
    const { contents, options } = fixture();
    expect(await saveBrowserPage(contents, options)).toBe('saved');
    expect(options.choosePath).toHaveBeenCalledExactlyOnceWith('An article.html');
    expect(contents.savePage).toHaveBeenCalledExactlyOnceWith('/chosen/article.html', 'HTMLOnly');
    expect(contents.listenerCount('did-start-navigation')).toBe(0);
  });

  it('does not call the native saver when the user cancels', async () => {
    const { contents, options } = fixture();
    options.choosePath.mockResolvedValue(undefined);
    expect(await saveBrowserPage(contents, options)).toBe('cancelled');
    expect(contents.savePage).not.toHaveBeenCalled();
  });

  it('does not save a replacement document even when a reload keeps the same URL', async () => {
    const { contents, options } = fixture();
    options.choosePath.mockImplementation(async () => {
      contents.emit('did-start-navigation', {}, contents.getURL(), false, true);
      return '/chosen/article.html';
    });
    expect(await saveBrowserPage(contents, options)).toBe('stale');
    expect(contents.savePage).not.toHaveBeenCalled();
  });

  it('does not save after the source tab is closed or moved to another window', async () => {
    const { contents, options } = fixture();
    let owned = true;
    options.owned = () => owned;
    options.choosePath.mockImplementation(async () => {
      owned = false;
      return '/chosen/article.html';
    });
    expect(await saveBrowserPage(contents, options)).toBe('stale');
    expect(contents.savePage).not.toHaveBeenCalled();
  });

  it('keeps internal/workspace pages unavailable and reports a native save failure', async () => {
    const internal = fixture();
    internal.contents.getURL = () => 'oryn://model-workspace/private';
    expect(await saveBrowserPage(internal.contents, internal.options)).toBe('unavailable');
    expect(internal.options.choosePath).not.toHaveBeenCalled();
    const failure = fixture();
    vi.mocked(failure.contents.savePage).mockRejectedValue(new Error('internal path details'));
    expect(await saveBrowserPage(failure.contents, failure.options)).toBe('failed');
    expect(failure.options.failed).toHaveBeenCalledExactlyOnceWith();
  });

  it('uses a bounded plain filename rather than a title-supplied path', () => {
    const name = suggestedPageFilename('../../private/path\nwith:*?"<>|special');
    expect(name).not.toMatch(/[/\\\n:*?"<>|]/u);
    expect(name).toMatch(/\.html$/u);
    expect(suggestedPageFilename('')).toBe('Saved page.html');
    expect(suggestedPageFilename('a'.repeat(1000)).length).toBe(125);
  });
});
