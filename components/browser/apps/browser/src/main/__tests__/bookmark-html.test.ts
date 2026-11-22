import { describe, expect, it } from 'vitest';
import { parseBookmarkHTML, exportBookmarkHTML } from '../data/bookmark-html';
describe('bounded bookmark HTML migration', () => {
  it('round-trips nested folders and escaped labels without importing active content', () => {
    const nodes = parseBookmarkHTML(
      '<DL><p><DT><H3>Folder &amp; more</H3><DL><p><DT><A HREF="https://example.com/?a=1&amp;b=2">Example</A></DL><p></DL>',
    );
    expect(nodes).toHaveLength(2);
    expect(nodes[1]?.parentId).toBe(nodes[0]?.bookmarkId);
    const restored = parseBookmarkHTML(exportBookmarkHTML(nodes));
    expect(restored.map((node) => node.title)).toEqual(['Folder & more', 'Example']);
    expect(restored[1]?.parentId).toBe(restored[0]?.bookmarkId);
  });
  it('rejects unsafe URLs and oversized input before changing a store', () => {
    expect(() => parseBookmarkHTML('<A HREF="javascript:alert(1)">bad</A>')).toThrow(
      'BOOKMARK_URL_UNSAFE',
    );
    expect(() => parseBookmarkHTML('x'.repeat(2 * 1024 * 1024 + 1))).toThrow(
      'BOOKMARK_IMPORT_TOO_LARGE',
    );
    expect(() => parseBookmarkHTML('<script>alert(1)</script>')).toThrow('BOOKMARK_IMPORT_EMPTY');
  });
});
