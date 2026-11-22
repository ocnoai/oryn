import { Parser } from 'htmlparser2';
import { randomUUID } from 'node:crypto';
import type { BookmarkNode } from './bookmark-store';

export const MAX_BOOKMARK_IMPORT_BYTES = 2 * 1024 * 1024;
export function parseBookmarkHTML(html: string): BookmarkNode[] {
  if (Buffer.byteLength(html) > MAX_BOOKMARK_IMPORT_BYTES)
    throw new Error('BOOKMARK_IMPORT_TOO_LARGE');
  const nodes: BookmarkNode[] = [];
  const parents: (string | null)[] = [];
  let parentId: string | null = null;
  let pendingFolder: string | undefined;
  let capture: { kind: 'folder' | 'bookmark'; text: string; url?: string } | undefined;
  let ignored = 0;
  const parser = new Parser(
    {
      onopentag(name, attributes) {
        if (['script', 'style', 'iframe', 'object', 'template'].includes(name)) {
          ignored++;
          return;
        }
        if (ignored) return;
        if (name === 'dl') {
          parents.push(parentId);
          parentId = pendingFolder ?? parentId;
          pendingFolder = undefined;
          if (parents.length > 9) throw new Error('BOOKMARK_DEPTH_EXCEEDED');
        }
        if (name === 'h3') capture = { kind: 'folder', text: '' };
        if (name === 'a') capture = { kind: 'bookmark', text: '', url: attributes.href };
      },
      ontext(text) {
        if (capture && !ignored) {
          capture.text += text;
          if (capture.text.length > 512) throw new Error('BOOKMARK_TITLE_TOO_LONG');
        }
      },
      onclosetag(name) {
        if (['script', 'style', 'iframe', 'object', 'template'].includes(name)) {
          ignored = Math.max(0, ignored - 1);
          return;
        }
        if (ignored) return;
        if (
          capture &&
          ((name === 'h3' && capture.kind === 'folder') ||
            (name === 'a' && capture.kind === 'bookmark'))
        ) {
          const base = {
            bookmarkId: `bookmark_${randomUUID().replaceAll('-', '')}`,
            parentId,
            title: capture.text.trim() || 'Untitled',
            order: nodes.filter((node) => node.parentId === parentId).length,
            favorite: parentId === null,
            createdAt: new Date().toISOString(),
          };
          if (capture.kind === 'folder') {
            nodes.push({ ...base, kind: 'folder' });
            pendingFolder = base.bookmarkId;
          } else {
            const url = new URL(capture.url ?? '');
            if (!['http:', 'https:'].includes(url.protocol) || url.toString().length > 8192)
              throw new Error('BOOKMARK_URL_UNSAFE');
            nodes.push({ ...base, kind: 'bookmark', url: url.toString() });
          }
          capture = undefined;
          if (nodes.length > 20_000) throw new Error('BOOKMARK_COUNT_EXCEEDED');
        }
        if (name === 'dl') {
          parentId = parents.pop() ?? null;
          pendingFolder = undefined;
        }
      },
    },
    { decodeEntities: true },
  );
  parser.write(html);
  parser.end();
  if (!nodes.length) throw new Error('BOOKMARK_IMPORT_EMPTY');
  return nodes;
}

export function exportBookmarkHTML(nodes: BookmarkNode[]): string {
  const escape = (value: string) =>
    value
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;');
  const render = (parentId: string | null, depth = 0): string => {
    if (depth > 8) throw new Error('BOOKMARK_DEPTH_EXCEEDED');
    return (
      '<DL><p>\n' +
      nodes
        .filter((node) => node.parentId === parentId && !node.trashedAt)
        .sort((a, b) => a.order - b.order)
        .map((node) =>
          node.kind === 'folder'
            ? `<DT><H3>${escape(node.title)}</H3>\n${render(node.bookmarkId, depth + 1)}`
            : `<DT><A HREF="${escape(node.url)}">${escape(node.title)}</A>\n`,
        )
        .join('') +
      '</DL><p>\n'
    );
  };
  return (
    '<!DOCTYPE NETSCAPE-Bookmark-file-1>\n<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">\n<TITLE>Oryn bookmarks</TITLE>\n<H1>Oryn bookmarks</H1>\n' +
    render(null)
  );
}
