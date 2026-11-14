import { createHash } from 'node:crypto';
import { Parser } from 'htmlparser2';
import type { BookmarkNode } from './bookmark-store';
import { isRecord } from './json-file-store';
import {
  BROWSER_IMPORT_LIMITS,
  type BrowserImportData,
  type BrowserImportHistoryEntry,
  type BrowserImportWorkerRequest,
} from './browser-import-types';

export function emptyImportData(): BrowserImportData {
  return { bookmarks: [], history: [], skipped: [], warnings: [] };
}

export function skipImport(data: BrowserImportData, reason: string, count = 1): void {
  if (!count) return;
  const item = data.skipped.find((entry) => entry.reason === reason);
  if (item) item.count += count;
  else data.skipped.push({ reason, count });
}

export function importIdentifier(sourceKey: string, value: string): string {
  return `bookmark_${createHash('sha256').update(sourceKey).update('\0').update(value).digest('hex')}`;
}

export function safeImportURL(value: unknown): string | undefined {
  if (typeof value !== 'string' || value.length > 8192) return undefined;
  try {
    const url = new URL(value);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password)
      return undefined;
    return url.toString();
  } catch {
    return undefined;
  }
}

export function importHistoryEntry(value: {
  url?: unknown;
  title?: unknown;
  visitedAt?: unknown;
}): BrowserImportHistoryEntry | undefined {
  const url = safeImportURL(value.url);
  const time = typeof value.visitedAt === 'string' ? Date.parse(value.visitedAt) : NaN;
  if (!url || !Number.isFinite(time) || time < 0 || time > 8.64e15) return undefined;
  return {
    url,
    title: typeof value.title === 'string' ? value.title.slice(0, 512) : new URL(url).hostname,
    visitedAt: new Date(time).toISOString(),
  };
}

interface CandidateBookmark {
  id: string;
  title: string;
  url?: unknown;
  parent?: string;
  folder: boolean;
}

/** Normalizes the new source graph, including its wrapper, against the actual destination budget. */
function appendCandidates(
  candidates: CandidateBookmark[],
  request: BrowserImportWorkerRequest,
  data: BrowserImportData,
): void {
  const existing = new Set(request.existingBookmarkIds);
  const wrapperId = importIdentifier(request.sourceKey, 'imported-root');
  const budget = Math.max(
    0,
    Math.min(BROWSER_IMPORT_LIMITS.bookmarkNodes, request.bookmarkCapacity),
  );
  const wrapperExists = existing.has(wrapperId);
  const nodes: BookmarkNode[] = [];
  const seen = new Set<string>();
  const targetParents = new Map<string, string>();
  const targetDepth = new Map<string, number>([[wrapperId, 1]]);
  const orders = new Map<string, number>();
  let used = wrapperExists ? 0 : 1;
  const createdAt = new Date().toISOString();
  for (const candidate of candidates) {
    if (seen.has(candidate.id)) {
      skipImport(data, 'duplicate-bookmark-id');
      continue;
    }
    seen.add(candidate.id);
    let parentId = candidate.parent ? targetParents.get(candidate.parent) : wrapperId;
    if (!parentId) parentId = wrapperId;
    // Depth includes both the wrapper and leaf. Folders at depth 8 cannot contain a leaf.
    if (candidate.folder && (targetDepth.get(parentId) ?? 1) >= 7) {
      targetParents.set(candidate.id, parentId);
      skipImport(data, 'flattened-folder');
      continue;
    }
    const bookmarkId = importIdentifier(request.sourceKey, candidate.id);
    if (existing.has(bookmarkId)) {
      if (candidate.folder) {
        targetParents.set(candidate.id, bookmarkId);
        targetDepth.set(bookmarkId, Math.min(7, (targetDepth.get(parentId) ?? 1) + 1));
      }
      skipImport(data, 'already-imported-bookmark');
      continue;
    }
    if (used >= budget) {
      if (candidate.folder) targetParents.set(candidate.id, parentId);
      skipImport(data, 'bookmark-capacity');
      continue;
    }
    const url = candidate.folder ? undefined : safeImportURL(candidate.url);
    if (!candidate.folder && !url) {
      skipImport(data, 'unsafe-bookmark-url');
      continue;
    }
    const title = candidate.title.trim().slice(0, 512) || 'Untitled';
    if (candidate.title.length > 512) skipImport(data, 'truncated-bookmark-title');
    const order = orders.get(parentId) ?? 0;
    orders.set(parentId, order + 1);
    const common = { bookmarkId, parentId, title, createdAt, order, favorite: false };
    nodes.push(
      candidate.folder ? { ...common, kind: 'folder' } : { ...common, kind: 'bookmark', url: url! },
    );
    used++;
    if (candidate.folder) {
      targetParents.set(candidate.id, bookmarkId);
      targetDepth.set(bookmarkId, (targetDepth.get(parentId) ?? 1) + 1);
    }
  }
  if (nodes.length && !wrapperExists) {
    nodes.unshift({
      kind: 'folder',
      bookmarkId: wrapperId,
      parentId: null,
      title: `Imported from ${request.sourceLabel}`.slice(0, 512),
      createdAt,
      order: 20_000,
      favorite: false,
    });
  }
  data.bookmarks.push(...nodes);
}

export function normalizeChromiumBookmarks(
  value: unknown,
  request: BrowserImportWorkerRequest,
  data: BrowserImportData,
): void {
  if (!isRecord(value) || !isRecord(value.roots)) throw new Error('IMPORT_BOOKMARKS_INVALID');
  const candidates: CandidateBookmark[] = [];
  const stack: { node: unknown; parent?: string; path: string; depth: number }[] = [];
  for (const [key, root] of Object.entries(value.roots).reverse())
    stack.push({ node: root, path: key, depth: 0 });
  let visited = 0;
  while (stack.length) {
    if (++visited > BROWSER_IMPORT_LIMITS.sourceNodes)
      throw new Error('IMPORT_BOOKMARK_SOURCE_LIMIT');
    const { node, parent, path, depth } = stack.pop()!;
    if (!isRecord(node) || depth > 128) {
      skipImport(data, 'malformed-bookmark');
      continue;
    }
    const id =
      typeof node.guid === 'string' && node.guid.length < 256
        ? `guid:${node.guid}`
        : typeof node.id === 'string' && node.id.length < 256
          ? `id:${node.id}`
          : `path:${path}`;
    const folder = node.type === 'folder' || Array.isArray(node.children);
    if (!folder && node.type !== 'url') {
      skipImport(data, 'malformed-bookmark');
      continue;
    }
    candidates.push({
      id,
      parent,
      folder,
      title: typeof node.name === 'string' ? node.name : 'Untitled',
      url: node.url,
    });
    if (folder && Array.isArray(node.children)) {
      for (let index = node.children.length - 1; index >= 0; index--) {
        stack.push({
          node: node.children[index],
          parent: id,
          path: `${path}/${index}`,
          depth: depth + 1,
        });
        if (stack.length > BROWSER_IMPORT_LIMITS.sourceNodes)
          throw new Error('IMPORT_BOOKMARK_SOURCE_LIMIT');
      }
    }
  }
  appendCandidates(candidates, request, data);
}

/** New import parser skips unsupported items, while legacy Import HTML retains its existing API. */
export function normalizeSafariBookmarks(
  html: string,
  request: BrowserImportWorkerRequest,
  data: BrowserImportData,
): void {
  const candidates: CandidateBookmark[] = [];
  const parents: (string | undefined)[] = [];
  let parent: string | undefined;
  let pendingFolder: string | undefined;
  let capture: { folder: boolean; text: string; url?: string } | undefined;
  let ignored = 0;
  const occurrence = new Map<string, number>();
  const parser = new Parser(
    {
      onopentag(name, attributes) {
        if (['script', 'style', 'iframe', 'object', 'template'].includes(name)) {
          ignored++;
          return;
        }
        if (ignored) return;
        if (name === 'dl') {
          parents.push(parent);
          parent = pendingFolder ?? parent;
          pendingFolder = undefined;
          if (parents.length > 128) throw new Error('IMPORT_BOOKMARK_SOURCE_LIMIT');
        }
        if (name === 'h3' || name === 'a')
          capture = { folder: name === 'h3', text: '', url: attributes.href };
      },
      ontext(text) {
        if (capture && !ignored) capture.text = (capture.text + text).slice(0, 1024);
      },
      onclosetag(name) {
        if (['script', 'style', 'iframe', 'object', 'template'].includes(name)) {
          ignored = Math.max(0, ignored - 1);
          return;
        }
        if (ignored) return;
        if (capture && ((name === 'h3' && capture.folder) || (name === 'a' && !capture.folder))) {
          const identity = `${parent ?? 'root'}\0${capture.folder ? 'folder' : (capture.url ?? '')}\0${capture.text}`;
          const index = occurrence.get(identity) ?? 0;
          occurrence.set(identity, index + 1);
          const id = createHash('sha256').update(identity).update(String(index)).digest('hex');
          candidates.push({
            id,
            parent,
            folder: capture.folder,
            title: capture.text,
            url: capture.url,
          });
          if (capture.folder) pendingFolder = id;
          capture = undefined;
          if (candidates.length > BROWSER_IMPORT_LIMITS.sourceNodes)
            throw new Error('IMPORT_BOOKMARK_SOURCE_LIMIT');
        }
        if (name === 'dl') {
          parent = parents.pop();
          pendingFolder = undefined;
        }
      },
    },
    { decodeEntities: true },
  );
  parser.end(html);
  appendCandidates(candidates, request, data);
}
