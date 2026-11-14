import { randomUUID } from 'node:crypto';

import { JsonFileStore, isRecord } from './json-file-store';

interface BookmarkBase {
  bookmarkId: string;
  parentId: string | null;
  title: string;
  createdAt: string;
  order: number;
  favorite: boolean;
  trashedAt?: string;
}

export interface Bookmark extends BookmarkBase {
  kind: 'bookmark';
  url: string;
  thumbnail?: string;
}

export interface BookmarkFolder extends BookmarkBase {
  kind: 'folder';
}

export type BookmarkNode = Bookmark | BookmarkFolder;

interface BookmarkData {
  version: 1;
  barVisible: boolean;
  nodes: BookmarkNode[];
}

const MAX_BOOKMARK_DEPTH = 8;
const MAX_THUMBNAIL_LENGTH = 512 * 1024;
const SAFE_THUMBNAIL_PATTERN = /^data:image\/(?:png|jpeg|webp);base64,[a-zA-Z0-9+/=]+$/;

export class BookmarkStore {
  private readonly store: JsonFileStore<BookmarkData>;

  constructor(filePath: string) {
    this.store = new JsonFileStore(
      filePath,
      () => ({ version: 1, barVisible: true, nodes: [] }),
      isBookmarkData,
    );
    const data = this.store.read();
    const normalized = normalizeNodes(data.nodes);
    if (JSON.stringify(data.nodes) !== JSON.stringify(normalized)) {
      this.store.write({ ...data, nodes: normalized });
    }
  }

  list(): BookmarkNode[] {
    return normalizeNodes(this.store.read().nodes);
  }

  isBarVisible(): boolean {
    return this.store.read().barVisible;
  }

  setBarVisible(visible: boolean): void {
    this.store.update((data) => ({ ...data, barVisible: visible }));
  }

  createBookmark(title: string, url: string, parentId: string | null = null): Bookmark {
    validateParent(this.store.read(), parentId);
    const bookmark: Bookmark = {
      kind: 'bookmark',
      bookmarkId: createBookmarkId(),
      parentId,
      title: title.trim() || new URL(url).hostname,
      url: safeBookmarkURL(url),
      createdAt: new Date().toISOString(),
      order: this.nextOrder(parentId),
      favorite: parentId === null,
    };
    const data = this.store.read();
    const updated = [...data.nodes, bookmark];
    validateGraph(updated);
    this.store.write({ ...data, nodes: updated });
    return bookmark;
  }

  createFolder(title: string, parentId: string | null = null): BookmarkFolder {
    validateParent(this.store.read(), parentId);
    const folder: BookmarkFolder = {
      kind: 'folder',
      bookmarkId: createBookmarkId(),
      parentId,
      title: title.trim() || 'New folder',
      createdAt: new Date().toISOString(),
      order: this.nextOrder(parentId),
      favorite: false,
    };
    const data = this.store.read();
    const updated = [...data.nodes, folder];
    validateGraph(updated);
    this.store.write({ ...data, nodes: updated });
    return folder;
  }

  edit(bookmarkId: string, title: string, url?: string): void {
    const cleanTitle = title.trim();
    if (!cleanTitle || cleanTitle.length > 512) throw new Error('BOOKMARK_TITLE_REQUIRED');
    this.store.update((data) => ({
      ...data,
      nodes: normalizeNodes(data.nodes).map((node) => {
        if (node.bookmarkId !== bookmarkId) return node;
        if (node.kind === 'bookmark' && url !== undefined) {
          return { ...node, title: cleanTitle, url: safeBookmarkURL(url) };
        }
        return { ...node, title: cleanTitle };
      }),
    }));
  }

  createFolderFromItems(bookmarkIds: string[], title = 'New folder'): BookmarkFolder {
    const selected = normalizeNodes(this.store.read().nodes).filter((node) =>
      bookmarkIds.includes(node.bookmarkId),
    );
    if (selected.length < 2) throw new Error('BOOKMARK_FOLDER_REQUIRES_ITEMS');
    const sharedParent = selected[0]?.parentId ?? null;
    if (!selected.every((node) => node.parentId === sharedParent)) {
      throw new Error('BOOKMARK_FOLDER_PARENT_MISMATCH');
    }
    const data = this.store.read();
    const folder: BookmarkFolder = {
      kind: 'folder',
      bookmarkId: createBookmarkId(),
      parentId: sharedParent,
      title: title.trim() || 'New folder',
      createdAt: new Date().toISOString(),
      order: this.nextOrder(sharedParent),
      favorite: false,
    };
    const ids = new Set(bookmarkIds);
    const updated = [
      ...normalizeNodes(data.nodes).map((node) =>
        ids.has(node.bookmarkId) ? { ...node, parentId: folder.bookmarkId } : node,
      ),
      folder,
    ];
    validateGraph(updated);
    this.store.write({ ...data, nodes: updated });
    return folder;
  }

  delete(bookmarkId: string): boolean {
    return this.trash([bookmarkId]) > 0;
  }

  trash(bookmarkIds: string[]): number {
    const selected = this.withDescendants(bookmarkIds);
    const trashedAt = new Date().toISOString();
    this.store.update((data) => ({
      ...data,
      nodes: normalizeNodes(data.nodes).map((node) =>
        selected.has(node.bookmarkId) ? { ...node, trashedAt } : node,
      ),
    }));
    return selected.size;
  }

  restore(bookmarkIds: string[]): number {
    const selected = this.withDescendants(bookmarkIds);
    const nodes = this.list();
    for (const id of [...selected]) {
      let parentId = nodes.find((node) => node.bookmarkId === id)?.parentId;
      const seen = new Set<string>();
      while (parentId && !seen.has(parentId)) {
        seen.add(parentId);
        selected.add(parentId);
        parentId = nodes.find((node) => node.bookmarkId === parentId)?.parentId;
      }
    }
    this.store.update((data) => ({
      ...data,
      nodes: normalizeNodes(data.nodes).map((node) => {
        if (!selected.has(node.bookmarkId)) return node;
        const restored: BookmarkNode = { ...node };
        delete restored.trashedAt;
        return restored;
      }),
    }));
    return selected.size;
  }

  deletePermanently(bookmarkIds: string[]): number {
    const removed = this.withDescendants(bookmarkIds);
    this.store.update((data) => ({
      ...data,
      nodes: data.nodes.filter((node) => !removed.has(node.bookmarkId)),
    }));
    return removed.size;
  }

  setFavorite(bookmarkId: string, favorite: boolean): void {
    this.store.update((data) => ({
      ...data,
      nodes: normalizeNodes(data.nodes).map((node) =>
        node.bookmarkId === bookmarkId ? { ...node, favorite } : node,
      ),
    }));
  }

  setThumbnail(bookmarkId: string, thumbnail?: string): void {
    if (
      thumbnail !== undefined &&
      (thumbnail.length > MAX_THUMBNAIL_LENGTH || !SAFE_THUMBNAIL_PATTERN.test(thumbnail))
    ) {
      throw new Error('BOOKMARK_THUMBNAIL_UNSAFE');
    }
    let found = false;
    this.store.update((data) => ({
      ...data,
      nodes: normalizeNodes(data.nodes).map((node) => {
        if (node.bookmarkId !== bookmarkId) return node;
        if (node.kind !== 'bookmark') throw new Error('BOOKMARK_THUMBNAIL_REQUIRES_BOOKMARK');
        found = true;
        const updated: Bookmark = { ...node };
        if (thumbnail === undefined) delete updated.thumbnail;
        else updated.thumbnail = thumbnail;
        return updated;
      }),
    }));
    if (!found) throw new Error('BOOKMARK_NOT_FOUND');
  }

  reorder(bookmarkId: string, order: number): void {
    this.store.update((data) => ({
      ...data,
      nodes: normalizeNodes(data.nodes).map((node) =>
        node.bookmarkId === bookmarkId ? { ...node, order: Math.max(0, Math.round(order)) } : node,
      ),
    }));
  }

  move(bookmarkIds: string[], parentId: string | null, copy = false): void {
    const data = this.store.read();
    validateParent(data, parentId);
    const normalized = normalizeNodes(data.nodes);
    const byId = new Map(normalized.map((node) => [node.bookmarkId, node]));
    const ids = new Set(bookmarkIds);
    for (const id of ids) {
      if (!byId.has(id)) throw new Error('BOOKMARK_NOT_FOUND');
      let parent = byId.get(id)?.parentId;
      while (parent) {
        if (ids.has(parent)) {
          ids.delete(id);
          break;
        }
        parent = byId.get(parent)?.parentId;
      }
    }
    if (copy) {
      const selected = this.withDescendants([...ids]);
      const remap = new Map([...selected].map((id) => [id, createBookmarkId()]));
      let order = this.nextOrder(parentId);
      const clones = normalized
        .filter((node) => selected.has(node.bookmarkId))
        .map((node) => ({
          ...node,
          bookmarkId: remap.get(node.bookmarkId)!,
          parentId: ids.has(node.bookmarkId) ? parentId : remap.get(node.parentId!)!,
          order: ids.has(node.bookmarkId) ? order++ : node.order,
          createdAt: new Date().toISOString(),
        }));
      const updated = [...normalized, ...clones];
      validateGraph(updated);
      this.store.write({ ...data, nodes: updated });
      return;
    }
    const nextOrder = this.nextOrder(parentId);
    const updated = normalized.map((node, index) =>
      ids.has(node.bookmarkId) ? { ...node, parentId, order: nextOrder + index } : node,
    );
    validateGraph(updated);
    this.store.write({ ...data, nodes: updated });
  }

  importNodes(nodes: BookmarkNode[]): void {
    const data = this.store.read();
    const updated = [...normalizeNodes(data.nodes), ...nodes];
    validateGraph(updated);
    this.store.write({ ...data, nodes: updated });
  }

  private withDescendants(bookmarkIds: string[]): Set<string> {
    const data = this.store.read();
    const removed = new Set(bookmarkIds);
    let changed = true;
    while (changed) {
      changed = false;
      for (const node of data.nodes) {
        if (node.parentId && removed.has(node.parentId) && !removed.has(node.bookmarkId)) {
          removed.add(node.bookmarkId);
          changed = true;
        }
      }
    }
    return removed;
  }

  private nextOrder(parentId: string | null): number {
    return normalizeNodes(this.store.read().nodes).filter((node) => node.parentId === parentId)
      .length;
  }
}

function normalizeNodes(nodes: BookmarkNode[]): BookmarkNode[] {
  return nodes
    .map((node, index) => {
      const normalized: BookmarkNode = {
        ...node,
        order: typeof node.order === 'number' ? node.order : index,
        favorite: typeof node.favorite === 'boolean' ? node.favorite : node.parentId === null,
      };
      if (
        normalized.kind === 'bookmark' &&
        normalized.thumbnail !== undefined &&
        (normalized.thumbnail.length > MAX_THUMBNAIL_LENGTH ||
          !SAFE_THUMBNAIL_PATTERN.test(normalized.thumbnail))
      ) {
        delete normalized.thumbnail;
      }
      return normalized;
    })
    .sort((left, right) => left.order - right.order);
}

function createBookmarkId(): string {
  return `bookmark_${randomUUID().replaceAll('-', '')}`;
}

function validateParent(data: BookmarkData, parentId: string | null): void {
  if (
    parentId &&
    !data.nodes.some((node) => node.bookmarkId === parentId && node.kind === 'folder')
  ) {
    throw new Error('BOOKMARK_PARENT_NOT_FOUND');
  }
  if (parentId && validateDepth(data.nodes, parentId) >= MAX_BOOKMARK_DEPTH) {
    throw new Error('BOOKMARK_DEPTH_EXCEEDED');
  }
}

function validateGraph(nodes: BookmarkNode[]): void {
  if (nodes.length > 20_000) throw new Error('BOOKMARK_COUNT_EXCEEDED');
  if (new Set(nodes.map((node) => node.bookmarkId)).size !== nodes.length)
    throw new Error('BOOKMARK_DUPLICATE_ID');
  for (const node of nodes) {
    if (!node.title.trim() || node.title.length > 512) throw new Error('BOOKMARK_TITLE_REQUIRED');
    if (node.kind === 'bookmark') safeBookmarkURL(node.url);
  }
  const byId = new Map(nodes.map((node) => [node.bookmarkId, node]));
  for (const node of nodes) validateDepth(nodes, node.bookmarkId, byId);
}

function validateDepth(
  nodes: BookmarkNode[],
  bookmarkId: string,
  byId = new Map(nodes.map((node) => [node.bookmarkId, node])),
): number {
  const visited = new Set<string>();
  let current = byId.get(bookmarkId);
  let depth = 0;
  while (current) {
    if (visited.has(current.bookmarkId)) throw new Error('BOOKMARK_CYCLE');
    visited.add(current.bookmarkId);
    depth += 1;
    if (depth > MAX_BOOKMARK_DEPTH) throw new Error('BOOKMARK_DEPTH_EXCEEDED');
    if (!current.parentId) break;
    const parent = byId.get(current.parentId);
    if (!parent || parent.kind !== 'folder') throw new Error('BOOKMARK_PARENT_NOT_FOUND');
    current = parent;
  }
  return depth;
}

function isBookmarkData(value: unknown): value is BookmarkData {
  const validShape =
    isRecord(value) &&
    value.version === 1 &&
    typeof value.barVisible === 'boolean' &&
    Array.isArray(value.nodes) &&
    value.nodes.every(
      (node) =>
        isRecord(node) &&
        (node.kind === 'bookmark' || node.kind === 'folder') &&
        typeof node.bookmarkId === 'string' &&
        (node.parentId === null || typeof node.parentId === 'string') &&
        typeof node.title === 'string' &&
        typeof node.createdAt === 'string' &&
        (node.order === undefined ||
          (typeof node.order === 'number' && Number.isFinite(node.order))) &&
        (node.favorite === undefined || typeof node.favorite === 'boolean') &&
        (node.trashedAt === undefined || typeof node.trashedAt === 'string') &&
        (node.thumbnail === undefined || typeof node.thumbnail === 'string') &&
        (node.kind === 'folder' || typeof node.url === 'string'),
    );
  if (!validShape) return false;
  try {
    validateGraph((value as unknown as BookmarkData).nodes);
    return true;
  } catch {
    return false;
  }
}

function safeBookmarkURL(value: string): string {
  if (value.length > 8192) throw new Error('BOOKMARK_URL_UNSAFE');
  const url = new URL(value);
  if (
    !['http:', 'https:', 'file:'].includes(url.protocol) ||
    (url.protocol === 'file:' && url.hostname && url.hostname !== 'localhost')
  )
    throw new Error('BOOKMARK_URL_UNSAFE');
  return url.toString();
}
