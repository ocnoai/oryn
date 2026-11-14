import { createHash, randomUUID } from 'node:crypto';
import { constants } from 'node:fs';
import { lstat, open, realpath } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import type { BookmarkStore, BookmarkNode } from './bookmark-store';
import type { BrowsingHistoryStore } from './history-store';
import type { BrowserAppearanceStore } from './appearance-store';
import { isRecord } from './json-file-store';
import { importHistoryEntry, safeImportURL } from './browser-import-normalize';
import { runBrowserImportWorker } from './browser-import-runner';
import { createImportSnapshot } from './browser-import-snapshot';
import { inspectSafariArchiveSource } from './browser-import-safari-source';
import {
  BROWSER_IMPORT_LIMITS,
  type BrowserImportBrowser,
  type BrowserImportCommitResult,
  type BrowserImportData,
  type BrowserImportDataType,
  type BrowserImportOwner,
  type BrowserImportPreview,
  type BrowserImportRequest,
  type BrowserImportSource,
  type BrowserImportWorkerRequest,
} from './browser-import-types';

const SOURCES: { browser: BrowserImportBrowser; label: string; path: string[] }[] = [
  { browser: 'chrome', label: 'Google Chrome', path: ['Google', 'Chrome'] },
  { browser: 'chromium', label: 'Chromium', path: ['Chromium'] },
  { browser: 'brave', label: 'Brave', path: ['BraveSoftware', 'Brave-Browser'] },
  { browser: 'edge', label: 'Microsoft Edge', path: ['Microsoft Edge'] },
  { browser: 'vivaldi', label: 'Vivaldi', path: ['Vivaldi'] },
  { browser: 'arc', label: 'Arc', path: ['Arc', 'User Data'] },
];

interface RegisteredSource extends BrowserImportSource {
  profileDirectory?: string;
  sourceKey: string;
}
interface PendingImport {
  preview: BrowserImportPreview;
  data: BrowserImportData;
  timer: ReturnType<typeof setTimeout>;
}

export interface BrowserImportServiceOptions {
  bookmarks: BookmarkStore;
  history: BrowsingHistoryStore;
  appearance?: BrowserAppearanceStore;
  homePath: string;
  workerPath: string;
  nativeSandboxPath: string;
  temporaryRoot?: string;
  now?: () => number;
  runWorker?: (
    request: BrowserImportWorkerRequest,
    options: { signal: AbortSignal },
  ) => Promise<BrowserImportData>;
}

function ownerKey(owner: BrowserImportOwner): string {
  if (
    !Number.isSafeInteger(owner.webContentsId) ||
    owner.webContentsId < 1 ||
    (owner.documentId !== undefined &&
      (typeof owner.documentId !== 'string' || owner.documentId.length > 256))
  )
    throw new Error('IMPORT_OWNER_INVALID');
  return `${owner.webContentsId}:${owner.documentId ?? ''}`;
}

async function regularFile(path: string): Promise<boolean> {
  return lstat(path).then(
    (value) => value.isFile() && !value.isSymbolicLink(),
    () => false,
  );
}

async function readLocalState(path: string): Promise<unknown> {
  const file = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const information = await file.stat();
    if (!information.isFile() || information.size > BROWSER_IMPORT_LIMITS.localStateBytes)
      throw new Error('IMPORT_PROFILE_LIST_INVALID');
    const buffer = Buffer.alloc(information.size + 1);
    const { bytesRead } = await file.read(buffer, 0, buffer.length, 0);
    if (bytesRead > information.size) throw new Error('IMPORT_PROFILE_LIST_INVALID');
    return JSON.parse(
      new TextDecoder('utf-8', { fatal: true }).decode(buffer.subarray(0, bytesRead)),
    );
  } finally {
    await file.close();
  }
}

/** Trusted main-process coordinator. It never accepts paths from website/renderer requests. */
export class BrowserImportService {
  private readonly sources = new Map<string, RegisteredSource>();
  private readonly sourceIds = new Map<string, string>();
  private readonly pending = new Map<string, PendingImport>();
  private active?: { owner: string; abort: AbortController };
  private readonly now: () => number;

  constructor(private readonly options: BrowserImportServiceOptions) {
    this.now = options.now ?? Date.now;
  }

  async discover(): Promise<BrowserImportSource[]> {
    const result: RegisteredSource[] = [];
    const supportRoot = join(resolve(this.options.homePath), 'Library', 'Application Support');
    for (const source of SOURCES) {
      const root = join(supportRoot, ...source.path);
      const directory = await lstat(root).catch(() => undefined);
      if (!directory?.isDirectory() || directory.isSymbolicLink()) continue;
      let state: unknown;
      try {
        state = await readLocalState(join(root, 'Local State'));
      } catch {
        state = undefined;
      }
      const profile = isRecord(state) && isRecord(state.profile) ? state.profile : undefined;
      const info = profile && isRecord(profile.info_cache) ? profile.info_cache : {};
      const names = [...new Set(['Default', ...Object.keys(info)])].slice(0, 64);
      for (const name of names) {
        if (name === '.' || name === '..' || !/^[^/\\\0]{1,128}$/.test(name)) continue;
        const path = join(root, name);
        const candidate = await lstat(path).catch(() => undefined);
        if (!candidate?.isDirectory() || candidate.isSymbolicLink()) continue;
        // Resolve configured source roots before storing them; child paths are never renderer-selected.
        const profileDirectory = await realpath(path);
        const rootPath = await realpath(root);
        if (!profileDirectory.startsWith(`${rootPath}/`)) continue;
        const dataTypes: BrowserImportDataType[] = [];
        if (await regularFile(join(profileDirectory, 'Bookmarks'))) dataTypes.push('bookmarks');
        if (await regularFile(join(profileDirectory, 'History'))) dataTypes.push('history');
        if (await regularFile(join(profileDirectory, 'Preferences')))
          dataTypes.push('searchEngine');
        if (!dataTypes.length) continue;
        const sourceKey = createHash('sha256')
          .update(source.browser)
          .update('\0')
          .update(profileDirectory)
          .digest('hex');
        const sourceId = this.sourceIds.get(sourceKey) ?? `import_${randomUUID()}`;
        this.sourceIds.set(sourceKey, sourceId);
        const record = info[name];
        result.push({
          sourceId,
          sourceKey,
          browser: source.browser,
          browserLabel: source.label,
          profileLabel:
            isRecord(record) && typeof record.name === 'string' ? record.name.slice(0, 256) : name,
          dataTypes,
          profileDirectory,
        });
      }
    }
    const safariKey = createHash('sha256').update('safari-export').digest('hex');
    const safariId = this.sourceIds.get(safariKey) ?? `import_${randomUUID()}`;
    this.sourceIds.set(safariKey, safariId);
    result.push({
      sourceId: safariId,
      sourceKey: safariKey,
      browser: 'safari',
      browserLabel: 'Safari',
      profileLabel: 'Exported browsing data',
      dataTypes: ['bookmarks', 'history'],
    });
    this.sources.clear();
    for (const source of result) this.sources.set(source.sourceId, source);
    return result.map(({ sourceId, browser, browserLabel, profileLabel, dataTypes }) => ({
      sourceId,
      browser,
      browserLabel,
      profileLabel,
      dataTypes: [...dataTypes],
    }));
  }

  async preview(
    owner: BrowserImportOwner,
    request: BrowserImportRequest,
  ): Promise<BrowserImportPreview> {
    const key = ownerKey(owner);
    this.cancel(owner);
    if (this.active) throw new Error('IMPORT_BUSY');
    const source = this.sources.get(request.sourceId);
    if (
      !source ||
      !Array.isArray(request.dataTypes) ||
      !request.dataTypes.length ||
      request.dataTypes.length > 3 ||
      new Set(request.dataTypes).size !== request.dataTypes.length ||
      request.dataTypes.some((type) => !source.dataTypes.includes(type))
    )
      throw new Error('IMPORT_REQUEST_INVALID');
    if ((source.browser === 'safari') !== (typeof request.safariArchivePath === 'string'))
      throw new Error('IMPORT_REQUEST_INVALID');
    const operation = { owner: key, abort: new AbortController() };
    this.active = operation;
    let snapshot: Awaited<ReturnType<typeof createImportSnapshot>> | undefined;
    let prepared: { preview: BrowserImportPreview; data: BrowserImportData } | undefined;
    let problem: unknown;
    try {
      const safariArchive =
        source.browser === 'safari'
          ? await inspectSafariArchiveSource(request.safariArchivePath!)
          : undefined;
      if (!safariArchive)
        snapshot = await createImportSnapshot({
          profileDirectory: source.profileDirectory,
          dataTypes: request.dataTypes,
          temporaryRoot: this.options.temporaryRoot,
        });
      if (operation.abort.signal.aborted) throw new Error('IMPORT_CANCELLED');
      const current = this.options.bookmarks.list();
      const workerRequest: BrowserImportWorkerRequest = {
        version: 2,
        kind: source.browser === 'safari' ? 'safari' : 'chromium',
        directory: snapshot?.directory ?? '',
        ...(safariArchive ? { safariArchive } : {}),
        sourceKey: source.sourceKey,
        sourceLabel: source.browserLabel,
        dataTypes: [...request.dataTypes],
        bookmarkCapacity: Math.max(0, BROWSER_IMPORT_LIMITS.bookmarkNodes - current.length),
        existingBookmarkIds: current.map((node) => node.bookmarkId),
      };
      const result = this.options.runWorker
        ? await this.options.runWorker(workerRequest, { signal: operation.abort.signal })
        : await runBrowserImportWorker(workerRequest, {
            workerPath: this.options.workerPath,
            nativeSandboxPath: this.options.nativeSandboxPath,
            signal: operation.abort.signal,
          });
      if (operation.abort.signal.aborted) throw new Error('IMPORT_CANCELLED');
      const data = validateImportResult(result, workerRequest);
      const preview: BrowserImportPreview = {
        token: randomUUID(),
        expiresAt: this.now() + BROWSER_IMPORT_LIMITS.previewLifetimeMs,
        sourceLabel: `${source.browserLabel} — ${source.profileLabel}`,
        counts: {
          bookmarks: data.bookmarks.filter((node) => node.kind === 'bookmark').length,
          folders: data.bookmarks.filter((node) => node.kind === 'folder').length,
          history: data.history.length,
        },
        skipped: data.skipped,
        warnings: data.warnings,
        ...(data.searchProvider ? { searchProvider: data.searchProvider } : {}),
      };
      prepared = { preview, data };
    } catch (error) {
      problem = error;
    } finally {
      try {
        if (snapshot) await snapshot.dispose();
      } catch {
        problem = new Error('IMPORT_CLEANUP_FAILED');
      }
      if (this.active === operation) this.active = undefined;
    }
    if (problem) throw problem;
    if (!prepared || operation.abort.signal.aborted) throw new Error('IMPORT_CANCELLED');
    if (this.pending.size >= 16) throw new Error('IMPORT_PREVIEW_LIMIT');
    const timer = setTimeout(
      () => this.pending.delete(key),
      BROWSER_IMPORT_LIMITS.previewLifetimeMs,
    );
    timer.unref?.();
    this.pending.set(key, { ...prepared, timer });
    return structuredClone(prepared.preview);
  }

  commit(owner: BrowserImportOwner, token: string): BrowserImportCommitResult {
    const key = ownerKey(owner);
    const pending = this.pending.get(key);
    if (
      !pending ||
      typeof token !== 'string' ||
      pending.preview.token !== token ||
      pending.preview.expiresAt <= this.now()
    )
      throw new Error('IMPORT_PREVIEW_EXPIRED');
    clearTimeout(pending.timer);
    this.pending.delete(key);
    const existingIds = new Set(this.options.bookmarks.list().map((node) => node.bookmarkId));
    const bookmarks = pending.data.bookmarks.filter((node) => !existingIds.has(node.bookmarkId));
    // BookmarkStore revalidates the combined current graph/capacity, including concurrent changes.
    if (bookmarks.length) this.options.bookmarks.importNodes(bookmarks);
    const history = pending.data.history.length
      ? this.options.history.mergeImported(pending.data.history)
      : { added: 0, duplicates: 0, dropped: 0 };
    if (pending.data.searchProvider && this.options.appearance)
      this.options.appearance.update({ searchProvider: pending.data.searchProvider });
    return {
      bookmarks: bookmarks.filter((node) => node.kind === 'bookmark').length,
      folders: bookmarks.filter((node) => node.kind === 'folder').length,
      history: history.added,
      historyDuplicates: history.duplicates,
      ...(pending.data.searchProvider ? { searchProvider: pending.data.searchProvider } : {}),
    };
  }

  cancel(owner: BrowserImportOwner, token?: string): void {
    const key = ownerKey(owner);
    const pending = this.pending.get(key);
    if (pending && (token === undefined || pending.preview.token === token)) {
      clearTimeout(pending.timer);
      this.pending.delete(key);
    }
    if (token === undefined && this.active?.owner === key) this.active.abort.abort();
  }

  disposeOwner(owner: BrowserImportOwner): void {
    this.cancel(owner);
  }
  dispose(): void {
    this.active?.abort.abort();
    for (const pending of this.pending.values()) clearTimeout(pending.timer);
    this.pending.clear();
  }
}

function validateImportResult(
  value: BrowserImportData,
  request: BrowserImportWorkerRequest,
): BrowserImportData {
  if (
    !isRecord(value) ||
    !Array.isArray(value.bookmarks) ||
    value.bookmarks.length > request.bookmarkCapacity ||
    !Array.isArray(value.history) ||
    value.history.length > BROWSER_IMPORT_LIMITS.historyEntries ||
    !Array.isArray(value.skipped) ||
    value.skipped.length > 64 ||
    !Array.isArray(value.warnings) ||
    value.warnings.length > 16
  )
    throw new Error('IMPORT_WORKER_INVALID');
  const bookmarks: BookmarkNode[] = value.bookmarks.map((node: unknown) => {
    if (
      !isRecord(node) ||
      !['folder', 'bookmark'].includes(String(node.kind)) ||
      typeof node.bookmarkId !== 'string' ||
      !/^bookmark_[a-f0-9]{64}$/.test(node.bookmarkId) ||
      (node.parentId !== null &&
        (typeof node.parentId !== 'string' || !/^bookmark_[a-f0-9]{64}$/.test(node.parentId))) ||
      typeof node.title !== 'string' ||
      !node.title.trim() ||
      node.title.length > 512 ||
      typeof node.createdAt !== 'string' ||
      !Number.isFinite(Date.parse(node.createdAt)) ||
      typeof node.order !== 'number' ||
      !Number.isSafeInteger(node.order) ||
      node.order < 0
    )
      throw new Error('IMPORT_WORKER_INVALID');
    const common = {
      bookmarkId: node.bookmarkId,
      parentId: node.parentId,
      title: node.title,
      createdAt: node.createdAt,
      order: node.order,
      favorite: false,
    };
    const url = node.kind === 'bookmark' ? safeImportURL(node.url) : undefined;
    if (node.kind === 'bookmark' && !url) throw new Error('IMPORT_WORKER_INVALID');
    return node.kind === 'folder'
      ? { ...common, kind: 'folder' }
      : { ...common, kind: 'bookmark', url: url! };
  });
  const history = value.history.map((entry: unknown) => {
    if (!isRecord(entry)) throw new Error('IMPORT_WORKER_INVALID');
    const normalized = importHistoryEntry(entry);
    if (!normalized) throw new Error('IMPORT_WORKER_INVALID');
    return normalized;
  });
  const skipped = value.skipped.map((entry: unknown) => {
    if (
      !isRecord(entry) ||
      typeof entry.reason !== 'string' ||
      !/^[a-z-]{1,64}$/.test(entry.reason) ||
      typeof entry.count !== 'number' ||
      !Number.isSafeInteger(entry.count) ||
      entry.count < 0
    )
      throw new Error('IMPORT_WORKER_INVALID');
    return { reason: entry.reason, count: entry.count };
  });
  if (
    value.warnings.some((item: unknown) => typeof item !== 'string' || item.length > 512) ||
    (value.searchProvider !== undefined &&
      !['google', 'duckduckgo', 'bing'].includes(String(value.searchProvider)))
  )
    throw new Error('IMPORT_WORKER_INVALID');
  if (
    (!request.dataTypes.includes('bookmarks') && bookmarks.length) ||
    (!request.dataTypes.includes('history') && history.length) ||
    (!request.dataTypes.includes('searchEngine') && value.searchProvider !== undefined)
  )
    throw new Error('IMPORT_WORKER_INVALID');
  return {
    bookmarks,
    history,
    skipped,
    warnings: value.warnings as string[],
    ...(value.searchProvider ? { searchProvider: value.searchProvider } : {}),
  };
}
