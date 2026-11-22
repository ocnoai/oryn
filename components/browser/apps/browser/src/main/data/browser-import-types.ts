import type { BookmarkNode } from './bookmark-store';

export type BrowserImportDataType = 'bookmarks' | 'history' | 'searchEngine';
export type BrowserImportBrowser =
  'chrome' | 'chromium' | 'brave' | 'edge' | 'vivaldi' | 'arc' | 'safari';
export type BrowserImportSearchProvider = 'google' | 'duckduckgo' | 'bing';

export interface BrowserImportSource {
  sourceId: string;
  browser: BrowserImportBrowser;
  browserLabel: string;
  profileLabel: string;
  dataTypes: BrowserImportDataType[];
}

export interface BrowserImportOwner {
  webContentsId: number;
  documentId?: string;
}

export interface BrowserImportRequest {
  sourceId: string;
  dataTypes: BrowserImportDataType[];
  /** Main-process native-file-dialog result. Never accept this property from renderer IPC. */
  safariArchivePath?: string;
}

export interface BrowserImportSkipped {
  reason: string;
  count: number;
}

export interface BrowserImportPreview {
  token: string;
  expiresAt: number;
  sourceLabel: string;
  counts: { bookmarks: number; folders: number; history: number };
  skipped: BrowserImportSkipped[];
  warnings: string[];
  searchProvider?: BrowserImportSearchProvider;
}

export interface BrowserImportCommitResult {
  bookmarks: number;
  folders: number;
  history: number;
  historyDuplicates: number;
  searchProvider?: BrowserImportSearchProvider;
}

export interface BrowserImportHistoryEntry {
  url: string;
  title: string;
  visitedAt: string;
}

export interface BrowserImportData {
  bookmarks: BookmarkNode[];
  history: BrowserImportHistoryEntry[];
  skipped: BrowserImportSkipped[];
  warnings: string[];
  searchProvider?: BrowserImportSearchProvider;
}

/** Main-derived native selection. No archive bytes or renderer-supplied paths. */
export interface SafariArchiveIdentity {
  path: string;
  dev: string;
  ino: string;
  size: number;
  mtimeNs: string;
  ctimeNs: string;
}

export interface BrowserImportWorkerRequest {
  version: 2;
  kind: 'chromium' | 'safari';
  directory: string;
  safariArchive?: SafariArchiveIdentity;
  sourceKey: string;
  sourceLabel: string;
  dataTypes: BrowserImportDataType[];
  /** Includes the imported wrapper folder, and is bounded against the current store. */
  bookmarkCapacity: number;
  existingBookmarkIds: string[];
}

export type BrowserImportWorkerReply =
  { version: 2; ok: true; data: BrowserImportData } | { version: 2; ok: false; code: string };

export const BROWSER_IMPORT_LIMITS = {
  localStateBytes: 2 * 1024 * 1024,
  bookmarksBytes: 16 * 1024 * 1024,
  preferencesBytes: 8 * 1024 * 1024,
  historyBytes: 256 * 1024 * 1024,
  archiveBytes: 64 * 1024 * 1024,
  archiveEntries: 1024,
  archiveMemberBytes: 16 * 1024 * 1024,
  archiveTotalBytes: 32 * 1024 * 1024,
  archiveRatio: 200,
  bookmarkNodes: 20_000,
  historyEntries: 10_000,
  sourceNodes: 100_000,
  workerTimeoutMs: 20_000,
  workerMemoryBytes: 512 * 1024 * 1024,
  previewLifetimeMs: 300_000,
} as const;
