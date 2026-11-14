import { EventEmitter } from 'node:events';
import { randomUUID } from 'node:crypto';

import { JsonFileStore, isRecord } from './json-file-store';
import type { DownloadFileIdentity, DownloadOpenError } from '../download-safety';

export type DownloadState = 'progressing' | 'paused' | 'completed' | 'cancelled' | 'interrupted';

export interface DownloadRecord {
  downloadId: string;
  partition?: string;
  url: string;
  filename: string;
  savePath?: string;
  receivedBytes: number;
  totalBytes: number;
  state: DownloadState;
  startedAt: string;
  updatedAt: string;
  openState?: 'pending' | 'ready' | 'error';
  openError?: DownloadOpenError;
  /** Browser-private identity; never used as a website capability. */
  fileIdentity?: DownloadFileIdentity;
}

interface DownloadData {
  version: 1;
  records: DownloadRecord[];
}

export class DownloadStore extends EventEmitter {
  private data: DownloadData;
  private pendingWrite?: ReturnType<typeof setTimeout>;
  private readonly store: JsonFileStore<DownloadData>;

  constructor(filePath: string) {
    super();
    this.store = new JsonFileStore(filePath, () => ({ version: 1, records: [] }), isDownloadData);
    this.data = this.store.read();
    let recovered = false;
    for (const record of this.data.records) {
      if (['progressing', 'paused'].includes(record.state)) {
        record.state = 'interrupted';
        record.updatedAt = new Date().toISOString();
        recovered = true;
      }
      if (record.state === 'completed') {
        record.openState = 'pending';
        delete record.openError;
        recovered = true;
      }
    }
    if (recovered) this.flush();
  }

  start(
    input: Pick<DownloadRecord, 'url' | 'filename' | 'savePath' | 'totalBytes' | 'partition'>,
  ): DownloadRecord {
    const now = new Date().toISOString();
    const record: DownloadRecord = {
      downloadId: `download_${randomUUID().replaceAll('-', '')}`,
      ...input,
      receivedBytes: 0,
      state: 'progressing',
      startedAt: now,
      updatedAt: now,
      openState: 'pending',
    };
    this.data.records = [...this.data.records, record];
    this.flush();
    this.emit('changed');
    return record;
  }

  update(
    downloadId: string,
    patch: Partial<
      Pick<
        DownloadRecord,
        | 'receivedBytes'
        | 'totalBytes'
        | 'state'
        | 'savePath'
        | 'openState'
        | 'openError'
        | 'fileIdentity'
      >
    >,
  ): DownloadRecord {
    const index = this.data.records.findIndex((record) => record.downloadId === downloadId);
    if (index < 0) throw new Error('DOWNLOAD_NOT_FOUND');
    const updated = { ...this.data.records[index]!, ...patch, updatedAt: new Date().toISOString() };
    this.data.records[index] = updated;
    if (['completed', 'cancelled', 'interrupted'].includes(updated.state)) this.flush();
    else if (!this.pendingWrite)
      this.pendingWrite = setTimeout(() => {
        this.flush();
        this.emit('changed');
      }, 250);
    if (!this.pendingWrite) this.emit('changed');
    return structuredClone(updated);
  }

  remove(downloadId: string): void {
    this.data.records = this.data.records.filter((record) => record.downloadId !== downloadId);
    this.flush();
    this.emit('changed');
  }

  flush(): void {
    clearTimeout(this.pendingWrite);
    this.pendingWrite = undefined;
    const active = this.data.records.filter((record) =>
      ['progressing', 'paused'].includes(record.state),
    );
    const finished = this.data.records.filter((record) => !active.includes(record)).slice(-1000);
    this.data.records = [...finished, ...active];
    this.store.write(this.data);
  }

  list(): DownloadRecord[] {
    return structuredClone(this.data.records).sort((a, b) =>
      b.startedAt.localeCompare(a.startedAt),
    );
  }
}

function isDownloadData(value: unknown): value is DownloadData {
  return (
    isRecord(value) &&
    value.version === 1 &&
    Array.isArray(value.records) &&
    value.records.every(
      (record) =>
        isRecord(record) &&
        typeof record.downloadId === 'string' &&
        (record.partition === undefined ||
          record.partition === 'default' ||
          (typeof record.partition === 'string' &&
            /^persist:oryn-sidebar-[a-z0-9-]{1,64}$/u.test(record.partition))) &&
        typeof record.url === 'string' &&
        typeof record.filename === 'string' &&
        (record.savePath === undefined || typeof record.savePath === 'string') &&
        typeof record.receivedBytes === 'number' &&
        typeof record.totalBytes === 'number' &&
        ['progressing', 'paused', 'completed', 'cancelled', 'interrupted'].includes(
          String(record.state),
        ) &&
        typeof record.startedAt === 'string' &&
        typeof record.updatedAt === 'string' &&
        (record.openState === undefined ||
          ['pending', 'ready', 'error'].includes(String(record.openState))) &&
        (record.openError === undefined ||
          ['quarantine', 'file-unavailable', 'file-changed', 'open-failed'].includes(
            String(record.openError),
          )) &&
        (record.fileIdentity === undefined || isFileIdentity(record.fileIdentity)),
    )
  );
}

function isFileIdentity(value: unknown): value is DownloadFileIdentity {
  return (
    isRecord(value) &&
    Object.keys(value).length === 5 &&
    ['device', 'inode', 'size', 'modified', 'changed'].every(
      (key) => typeof value[key] === 'string' && /^\d{1,40}$/u.test(value[key]),
    )
  );
}
