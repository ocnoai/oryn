import { JsonFileStore, isRecord } from './json-file-store';

export type SitePermissionDecision = 'allow' | 'deny';

const SUPPORTED_SITE_PERMISSIONS = new Set([
  'clipboard-read',
  'clipboard-sanitized-write',
  'popups',
  'sound',
  'display-capture',
  'fullscreen',
  'geolocation',
  'idle-detection',
  'media',
  'camera',
  'microphone',
  'midi',
  'notifications',
  'pointerLock',
  'speaker-selection',
]);

export function isSupportedSitePermission(permission: string): boolean {
  return SUPPORTED_SITE_PERMISSIONS.has(permission);
}

export interface SitePermissionRecord {
  partition?: string;
  origin: string;
  embeddedOrigin?: string;
  ancestorOrigins?: string[];
  permission: string;
  decision: SitePermissionDecision;
  updatedAt: string;
}

interface SitePermissionData {
  version: 1;
  records: SitePermissionRecord[];
}

export class SitePermissionStore {
  private readonly store: JsonFileStore<SitePermissionData>;

  constructor(filePath: string) {
    this.store = new JsonFileStore(
      filePath,
      () => ({ version: 1, records: [] }),
      isSitePermissionData,
    );
  }

  get(
    origin: string,
    permission: string,
    partition = 'default',
  ): SitePermissionDecision | undefined {
    if (!isSupportedSitePermission(permission)) return undefined;
    return this.store
      .read()
      .records.find(
        (record) =>
          !record.embeddedOrigin &&
          record.origin === origin &&
          record.permission === permission &&
          (record.partition ?? 'default') === partition,
      )?.decision;
  }

  list(origin?: string, partition = 'default'): SitePermissionRecord[] {
    return this.store
      .read()
      .records.filter(
        (record) =>
          !record.embeddedOrigin &&
          (!origin || record.origin === origin) &&
          (record.partition ?? 'default') === partition,
      )
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  }

  set(
    origin: string,
    permission: string,
    decision: SitePermissionDecision,
    partition = 'default',
  ): void {
    if (!isSupportedSitePermission(permission)) {
      throw new Error('UNSUPPORTED_SITE_PERMISSION');
    }
    const record: SitePermissionRecord = {
      partition,
      origin: new URL(origin).origin,
      permission,
      decision,
      updatedAt: new Date().toISOString(),
    };
    this.store.update((data) => ({
      ...data,
      records: [
        ...data.records.filter(
          (existing) =>
            existing.embeddedOrigin !== undefined ||
            existing.origin !== record.origin ||
            existing.permission !== record.permission ||
            (existing.partition ?? 'default') !== partition,
        ),
        record,
      ],
    }));
  }

  getEmbedded(
    topOrigin: string,
    embeddedOrigin: string,
    permission: string,
    ancestorOrigins: string[],
    partition = 'default',
  ): SitePermissionDecision | undefined {
    return this.listEmbedded(topOrigin, partition).find(
      (record) =>
        record.embeddedOrigin === embeddedOrigin &&
        record.permission === permission &&
        JSON.stringify(record.ancestorOrigins) === JSON.stringify(ancestorOrigins),
    )?.decision;
  }

  listEmbedded(topOrigin?: string, partition = 'default'): SitePermissionRecord[] {
    return this.store
      .read()
      .records.filter(
        (record) =>
          record.embeddedOrigin !== undefined &&
          (!topOrigin || record.origin === topOrigin) &&
          (record.partition ?? 'default') === partition,
      );
  }

  setEmbedded(
    topOrigin: string,
    embeddedOrigin: string,
    permission: string,
    decision: SitePermissionDecision,
    ancestorOrigins: string[],
    partition = 'default',
  ): void {
    if (
      !isSupportedSitePermission(permission) ||
      !ancestorOrigins.length ||
      ancestorOrigins.length > 31
    )
      throw new Error('UNSUPPORTED_EMBEDDED_SITE_PERMISSION');
    const record: SitePermissionRecord = {
      partition,
      origin: canonicalOrigin(topOrigin),
      embeddedOrigin: canonicalOrigin(embeddedOrigin),
      ancestorOrigins: ancestorOrigins.map(canonicalOrigin),
      permission,
      decision,
      updatedAt: new Date().toISOString(),
    };
    this.store.update((data) => ({
      ...data,
      records: [
        ...data.records.filter(
          (existing) =>
            !(
              existing.origin === record.origin &&
              existing.embeddedOrigin === record.embeddedOrigin &&
              existing.permission === permission &&
              (existing.partition ?? 'default') === partition &&
              JSON.stringify(existing.ancestorOrigins) === JSON.stringify(record.ancestorOrigins)
            ),
        ),
        record,
      ],
    }));
  }

  clearEmbedded(
    topOrigin: string,
    embeddedOrigin: string,
    permission: string,
    partition = 'default',
  ): void {
    this.store.update((data) => ({
      ...data,
      records: data.records.filter(
        (record) =>
          record.origin !== topOrigin ||
          record.embeddedOrigin !== embeddedOrigin ||
          record.permission !== permission ||
          (record.partition ?? 'default') !== partition,
      ),
    }));
  }

  clear(origin?: string, partition = 'default', permission?: string): void {
    this.store.update((data) => ({
      ...data,
      records: data.records.filter(
        (record) =>
          (record.partition ?? 'default') !== partition ||
          (origin !== undefined && record.origin !== origin && record.embeddedOrigin !== origin) ||
          (permission !== undefined && record.permission !== permission),
      ),
    }));
  }
}

function isSitePermissionData(value: unknown): value is SitePermissionData {
  return (
    isRecord(value) &&
    value.version === 1 &&
    Array.isArray(value.records) &&
    value.records.every(
      (record) =>
        isRecord(record) &&
        typeof record.origin === 'string' &&
        (record.partition === undefined || typeof record.partition === 'string') &&
        (record.embeddedOrigin === undefined ||
          (typeof record.embeddedOrigin === 'string' &&
            Array.isArray(record.ancestorOrigins) &&
            record.ancestorOrigins.length > 0 &&
            record.ancestorOrigins.length < 32 &&
            record.ancestorOrigins.every((origin: unknown) => typeof origin === 'string'))) &&
        typeof record.permission === 'string' &&
        isSupportedSitePermission(record.permission) &&
        (record.decision === 'allow' || record.decision === 'deny') &&
        typeof record.updatedAt === 'string',
    )
  );
}

function canonicalOrigin(value: string): string {
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('INVALID_SITE_ORIGIN');
  return url.origin;
}
