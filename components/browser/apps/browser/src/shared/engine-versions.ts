export type EngineVersionStatus =
  'unchecked' | 'current' | 'update' | 'ahead' | 'unknown' | 'error' | 'managed';

export interface EngineVersion {
  id: string;
  name: string;
  group: 'AI engines' | 'Oryn runtime' | 'Browser runtime';
  currentVersion: string | null;
  versionOrigin: 'bundled' | 'running' | 'pinned';
  detail: string;
  source: string;
  checkable: boolean;
  status: EngineVersionStatus;
  latestVersion?: string;
  checkedAt?: string;
  message?: string;
}
