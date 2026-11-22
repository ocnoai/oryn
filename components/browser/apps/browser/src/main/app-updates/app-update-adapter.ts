import type { AppUpdateChannel } from './app-update-settings-store';

export interface AppUpdateCandidate {
  version: string;
  bundleVersion?: string;
  releaseName?: string;
  releaseNotes?: string;
  releaseDate?: string;
  sizeBytes?: number;
  unpackedSizeBytes?: number;
}

export interface AppUpdateProgress {
  transferredBytes: number;
  totalBytes: number;
  bytesPerSecond?: number;
}

export interface AppUpdateAdapter {
  configure(channel: AppUpdateChannel): void;
  check(): Promise<AppUpdateCandidate | undefined>;
  download(onProgress: (progress: AppUpdateProgress) => void): Promise<void>;
  cancelDownload(): void;
  install(): Promise<void>;
  dispose(): void;
}
