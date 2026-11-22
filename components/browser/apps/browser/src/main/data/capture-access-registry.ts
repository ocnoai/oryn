import type { WebContents } from 'electron';
import { randomUUID } from 'node:crypto';
import type { PermissionFrameContext } from './permission-frame-policy';

export type CapturePermission = 'camera' | 'microphone' | 'screen';
export interface CaptureAccessSnapshot {
  generation: string;
  webContentsId: number;
  partition: string;
  topOrigin: string;
  origin: string;
  permissions: CapturePermission[];
  stopping: boolean;
}
interface Entry {
  generation: string;
  contents: WebContents;
  partition: string;
  context: PermissionFrameContext;
  permissions: Set<CapturePermission>;
  stopping: boolean;
}

/** An access registry, not an assertion that a native media stream is recording. */
export class CaptureAccessRegistry {
  private readonly entries = new Map<WebContents, Map<string, Entry>>();
  private readonly listeners = new Set<() => void>();
  private readonly watched = new WeakSet<WebContents>();

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  record(
    contents: WebContents,
    partition: string,
    context: PermissionFrameContext,
    permission: string,
  ): void {
    const capture = permission === 'display-capture' ? 'screen' : permission;
    if (!['camera', 'microphone', 'screen'].includes(capture)) return;
    if (!this.watched.has(contents)) {
      this.watched.add(contents);
      contents.on('did-navigate', () => this.clear(contents));
      contents.once('destroyed', () => this.clear(contents));
      contents.on('render-process-gone', () => this.clear(contents));
    }
    let frames = this.entries.get(contents);
    if (!frames) {
      frames = new Map();
      this.entries.set(contents, frames);
    }
    let entry = frames.get(context.frameKey);
    if (!entry) {
      entry = {
        generation: randomUUID(),
        contents,
        partition,
        context,
        permissions: new Set(),
        stopping: false,
      };
      frames.set(context.frameKey, entry);
    }
    const changed = !entry.permissions.has(capture as CapturePermission);
    entry.permissions.add(capture as CapturePermission);
    if (changed) this.emit();
  }

  snapshot(): CaptureAccessSnapshot[] {
    return [...this.entries.values()].flatMap((frames) =>
      [...frames.values()].map((entry) => ({
        generation: entry.generation,
        webContentsId: entry.contents.id,
        partition: entry.partition,
        topOrigin: entry.context.topOrigin,
        origin: entry.context.origin,
        permissions: [...entry.permissions].sort(),
        stopping: entry.stopping,
      })),
    );
  }

  /** Revocation is not stream termination. Only actual navigation/destruction clears state. */
  markStopping(webContentsId: number): WebContents | undefined {
    const contents = [...this.entries.keys()].find((entry) => entry.id === webContentsId);
    if (!contents || contents.isDestroyed()) return undefined;
    for (const entry of this.entries.get(contents)!.values()) entry.stopping = true;
    this.emit();
    return contents;
  }

  clear(contents: WebContents): void {
    if (this.entries.delete(contents)) this.emit();
  }

  private emit(): void {
    for (const listener of this.listeners) listener();
  }
}
