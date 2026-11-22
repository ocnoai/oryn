import { EventEmitter } from 'node:events';
import type { WebContents } from 'electron';
import { describe, expect, it, vi } from 'vitest';
import { CaptureAccessRegistry } from '../data/capture-access-registry';

const context = {
  topOrigin: 'https://top.example',
  origin: 'https://embed.example',
  embedded: true,
  isMainFrame: false,
  ancestorOrigins: ['https://top.example'],
  frameKey: 'frame:token',
  identity: 'live-chain',
};
function contents(id: number) {
  return Object.assign(new EventEmitter(), {
    id,
    isDestroyed: () => false,
  }) as unknown as WebContents;
}
describe('capture access visibility', () => {
  it('keeps revoked access visible while before-unload can still preserve the document', () => {
    const registry = new CaptureAccessRegistry();
    const page = contents(1);
    const change = vi.fn();
    registry.subscribe(change);
    registry.record(page, 'default', context, 'camera');
    expect(registry.markStopping(1)).toBe(page);
    page.emit('will-prevent-unload');
    expect(registry.snapshot()).toEqual([
      expect.objectContaining({ stopping: true, permissions: ['camera'] }),
    ]);
    page.emit('did-navigate');
    expect(registry.snapshot()).toEqual([]);
    expect(change).toHaveBeenCalledTimes(3);
  });
  it('aggregates media without duplicating checks and isolates other windows', () => {
    const registry = new CaptureAccessRegistry();
    const a = contents(1);
    const b = contents(2);
    registry.record(a, 'private-a', context, 'camera');
    registry.record(a, 'private-a', context, 'camera');
    registry.record(a, 'private-a', context, 'microphone');
    registry.record(b, 'private-b', context, 'display-capture');
    expect(registry.snapshot()).toEqual([
      expect.objectContaining({ webContentsId: 1, permissions: ['camera', 'microphone'] }),
      expect.objectContaining({ webContentsId: 2, permissions: ['screen'] }),
    ]);
    a.emit('destroyed');
    expect(registry.snapshot()).toEqual([expect.objectContaining({ webContentsId: 2 })]);
  });
});
