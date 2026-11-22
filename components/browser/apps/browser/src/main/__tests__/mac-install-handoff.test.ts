import { EventEmitter } from 'node:events';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { monitorMacInstall } from '../app-updates/mac-install-handoff';

afterEach(() => vi.useRealTimers());

describe('native installation handoff', () => {
  function fixture() {
    const updater = new EventEmitter();
    const native = new EventEmitter();
    const installed = vi.fn();
    const observed = vi.fn();
    native.on('update-downloaded', observed);
    const begin = () => native.on('update-downloaded', installed);
    return { updater, native, installed, observed, begin };
  }

  it('removes the pending quit callback after a native error, including late completion', async () => {
    const { updater, native, installed, observed, begin } = fixture();
    const pending = monitorMacInstall(updater, native, begin);
    const rejection = expect(pending).rejects.toThrow('APP_UPDATE_NATIVE_INSTALL_FAILED');
    native.emit('error', new Error('wrong publisher'));
    await rejection;
    native.emit('update-downloaded');
    expect(installed).not.toHaveBeenCalled();
    expect(observed).toHaveBeenCalledOnce();
    expect(updater.listenerCount('error')).toBe(0);
    expect(native.listenerCount('before-quit-for-update')).toBe(0);
  });

  it('handles synchronous handoff errors and removes callbacks registered before the throw', async () => {
    const { updater, native, installed, begin } = fixture();
    await expect(
      monitorMacInstall(updater, native, () => {
        begin();
        throw new Error('failed');
      }),
    ).rejects.toThrow('APP_UPDATE_NATIVE_INSTALL_FAILED');
    native.emit('update-downloaded');
    expect(installed).not.toHaveBeenCalled();
  });

  it('bounds a stalled native handoff and prevents a late quit', async () => {
    vi.useFakeTimers();
    const { updater, native, installed, begin } = fixture();
    const pending = monitorMacInstall(updater, native, begin, 100);
    const rejection = expect(pending).rejects.toThrow('APP_UPDATE_NATIVE_INSTALL_FAILED');
    await vi.advanceTimersByTimeAsync(100);
    await rejection;
    native.emit('update-downloaded');
    expect(installed).not.toHaveBeenCalled();
  });

  it('resolves only when the native installer starts quitting, not at download completion', async () => {
    const { updater, native } = fixture();
    let resolved = false;
    const pending = monitorMacInstall(updater, native, () => undefined).then(() => {
      resolved = true;
    });
    updater.emit('update-downloaded');
    await Promise.resolve();
    expect(resolved).toBe(false);
    native.emit('before-quit-for-update');
    await pending;
    expect(resolved).toBe(true);
    expect(native.listenerCount('error')).toBe(0);
  });
});
