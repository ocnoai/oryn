import { EventEmitter } from 'node:events';
import { dialog, type BrowserWindow, type WebContents } from 'electron';
import { describe, expect, it, vi } from 'vitest';
import { requestPageClose } from '../page-close';
vi.mock('electron', () => ({ dialog: { showMessageBoxSync: vi.fn() } }));
it('uses native close and leaves retained pages alive until explicitly accepted', async () => {
  let destroyed = false;
  const contents = Object.assign(new EventEmitter(), {
    getTitle: () => 'Draft',
    getURL: () => 'https://example.com',
    isDestroyed: () => destroyed,
    close: vi.fn((options?: { waitForBeforeUnload: boolean }) => {
      const event = {
        defaultPrevented: false,
        preventDefault() {
          this.defaultPrevented = true;
        },
      };
      if (options?.waitForBeforeUnload) contents.emit('will-prevent-unload', event);
      if (event.defaultPrevented || !options?.waitForBeforeUnload) {
        destroyed = true;
        contents.emit('destroyed');
      }
    }),
  });
  vi.mocked(dialog.showMessageBoxSync).mockReturnValue(0);
  expect(await requestPageClose(contents as unknown as WebContents, {} as BrowserWindow)).toBe(
    false,
  );
  expect(destroyed).toBe(false);
  expect(contents.close).toHaveBeenCalledWith({ waitForBeforeUnload: true });
  vi.mocked(dialog.showMessageBoxSync).mockReturnValue(1);
  expect(await requestPageClose(contents as unknown as WebContents, {} as BrowserWindow)).toBe(
    true,
  );
  expect(destroyed).toBe(true);
  expect(contents.listenerCount('will-prevent-unload')).toBe(0);
});
describe('close coordination', () => {
  it('coalesces concurrent requests for the same document', async () => {
    const contents = Object.assign(new EventEmitter(), {
      getTitle: () => '',
      getURL: () => 'about:blank',
      isDestroyed: () => false,
      close: vi.fn(),
    });
    const one = requestPageClose(contents as unknown as WebContents, {} as BrowserWindow);
    const two = requestPageClose(contents as unknown as WebContents, {} as BrowserWindow);
    expect(one).toBe(two);
    expect(contents.close).toHaveBeenCalledTimes(1);
    contents.emit('destroyed');
    expect(await one).toBe(true);
  });
});
