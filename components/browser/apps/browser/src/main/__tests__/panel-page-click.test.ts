import { describe, expect, it, vi } from 'vitest';
import { BrowserWindowController } from '../browser-window';
import { IPC } from '../../shared/ipc';

function setup(coverWindow = true, sidebarInteractive = true) {
  const contents = { focus: vi.fn(), sendInputEvent: vi.fn() };
  const controller = Object.assign(Object.create(BrowserWindowController.prototype), {
    chromeCoversWindow: coverWindow,
    chromeAllowsSidebarInteraction: sidebarInteractive,
    tabs: {
      getActiveTabId: () => 'page',
      getTab: () => ({
        view: {
          getBounds: () => ({ x: 60, y: 112, width: 900, height: 600 }),
          webContents: contents,
        },
      }),
    },
    trustedView: {
      getBounds: () => ({ x: 60, y: 0 }),
      webContents: { send: vi.fn() },
    },
    layoutViews: vi.fn(),
  }) as BrowserWindowController;
  return { controller, contents };
}

describe('clicking the page outside a browser panel', () => {
  it('dismisses the sheet and delivers the same click in website coordinates', () => {
    const { controller, contents } = setup();
    controller.dismissPanelAndClickPage(100, 212, 'left');
    expect(controller.trustedWebContents.send).toHaveBeenCalledWith(IPC.browserOpenPanel, null);
    expect(contents.focus).toHaveBeenCalledOnce();
    expect(contents.sendInputEvent.mock.calls).toEqual([
      [{ type: 'mouseDown', x: 100, y: 100, button: 'left', clickCount: 1 }],
      [{ type: 'mouseUp', x: 100, y: 100, button: 'left', clickCount: 1 }],
    ]);
  });

  it('does not click through modal dialogs or already closed panels', () => {
    for (const flags of [
      [true, false],
      [false, true],
    ]) {
      const { controller, contents } = setup(...flags);
      controller.dismissPanelAndClickPage(100, 212, 'left');
      expect(contents.sendInputEvent).not.toHaveBeenCalled();
    }
  });

  it('ignores clicks outside the website', () => {
    const { controller, contents } = setup();
    controller.dismissPanelAndClickPage(100, 50, 'left');
    controller.dismissPanelAndClickPage(1000, 212, 'left');
    expect(contents.sendInputEvent).not.toHaveBeenCalled();
  });
});
