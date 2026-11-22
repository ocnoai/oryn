import { EventEmitter } from 'node:events';
import { beforeEach, describe, expect, it, vi } from 'vitest';
const { prompt } = vi.hoisted(() => ({ prompt: vi.fn() }));
vi.mock('electron', () => ({
  dialog: { showMessageBox: prompt },
  BrowserWindow: { fromWebContents: () => null },
  webContents: { fromId: () => undefined },
}));
import {
  guardInsecureNavigation,
  installBrowserRequestGate,
  registerPopupNavigationSource,
  consumeInsecureNavigationCancellation,
  type NavigationRequest,
} from '../insecure-form-guard';

function fixture(url = 'https://secure.test/form'): NavigationRequest {
  const contents = Object.assign(new EventEmitter(), {
    getURL: () => url,
    isDestroyed: () => false,
    stop: vi.fn(),
  });
  return {
    url: 'http://insecure.test/submit',
    resourceType: 'mainFrame',
    initiatorOrigin: 'https://secure.test',
    webContents: contents as never,
    frame: { url } as never,
  };
}
beforeEach(() => {
  prompt.mockReset().mockResolvedValue({ response: 0 });
});
describe('native insecure navigation boundary', () => {
  it('blocks a secure-page document request by default without reading or reconstructing a body', async () => {
    const request = fixture();
    Object.defineProperty(request, 'uploadData', {
      get() {
        throw new Error('body must stay opaque');
      },
    });
    const decision = vi.fn();
    await guardInsecureNavigation(request, decision);
    expect(prompt).toHaveBeenCalledOnce();
    expect(decision).toHaveBeenCalledWith({ cancel: true });
    expect(request.webContents!.stop).toHaveBeenCalledOnce();
  });
  it('continues the original request only after explicit Send anyway', async () => {
    prompt.mockResolvedValue({ response: 1 });
    const decision = vi.fn();
    await guardInsecureNavigation(fixture(), decision);
    expect(decision).toHaveBeenCalledWith({ cancel: false });
  });
  it('covers OOPIFs and no-referrer new targets using the native initiator origin', async () => {
    for (const resourceType of ['mainFrame', 'subFrame']) {
      const decision = vi.fn();
      await guardInsecureNavigation({ ...fixture('about:blank'), resourceType }, decision);
      expect(decision).toHaveBeenCalledWith({ cancel: true });
    }
  });
  it('rejects an answer after another navigation while the prompt is pending', async () => {
    let answer!: (value: { response: number }) => void;
    prompt.mockImplementation(
      () =>
        new Promise((resolve) => {
          answer = resolve;
        }),
    );
    const request = fixture();
    const decision = vi.fn();
    const pending = guardInsecureNavigation(request, decision);
    request.webContents!.emit('did-start-navigation');
    answer({ response: 1 });
    await pending;
    expect(decision).toHaveBeenCalledWith({ cancel: true });
  });
  it('preserves loopback, initial HTTP and subresource requests without a prompt', async () => {
    for (const request of [
      { ...fixture(), url: 'http://127.0.0.1/submit' },
      { ...fixture(''), initiatorOrigin: undefined },
      { ...fixture(), resourceType: 'image' },
    ]) {
      const decision = vi.fn();
      await guardInsecureNavigation(request, decision);
      expect(decision).toHaveBeenCalledWith({ cancel: false });
    }
    expect(prompt).not.toHaveBeenCalled();
  });
  it('preserves native secure-opener provenance when fallback loadURL has no initiator or referrer', async () => {
    const opener = fixture().webContents!;
    const childRequest = { ...fixture('about:blank'), initiatorOrigin: undefined };
    registerPopupNavigationSource(childRequest.webContents!, opener);
    const decision = vi.fn();
    await guardInsecureNavigation(childRequest, decision);
    expect(prompt).toHaveBeenCalledOnce();
    expect(decision).toHaveBeenCalledWith({ cancel: true });
    expect(
      consumeInsecureNavigationCancellation(childRequest.webContents!, childRequest.url, -20),
    ).toBe(true);
    expect(
      consumeInsecureNavigationCancellation(childRequest.webContents!, childRequest.url, -20),
    ).toBe(false);
    childRequest.webContents!.emit('destroyed');
  });
  it('invalidates fallback popup source when its opener navigates during confirmation', async () => {
    let answer!: (value: { response: number }) => void;
    prompt.mockImplementation(
      () =>
        new Promise((resolve) => {
          answer = resolve;
        }),
    );
    const opener = fixture().webContents!;
    const request = { ...fixture('about:blank'), initiatorOrigin: undefined };
    registerPopupNavigationSource(request.webContents!, opener);
    const decision = vi.fn();
    const pending = guardInsecureNavigation(request, decision);
    opener.emit('did-start-navigation');
    answer({ response: 1 });
    await pending;
    expect(decision).toHaveBeenCalledWith({ cancel: true });
    request.webContents!.emit('destroyed');
  });
  it('keeps one listener and preserves artifact authorization when installed in either order', () => {
    let handler!: (
      request: NavigationRequest,
      callback: (value: { cancel: boolean }) => void,
    ) => void;
    const onBeforeRequest = vi.fn((_filter, callback) => {
      handler = callback;
    });
    const target = { webRequest: { onBeforeRequest } };
    installBrowserRequestGate(target as never);
    installBrowserRequestGate(
      target as never,
      (request) => request.url === 'oryn-artifact://capability/owned',
    );
    installBrowserRequestGate(target as never);
    expect(onBeforeRequest).toHaveBeenCalledOnce();
    const decision = vi.fn();
    handler({ url: 'oryn-artifact://capability/owned', resourceType: 'other' }, decision);
    expect(decision).toHaveBeenLastCalledWith({ cancel: false });
    handler({ url: 'oryn-artifact://capability/foreign', resourceType: 'other' }, decision);
    expect(decision).toHaveBeenLastCalledWith({ cancel: true });
  });
});
