import { EventEmitter } from 'node:events';

import type { IpcMainInvokeEvent, WebContents } from 'electron';
import { describe, expect, it } from 'vitest';

import { BrowserCallerContextService, isAllowedLocalAIUrl } from '../browser-caller-context';
import { normalizeNavigationInput } from '../browser-window';
import {
  createOrynNetworkErrorURL,
  isOrynNewTabURL,
  migrateLegacyDevelopmentStartURL,
  ORYN_NEW_TAB_URL,
  readOrynNetworkErrorURL,
} from '../../shared/browser-urls';
import { networkErrorHTML } from '../new-tab-protocol';

describe('isAllowedLocalAIUrl', () => {
  it('allows HTTPS and loopback development origins', () => {
    expect(isAllowedLocalAIUrl('https://example.com/app')).toBe(true);
    expect(isAllowedLocalAIUrl('http://localhost:5173')).toBe(true);
    expect(isAllowedLocalAIUrl('http://127.0.0.1:8080')).toBe(true);
  });

  it('rejects insecure remote and privileged schemes', () => {
    expect(isAllowedLocalAIUrl('http://example.com')).toBe(false);
    expect(isAllowedLocalAIUrl('file:///tmp/demo.html')).toBe(false);
    expect(isAllowedLocalAIUrl('data:text/html,hello')).toBe(false);
  });
});

describe('normalizeNavigationInput', () => {
  it('preserves URLs and adds HTTPS to hostnames', () => {
    expect(normalizeNavigationInput('https://example.com/path')).toBe('https://example.com/path');
    expect(normalizeNavigationInput('example.com')).toBe('https://example.com/');
    expect(normalizeNavigationInput('example.com:3000/path')).toBe('https://example.com:3000/path');
  });

  it('uses HTTP for local development addresses without a scheme', () => {
    expect(normalizeNavigationInput('localhost:3000')).toBe('http://localhost:3000/');
    expect(normalizeNavigationInput(' localhost:3000/path?ready=1 ')).toBe(
      'http://localhost:3000/path?ready=1',
    );
    expect(normalizeNavigationInput('localhost')).toBe('http://localhost/');
    expect(normalizeNavigationInput('127.0.0.1:8080')).toBe('http://127.0.0.1:8080/');
    expect(normalizeNavigationInput('[::1]:5173')).toBe('http://[::1]:5173/');
    expect(normalizeNavigationInput('http://localhost:3000/')).toBe('http://localhost:3000/');
    expect(normalizeNavigationInput('https://localhost:3000/')).toBe('https://localhost:3000/');
  });

  it('uses search fallback for non-URL text', () => {
    expect(normalizeNavigationInput('local ai browser')).toBe(
      'https://www.google.com/search?q=local%20ai%20browser',
    );
  });
});

describe('Oryn new-tab URL', () => {
  it('recognizes the internal page and preserves it through omnibox normalization', () => {
    expect(isOrynNewTabURL(ORYN_NEW_TAB_URL)).toBe(true);
    expect(isOrynNewTabURL('https://new-tab.example')).toBe(false);
    expect(normalizeNavigationInput(ORYN_NEW_TAB_URL)).toBe(ORYN_NEW_TAB_URL);
  });

  it('migrates only the retired development placeholder URLs', () => {
    expect(migrateLegacyDevelopmentStartURL('https://example.com/')).toBe(ORYN_NEW_TAB_URL);
    expect(migrateLegacyDevelopmentStartURL('https://www.iana.org/help/example-domains')).toBe(
      ORYN_NEW_TAB_URL,
    );
    expect(migrateLegacyDevelopmentStartURL('https://example.com/real-page')).toBe(
      'https://example.com/real-page',
    );
  });
});

describe('Oryn network-error page', () => {
  it('round-trips HTTP failure details without granting arbitrary internal URLs', () => {
    const details = {
      url: 'http://localhost:5175/',
      code: -102,
      description: 'net::ERR_CONNECTION_REFUSED',
    };
    expect(readOrynNetworkErrorURL(createOrynNetworkErrorURL(details))).toEqual(details);
    expect(
      readOrynNetworkErrorURL('oryn://network-error/?url=file:///tmp/private'),
    ).toBeUndefined();
  });

  it('renders an actionable, escaped local connection failure', () => {
    const html = networkErrorHTML({
      url: 'http://localhost:5175/?value=%3Cscript%3E',
      code: -102,
      description: 'net::ERR_CONNECTION_REFUSED<script>',
    });
    expect(html).toContain('This page could not be reached');
    expect(html).toContain('This is a local address');
    expect(html).toContain('Try again');
    expect(html).not.toContain('REFUSED<script>');
  });
});

class FakeWebContents extends EventEmitter {
  readonly id: number;
  readonly mainFrame: { processId: number; routingId: number; url: string };

  constructor(id: number, url: string) {
    super();
    this.id = id;
    this.mainFrame = { processId: id * 10, routingId: id * 100, url };
  }

  isDestroyed(): boolean {
    return false;
  }

  asWebContents(): WebContents {
    return this as unknown as WebContents;
  }

  asInvokeEvent(): IpcMainInvokeEvent {
    return {
      sender: this.asWebContents(),
      senderFrame: this.mainFrame,
    } as unknown as IpcMainInvokeEvent;
  }
}

describe('BrowserCallerContextService', () => {
  it('binds caller authority to the correct tab and active WebContents', () => {
    const service = new BrowserCallerContextService();
    const first = new FakeWebContents(1, 'https://one.example/app');
    const second = new FakeWebContents(2, 'https://two.example/app');
    service.register('tab_one', first.asWebContents());
    service.register('tab_two', second.asWebContents());

    const firstContext = service.derive(first.asInvokeEvent(), first.asWebContents());
    const secondContext = service.derive(second.asInvokeEvent(), second.asWebContents());

    expect(firstContext.tabId).toBe('tab_one');
    expect(firstContext.origin).toBe('https://one.example');
    expect(secondContext.tabId).toBe('tab_two');
    expect(service.hasDocumentAuthority('tab_one', 1, firstContext.documentId)).toBe(true);
    expect(service.hasDocumentAuthority('tab_two', 1, firstContext.documentId)).toBe(false);
    expect(() => service.derive(first.asInvokeEvent(), second.asWebContents())).toThrow(
      'LOCAL_AI_CALLER_NOT_ACTIVE',
    );
    expect(service.deriveManaged(first.asInvokeEvent())).toEqual(firstContext);
  });

  it('rotates document authority for reload/cross-origin navigation but not same-document changes', () => {
    const service = new BrowserCallerContextService();
    const contents = new FakeWebContents(3, 'https://one.example/app');
    service.register('tab_three', contents.asWebContents());
    const initial = service.getDocumentId(contents.asWebContents());

    contents.emit('did-start-navigation', {}, 'https://one.example/app#section', true, true);
    expect(service.getDocumentId(contents.asWebContents())).toBe(initial);

    contents.mainFrame.url = 'https://other.example/home';
    contents.emit('did-start-navigation', {}, contents.mainFrame.url, false, true);
    expect(() => service.deriveManaged(contents.asInvokeEvent())).toThrow(
      'LOCAL_AI_DOCUMENT_TRANSITION',
    );
    contents.emit('did-navigate', {}, contents.mainFrame.url);
    const afterNavigation = service.getDocumentId(contents.asWebContents());
    expect(afterNavigation).not.toBe(initial);
    expect(service.hasDocumentAuthority('tab_three', 3, initial)).toBe(false);
    expect(service.derive(contents.asInvokeEvent(), contents.asWebContents()).origin).toBe(
      'https://other.example',
    );
  });

  it('does not let a cross-origin iframe inherit top-level authority', () => {
    const service = new BrowserCallerContextService();
    const contents = new FakeWebContents(4, 'https://top.example/app');
    service.register('tab_four', contents.asWebContents());
    const iframeEvent = {
      sender: contents.asWebContents(),
      senderFrame: {
        processId: contents.mainFrame.processId + 1,
        routingId: contents.mainFrame.routingId + 1,
        url: 'https://iframe.example/embed',
      },
    } as unknown as IpcMainInvokeEvent;

    expect(() => service.derive(iframeEvent, contents.asWebContents())).toThrow(
      'LOCAL_AI_FRAME_NOT_ALLOWED',
    );
  });

  it('grants the narrow workspace principal only to a managed opaque model route', () => {
    const service = new BrowserCallerContextService();
    const workspace = new FakeWebContents(5, `oryn://model/${'a'.repeat(32)}`);
    const website = new FakeWebContents(6, 'https://huggingface.co/google/gemma-3-1b-it');
    service.register('tab_workspace', workspace.asWebContents(), 'model-workspace');
    service.register('tab_website', website.asWebContents());

    expect(service.deriveModelWorkspace(workspace.asInvokeEvent())).toMatchObject({
      origin: 'oryn://model-workspace',
      reference: 'a'.repeat(32),
      tabId: 'tab_workspace',
    });
    expect(() => service.deriveModelWorkspace(website.asInvokeEvent())).toThrow(
      'WORKSPACE_CALLER_NOT_MANAGED',
    );
    website.mainFrame.url = workspace.mainFrame.url;
    expect(() => service.deriveModelWorkspace(website.asInvokeEvent())).toThrow(
      'WORKSPACE_CALLER_NOT_MANAGED',
    );
  });
});

describe('Sprint 60 committed-document lifetime', () => {
  it('B14 retains unchanged document authority after an attachment or aborted navigation', () => {
    const service = new BrowserCallerContextService();
    const contents = new FakeWebContents(90, 'https://one.example/app');
    service.register('tab_download', contents.asWebContents());
    const before = service.getDocumentId(contents.asWebContents());
    contents.emit('did-start-navigation', {}, 'https://one.example/download', false, true);
    expect(service.hasDocumentAuthority('tab_download', 90, before)).toBe(false);
    contents.emit('did-stop-loading');
    expect(service.hasDocumentAuthority('tab_download', 90, before)).toBe(true);
    contents.emit('did-start-navigation', {}, contents.mainFrame.url, false, true);
    contents.emit('did-navigate', {}, contents.mainFrame.url);
    expect(service.hasDocumentAuthority('tab_download', 90, before)).toBe(false);
  });
  it('revokes stale capabilities on renderer death', () => {
    const service = new BrowserCallerContextService();
    const contents = new FakeWebContents(91, 'https://one.example');
    service.register('tab_crash', contents.asWebContents());
    const before = service.getDocumentId(contents.asWebContents());
    contents.emit('render-process-gone');
    expect(service.hasDocumentAuthority('tab_crash', 91, before)).toBe(false);
    expect(() => service.deriveManaged(contents.asInvokeEvent())).toThrow(
      'LOCAL_AI_DOCUMENT_TRANSITION',
    );
  });
});
