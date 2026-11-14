import { EventEmitter } from 'node:events';
import { fileURLToPath } from 'node:url';
import type { IpcMainInvokeEvent, WebContents } from 'electron';
import { describe, expect, it } from 'vitest';
import { BrowserCallerContextService, isAllowedLocalAIUrl } from '../browser-caller-context';
import { normalizeNavigationInput, searchNavigationInput } from '../tab-manager';

describe('trusted local-file navigation and search separation', () => {
  it('encodes local filenames without treating their delimiters as URL query or fragment', () => {
    const path = '/private/tmp/Oryn examples/žluťoučký #1?.html';
    const result = normalizeNavigationInput(`  ${path}  `);
    expect(fileURLToPath(result)).toBe(path);
    expect(new URL(result).search).toBe('');
    expect(new URL(result).hash).toBe('');
  });

  it('accepts local file URLs while denying remote file-server authorities', () => {
    expect(fileURLToPath(normalizeNavigationInput('file://localhost/private/tmp/demo.html'))).toBe(
      '/private/tmp/demo.html',
    );
    expect(() => normalizeNavigationInput('file://server.example/private/demo.html')).toThrow(
      'REMOTE_FILE_URL_DENIED',
    );
    expect(() =>
      normalizeNavigationInput('file://user:password@server.example/demo.html'),
    ).toThrow();
  });

  it.each(['google', 'duckduckgo', 'bing'] as const)(
    '%s searches webpage text as a query even when the text names a local file',
    (provider) => {
      for (const text of [
        '/Users/example/private.html',
        'file:///private/tmp/confidential.html#section',
        'https://example.test/?other=value&secret=selected text',
        'javascript:location.href="file:///private/tmp/private.html"',
      ]) {
        const result = new URL(searchNavigationInput(text, provider));
        expect(result.protocol).toBe('https:');
        expect(result.searchParams.get('q')).toBe(text);
        expect(result.searchParams.size).toBe(1);
        expect(result.hash).toBe('');
      }
    },
  );
});

describe('local files and private contexts have no Local AI admission', () => {
  it.each([
    'file:///private/tmp/demo.html',
    'file://localhost/private/tmp/demo.html',
    'data:text/html,<h1>Potentially local</h1>',
    'about:blank',
    'oryn://model-workspace/test',
  ])('rejects %s as a Local AI website URL', (url) => {
    expect(isAllowedLocalAIUrl(url)).toBe(false);
  });

  it.each(['local-file', 'private'] as const)(
    'does not admit a %s principal even if its frame claims an otherwise eligible HTTPS URL',
    (securityClass) => {
      const contents = Object.assign(new EventEmitter(), {
        id: 100,
        mainFrame: { processId: 10, routingId: 20, url: 'https://example.test/' },
      }) as unknown as WebContents;
      const service = new BrowserCallerContextService();
      const unregister = service.register('tab_fixture', contents, securityClass);
      try {
        expect(() =>
          service.deriveManaged({
            sender: contents,
            senderFrame: contents.mainFrame,
          } as IpcMainInvokeEvent),
        ).toThrow('LOCAL_AI_CALLER_NOT_MANAGED');
      } finally {
        unregister();
      }
    },
  );
});
