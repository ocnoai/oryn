import { EventEmitter } from 'node:events';
import type { WebContents, WebFrameMain } from 'electron';
import { describe, expect, it, vi } from 'vitest';
import {
  ExternalApplicationHandoff,
  externalApplicationURL,
  isExternalApplicationNavigation,
} from '../external-application-handoff';

function fixture(enabled = true) {
  const frame = {
    url: 'https://website.test/page',
    frameToken: 'main-document',
    parent: null,
    detached: false,
  };
  const contents = Object.assign(new EventEmitter(), {
    mainFrame: frame,
    session: {},
    isDestroyed: () => false,
    getURL: () => frame.url,
  }) as unknown as WebContents;
  const dependencies = {
    resolveHandler: vi.fn().mockResolvedValue({ name: 'Mail', path: '/Applications/Mail.app' }),
    confirm: vi.fn().mockResolvedValue(true),
    notify: vi.fn().mockResolvedValue(undefined),
    openExternal: vi.fn().mockResolvedValue(undefined),
  };
  const service = new ExternalApplicationHandoff(dependencies, enabled);
  const unregister = service.register(contents, () => true);
  const navigation = (initiator: WebFrameMain | null = contents.mainFrame) => ({
    url: 'mailto:someone@example.test?subject=Hello%20there',
    isMainFrame: true,
    frame: contents.mainFrame,
    initiator,
    isSameDocument: false,
  });
  return { contents, dependencies, service, unregister, navigation, frame };
}

describe('reviewed external application URLs', () => {
  it.each([
    'mailto:user@example.test',
    'tel:+421123456789',
    'zoommtg://zoom.us/join?confno=123',
    'slack://channel?id=test',
    'msteams://teams.test/join',
  ])('allows %s as bounded data', (url) => {
    expect(externalApplicationURL(url)).toBe(new URL(url).href);
    expect(isExternalApplicationNavigation(url)).toBe(true);
  });

  it.each([
    'file:///private/secret',
    'javascript:alert(1)',
    'https://website.test',
    'unknown://application',
    'mailto:',
    'mailto:user@example.test\n',
    'slack://user:password@application',
    `mailto:${'a'.repeat(8200)}`,
  ])('does not admit %s', (url) => {
    expect(externalApplicationURL(url)).toBeUndefined();
  });

  it('recognizes unknown external schemes for blocking without handing them to the OS', () => {
    expect(isExternalApplicationNavigation('unknown://application')).toBe(true);
    expect(isExternalApplicationNavigation('oryn://new-tab/')).toBe(false);
    expect(isExternalApplicationNavigation('https://website.test')).toBe(false);
  });
});

describe('document-bound external application handoff', () => {
  it('confirms the attributed site/application and rechecks the handler before one exact URL is opened', async () => {
    const { service, contents, dependencies, navigation } = fixture();
    expect(await service.requestNavigation(contents, navigation())).toBe('opened');
    expect(dependencies.confirm).toHaveBeenCalledWith({
      origin: 'https://website.test',
      application: 'Mail',
      signal: expect.any(AbortSignal),
    });
    expect(dependencies.resolveHandler).toHaveBeenCalledTimes(2);
    expect(dependencies.openExternal).toHaveBeenCalledExactlyOnceWith(navigation().url);
  });

  it.each(['child', 'unknown'] as const)(
    'denies %s initiation even when the target is the main frame',
    async (kind) => {
      const { service, contents, dependencies, navigation } = fixture();
      const initiator =
        kind === 'unknown' ? null : ({ parent: contents.mainFrame } as WebFrameMain);
      expect(await service.requestNavigation(contents, navigation(initiator))).toBe('denied');
      expect(dependencies.resolveHandler).not.toHaveBeenCalled();
      expect(dependencies.openExternal).not.toHaveBeenCalled();
    },
  );

  it('denies subframe targets and an unmanaged contents', async () => {
    const { service, contents, dependencies, navigation, unregister } = fixture();
    expect(await service.requestNavigation(contents, { ...navigation(), isMainFrame: false })).toBe(
      'denied',
    );
    unregister();
    expect(await service.requestNavigation(contents, navigation())).toBe('denied');
    expect(dependencies.confirm).not.toHaveBeenCalled();
  });

  it('keeps private windows and non-website principals unavailable', async () => {
    const privateFixture = fixture(false);
    expect(
      await privateFixture.service.requestNavigation(
        privateFixture.contents,
        privateFixture.navigation(),
      ),
    ).toBe('denied');
    const { service, contents, frame, dependencies, navigation } = fixture();
    for (const url of ['file:///private/page.html', 'oryn://new-tab/', 'about:blank']) {
      frame.url = url;
      expect(await service.requestNavigation(contents, navigation())).toBe('denied');
    }
    service.register(contents, () => false);
    frame.url = 'https://website.test/page';
    expect(await service.requestNavigation(contents, navigation())).toBe('denied');
    expect(dependencies.openExternal).not.toHaveBeenCalled();
  });

  it('does not open on Cancel or a missing installed application', async () => {
    const cancelled = fixture();
    cancelled.dependencies.confirm.mockResolvedValue(false);
    expect(
      await cancelled.service.requestNavigation(cancelled.contents, cancelled.navigation()),
    ).toBe('cancelled');
    expect(cancelled.dependencies.openExternal).not.toHaveBeenCalled();
    const missing = fixture();
    missing.dependencies.resolveHandler.mockResolvedValue(null);
    expect(await missing.service.requestNavigation(missing.contents, missing.navigation())).toBe(
      'missing-handler',
    );
    expect(missing.dependencies.notify).toHaveBeenCalledWith('missing-handler');
    expect(missing.dependencies.confirm).not.toHaveBeenCalled();
    expect(missing.dependencies.openExternal).not.toHaveBeenCalled();
  });

  it.each(['navigation', 'crash', 'close'] as const)(
    'invalidates and aborts an outstanding confirmation on %s',
    async (change) => {
      const { service, contents, dependencies, navigation, unregister } = fixture();
      let answer!: (value: boolean) => void;
      dependencies.confirm.mockImplementation(
        () =>
          new Promise((resolve) => {
            answer = resolve;
          }),
      );
      const result = service.requestNavigation(contents, navigation());
      await Promise.resolve();
      const signal = dependencies.confirm.mock.calls[0]![0].signal as AbortSignal;
      if (change === 'navigation')
        contents.emit('did-start-navigation', {}, contents.getURL(), false, true);
      else if (change === 'crash') contents.emit('render-process-gone');
      else unregister();
      expect(signal.aborted).toBe(true);
      answer(true);
      expect(await result).toBe('stale');
      expect(dependencies.openExternal).not.toHaveBeenCalled();
    },
  );

  it('refuses a changed installed handler or partition after confirmation', async () => {
    const changedHandler = fixture();
    changedHandler.dependencies.resolveHandler
      .mockResolvedValueOnce({ name: 'Mail', path: '/Applications/Mail.app' })
      .mockResolvedValueOnce({ name: 'Mail', path: '/Other/Mail.app' });
    expect(
      await changedHandler.service.requestNavigation(
        changedHandler.contents,
        changedHandler.navigation(),
      ),
    ).toBe('stale');
    expect(changedHandler.dependencies.openExternal).not.toHaveBeenCalled();
    const changedPartition = fixture();
    changedPartition.dependencies.confirm.mockImplementation(async () => {
      Object.defineProperty(changedPartition.contents, 'session', { value: {} });
      return true;
    });
    expect(
      await changedPartition.service.requestNavigation(
        changedPartition.contents,
        changedPartition.navigation(),
      ),
    ).toBe('stale');
    expect(changedPartition.dependencies.openExternal).not.toHaveBeenCalled();
  });

  it('suppresses concurrent requests and repeated launches without creating an always-allow grant', async () => {
    const { service, contents, dependencies, navigation } = fixture();
    let answer!: (value: boolean) => void;
    dependencies.confirm.mockImplementation(
      () =>
        new Promise((resolve) => {
          answer = resolve;
        }),
    );
    const first = service.requestNavigation(contents, navigation());
    expect(await service.requestNavigation(contents, navigation())).toBe('duplicate');
    await Promise.resolve();
    answer(true);
    expect(await first).toBe('opened');
    expect(await service.requestNavigation(contents, navigation())).toBe('duplicate');
    expect(dependencies.openExternal).toHaveBeenCalledTimes(1);
  });

  it('reports an OS launch failure without passing an Error or URL to the notification', async () => {
    const { service, contents, dependencies, navigation } = fixture();
    dependencies.openExternal.mockRejectedValue(new Error('private URL details'));
    expect(await service.requestNavigation(contents, navigation())).toBe('open-failed');
    expect(dependencies.notify).toHaveBeenCalledExactlyOnceWith('open-failed');
  });
});
