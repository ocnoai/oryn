import { EventEmitter } from 'node:events';
import { afterEach, beforeEach, expect, it } from 'vitest';
import { electronDouble, FakeSession, installFixture, page } from './security-audit-ipc-fixture';

let fixture: ReturnType<typeof installFixture>;
let website: ReturnType<typeof page>;
beforeEach(() => {
  website = page(10, new FakeSession());
  fixture = installFixture({}, new EventEmitter(), new EventEmitter(), [website]);
});
afterEach(() => fixture.cleanup());

it.each(['local-ai:engine-versions-get', 'local-ai:engine-version-check'])(
  'restricts %s to the trusted main frame',
  (channel) => {
    const handler = electronDouble.handlers.get(channel)!;
    expect(() => handler({ sender: website, senderFrame: website.mainFrame }, 'torch')).toThrow(
      'TRUSTED_BROWSER_UI_REQUIRED',
    );
    expect(() => handler({ sender: fixture.trusted, senderFrame: {} }, 'torch')).toThrow(
      'TRUSTED_BROWSER_UI_REQUIRED',
    );
  },
);

it('exposes the version inventory through the trusted bridge', async () => {
  expect(await fixture.invoke('local-ai:engine-versions-get')).toEqual(
    expect.arrayContaining([expect.objectContaining({ id: 'torch', versionOrigin: 'pinned' })]),
  );
  await expect(fixture.invoke('local-ai:engine-version-check', '../arbitrary')).rejects.toThrow(
    'INVALID_ENGINE_ID',
  );
});
