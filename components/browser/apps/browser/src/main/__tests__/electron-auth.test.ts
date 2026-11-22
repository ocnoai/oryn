import { EventEmitter } from 'node:events';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { electronDouble, FakeSession, installFixture, page } from './security-audit-ipc-fixture';

let fixture: ReturnType<typeof installFixture>;
let contents: ReturnType<typeof page>;

beforeEach(() => {
  vi.clearAllMocks();
  contents = page(10, new FakeSession());
  fixture = installFixture({}, new EventEmitter(), new EventEmitter(), [contents]);
});

afterEach(() => fixture?.cleanup());

function listener(name: string): (...args: unknown[]) => void {
  const registration = electronDouble.app.on.mock.calls.find(([event]) => event === name);
  if (!registration) throw new Error(`Missing handler: ${name}`);
  return registration[1];
}

const certificate = { subjectName: 'Client', issuerName: 'Local test CA' };

it('declines background client-certificate requests without selecting a certificate', () => {
  const event = { preventDefault: vi.fn() };
  const callback = vi.fn();
  listener('select-client-certificate')(
    event,
    null,
    'https://example.com',
    [certificate],
    callback,
  );
  expect(event.preventDefault).toHaveBeenCalledOnce();
  expect(callback).toHaveBeenCalledExactlyOnceWith();
  expect(electronDouble.dialog.showMessageBox).not.toHaveBeenCalled();
});

it('still requires the user to select a certificate for an owned tab', async () => {
  electronDouble.dialog.showMessageBox.mockResolvedValue({ response: 1 });
  const callback = vi.fn();
  listener('select-client-certificate')(
    { preventDefault: vi.fn() },
    contents,
    'https://example.com',
    [certificate],
    callback,
  );
  await vi.waitFor(() => expect(callback).toHaveBeenCalledExactlyOnceWith(certificate));
  expect(electronDouble.dialog.showMessageBox).toHaveBeenCalledOnce();
});

it('does not send a certificate if the tab navigates while the chooser is open', async () => {
  let resolveChoice!: (choice: { response: number }) => void;
  electronDouble.dialog.showMessageBox.mockReturnValue(
    new Promise((resolve) => {
      resolveChoice = resolve;
    }),
  );
  const callback = vi.fn();
  listener('select-client-certificate')(
    { preventDefault: vi.fn() },
    contents,
    'https://example.com',
    [certificate],
    callback,
  );
  contents.emit('did-start-navigation');
  resolveChoice({ response: 1 });
  await vi.waitFor(() => expect(callback).toHaveBeenCalledExactlyOnceWith(undefined));
});

it('does not attach background HTTP authentication to an unrelated browser tab', () => {
  const callback = vi.fn();
  expect(() =>
    listener('login')(
      { defaultPrevented: false, preventDefault: vi.fn() },
      null,
      { url: 'https://example.com' },
      { isProxy: false },
      callback,
    ),
  ).not.toThrow();
  expect(fixture.trusted.send).not.toHaveBeenCalled();
  expect(callback).not.toHaveBeenCalled();
});
