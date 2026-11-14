import { useEffect, useRef, useState, useSyncExternalStore } from 'react';

type Request = {
  title: string;
  initial?: string;
  credentials?: boolean;
  choices?: { label: string; value: string }[];
  resolve: (value: string | null) => void;
};
let pending: Request | null = null;
const listeners = new Set<() => void>();
function publish() {
  for (const listener of listeners) listener();
}
export async function requestCredentials(
  title: string,
): Promise<{ username: string; password: string } | null> {
  const value = await request({ title, credentials: true });
  return value ? (JSON.parse(value) as { username: string; password: string }) : null;
}
export function requestText(title: string, initial = ''): Promise<string | null> {
  return request({ title, initial });
}
export function requestChoice(
  title: string,
  choices: { label: string; value: string }[],
): Promise<string | null> {
  return request({ title, choices });
}
function request(value: Omit<Request, 'resolve'>): Promise<string | null> {
  if (pending) return Promise.resolve(null);
  return new Promise((resolve) => {
    pending = { ...value, resolve };
    publish();
  });
}
export function useBrowserDialogOpen(): boolean {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => pending !== null,
  );
}
export function BrowserDialog() {
  const current = useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => pending,
  );
  return current ? <Dialog key={current.title} request={current} /> : null;
}
function Dialog({ request }: { request: Request }) {
  const element = useRef<HTMLDialogElement>(null);
  const [value, setValue] = useState(request.initial ?? '');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const finish = (result: string | null) => {
    pending = null;
    publish();
    request.resolve(result);
  };
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    element.current?.showModal();
    element.current?.querySelector<HTMLElement>('input, button')?.focus();
    return () => {
      if (previous?.isConnected) previous.focus();
    };
  }, []);
  return (
    <dialog
      ref={element}
      className="browser-input-dialog"
      aria-labelledby="browser-dialog-title"
      data-trusted-overlay="true"
      onCancel={(event) => {
        event.preventDefault();
        finish(null);
      }}
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (request.title.includes('URL')) {
            try {
              if (!['http:', 'https:'].includes(new URL(value).protocol)) throw new Error();
            } catch {
              setError('Enter a valid HTTP or HTTPS address.');
              return;
            }
          }
          if (value.trim())
            finish(
              request.credentials ? JSON.stringify({ username: value, password }) : value.trim(),
            );
        }}
      >
        <h2 id="browser-dialog-title">{request.title}</h2>
        {request.choices ? (
          request.choices.map((choice) => (
            <button key={choice.value} type="button" onClick={() => finish(choice.value)}>
              {choice.label}
            </button>
          ))
        ) : (
          <label>
            {request.credentials ? 'Username' : request.title}
            <input
              autoComplete="off"
              required
              maxLength={
                request.title.includes('URL')
                  ? 8192
                  : /workspace|island/iu.test(request.title)
                    ? 64
                    : 512
              }
              value={value}
              onChange={(event) => setValue(event.target.value)}
            />
          </label>
        )}
        {request.credentials && (
          <label>
            Password
            <input
              type="password"
              autoComplete="off"
              maxLength={2048}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>
        )}
        {error && <p role="alert">{error}</p>}
        <div className="browser-dialog-actions">
          <button type="button" onClick={() => finish(null)}>
            Cancel
          </button>
          {!request.choices && (
            <button type="submit" disabled={!value.trim()}>
              {request.credentials ? 'Sign in' : 'Save'}
            </button>
          )}
        </div>
      </form>
    </dialog>
  );
}
