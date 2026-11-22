import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { Button, Icon, IconButton, type IconName } from './ui';

type Choice = {
  label: string;
  value: string;
  description?: string;
  icon?: IconName;
  tone?: 'danger';
  disabled?: boolean;
};

type DialogOptions = {
  description?: string;
  icon?: IconName;
};

type Request = DialogOptions & {
  title: string;
  initial?: string;
  credentials?: boolean;
  choices?: Choice[];
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
  choices: Choice[],
  options: DialogOptions = {},
): Promise<string | null> {
  return request({ ...options, title, choices });
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
    element.current
      ?.querySelector<HTMLElement>(
        'input, .browser-dialog-choice:not(:disabled), .browser-dialog-actions button',
      )
      ?.focus();
    return () => {
      if (previous?.isConnected) previous.focus();
    };
  }, []);
  return (
    <dialog
      ref={element}
      className="browser-input-dialog"
      aria-labelledby="browser-dialog-title"
      aria-describedby={request.description ? 'browser-dialog-description' : undefined}
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
        <div className="browser-dialog-header">
          {request.icon && (
            <span className="browser-dialog-mark">
              <Icon name={request.icon} />
            </span>
          )}
          <div className="browser-dialog-heading">
            <h2 id="browser-dialog-title">{request.title}</h2>
            {request.description && <p id="browser-dialog-description">{request.description}</p>}
          </div>
          <IconButton
            className="browser-dialog-close"
            icon="close"
            label={`Close ${request.title}`}
            onClick={() => finish(null)}
            type="button"
          />
        </div>
        {request.choices ? (
          <div className="browser-dialog-choices">
            {request.choices.map((choice, index) => (
              <button
                className={`browser-dialog-choice${choice.tone === 'danger' ? ' browser-dialog-choice-danger' : ''}`}
                key={choice.value}
                type="button"
                disabled={choice.disabled}
                aria-labelledby={`browser-dialog-choice-${index}`}
                aria-describedby={
                  choice.description ? `browser-dialog-choice-description-${index}` : undefined
                }
                onClick={() => finish(choice.value)}
              >
                {choice.icon && (
                  <span className="browser-dialog-choice-icon">
                    <Icon name={choice.icon} />
                  </span>
                )}
                <span className="browser-dialog-choice-copy">
                  <span id={`browser-dialog-choice-${index}`}>{choice.label}</span>
                  {choice.description && (
                    <small id={`browser-dialog-choice-description-${index}`}>
                      {choice.description}
                    </small>
                  )}
                </span>
                <Icon name="chevron-right" className="browser-dialog-choice-chevron" />
              </button>
            ))}
          </div>
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
          <Button type="button" onClick={() => finish(null)}>
            Cancel
          </Button>
          {!request.choices && (
            <Button variant="primary" type="submit" disabled={!value.trim()}>
              {request.credentials ? 'Sign in' : 'Save'}
            </Button>
          )}
        </div>
      </form>
    </dialog>
  );
}
