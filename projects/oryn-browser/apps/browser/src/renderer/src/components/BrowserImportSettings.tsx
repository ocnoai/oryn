import { useEffect, useState } from 'react';
import type {
  BrowserImportDataType,
  BrowserImportPreview,
  BrowserImportSource,
} from '../../../main/data/browser-import-types';
import { Button } from './ui';

export function BrowserImportSettings({ onImported }: { onImported: () => Promise<void> }) {
  const [sources, setSources] = useState<BrowserImportSource[]>([]);
  const [sourceId, setSourceId] = useState('');
  const [types, setTypes] = useState<BrowserImportDataType[]>(['bookmarks', 'history']);
  const [preview, setPreview] = useState<BrowserImportPreview>();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const source = sources.find((item) => item.sourceId === sourceId);
  useEffect(() => {
    let current = true;
    void window.browser
      .discoverBrowserImports()
      .then((items) => {
        if (current) {
          setSources(items);
          setSourceId(items[0]?.sourceId ?? '');
        }
      })
      .catch(() => {
        if (current) setMessage('Browser profiles could not be read. Try again later.');
      });
    return () => {
      current = false;
      void window.browser.cancelBrowserImport();
    };
  }, []);
  const cancel = async () => {
    await window.browser.cancelBrowserImport(preview?.token);
    setPreview(undefined);
    setBusy(false);
  };
  const start = async () => {
    setBusy(true);
    setMessage('');
    try {
      const result = await window.browser.previewBrowserImport({
        sourceId,
        dataTypes: types.filter((type) => source?.dataTypes.includes(type)),
      });
      setPreview(result);
    } catch (error) {
      const code = String(error);
      setMessage(
        code.includes('SOURCE_CHANGED')
          ? 'The source browser is changing its data. Close it and try again.'
          : code.includes('SANDBOX_UNAVAILABLE')
            ? 'Secure browser-data import is unavailable in this build.'
            : 'Import could not be prepared. Check the selected profile or export and try again.',
      );
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="settings-grid" aria-label="Import browser data">
      <h3>Import browser data</h3>
      <p>
        Bring bookmarks and history from another browser. Passwords, cookies, payment details and
        extensions are excluded.
      </p>
      <label>
        Browser and profile
        <select
          value={sourceId}
          disabled={busy || Boolean(preview)}
          onChange={(event) => setSourceId(event.target.value)}
        >
          {sources.map((item) => (
            <option key={item.sourceId} value={item.sourceId}>
              {item.browserLabel} — {item.profileLabel}
            </option>
          ))}
        </select>
      </label>
      {source?.browser === 'safari' && (
        <p>
          In Safari 18.2 or later, choose File → Export Browsing Data to File, then select that ZIP
          here. The export is unencrypted; delete it when finished. Older Safari supports bookmark
          HTML import in Bookmarks.
        </p>
      )}
      {(['bookmarks', 'history', 'searchEngine'] as const)
        .filter((type) => source?.dataTypes.includes(type))
        .map((type) => (
          <label key={type}>
            <input
              type="checkbox"
              disabled={busy || Boolean(preview)}
              checked={types.includes(type)}
              onChange={(event) =>
                setTypes((current) =>
                  event.target.checked
                    ? [...current, type]
                    : current.filter((item) => item !== type),
                )
              }
            />
            {type === 'searchEngine'
              ? 'Supported default search engine'
              : type === 'history'
                ? 'Browsing history'
                : 'Bookmarks'}
          </label>
        ))}
      {!preview && (
        <Button
          disabled={busy || !sourceId || !types.some((type) => source?.dataTypes.includes(type))}
          onClick={() => void start()}
        >
          {busy
            ? 'Preparing preview…'
            : source?.browser === 'safari'
              ? 'Choose Safari export and preview…'
              : 'Preview import'}
        </Button>
      )}
      {busy && (
        <Button variant="secondary" onClick={() => void cancel()}>
          Cancel import
        </Button>
      )}
      {preview && (
        <div role="region" aria-label="Import preview">
          <p>
            {preview.sourceLabel}: {preview.counts.bookmarks} bookmarks, {preview.counts.folders}{' '}
            folders and {preview.counts.history} history entries.
          </p>
          {preview.skipped.map((entry) => (
            <p key={entry.reason}>
              Skipped {entry.count}: {entry.reason.replaceAll('-', ' ')}.
            </p>
          ))}
          {preview.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
          <p>This preview expires after five minutes. Import starts only when you confirm.</p>
          <Button
            disabled={busy}
            onClick={() => {
              setBusy(true);
              void window.browser
                .commitBrowserImport(preview.token)
                .then(async (result) => {
                  setPreview(undefined);
                  setMessage(
                    `Imported ${result.bookmarks} bookmarks and ${result.history} history entries.`,
                  );
                  await onImported();
                })
                .catch((error) => {
                  setPreview(undefined);
                  setMessage(
                    String(error).includes('IMPORT_PREVIEW_EXPIRED')
                      ? 'The preview expired. Prepare a new preview.'
                      : 'Import could not finish. Check the imported data before retrying.',
                  );
                })
                .finally(() => setBusy(false));
            }}
          >
            Confirm import
          </Button>
          <Button variant="secondary" disabled={busy} onClick={() => void cancel()}>
            Cancel
          </Button>
        </div>
      )}
      {message && <p role="status">{message}</p>}
    </section>
  );
}
