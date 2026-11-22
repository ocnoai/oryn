import { useEffect, useRef, useState } from 'react';
import type { TrustedAlternativeSearch } from '../../../shared/ipc';
import { formatStorageSize } from '../../../shared/format-storage-size';
import { ModelMemoryRequirements } from './ModelMemoryRequirements';

export function ModelAlternatives({
  installationId,
  availableBytes,
}: {
  installationId: string;
  availableBytes?: number;
}) {
  const [state, setState] = useState<'idle' | 'searching' | 'done' | 'cancelled' | 'error'>('idle');
  const [result, setResult] = useState<TrustedAlternativeSearch>();
  const [role, setRole] = useState('');
  const [choice, setChoice] = useState('');
  const generation = useRef(0);
  useEffect(
    () => () => {
      generation.current++;
      void window.browser.cancelBetterAlternatives();
    },
    [],
  );
  const search = async () => {
    const current = ++generation.current;
    setState('searching');
    setResult(undefined);
    try {
      const value = await window.browser.checkBetterAlternatives(installationId);
      if (current === generation.current) {
        setResult(value);
        setRole(value.suggestedRole);
        setChoice('');
        setState('done');
      }
    } catch {
      if (current === generation.current) setState('error');
    }
  };
  return (
    <section aria-label="Alternative model representations">
      <button type="button" disabled={state === 'searching'} onClick={() => void search()}>
        Check better alternatives
      </button>
      {state === 'searching' && (
        <button
          type="button"
          onClick={() => {
            generation.current++;
            void window.browser.cancelBetterAlternatives();
            setState('cancelled');
          }}
        >
          Cancel search
        </button>
      )}
      <p role="status">
        {state === 'searching'
          ? 'Searching public model metadata and checking local support…'
          : state === 'cancelled'
            ? 'Search cancelled.'
            : state === 'error'
              ? 'Search unavailable. Check your connection and try again.'
              : result?.status === 'no-results'
                ? 'No same-checkpoint alternatives were found within this search.'
                : result?.status === 'no-better-option'
                  ? 'No locally executable improvement was found.'
                  : 'Search sends the public checkpoint identity to Hugging Face. It does not download model weights or switch sessions.'}
      </p>
      {!!result?.variants.length && (
        <>
          <label>
            Preference role{' '}
            <input value={role} onChange={(e) => setRole(e.target.value)} maxLength={64} />
          </label>
          <p>
            Saving a preference applies to future requests for this role. Oryn checks support again
            and requests download approval when needed. Exact artifact pins remain binding.
          </p>
          {choice && <p role="status">{choice}</p>}
        </>
      )}
      {result?.variants.map((v) => (
        <article key={v.model.repository + v.model.revision + JSON.stringify(v.model.files)}>
          <strong>{v.model.repository}</strong>
          <p>
            {v.model.files?.join(', ')} · Lineage: {v.lineage} ·{' '}
            {v.executable ? 'Locally executable; performance unknown' : 'Not currently executable'}
          </p>
          <p>
            {v.downloadBytes === undefined
              ? 'Download size unknown'
              : `Download: ${formatStorageSize(v.downloadBytes)}`}
          </p>
          <p>{v.reasons.join(' ')}</p>
          <ModelMemoryRequirements
            requiredBytes={v.requiredMemoryBytes}
            availableBytes={availableBytes}
          />
          <button
            type="button"
            disabled={!v.executable || !/^[a-z][a-z0-9-]{0,63}$/u.test(role)}
            onClick={async () => {
              setChoice('Checking this exact representation…');
              try {
                await window.browser.updateOrynRolePreference({ role, model: v.model });
                setChoice(`Saved ${v.model.repository} for future ${role} requests.`);
              } catch {
                setChoice(
                  'This representation could not be selected. Search again to refresh its support.',
                );
              }
            }}
          >
            Prefer this representation
          </button>
          <button
            type="button"
            onClick={() =>
              void window.browser.openModelSource(v.model.repository, v.model.revision)
            }
          >
            Review source
          </button>
        </article>
      ))}
    </section>
  );
}
