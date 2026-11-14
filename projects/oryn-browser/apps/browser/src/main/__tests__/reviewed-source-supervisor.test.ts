import { afterEach, describe, expect, it, vi } from 'vitest';
import type { CandidateExecutionProfile, EngineRuntimeCapabilities } from '@ocno/oryn-protocol';
import { EngineSupervisor } from '../engine-supervisor';
import profiles from './fixtures/unverified-chat-profiles.json';

afterEach(() => vi.useRealTimers());

const context = {
  sourceURL: `https://huggingface.co/${profiles[0]!.model.repository}`,
  tabId: 'tab_hf',
  documentId: 'doc_hf',
};
const candidate = () => structuredClone(profiles[0]) as CandidateExecutionProfile;

function supervisor(enabled: boolean) {
  const engine = new EngineSupervisor();
  vi.spyOn(engine, 'getRuntimeCapabilities').mockResolvedValue({
    features: { reviewedSourceInspection: enabled },
  } as EngineRuntimeCapabilities);
  return engine;
}

describe('private reviewed-source supervision', () => {
  it('keeps an older peer on ordinary inspection and reports unavailable recognition', async () => {
    const engine = supervisor(false);
    const profile = { ...candidate(), requiresCustomCode: true };
    vi.spyOn(engine, 'inspectModel').mockResolvedValue(profile);
    const request = vi.spyOn(engine, 'request');
    const result = await engine.inspectHuggingFaceModel(profile.model, context);
    expect(result.sourceRecognition?.state).toBe('inspection-unavailable');
    expect(request).not.toHaveBeenCalled();
  });

  it('retries document cancellation if it overtakes Engine registration', async () => {
    vi.useFakeTimers();
    const engine = supervisor(true);
    const profile = candidate();
    let finish!: (value: unknown) => void;
    vi.spyOn(engine, 'request').mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const cancel = vi
      .spyOn(engine, 'cancelModelInstall')
      .mockResolvedValueOnce(false)
      .mockResolvedValue(true);
    const abort = new AbortController();
    const pending = engine.inspectHuggingFaceModel(profile.model, context, abort.signal);
    await vi.advanceTimersByTimeAsync(0);
    abort.abort();
    await vi.advanceTimersByTimeAsync(100);
    expect(cancel).toHaveBeenCalledTimes(2);
    expect(cancel.mock.calls.at(0)?.[0]).toBe(
      'source_5091ce246f93c9546725de079218b6035f71735251eb5e15cedf10bf1e290675',
    );
    finish(profile);
    await expect(pending).resolves.toEqual(profile);
    await vi.advanceTimersByTimeAsync(1000);
    expect(cancel).toHaveBeenCalledTimes(2);
  });

  it('validates the private install result with the ordinary closed result contract', async () => {
    const engine = supervisor(true);
    vi.spyOn(engine, 'request').mockResolvedValue({ accepted: true });
    await expect(
      engine.installHuggingFaceModel('install_source', candidate().model, context),
    ).rejects.toThrow();
  });
});
