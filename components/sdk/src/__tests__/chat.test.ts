import { afterEach, describe, expect, it, vi } from 'vitest';
import fixture from './fixtures/chat-v2.json';
import { invokeChat, normalizeChatResult, selectChatContract, validateChatInput } from '../chat.js';
import type { InferenceResult, LocalAIClient, LocalAIEvent, ModelSession, RuntimeCapabilities } from '../types.js';
import { createLocalAIClient } from '../client.js';
import type { InjectedLocalAI } from '../types.js';

const capabilities = { features: { taskProfiles: true }, taskContracts: [{ id: 'chat-v2', revision: 1 }] } as RuntimeCapabilities;
describe('versioned chat helper', () => {
  afterEach(() => vi.unstubAllGlobals());
  it('uses the actual injected SDK boundary and preserves unsupported explicit false', async () => {
    let listener: ((event: LocalAIEvent) => void) | undefined;
    const output = fixture.cases.find(entry => entry.name === 'unsupported')!.result as InferenceResult;
    const unavailable = async (): Promise<never> => { throw new Error('Unexpected media operation.'); };
    const invokeJob = vi.fn<InjectedLocalAI['invokeJob']>(async request => {
      expect(request.input).toMatchObject({ thinking: false, maxTokens: 4096 });
      for (const [index, channel] of ['reasoning', 'text'].entries()) listener!({ event: 'job.output.delta', jobId: 'job', sessionId: 'session', sequence: index + 1, timestamp: 1, data: { channel, delta: channel === 'text' ? 'A final synthetic answer.' : 'Synthetic reasoning.' } });
      listener!({ event: 'job.completed', jobId: 'job', sessionId: 'session', sequence: 3, timestamp: 1, data: { result: output } });
      return { ok: true, value: { jobId: 'job', sessionId: 'session', state: 'running' } };
    });
    const bridge: InjectedLocalAI = {
      sdkVersion: '0.1.0', getCapabilities: async () => capabilities,
      requestModel: async () => ({ ok: true, value: { sessionId: 'session', taskProfile: fixture.profiles.unsupported, capabilities: { task: 'text-generation' } } as unknown as ModelSession }),
      cancelModelRequest: async () => undefined, releaseModelSession: async () => ({ ok: true, value: undefined }),
      onModelRequestProgress: () => () => undefined,
      invokeJob, cancelJob: async () => ({ ok: true, value: undefined }),
      onJobEvent: callback => { listener = callback; return () => { listener = undefined; }; },
      beginArtifactCreate: unavailable, appendArtifactChunk: unavailable, finishArtifactCreate: unavailable,
      abortArtifactCreate: unavailable, getArtifact: unavailable, getArtifactUrl: unavailable, saveArtifactAs: unavailable, releaseArtifact: unavailable,
    };
    vi.stubGlobal('window', { localAI: bridge });
    const client = createLocalAIClient();
    const session = await client.models.request({ role: 'chat', requirements: { task: 'text-generation', outputModalities: ['text'], contract: { id: 'chat-v2', revision: 1 } } });
    const onEvent = vi.fn();
    const response = await invokeChat(client, session, { messages: [{ role: 'user', content: 'Synthetic question' }], thinking: false, maxTokens: 4096 }, { onEvent });
    expect(response.metadata.thinking).toMatchObject({ requested: false, applied: false, reason: 'unsupported_model', source: 'engine' });
    expect(onEvent.mock.calls.map(([event]) => event.event)).toEqual(['job.output.delta', 'job.output.delta', 'job.completed']);
    await expect(client.invoke({ sessionId: session.sessionId, input: { messages: [{ role: 'user', content: 'Question' }], thinking: 'false' } })).rejects.toMatchObject({ code: 'JOB_INPUT_INVALID' });
    expect(invokeJob).toHaveBeenCalledOnce(); expect(listener).toBeUndefined();
  });
  it('uses explicit discovery and never shrinks unsupported budgets', () => {
    expect(selectChatContract(capabilities)).toEqual({ id: 'chat-v2', revision: 1 });
    expect(selectChatContract({ ...capabilities, taskContracts: undefined })).toEqual({ id: 'chat-v1', revision: 1 });
    expect(() => selectChatContract({ ...capabilities, taskContracts: undefined }, { maxTokens: 4096 })).toThrow(/512/);
    expect(() => selectChatContract(capabilities, { maxTokens: 8193 })).toThrow();
    expect(() => validateChatInput({ messages: [{ role: 'user', content: 'Question' }], thinking: 'false' })).toThrow();
  });
  for (const entry of fixture.cases) it(`normalizes ${entry.name} without losing false`, () => {
    const response = normalizeChatResult(entry.result as InferenceResult, { id: 'chat-v2', revision: 1 }, entry.input.thinking);
    expect(response.metadata.thinking.requested).toBe(entry.input.thinking ?? null);
    expect(response.result).toBe(entry.result);
    expect(response.finishReason).toBe('stop');
  });
  it('keeps legacy result unchanged and metadata explicitly SDK-authored', () => {
    const result = { kind: 'structured', value: { text: 'Legacy answer' } } as const;
    const output = normalizeChatResult(result, { id: 'chat-v1', revision: 1 }, false);
    expect(output.result).toBe(result); expect(result.value).not.toHaveProperty('metadata');
    expect(output.metadata.thinking).toEqual({ requested: false, applied: false, effective: null, reason: 'unsupported_peer', source: 'sdk_compatibility' });
    expect(output.finishReason).toBe('unknown'); expect(output.usage.outputTokens).toBeNull();
    for (const text of ['<thi', '<think', '<think>reasoning</think>answer', 'Reasoning first</think>answer']) {
      expect(normalizeChatResult({ kind: 'structured', value: { text } }, { id: 'chat-v1', revision: 1 }, false).text).toBe('');
    }
    expect(normalizeChatResult({ kind: 'structured', value: { text: 'A literal <think> example.' } }, { id: 'chat-v1', revision: 1 }).text).toBe('A literal <think> example.');
  });
  it('rejects incompatible metadata, invented peer attestation and overlong results', () => {
    const result = structuredClone(fixture.cases[0]!.result);
    result.value.metadata.thinking.effective = true;
    expect(() => normalizeChatResult(result as InferenceResult, { id: 'chat-v2', revision: 1 }, false)).toThrow();
    expect(() => normalizeChatResult(fixture.cases[0]!.result as InferenceResult, { id: 'chat-v2', revision: 1 }, true)).toThrow();
    expect(() => normalizeChatResult({ kind: 'structured', value: { text: 'x'.repeat(262144) } }, { id: 'chat-v1', revision: 1 })).toThrow(/bounded/);
  });
  it('subscribes before invoking, strips only the unsupported legacy field, and performs one job', async () => {
    let listener: ((event: LocalAIEvent) => void) | undefined;
    let calls = 0;
    const client = {
      subscribe: (value: (event: LocalAIEvent) => void) => { listener = value; return () => { listener = undefined; }; },
      invoke: async (request: { sessionId: string; input: Record<string, unknown> }) => {
        calls++; expect(request.input).not.toHaveProperty('thinking'); expect(request.input.maxTokens).toBe(256);
        listener!({ event: 'job.completed', jobId: 'job', sessionId: 'session', sequence: 1, timestamp: 1, data: { result: { kind: 'structured', value: { text: 'Done' } } } });
        return { jobId: 'job', sessionId: 'session', state: 'running' };
      }, cancel: async () => undefined,
    } as unknown as LocalAIClient;
    const response = await invokeChat(client, { sessionId: 'session' } as ModelSession, { messages: [{ role: 'user', content: 'Question' }], thinking: false, maxTokens: 256 });
    expect(response.metadata.thinking.applied).toBe(false); expect(calls).toBe(1); expect(listener).toBeUndefined();
  });
});
