import { invokeAndWait, LocalAIError } from './client.js';
import { CHAT_V1_RESULT_SCHEMA, CHAT_V2_INPUT_SCHEMA, CHAT_V2_RESULT_SCHEMA, type ChatV2Input, type ChatV2Value, type EngineThinkingMetadata } from './chat-contracts.js';
import { matchesSchema } from './schema-match.js';
import { validTaskOutputRelations } from './task-relations.js';
import type { InferenceResult, LocalAIClient, LocalAIEvent, ModelSession, RuntimeCapabilities } from './types.js';

export type ChatContract = { id: 'chat-v1' | 'chat-v2'; revision: 1 };
export type ChatThinkingMetadata = EngineThinkingMetadata | {
  requested: boolean | null; applied: false; effective: null;
  reason: 'unsupported_peer'; source: 'sdk_compatibility';
};
export interface ChatResponse {
  contract: ChatContract;
  /** Original wire result. Legacy responses are never relabeled as v2. */
  result: InferenceResult;
  text: string;
  reasoning?: string;
  finishReason: 'stop' | 'length' | 'unknown';
  usage: ChatV2Value['usage'];
  metadata: { thinking: ChatThinkingMetadata };
}

function invalid(message: string): never {
  throw new LocalAIError({ code: 'JOB_INPUT_INVALID', message });
}

/** Discovery only: this function does not request a model, permission or download. */
export function selectChatContract(
  capabilities: RuntimeCapabilities,
  options: { maxTokens?: number; requireV2?: boolean } = {},
): ChatContract {
  const limit = options.maxTokens;
  if (limit !== undefined && (!Number.isInteger(limit) || limit < 1 || limit > 8192)) invalid('Choose a total generation budget from 1 to 8192 tokens.');
  if (capabilities.features.taskProfiles && capabilities.taskContracts?.some(contract => contract.id === 'chat-v2' && contract.revision === 1)) {
    return { id: 'chat-v2', revision: 1 };
  }
  if (options.requireV2 || (limit !== undefined && limit > 512)) {
    throw new LocalAIError({ code: 'MODEL_NOT_EXECUTABLE', message: 'This Oryn peer does not advertise chat-v2. Its legacy chat limit is 512 tokens and completion/reasoning metadata is unavailable.' });
  }
  return { id: 'chat-v1', revision: 1 };
}

/** Validates the canonical v2 input without treating thinking as a strict model filter. */
export function validateChatInput(input: unknown): ChatV2Input {
  if (!matchesSchema(CHAT_V2_INPUT_SCHEMA, input) || new TextEncoder().encode(JSON.stringify(input)).length > 256 * 1024) invalid('Chat input is invalid or too large.');
  return input as ChatV2Input;
}

export function normalizeChatResult(
  result: InferenceResult, contract: { id: string; revision: number }, requestedThinking?: boolean,
): ChatResponse {
  if (requestedThinking !== undefined && typeof requestedThinking !== 'boolean') invalid('Thinking must be a boolean or omitted.');
  if (contract.revision !== 1 || !['chat-v1', 'chat-v2'].includes(contract.id)) invalid('Unsupported chat contract.');
  if (new TextEncoder().encode(JSON.stringify(result)).length > 256 * 1024) throw new LocalAIError({ code: 'RUNTIME_PROTOCOL_ERROR', message: 'The chat result exceeds its bounded size.' });
  if (contract.id === 'chat-v2') {
    if (!matchesSchema(CHAT_V2_RESULT_SCHEMA, result) || result.kind !== 'structured' || !validTaskOutputRelations('chat-v2', result.value as Record<string, unknown>)) {
      throw new LocalAIError({ code: 'RUNTIME_PROTOCOL_ERROR', message: 'The local model returned an invalid chat-v2 result.' });
    }
    const value = result.value as ChatV2Value;
    if (value.metadata.thinking.requested !== (requestedThinking ?? null)) throw new LocalAIError({ code: 'RUNTIME_PROTOCOL_ERROR', message: 'The thinking result does not match the request.' });
    return { contract: { id: 'chat-v2', revision: 1 }, result, ...value };
  }
  if (!matchesSchema(CHAT_V1_RESULT_SCHEMA, result) || result.kind !== 'structured') throw new LocalAIError({ code: 'RUNTIME_PROTOCOL_ERROR', message: 'The peer returned an invalid legacy chat result.' });
  const raw = (result.value as { text: string }).text;
  // Never strip arbitrary literal tags. A recognized reasoning opener makes the
  // legacy output unsafe as an answer; preserve it solely in the untouched result.
  const leading = raw.trimStart().toLowerCase();
  const ambiguous = ['<think>', '<analysis>'].some(marker => leading.startsWith(marker) || marker.startsWith(leading)) || /<\/(?:think|analysis)>/i.test(raw);
  const text = ambiguous ? '' : raw;
  return {
    contract: { id: 'chat-v1', revision: 1 }, result, text, finishReason: 'unknown',
    usage: { inputTokens: null, outputTokens: null, reasoningTokens: null, answerTokens: null },
    metadata: { thinking: { requested: requestedThinking ?? null, applied: false, effective: null, reason: 'unsupported_peer', source: 'sdk_compatibility' } },
  };
}

/** Uses a negotiated session; fallback is chosen before invoking, never by retrying a job. */
export async function invokeChat(
  client: LocalAIClient, session: ModelSession, input: ChatV2Input,
  options: { signal?: AbortSignal; onEvent?: (event: LocalAIEvent) => void } = {},
): Promise<ChatResponse> {
  validateChatInput(input);
  const contract = session.taskProfile?.contract ?? { id: 'chat-v1', revision: 1 };
  if (contract.revision !== 1 || !['chat-v1', 'chat-v2'].includes(contract.id)) invalid('This session does not expose a supported chat contract.');
  if (contract.id !== 'chat-v2' && input.maxTokens !== undefined && input.maxTokens > 512) invalid('This legacy session supports at most 512 output tokens; the budget was not reduced.');
  const { thinking, ...legacy } = input;
  const result = await invokeAndWait(client, { sessionId: session.sessionId, input: contract.id === 'chat-v2' ? input : legacy }, options);
  const response = normalizeChatResult(result, contract, thinking);
  const defaultBudget = session.taskProfile?.parameters.maxTokens?.default;
  const budget = input.maxTokens ?? (typeof defaultBudget === 'number' && Number.isFinite(defaultBudget) ? defaultBudget : contract.id === 'chat-v2' ? 4096 : 128);
  if (response.usage.outputTokens !== null && response.usage.outputTokens > budget) {
    throw new LocalAIError({ code: 'RUNTIME_PROTOCOL_ERROR', message: 'The returned usage exceeds the requested output budget.' });
  }
  return response;
}
