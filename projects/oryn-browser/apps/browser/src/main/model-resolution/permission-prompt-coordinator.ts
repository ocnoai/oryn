import { randomUUID } from 'node:crypto';
import { EventEmitter } from 'node:events';

import type { LocalAIPermissionPrompt, LocalAIPermissionPromptResponse } from '../../shared/ipc';
import {
  isOrynVerified,
  isUserTrustEligible,
} from '../model-workspace/model-workspace-eligibility';
import type { ModelResolutionResult } from './types';

interface PendingPrompt {
  origin: string;
  resolve(value: LocalAIPermissionPromptResponse): void;
  reject(error: Error): void;
  timeout: NodeJS.Timeout;
  signal?: AbortSignal;
  abort?: () => void;
  choiceRequested?: () => void;
}

export class PermissionPromptCoordinator extends EventEmitter {
  static readonly MAX_PENDING_PROMPTS = 32;
  private readonly pending = new Map<string, PendingPrompt>();

  request(
    origin: string,
    role: string,
    result: ModelResolutionResult,
    signal?: AbortSignal,
    choiceRequested?: () => void,
  ): Promise<LocalAIPermissionPromptResponse> {
    const promptId = `permission_${randomUUID().replaceAll('-', '')}`;
    const prompt = toPrompt(promptId, origin, role, result);
    return new Promise((resolve, reject) => {
      if (signal?.aborted) {
        reject(new Error('REQUEST_ABORTED'));
        return;
      }
      if (
        this.pending.size >= PermissionPromptCoordinator.MAX_PENDING_PROMPTS ||
        [...this.pending.values()].some((item) => item.origin === origin)
      ) {
        reject(new Error('LOCAL_AI_PERMISSION_PROMPT_BUSY'));
        return;
      }
      const abort = () => {
        const active = this.pending.get(promptId);
        if (!active) return;
        clearTimeout(active.timeout);
        this.pending.delete(promptId);
        this.emit('cancelled', promptId);
        reject(new Error('REQUEST_ABORTED'));
      };
      const timeout = setTimeout(() => {
        this.pending.delete(promptId);
        signal?.removeEventListener('abort', abort);
        reject(new Error('LOCAL_AI_PERMISSION_PROMPT_TIMEOUT'));
      }, 5 * 60_000);
      this.pending.set(promptId, {
        origin,
        resolve,
        reject,
        timeout,
        signal,
        abort,
        choiceRequested,
      });
      signal?.addEventListener('abort', abort, { once: true });
      this.emit('prompt', prompt);
    });
  }

  requestChoiceFromTrustedUI(promptId: string): boolean {
    const pending = this.pending.get(promptId);
    if (!pending) return false;
    pending.choiceRequested?.();
    return true;
  }

  respondFromTrustedUI(value: LocalAIPermissionPromptResponse): boolean {
    const pending = this.pending.get(value.promptId);
    if (!pending || !['allow-once', 'always-allow', 'deny'].includes(value.action)) return false;
    clearTimeout(pending.timeout);
    if (pending.abort) pending.signal?.removeEventListener('abort', pending.abort);
    this.pending.delete(value.promptId);
    pending.resolve(value);
    return true;
  }
}

function toPrompt(
  promptId: string,
  origin: string,
  role: string,
  result: ModelResolutionResult,
): LocalAIPermissionPrompt {
  const selected = result.selected;
  return {
    promptId,
    origin,
    role,
    ...(result.resolution.selection ? { selection: result.resolution.selection } : {}),
    requestedRepository: result.resolution.preferredModel.repository,
    resolvedModel: result.resolution.resolvedModel,
    resolvedRepository: result.resolution.resolvedModel.repository,
    resolutionReason: result.resolution.resolutionReason,
    compatibility: result.resolution.compatibility,
    developerCertified: result.resolution.developerCertified,
    substituted: result.resolution.substituted,
    provider: 'huggingface',
    exactRevision: result.resolution.resolvedModel.revision,
    installed: selected.installed,
    downloadBytes: selected.profile.estimatedDiskBytes,
    estimatedMemoryBytes: selected.profile.estimatedMemoryBytes,
    candidates: result.candidates.map((candidate) => ({
      model: candidate.model,
      compatibility: candidate.compatibility,
      developerCertified: candidate.developerCertified,
      installed: candidate.installed,
      executable: candidate.executability === 'executable',
      securityAllowed: candidate.security === 'allowed',
      trustRequired: !isOrynVerified(candidate.profile) && isUserTrustEligible(candidate.profile),
      blockingReasons: candidate.profile.blockingReasons ?? [],
      estimatedDiskBytes: candidate.profile.estimatedDiskBytes,
      estimatedMemoryBytes: candidate.profile.estimatedMemoryBytes,
    })),
  };
}
