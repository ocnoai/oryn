import { validateTaskProfile, type CandidateExecutionProfile } from '@ocno/oryn-protocol';

import type { ModelWorkspaceTaskContract } from '../../shared/ipc';
import { isModelWorkspaceTaskContract } from './task-contract-registry';

export type ModelWorkspaceTrustMode = 'oryn-verified' | 'user-trusted';

export function taskContractForProfile(
  profile: CandidateExecutionProfile,
): ModelWorkspaceTaskContract | undefined {
  const signedContract = profile.runtimePack?.taskContract;
  if (profile.taskProfile !== undefined) {
    try {
      // The Engine's effective profile is authoritative across bundled runners. Older
      // capability flags can be incomplete even when this versioned contract is supported.
      const contract = validateTaskProfile(profile.taskProfile).contract.id;
      if (
        !isModelWorkspaceTaskContract(contract) ||
        (signedContract !== undefined && signedContract !== contract)
      ) {
        return undefined;
      }
      return contract;
    } catch {
      // A present invalid profile must not acquire authority through a legacy fallback.
      return undefined;
    }
  }
  if (signedContract !== undefined) {
    return isModelWorkspaceTaskContract(signedContract) ? signedContract : undefined;
  }
  if (profile.runner !== 'transformers') return undefined;

  const technical = profile.technicalCapabilities;
  const capabilities = technical.capabilities ?? {};
  const inputModalities = technical.inputModalities ?? [];
  const outputModalities = technical.outputModalities ?? [];
  if (
    technical.task === 'text-generation' &&
    inputModalities.includes('text') &&
    outputModalities.includes('text') &&
    capabilities.chat === true
  ) {
    return 'chat-v1';
  }
  if (
    technical.task === 'automatic-speech-recognition' &&
    inputModalities.includes('audio') &&
    outputModalities.includes('text')
  ) {
    return 'transcription-v1';
  }
  if (
    technical.task === 'text-to-audio' &&
    inputModalities.includes('text') &&
    outputModalities.includes('audio') &&
    capabilities.audioOutput === true
  ) {
    return 'text-to-music-v1';
  }
  if (
    technical.task === 'feature-extraction' &&
    inputModalities.includes('text') &&
    outputModalities.includes('embedding')
  ) {
    return 'embeddings-v1';
  }
  if (
    technical.task === 'text-ranking' &&
    inputModalities.includes('text') &&
    outputModalities.includes('structured')
  ) {
    return 'reranking-v1';
  }
  const boundedTaskContracts: Partial<Record<string, ModelWorkspaceTaskContract>> = {
    'zero-shot-classification': 'zero-shot-classification-v1',
    'image-feature-extraction': 'image-embeddings-v1',
    'image-segmentation': 'image-segmentation-v1',
    'mask-generation': 'mask-generation-v1',
    'zero-shot-object-detection': 'zero-shot-object-detection-v1',
    'image-to-text': 'image-to-text-v1',
    'visual-question-answering': 'visual-question-answering-v1',

    'text-classification': 'text-classification-v1',
    'token-classification': 'token-classification-v1',
    'question-answering': 'question-answering-v1',
    'fill-mask': 'fill-mask-v1',
    translation: 'translation-v1',
    summarization: 'summarization-v1',
    'image-classification': 'image-classification-v1',
    'audio-classification': 'audio-classification-v1',
    'object-detection': 'object-detection-v1',
    'depth-estimation': 'depth-estimation-v1',
    'zero-shot-image-classification': 'zero-shot-image-classification-v1',
  };
  return technical.task ? boundedTaskContracts[technical.task] : undefined;
}

export function isUserTrustEligible(profile: CandidateExecutionProfile): boolean {
  const nonOverridableReasons = (profile.blockingReasons ?? []).filter(
    (reason) => reason !== 'CONFORMANCE_REQUIRED',
  );
  if (
    !profile.executable ||
    profile.requiresCustomCode ||
    nonOverridableReasons.length > 0 ||
    !taskContractForProfile(profile)
  ) {
    return false;
  }
  if (profile.loadDisposition !== undefined) {
    return profile.loadDisposition === 'trust-required';
  }
  if (profile.runtimePack) {
    return profile.runtimePack.trustedMetadata?.conformance.status === 'fixture-only';
  }
  return profile.runner === 'transformers';
}

export function isOrynVerified(profile: CandidateExecutionProfile): boolean {
  if (profile.loadDisposition !== undefined && profile.loadDisposition !== 'verified-load') {
    return false;
  }
  return Boolean(
    profile.executable &&
    profile.runtimePack &&
    isModelWorkspaceTaskContract(profile.runtimePack.taskContract) &&
    taskContractForProfile(profile) === profile.runtimePack.taskContract &&
    profile.runtimePack.trustedMetadata?.conformance.status !== 'fixture-only',
  );
}
