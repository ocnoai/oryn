import {
  validateModelSelection,
  validateSelectionModelRef,
  taskRequirementsFrom,
} from '@ocno/oryn-protocol';
import type { InputModality, OutputModality, ResolvedModelRef } from '@ocno/oryn-protocol';

import type { BehavioralCapability, ModelRef, ModelRequest, ModelRequirements } from './types';

const INPUT_MODALITIES = new Set<InputModality>(['text', 'image', 'audio', 'video', 'document']);
const OUTPUT_MODALITIES = new Set<OutputModality>([
  'text',
  'image',
  'audio',
  'video',
  'embedding',
  'structured',
]);
const CAPABILITIES = new Set<BehavioralCapability>([
  'chat',
  'streaming',
  'structuredOutput',
  'toolCalling',
  'imageEditing',
  'inpainting',
  'outpainting',
  'speechRecognition',
  'speechSynthesis',
  'audioOutput',
  'lyrics',
  'deterministicSeed',
  'multimodalConversation',
]);

export class ModelRequestValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelRequestValidationError';
  }
}

export function validateModelRequest(value: unknown): ModelRequest {
  const request = requireRecord(value, 'ModelRequest');
  rejectUnknown(request, [
    'role',
    'preferredModel',
    'certifiedModels',
    'requirements',
    'substitution',
    'selection',
  ]);
  if (typeof request.role !== 'string' || !/^[a-z][a-z0-9-]{0,63}$/u.test(request.role)) {
    throw new ModelRequestValidationError('ModelRequest.role must use lowercase kebab-case.');
  }
  let preferredModel =
    request.preferredModel === undefined ? undefined : validateModelRef(request.preferredModel);
  let selection: ModelRequest['selection'];
  try {
    selection =
      request.selection === undefined
        ? undefined
        : validateModelSelection(request.selection, preferredModel);
    if (selection?.intent === 'source-checkpoint') {
      if (
        preferredModel &&
        (preferredModel.repository !== selection.sourceCheckpoint.repository ||
          preferredModel.revision !== selection.sourceCheckpoint.revision ||
          preferredModel.files)
      )
        throw new Error('Source-only requests cannot also prefer an artifact.');
      preferredModel = selection.sourceCheckpoint;
    }
  } catch (error) {
    throw new ModelRequestValidationError(
      error instanceof Error ? error.message : 'Invalid selection.',
    );
  }
  let certifiedModels: ModelRef[] | undefined;
  if (request.certifiedModels !== undefined) {
    if (!Array.isArray(request.certifiedModels) || request.certifiedModels.length > 32) {
      throw new ModelRequestValidationError('certifiedModels must contain at most 32 models.');
    }
    certifiedModels = request.certifiedModels.map(validateModelRef);
  }
  const requirements =
    request.requirements === undefined ? undefined : validateRequirements(request.requirements);
  if (
    preferredModel === undefined &&
    (!requirements?.task || !requirements.outputModalities?.length)
  ) {
    throw new ModelRequestValidationError(
      'Capability-first requests require task and outputModalities.',
    );
  }
  if (preferredModel === undefined && certifiedModels !== undefined) {
    throw new ModelRequestValidationError(
      'Capability-first requests cannot declare certifiedModels.',
    );
  }
  let substitution: ModelRequest['substitution'];
  if (request.substitution !== undefined) {
    const raw = requireRecord(request.substitution, 'substitution');
    rejectUnknown(raw, ['automatic']);
    const automatic = raw.automatic ?? 'none';
    if (!['none', 'certified-only', 'same-checkpoint', 'compatible'].includes(String(automatic))) {
      throw new ModelRequestValidationError('Unknown automatic substitution mode.');
    }
    substitution = {
      automatic: automatic as 'none' | 'certified-only' | 'same-checkpoint' | 'compatible',
    };
  }
  if (selection?.intent === 'source-checkpoint' && !substitution)
    substitution = { automatic: 'same-checkpoint' };
  if (preferredModel === undefined && substitution?.automatic !== undefined) {
    if (substitution.automatic !== 'compatible') {
      throw new ModelRequestValidationError(
        'Capability-first requests require compatible selection.',
      );
    }
  }
  return preferredModel === undefined
    ? {
        role: request.role,
        requirements: requirements as ModelRequest['requirements'] & {
          task: string;
          outputModalities: OutputModality[];
        },
        substitution: { automatic: 'compatible' },
      }
    : {
        role: request.role,
        preferredModel,
        certifiedModels,
        requirements,
        substitution,
        ...(selection ? { selection } : {}),
      };
}

export function normalizeModelRef(model: ModelRef): ResolvedModelRef {
  return {
    provider: 'huggingface',
    repository: model.repository,
    revision: model.revision ?? 'main',
    ...(model.files ? { files: [...model.files].sort() } : {}),
  };
}

function validateModelRef(value: unknown): ModelRef {
  try {
    return validateSelectionModelRef(value);
  } catch (error) {
    throw new ModelRequestValidationError(
      error instanceof Error ? error.message : 'Invalid model reference.',
    );
  }
}

function validateRequirements(value: unknown): ModelRequirements {
  const raw = requireRecord(value, 'requirements');
  rejectUnknown(raw, [
    'task',
    'contract',
    'requiredFeatures',
    'parameterValues',
    'architectures',
    'parameters',
    'minimumContextLength',
    'inputModalities',
    'outputModalities',
    'capabilities',
    'memory',
  ]);
  let common;
  try {
    common = taskRequirementsFrom(raw);
  } catch {
    throw new ModelRequestValidationError('Invalid task contract requirements.');
  }
  const result: ModelRequirements = { ...common };
  if (raw.task !== undefined) result.task = boundedString(raw.task, 'task', 128);
  if (raw.architectures !== undefined) {
    result.architectures = stringArray(raw.architectures, 'architectures', 32, 128);
  }
  if (raw.parameters !== undefined) {
    const parameters = requireRecord(raw.parameters, 'parameters');
    rejectUnknown(parameters, ['minBillions', 'maxBillions']);
    const minBillions = optionalNonNegativeNumber(parameters.minBillions, 'minBillions');
    const maxBillions = optionalNonNegativeNumber(parameters.maxBillions, 'maxBillions');
    if (minBillions !== undefined && maxBillions !== undefined && minBillions > maxBillions) {
      throw new ModelRequestValidationError('minBillions cannot exceed maxBillions.');
    }
    result.parameters = { minBillions, maxBillions };
  }
  if (raw.minimumContextLength !== undefined) {
    result.minimumContextLength = positiveInteger(raw.minimumContextLength, 'minimumContextLength');
  }
  if (raw.inputModalities !== undefined) {
    result.inputModalities = enumArray(raw.inputModalities, INPUT_MODALITIES, 'inputModalities');
  }
  if (raw.outputModalities !== undefined) {
    result.outputModalities = enumArray(
      raw.outputModalities,
      OUTPUT_MODALITIES,
      'outputModalities',
    );
  }
  if (raw.capabilities !== undefined) {
    const capabilities = requireRecord(raw.capabilities, 'capabilities');
    rejectUnknown(capabilities, [...CAPABILITIES]);
    result.capabilities = {};
    for (const [name, expected] of Object.entries(capabilities)) {
      if (typeof expected !== 'boolean') {
        throw new ModelRequestValidationError(`Capability ${name} must be boolean.`);
      }
      result.capabilities[name as BehavioralCapability] = expected;
    }
  }
  if (raw.memory !== undefined) {
    const memory = requireRecord(raw.memory, 'memory');
    rejectUnknown(memory, ['maximumRecommendedBytes']);
    result.memory = {
      maximumRecommendedBytes: positiveInteger(
        memory.maximumRecommendedBytes,
        'maximumRecommendedBytes',
      ),
    };
  }
  return result;
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new ModelRequestValidationError(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function rejectUnknown(value: Record<string, unknown>, allowed: string[]): void {
  const unknown = Object.keys(value).find((key) => !allowed.includes(key));
  if (unknown) throw new ModelRequestValidationError(`Unknown field: ${unknown}.`);
}

function boundedString(value: unknown, label: string, maximum: number): string {
  if (typeof value !== 'string' || value.length < 1 || value.length > maximum) {
    throw new ModelRequestValidationError(`${label} is invalid.`);
  }
  return value;
}

function stringArray(value: unknown, label: string, maxItems: number, maxLength: number): string[] {
  if (!Array.isArray(value) || value.length > maxItems) {
    throw new ModelRequestValidationError(`${label} is invalid.`);
  }
  return [...new Set(value.map((item) => boundedString(item, label, maxLength)))];
}

function enumArray<T extends string>(value: unknown, allowed: Set<T>, label: string): T[] {
  if (!Array.isArray(value) || value.length > allowed.size) {
    throw new ModelRequestValidationError(`${label} is invalid.`);
  }
  const result = value.map((item) => {
    if (typeof item !== 'string' || !allowed.has(item as T)) {
      throw new ModelRequestValidationError(`${label} contains an unknown value.`);
    }
    return item as T;
  });
  return [...new Set(result)];
}

function optionalNonNegativeNumber(value: unknown, label: string): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new ModelRequestValidationError(`${label} must be a non-negative finite number.`);
  }
  return value;
}

function positiveInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || Number(value) <= 0) {
    throw new ModelRequestValidationError(`${label} must be a positive safe integer.`);
  }
  return Number(value);
}
