import type { Writable } from 'node:stream';

import type { ProtocolError } from '@ocno/oryn-protocol';

export const MAX_PENDING_PROTOCOL_REQUESTS = 1024;
export const MAX_DIAGNOSTIC_BYTES = 64 * 1024;

export class SerializedProtocolWriter {
  private tail = Promise.resolve();
  private queued = 0;

  constructor(
    private readonly stream: Writable,
    private readonly maxQueued = MAX_PENDING_PROTOCOL_REQUESTS,
  ) {}

  write(serialized: string): Promise<void> {
    if (this.queued >= this.maxQueued) {
      return Promise.reject(new Error('Protocol write queue is full.'));
    }
    this.queued += 1;
    const operation = this.tail
      .catch(() => undefined)
      .then(
        () =>
          new Promise<void>((resolve, reject) => {
            if (!this.stream.writable) {
              reject(new Error('Local AI Engine stdin is unavailable.'));
              return;
            }
            this.stream.write(serialized, 'utf8', (error) => {
              if (error) reject(error);
              else resolve();
            });
          }),
      );
    this.tail = operation.finally(() => {
      this.queued -= 1;
    });
    return operation;
  }
}

export class EngineDiagnosticBuffer {
  private value = '';

  append(text: string): void {
    this.value = `${this.value}${text}`.slice(-MAX_DIAGNOSTIC_BYTES);
  }

  read(): string {
    return this.value;
  }

  clear(): void {
    this.value = '';
  }
}

export function sanitizeProtocolError(error: ProtocolError): ProtocolError {
  const safeDetails = sanitizeResourceDetails(error.details);
  return {
    code: error.code,
    category: error.category,
    message: containsSensitiveDiagnostic(error.message)
      ? 'The local AI runtime could not complete the request.'
      : error.message,
    retryable: error.retryable,
    ...(safeDetails ? { details: safeDetails } : {}),
    ...(error.diagnosticId ? { diagnosticId: error.diagnosticId } : {}),
  };
}

function sanitizeResourceDetails(
  value: unknown,
): Record<string, number | string | null> | undefined {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined;
  const input = value as Record<string, unknown>;
  const result: Record<string, number | string | null> = {};
  for (const key of [
    'requiredBytes',
    'availableBytes',
    'availableForModelBytes',
    'memoryBudgetBytes',
    'currentResidentModelBytes',
    'processOverheadBytes',
  ]) {
    const item = input[key];
    if (item === null || (typeof item === 'number' && Number.isSafeInteger(item) && item >= 0)) {
      result[key] = item;
    }
  }
  if (['normal', 'high', 'critical'].includes(String(input.memoryPressure))) {
    result.memoryPressure = String(input.memoryPressure);
  }
  return Object.keys(result).length > 0 ? result : undefined;
}

function containsSensitiveDiagnostic(message: string): boolean {
  return (
    /(?:^|\s)(?:\/[\w.-]+){2,}/u.test(message) ||
    /[A-Za-z]:\\/u.test(message) ||
    /traceback|environment variable|auth(?:entication)? token/iu.test(message)
  );
}
