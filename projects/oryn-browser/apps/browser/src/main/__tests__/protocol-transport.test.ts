import { Writable } from 'node:stream';

import { describe, expect, it } from 'vitest';

import {
  EngineDiagnosticBuffer,
  sanitizeProtocolError,
  SerializedProtocolWriter,
} from '../protocol-transport';

describe('SerializedProtocolWriter', () => {
  it('serializes concurrent writes and bounds the queue', async () => {
    const chunks: string[] = [];
    const stream = new Writable({
      write(chunk, _encoding, callback) {
        setTimeout(
          () => {
            chunks.push(chunk.toString());
            callback();
          },
          chunk.toString().includes('one') ? 5 : 0,
        );
      },
    });
    const writer = new SerializedProtocolWriter(stream, 2);
    const first = writer.write('one\n');
    const second = writer.write('two\n');
    await expect(writer.write('three\n')).rejects.toThrow('queue is full');
    await Promise.all([first, second]);
    expect(chunks).toEqual(['one\n', 'two\n']);
  });

  it('retains only allowlisted resource measurements for trusted error handling', () => {
    expect(
      sanitizeProtocolError({
        code: 'MODEL_MEMORY_INSUFFICIENT',
        category: 'resource',
        message: 'The model cannot be admitted under the current memory policy.',
        retryable: true,
        details: {
          requiredBytes: 20_000_000_000,
          availableForModelBytes: 15_000_000_000,
          memoryBudgetBytes: null,
          memoryPressure: 'normal',
          secretPath: '/Users/alice/private',
        },
      }),
    ).toEqual({
      code: 'MODEL_MEMORY_INSUFFICIENT',
      category: 'resource',
      message: 'The model cannot be admitted under the current memory policy.',
      retryable: true,
      details: {
        requiredBytes: 20_000_000_000,
        availableForModelBytes: 15_000_000_000,
        memoryBudgetBytes: null,
        memoryPressure: 'normal',
      },
    });
  });
});

describe('engine diagnostics and website-safe errors', () => {
  it('bounds diagnostic history and removes sensitive messages from public errors', () => {
    const buffer = new EngineDiagnosticBuffer();
    buffer.append('x'.repeat(70 * 1024));
    expect(Buffer.byteLength(buffer.read())).toBeLessThanOrEqual(64 * 1024);

    expect(
      sanitizeProtocolError({
        code: 'PROTOCOL_INTERNAL_ERROR',
        category: 'protocol',
        message: 'Traceback in /Users/alice/private/model.py',
        retryable: false,
        details: { secret: 'not forwarded' },
        diagnosticId: 'diag_test',
      }),
    ).toEqual({
      code: 'PROTOCOL_INTERNAL_ERROR',
      category: 'protocol',
      message: 'The local AI runtime could not complete the request.',
      retryable: false,
      diagnosticId: 'diag_test',
    });
  });
});
