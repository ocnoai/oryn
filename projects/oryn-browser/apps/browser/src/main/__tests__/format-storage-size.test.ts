import { describe, expect, it } from 'vitest';

import { formatStorageSize } from '../../shared/format-storage-size';

describe('formatStorageSize', () => {
  it('promotes byte counts through one decimal storage-size pipeline', () => {
    expect(formatStorageSize(0)).toBe('0 B');
    expect(formatStorageSize(999)).toBe('999 B');
    expect(formatStorageSize(1_500)).toBe('1.5 KB');
    expect(formatStorageSize(999_999)).toBe('1 MB');
    expect(formatStorageSize(766_200_000)).toBe('766.2 MB');
    expect(formatStorageSize(3_241_200_000)).toBe('3.2 GB');
    expect(formatStorageSize(4_051_500_000)).toBe('4.1 GB');
  });

  it('clamps invalid or negative display values to zero', () => {
    expect(formatStorageSize(-1)).toBe('0 B');
    expect(formatStorageSize(Number.NaN)).toBe('0 B');
    expect(formatStorageSize(Number.POSITIVE_INFINITY)).toBe('0 B');
  });
});
