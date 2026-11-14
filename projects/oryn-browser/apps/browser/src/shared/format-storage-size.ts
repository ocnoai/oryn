const STORAGE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'] as const;
const BYTES_PER_UNIT = 1_000;

/**
 * Formats byte counts with decimal storage units, promoting values at each 1000-unit boundary.
 * A single fractional digit is kept when it carries information.
 */
export function formatStorageSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';

  let unitIndex = Math.min(
    Math.floor(Math.log(bytes) / Math.log(BYTES_PER_UNIT)),
    STORAGE_UNITS.length - 1,
  );
  let value = bytes / BYTES_PER_UNIT ** unitIndex;
  if (unitIndex < STORAGE_UNITS.length - 1 && Number(value.toFixed(1)) >= BYTES_PER_UNIT) {
    unitIndex += 1;
    value /= BYTES_PER_UNIT;
  }
  const formatted =
    unitIndex === 0 ? Math.round(value).toString() : value.toFixed(1).replace(/\.0$/u, '');

  return `${formatted} ${STORAGE_UNITS[unitIndex]}`;
}
