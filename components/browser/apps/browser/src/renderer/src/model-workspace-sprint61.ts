/** Workspace data entry for additive tasks; Protocol validates before execution. */
export function forecastInput(
  context: string,
  horizon: number,
  quantiles: string,
): Record<string, unknown> {
  const values = numbers(context);
  const levels = quantiles.trim() ? numbers(quantiles) : [];
  return { context: values, horizon, ...(levels.length ? { quantiles: levels } : {}) };
}

export function poseBoxes(value: string): number[][] {
  const rows = value
    .trim()
    .split('\n')
    .map((row) => numbers(row).map((n) => n / 100));
  if (
    !rows.length ||
    rows.length > 8 ||
    rows.some(
      (row) =>
        row.length !== 4 ||
        row.some((n) => n < 0 || n > 1) ||
        row[0]! >= row[2]! ||
        row[1]! >= row[3]!,
    )
  )
    throw new Error('Enter up to eight person boxes as left, top, right, bottom percentages.');
  return rows;
}

export function tabularInput(
  columns: string,
  training: string,
  targets: string,
  query: string,
  classification: boolean,
): Record<string, unknown> {
  const features = columns
    .trim()
    .split('\n')
    .map((row) => {
      const [name, type, ...extra] = row.split('\t');
      if (!name?.trim() || !['numeric', 'categorical'].includes(type ?? '') || extra.length)
        throw new Error(
          'Enter each feature name and numeric or categorical type, separated by a tab.',
        );
      return { name: name.trim(), type };
    });
  const rows = (value: string) =>
    value
      .replace(/\r\n?/gu, '\n')
      .replace(/\n$/u, '')
      .split('\n')
      .map((row) => {
        const cells = row.split('\t');
        if (cells.length !== features.length)
          throw new Error('Every row must have one cell for each feature.');
        return cells.map((cell, i) => {
          if (!cell.trim()) return null;
          if (features[i]!.type !== 'numeric') return cell;
          const parsed = numbers(cell);
          if (parsed.length !== 1)
            throw new Error('Each numeric cell must contain exactly one value.');
          return parsed[0]!;
        });
      });
  // Targets are a separate training-only field. There is no query-target input.
  return {
    columns: features,
    trainingRows: rows(training),
    queryRows: rows(query),
    targets: classification ? targets.trim().split('\n') : numbers(targets),
  };
}

export function numericMatrix(value: string, integral = false): number[][] {
  const result = value.trim().split('\n').map(numbers);
  if (
    result.some(
      (row) =>
        row.length !== result[0]!.length || (integral && row.some((n) => !Number.isInteger(n))),
    )
  )
    throw new Error(
      'Enter a rectangular grid of ' + (integral ? 'integer codes.' : 'numeric values.'),
    );
  return result;
}

export function choiceValues(
  schema: Record<string, unknown> | undefined,
  defaults: string[],
): string[] {
  let result = defaults;
  const visit = (value: Record<string, unknown>) => {
    if (typeof value.const === 'string') result = result.filter((v) => v === value.const);
    if (Array.isArray(value.enum))
      result = result.filter((v) => (value.enum as unknown[]).includes(v));
    if (Array.isArray(value.allOf)) value.allOf.forEach((v) => visit(v as Record<string, unknown>));
  };
  if (schema) visit(schema);
  return result;
}

function numbers(value: string): number[] {
  if (value.includes(',') && value.split(',').some((part) => !part.trim()))
    throw new Error('Enter every numeric value; missing entries are not supported.');
  const pieces = value.trim().split(/[\s,]+/u);
  if (
    !value.trim() ||
    pieces.some((p) => !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/iu.test(p))
  )
    throw new Error('Enter finite numeric values separated by commas or spaces.');
  const result = pieces.map(Number);
  if (result.some((v) => !Number.isFinite(v))) throw new Error('Enter finite numeric values.');
  return result;
}

export function numericControl(
  schema: Record<string, unknown> | undefined,
  defaults: { min: number; max: number; step: number },
) {
  const result = { ...defaults };
  const visit = (value: Record<string, unknown>) => {
    if (typeof value.minimum === 'number') result.min = Math.max(result.min, value.minimum);
    if (typeof value.maximum === 'number') result.max = Math.min(result.max, value.maximum);
    if (typeof value.multipleOf === 'number') result.step = Math.max(result.step, value.multipleOf);
    if (typeof value.const === 'number') result.min = result.max = value.const;
    if (Array.isArray(value.enum)) {
      const allowed = value.enum.filter((v): v is number => typeof v === 'number');
      if (allowed.length) {
        result.min = Math.max(result.min, Math.min(...allowed));
        result.max = Math.min(result.max, Math.max(...allowed));
        const sorted = [...new Set(allowed)].sort((a, b) => a - b);
        const gaps = sorted.slice(1).map((v, i) => v - sorted[i]!);
        if (gaps.length && gaps.every((g) => g === gaps[0]))
          result.step = Math.max(result.step, gaps[0]!);
      }
    }
    if (Array.isArray(value.allOf)) value.allOf.forEach((v) => visit(v as Record<string, unknown>));
  };
  if (schema) visit(schema);
  return result;
}
