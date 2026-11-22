/** Closed schema evaluator shared by generated SDK task validators. */
export function matchesSchema(raw: unknown, value: unknown): boolean {
  if (!record(raw)) return false;
  if ('const' in raw && value !== raw.const) return false;
  if (Array.isArray(raw.enum) && !raw.enum.includes(value)) return false;
  if (Array.isArray(raw.anyOf)) return raw.anyOf.some(schema => matchesSchema(schema, value));
  switch (raw.type) {
    case 'object': {
      if (!record(value) || (raw.properties !== undefined && !record(raw.properties))) return false;
      const properties = (raw.properties ?? {}) as Record<string, unknown>;
      if (raw.additionalProperties === false && Object.keys(value).some(key => !(Object.hasOwn(properties, key)))) return false;
      if (Array.isArray(raw.required) && raw.required.some(key => typeof key !== 'string' || !Object.hasOwn(value, key))) return false;
      return Object.entries(value).every(([key, item]) => !Object.hasOwn(properties, key) || matchesSchema(properties[key], item));
    }
    case 'array': {
      if (!Array.isArray(value) || value.length < Number(raw.minItems ?? 0) || value.length > Number(raw.maxItems ?? Infinity)) return false;
      if (raw.uniqueItems && new Set(value.map(item => JSON.stringify(item))).size !== value.length) return false;
      if (raw.contains && value.filter(item => matchesSchema(raw.contains, item)).length < Number(raw.minContains ?? 1)) return false;
      return value.every(item => matchesSchema(raw.items, item));
    }
    case 'string':
      return typeof value === 'string' && [...value].length >= Number(raw.minLength ?? 0) &&
        [...value].length <= Number(raw.maxLength ?? Infinity) &&
        (typeof raw.pattern !== 'string' || new RegExp(raw.pattern, 'u').test(value));
    case 'integer':
    case 'number':
      return typeof value === 'number' && Number.isFinite(value) &&
        (raw.type !== 'integer' || Number.isInteger(value)) &&
        value >= Number(raw.minimum ?? -Infinity) && value <= Number(raw.maximum ?? Infinity) &&
        value > Number(raw.exclusiveMinimum ?? -Infinity) &&
        (raw.multipleOf === undefined || value % Number(raw.multipleOf) === 0);
    case 'boolean': return typeof value === 'boolean';
    case 'null': return value === null;
    case undefined: return 'const' in raw || Array.isArray(raw.enum);
    default: return false;
  }
}
function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value) && Object.getPrototypeOf(value) === Object.prototype;
}
