import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
const root = new URL('../', import.meta.url);
const registry = JSON.parse(await readFile(new URL('schemas/task-contracts/registry.json', root), 'utf8'));
const profile = JSON.parse(await readFile(new URL('schemas/task-contracts/task-profile.schema.json', root), 'utf8'));
function type(schema) {
  if (schema.anyOf) return schema.anyOf.map(type).join(' | ');
  if (Object.hasOwn(schema, 'const')) return JSON.stringify(schema.const);
  if (schema.enum) return schema.enum.map(value => JSON.stringify(value)).join(' | ');
  if (schema.type === 'object') return '{ ' + Object.entries(schema.properties).map(([key, value]) => JSON.stringify(key) + ((schema.required ?? []).includes(key) ? '' : '?') + ': ' + type(value)).join('; ') + ' }';
  if (schema.type === 'array') return '(' + type(schema.items) + ')[]';
  const result = { string: 'string', boolean: 'boolean', null: 'null', number: 'number', integer: 'number' }[schema.type];
  if (!result) throw new Error('Unsupported chat schema type');
  return result;
}
const contract = registry.contracts['chat-v2'];
const value = contract.resultSchema.properties.value;
const source = '// Generated from canonical Protocol chat schemas; do not hand-edit.\n' +
  `export type ChatV2Input = ${type(contract.inputSchema)};\n` +
  `export type ChatV2Value = ${type(value)};\n` +
  `export type ChatThinkingCapability = ${type(profile.properties.thinking)};\n` +
  `export type EngineThinkingMetadata = ${type(value.properties.metadata.properties.thinking)};\n` +
  `export const CHAT_V2_INPUT_SCHEMA: unknown = ${JSON.stringify(contract.inputSchema, null, 2)};\n` +
  `export const CHAT_V2_RESULT_SCHEMA: unknown = ${JSON.stringify(contract.resultSchema, null, 2)};\n` +
  `export const CHAT_V1_RESULT_SCHEMA: unknown = ${JSON.stringify(registry.contracts['chat-v1'].resultSchema, null, 2)};\n`;
await writeFile(new URL('typescript/src/chat-contracts.ts', root), source);
if (process.argv[2]) await writeFile(resolve(process.argv[2], 'src/chat-contracts.ts'), source);
