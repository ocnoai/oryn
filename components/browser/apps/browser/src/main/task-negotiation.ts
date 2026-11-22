import { SHARED_TASK_CONTRACTS } from '@ocno/oryn-protocol';
export function supportsTaskNegotiation(version: string): boolean {
  if (!/^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/u.test(version)) return false;
  const [major = 0, minor = 0] = version.split('.').map(Number);
  return major > 0 || (major === 0 && minor >= 2);
}
export function validateNegotiatedTasks(value: unknown): string[] {
  if (typeof value !== 'object' || value === null || Array.isArray(value))
    throw new Error('ENGINE_TASK_NEGOTIATION_INVALID');
  const contracts = (value as Record<string, unknown>).contracts;
  if (
    Object.keys(value).length !== 1 ||
    !Array.isArray(contracts) ||
    contracts.some(
      (c) => typeof c !== 'string' || ![...SHARED_TASK_CONTRACTS, 'chat-v2'].includes(c),
    ) ||
    new Set(contracts).size !== contracts.length
  )
    throw new Error('ENGINE_TASK_NEGOTIATION_INVALID');
  return contracts as string[];
}
