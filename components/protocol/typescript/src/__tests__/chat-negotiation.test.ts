import Ajv2020 from 'ajv/dist/2020.js';
import { expect, it } from 'vitest';
import schema from '../../../schemas/v1/protocol-v1.schema.json';
import fixture from '../../../fixtures/task-contracts/chat-negotiation.json';
import capabilityFixture from '../../../fixtures/v1/runtime-capabilities.json';

it('uses the existing bounded negotiation shape for old and new chat peers', () => {
  const ajv = new Ajv2020({ strict: true }); ajv.addSchema(schema);
  const request = ajv.compile({ $ref: `${schema.$id}#/$defs/taskContractsRequest` });
  const result = ajv.compile({ $ref: `${schema.$id}#/$defs/taskContractsResult` });
  expect(request(fixture.legacyRequest)).toBe(true);
  expect(request(fixture.currentRequest)).toBe(true);
  // Older Engines can return the supported subset, rather than rejecting v2.
  expect(result(fixture.legacyResult)).toBe(true);
  expect(result(fixture.currentResult)).toBe(true);
  expect(request({ ...fixture.currentRequest, params: { contracts: ['chat-v2', 'chat-v2'] } })).toBe(false);
});

it('demonstrates why Engine capability additions must remain opt-in for old Browsers', () => {
  const oldSchema = structuredClone(schema);
  delete (oldSchema.$defs.runtimeCapabilitiesResult.properties as Record<string, unknown>).taskContracts;
  const ajv = new Ajv2020({ strict: true }); ajv.addSchema(oldSchema);
  const old = ajv.compile({ $ref: `${oldSchema.$id}#/$defs/runtimeCapabilitiesResult` });
  const current = new Ajv2020({ strict: true }); current.addSchema(schema);
  const latest = current.compile({ $ref: `${schema.$id}#/$defs/runtimeCapabilitiesResult` });
  const optedIn = { ...capabilityFixture.result, taskContracts: [{ id: 'chat-v2', revision: 1 }] };
  expect(old(capabilityFixture.result)).toBe(true);
  expect(latest(capabilityFixture.result)).toBe(true);
  expect(old(optedIn)).toBe(false);
  expect(latest(optedIn)).toBe(true);
});
