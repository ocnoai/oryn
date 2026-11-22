import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { assertModelInventoryResult } from "../index.js";
import type { ModelComposition } from "../index.js";

const cases = JSON.parse(readFileSync(new URL("../../../fixtures/v1/model-compositions.json", import.meta.url), "utf8")) as ModelComposition[];
it.each(cases)("preserves pinned adapter and base identities: $model.repository", (composition) => {
  const inventory = JSON.parse(readFileSync(new URL("../../../fixtures/v1/model-inventory-response.json", import.meta.url), "utf8")).result;
  inventory.models[0].runtimePack = {
    packId: "oryn.diffusers-pytorch", packVersion: "0.6.0", runnerId: "diffusers.pipeline",
    taskContract: "image-editing-v1", compatibilityLevel: "native-declarative", manifestDigest: "a".repeat(64),
    trustedMetadata: {license: "See source model cards", composition, conformance: {status: "fixture-only", platforms: ["cpu"]}},
  };
  expect(() => assertModelInventoryResult(inventory)).not.toThrow();
  for (const change of [
    {base: {...composition.base, revision: "main"}},
    {model: {...composition.model, repository: "https://other.example/model"}},
    {adapterFile: "../adapter.safetensors"}, {scale: 0}, {family: "RepositoryPipeline"},
    {overrides: ["../config.json"]}, {trust_remote_code: true},
  ]) {
    inventory.models[0].runtimePack.trustedMetadata.composition = {...composition, ...change};
    expect(() => assertModelInventoryResult(inventory)).toThrow();
  }
});
