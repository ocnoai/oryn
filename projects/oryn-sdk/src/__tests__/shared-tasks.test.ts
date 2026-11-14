import { describe, expect, it } from "vitest";
import { SHARED_TASK_CONTRACTS, validateSharedTaskInput, type SharedTaskOutputs } from "../index.js";
import cases from "./fixtures/shared-task-inputs.json";
describe("shared task contracts", () => {
  it("exports public artifact metadata at the existing result shape", () => {
    const output: SharedTaskOutputs['unconditional-image-generation-v1'] = {
      kind: 'artifact', artifact: {id: 'artifact_aaaaaaaaaaaaaaaaaaaaaaaa', mimeType: 'image/png', sizeBytes: 128, createdAt: 1},
    };
    expect(output.artifact.id).toMatch(/^artifact_/);
    // Typecheck the website boundary as well as the runtime fixture.
    // @ts-expect-error Internal worker paths are not public artifact metadata.
    const privatePath: keyof typeof output.artifact = 'internalPath';
    expect(Object.hasOwn(output.artifact, privatePath)).toBe(false);
  });
  for (const contract of SHARED_TASK_CONTRACTS) {
    it(`validates ${contract} before invoking the bridge`, () => {
      expect(validateSharedTaskInput(contract, cases[contract])).toEqual(cases[contract]);
      for (const field of ["backend", "foundation", "runtimeOptions", "modelRoot", "path", "trust_remote_code"]) {
        expect(() => validateSharedTaskInput(contract, {...cases[contract], [field]: "forbidden"})).toThrow();
      }
    });
  }
  it("rejects nonfinite coordinates, empty and equivalent labels", () => {
    expect(() => validateSharedTaskInput("mask-generation-v1", {...cases["mask-generation-v1"], points: [{x: NaN, y: .5, label: 1}]})).toThrow();
    for (const labels of [["cat", " cat "], ["cat", " "], Array(33).fill("cat")]) {
      expect(() => validateSharedTaskInput("zero-shot-classification-v1", {text: "a cat", labels})).toThrow();
    }
  });
});


it('validates bounded ordered image references without exposing file paths', () => {
  const ids = [...'abcd'].map(c => 'artifact_' + c.repeat(24));
  const base = {imageArtifactId:ids[0], prompt:'Use image 1 and image 2'};
  const value = {...base, referenceImageArtifactIds:[ids[2],ids[1]]};
  expect(validateSharedTaskInput('image-editing-v1',value)).toEqual(value);
  for (const refs of [[ids[0]], [ids[1],ids[1]], ids.slice(1), ['/tmp/image.png'], null, 'bad'])
    expect(() => validateSharedTaskInput('image-editing-v1', {...base,referenceImageArtifactIds:refs})).toThrow();
});
