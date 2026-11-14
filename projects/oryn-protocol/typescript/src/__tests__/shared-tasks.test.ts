import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { SHARED_TASK_CONTRACTS, validateSharedTaskInput, validateSharedTaskOutput } from "../index.js";
const cases = JSON.parse(readFileSync(new URL("../../../fixtures/shared-tasks/inputs.json", import.meta.url), "utf8"));
describe("versioned shared task inputs", () => {
  for (const contract of SHARED_TASK_CONTRACTS) {
    it(`accepts golden ${contract} and rejects runtime authority`, () => {
      expect(validateSharedTaskInput(contract, cases[contract])).toEqual(cases[contract]);
      for (const field of ["backend", "modelRoot", "trust_remote_code", "runtimeOptions", "path"]) {
        expect(() => validateSharedTaskInput(contract, {...cases[contract], [field]: "forbidden"})).toThrow();
      }
    });
  }
  it("requires a positive bounded mask point", () => {
    for (const point of [{x: NaN, y: .5, label: 1}, {x: 2, y: .5, label: 1}, {x: .5, y: .5, label: 0}]) {
      expect(() => validateSharedTaskInput("mask-generation-v1", {...cases["mask-generation-v1"], points: [point]})).toThrow();
    }
  });
});

const outputs = JSON.parse(readFileSync(new URL("../../../fixtures/shared-tasks/outputs.json", import.meta.url), "utf8"));
it.each([128, 320, 4096])("preserves the declared %s-dimensional retrieval graph", dimensions => {
  const vector = [1, ...Array(dimensions - 1).fill(0)];
  const value = {dimensions, normalized: true, scoring: "sum-max-cosine", documentVectors: [vector], queries: [{vectors: [vector], score: 1}]};
  expect(validateSharedTaskOutput("visual-document-retrieval-v1", value)).toEqual(value);
  for (const location of ["page", "query", "declaration"]) {
    const bad = structuredClone(value);
    if (location === "page") bad.documentVectors[0]!.push(0);
    else if (location === "query") bad.queries[0]!.vectors[0]!.pop();
    else bad.dimensions = dimensions === 128 ? 320 : 128;
    expect(() => validateSharedTaskOutput("visual-document-retrieval-v1", bad)).toThrow();
  }
});
it("bounds dense retrieval output within the existing worker frame", () => {
  const vector = Array(320).fill(Number((1 / Math.sqrt(320)).toFixed(6)));
  const value = {dimensions: 320, normalized: true, scoring: "sum-max-cosine", documentVectors: Array(144).fill(vector), queries: Array(2).fill({vectors: Array(48).fill(vector), score: 48 * vector.reduce((sum,x) => sum+x*x,0)})};
  expect(validateSharedTaskOutput("visual-document-retrieval-v1", value)).toEqual(value);
  const dense = Array(320).fill(1 / Math.sqrt(320));
  value.documentVectors = Array(256).fill(dense);
  value.queries = Array(2).fill({vectors: Array(64).fill(dense), score: 64});
  expect(() => validateSharedTaskOutput("visual-document-retrieval-v1", value)).toThrow();
});
it.each([[308,true],[512,true],[513,false]])("bounds dense pose joints at %s", (count, valid) => {
  const points = Array.from({length: Number(count)}, (_, id) => ({id,label:`joint_${id}`,x:1,y:1,confidence:0.5,visible:true}));
  const value = {width:20,height:20,poses:[{box:[0,0,20,20],keypoints:points}]};
  if (valid) expect(validateSharedTaskOutput("pose-estimation-v1",value)).toEqual(value);
  else expect(() => validateSharedTaskOutput("pose-estimation-v1",value)).toThrow();
});
it("bounds aggregate dense pose output below the worker frame limit", () => {
  const points = Array.from({length:308}, (_,id) => ({id,label:`joint_${id}`.padEnd(64,"x"),x:1.123456789,y:2.123456789,confidence:0.123456789,visible:true}));
  const value = {width:20,height:20,poses:Array.from({length:8}, () => ({box:[0,0,20,20],keypoints:structuredClone(points)}))};
  expect(JSON.stringify(value).length).toBeGreaterThan(256 * 1024);
  expect(validateSharedTaskOutput("pose-estimation-v1",value)).toEqual(value);
  const tooLarge = structuredClone(value);
  for (const pose of tooLarge.poses) pose.keypoints = Array.from({length:512}, (_,id) => ({...points[0]!,id,label:"x".repeat(128),x:1.1234567890123456,y:2.1234567890123456}));
  expect(JSON.stringify(tooLarge).length).toBeGreaterThan(900 * 1024);
  expect(() => validateSharedTaskOutput("pose-estimation-v1",tooLarge)).toThrow();
});
it("rejects visual retrieval scores inconsistent with the returned unit vectors", () => {
  const value = structuredClone(outputs["visual-document-retrieval-v1"]);
  value.queries[0].score = 0.5;
  expect(() => validateSharedTaskOutput("visual-document-retrieval-v1", value)).toThrow();
  value.queries[0].score = 1;
  value.documentVectors[0][0] = 0.5;
  expect(() => validateSharedTaskOutput("visual-document-retrieval-v1", value)).toThrow();
});
it.each([["sum-max-cosine", 2], ["mean-max-cosine", 1]])("validates %s against all query vectors", (scoring, score) => {
  const vector = [1, ...Array(127).fill(0)];
  const value = {dimensions: 128, normalized: true, scoring, documentVectors: [vector], queries: [{vectors: [vector, vector], score}]};
  expect(validateSharedTaskOutput("visual-document-retrieval-v1", value)).toEqual(value);
  expect(() => validateSharedTaskOutput("visual-document-retrieval-v1", {...value, scoring: "unknown"})).toThrow();
  value.queries[0]!.score = Number(score) === 1 ? 2 : 1;
  expect(() => validateSharedTaskOutput("visual-document-retrieval-v1", value)).toThrow();
});
for (const contract of SHARED_TASK_CONTRACTS) {
  it(`validates closed ${contract} output`, () => {
    expect(validateSharedTaskOutput(contract, outputs[contract])).toEqual(outputs[contract]);
    expect(() => validateSharedTaskOutput(contract, {...outputs[contract], backend: "mps"})).toThrow();
  });
}

it("binds mask metadata to complete unique binary label counts", () => {
  const value = {kind: "multi", outputs: [outputs["mask-generation-v1"],
    {kind: "structured", value: {width: 2, height: 2, encoding: "binary-8bit", segments: [
      {id: 0, label: "background", pixelCount: 1}, {id: 255, label: "foreground", pixelCount: 3}
    ]}}]};
  expect(validateSharedTaskOutput("mask-generation-v1", value)).toEqual(value);
  for (const change of [{id: 0}, {id: 1}, {pixelCount: 4}]) {
    const bad = structuredClone(value);
    Object.assign(bad.outputs[1].value.segments[1], change);
    expect(() => validateSharedTaskOutput("mask-generation-v1", bad)).toThrow();
  }
});

it("preserves wide retrieval vectors above the worker wire limit with bounded token counts", () => {
  const vector = Array(4096).fill(1 / 64);
  const value = {dimensions: 4096, normalized: true, scoring: "sum-max-cosine", documentVectors: Array(144).fill(vector), queries: [{vectors: Array(48).fill(vector), score: 48}]};
  expect(JSON.stringify(value).length).toBeGreaterThan(1024 * 1024);
  expect(validateSharedTaskOutput("visual-document-retrieval-v1", value)).toEqual(value);
  for (const bad of [ {...value, documentVectors: Array(145).fill(vector)}, {...value, queries: [{vectors: Array(49).fill(vector), score: 49}]} ]) {
    expect(() => validateSharedTaskOutput("visual-document-retrieval-v1", bad)).toThrow();
  }
});
