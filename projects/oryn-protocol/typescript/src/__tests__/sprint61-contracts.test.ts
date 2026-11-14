import { expect, it } from "vitest";
import { validateSharedTaskInput, validateSharedTaskOutput } from "../index.js";
import { validSprint61ResultForInput } from "../sprint61-relations.js";

const classification = {columns: [{name: "x", type: "numeric"}], trainingRows: [[1], [2]],
  targets: ["cold", "warm"], queryRows: [[1.5]]};
it("binds labeled pose joints to ordered caller person boxes", () => {
  const input = {imageArtifactId: "artifact_aaaaaaaaaaaaaaaaaaaaaaaa", boxes: [[0, 0, 1, 1]]};
  const output = {width: 32, height: 32, poses: [{box: [0, 0, 32, 32], keypoints: [
    {id: 0, label: "nose", x: 16, y: 16, confidence: .75, visible: true}]}]};
  expect(validateSharedTaskInput("pose-estimation-v1", input)).toEqual(input);
  expect(validateSharedTaskOutput("pose-estimation-v1", output)).toEqual(output);
  expect(validSprint61ResultForInput("pose-estimation-v1", {kind: "structured", value: output}, input)).toBe(true);
  expect(validSprint61ResultForInput("pose-estimation-v1", {kind: "structured", value: output}, {...input, boxes: [[0, 0, .5, 1]]})).toBe(false);
  expect(() => validateSharedTaskInput("pose-estimation-v1", {...input, boxes: [[.5, 0, .2, 1]]})).toThrow();
  expect(() => validateSharedTaskOutput("pose-estimation-v1", {...output, poses: [{...output.poses[0], keypoints: output.poses[0]!.keypoints.map(p => ({...p, visible: false}))}]})).toThrow();
});
it("rejects target leakage, column collisions and mismatched tabular types", () => {
  expect(validateSharedTaskInput("tabular-classification-v1", classification)).toEqual(classification);
  for (const change of [{queryTargets: ["warm"]}, {targets: ["cold"]}, {targets: ["cold", "cold"]},
    {queryRows: [[true]]}, {queryRows: [["1"]]}, {queryRows: [[1, 2]]},
    {columns: [{name: "x", type: "numeric"}, {name: " x ", type: "numeric"}]}])
    expect(() => validateSharedTaskInput("tabular-classification-v1", {...classification, ...change})).toThrow();
});
it("binds tabular probabilities, native class order and query count", () => {
  const output = {classes: ["cold", "warm"], predictions: [{label: "warm", probabilities: [.4, .6]}]};
  expect(validateSharedTaskOutput("tabular-classification-v1", output)).toEqual(output);
  for (const probabilities of [[.4, .4], [.6, .4], [.4, .6, 0]])
    expect(() => validateSharedTaskOutput("tabular-classification-v1", {...output, predictions: [{label: "warm", probabilities}]})).toThrow();
  expect(validSprint61ResultForInput("tabular-classification-v1", {kind: "structured", value: output}, classification)).toBe(true);
  expect(validSprint61ResultForInput("tabular-classification-v1", {kind: "structured", value: output}, {...classification, queryRows: [[1], [2]]})).toBe(false);
});
it("rejects forecast lookahead, irregular context and crossed quantiles", () => {
  const input = {context: [1, 2, 3], horizon: 2, timestamps: [0, 10, 20], frequencyMilliseconds: 10, quantiles: [.1, .9]};
  const output = {predictions: [4, 5], timestamps: [30, 40], quantiles: [{level: .1, values: [3, 4]}, {level: .9, values: [5, 6]}]};
  expect(validateSharedTaskInput("time-series-forecasting-v1", input)).toEqual(input);
  expect(validateSharedTaskOutput("time-series-forecasting-v1", output)).toEqual(output);
  for (const change of [{futureTargets: [4, 5]}, {timestamps: [0, 10, 30]}, {quantiles: [.9, .1]},
    {pastCovariates: [[1], [2], [3]]}, {futureCovariates: [[1], [2]]}])
    expect(() => validateSharedTaskInput("time-series-forecasting-v1", {...input, ...change})).toThrow();
  expect(() => validateSharedTaskOutput("time-series-forecasting-v1", {...output,
    quantiles: [{level: .1, values: [7, 4]}, {level: .9, values: [5, 6]}]})).toThrow();
  expect(validSprint61ResultForInput("time-series-forecasting-v1", {kind: "structured", value: output}, input)).toBe(true);
  expect(validSprint61ResultForInput("time-series-forecasting-v1", {kind: "structured", value: {...output, timestamps: [20, 30]}}, input)).toBe(false);
});
it("requires typed audio sources and real multimodal grounding", () => {
  expect(validateSharedTaskInput("audio-transformation-v1", {operation: "decode", codes: [[1, 2]]})).toEqual({operation: "decode", codes: [[1, 2]]});
  expect(() => validateSharedTaskInput("audio-transformation-v1", {operation: "enhance", codes: [[1]]})).toThrow();
  expect(() => validateSharedTaskInput("multimodal-generation-v1", {instruction: "describe"})).toThrow();
  const image = {instruction: "describe", inputMode: "image", imageArtifactId: "artifact_aaaaaaaaaaaaaaaaaaaaaaaa"};
  expect(validateSharedTaskInput("multimodal-generation-v1", image)).toEqual(image);
  expect(() => validateSharedTaskInput("multimodal-generation-v1", {...image, audioArtifactId: "artifact_bbbbbbbbbbbbbbbbbbbbbbbb"})).toThrow();
  expect(() => validateSharedTaskInput("multimodal-generation-v1", {...image, inputMode: "audio"})).toThrow();
  expect(() => validateSharedTaskInput("unconditional-image-generation-v1", {prompt: ""})).toThrow();
  expect(() => validateSharedTaskOutput("audio-transformation-v1", {kind: "artifact", artifact: {
    outputId: "output_fixture", internalPath: "/controlled/result.png", mimeType: "image/png", sizeBytes: 128}})).toThrow();
});
