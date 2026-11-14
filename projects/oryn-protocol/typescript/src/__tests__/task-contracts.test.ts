import { describe, it, expect } from "vitest";
import fixtures from "../../../fixtures/task-contracts/negotiation.json";
import decisionProfile from "../../../fixtures/task-contracts/decision-labels.json";
import embeddingModes from "../../../fixtures/task-contracts/embedding-modes.json";
import forecastLimits from "../../../fixtures/task-contracts/forecast-limits.json";
import audioDuration from "../../../fixtures/task-contracts/audio-duration.json";
import audioHeads from "../../../fixtures/task-contracts/audio-heads.json";
import {
  TASK_CONTRACTS,
  taskRequirementFailures,
  validateTaskInvocation,
  validateTaskRequirements,
  validateTaskProfile,
  type TaskProfile,
  type TaskRequirements,
} from "../task-contracts.js";

it("requires explicit adapter support for audio head selection", () => {
  const legacy = audioHeads.legacy as unknown as TaskProfile;
  const multihead = audioHeads.multihead as unknown as TaskProfile;
  const value = { audioArtifactId: "artifact_" + "a".repeat(24) };
  expect(Object.keys(legacy.parameters)).toEqual(["topK"]);
  expect(() => validateTaskInvocation(legacy, value)).not.toThrow();
  for (const head of ["language", "gender"]) {
    expect(() => validateTaskInvocation(multihead, { ...value, classificationHead: head })).not.toThrow();
    expect(() => validateTaskInvocation(legacy, { ...value, classificationHead: head })).toThrow();
  }
  expect(() => validateTaskInvocation(multihead, { ...value, classificationHead: "other" })).toThrow();
});

it("rejects new audio head profiles under an older registry", () => {
  const profile = audioHeads.multihead as unknown as TaskProfile;
  const contract = TASK_CONTRACTS["audio-classification-v1"]!;
  const previous = contract.parameterNames;
  try {
    contract.parameterNames = ["topK"];
    expect(() => validateTaskProfile(profile)).toThrow();
  } finally {
    contract.parameterNames = previous;
  }
});

it("declares audio classifier duration and rejects it under an older registry", () => {
  const profile = audioDuration as unknown as TaskProfile;
  expect(validateTaskProfile(profile).limits).toEqual({ audioDurationSeconds: 12 });
  expect(() => validateTaskInvocation(profile, { audioArtifactId: "artifact_" + "a".repeat(24) })).not.toThrow();
  const contract = TASK_CONTRACTS["audio-classification-v1"]!;
  const previous = contract.limitNames;
  try {
    contract.limitNames = [];
    expect(() => validateTaskProfile(profile)).toThrow();
  } finally {
    contract.limitNames = previous;
  }
});

it("enforces declared forecast context, horizon and covariate limits", () => {
  const profile = forecastLimits as unknown as TaskProfile;
  const value = { context: Array(16).fill(1), horizon: 8 };
  expect(() => validateTaskInvocation(profile, value)).not.toThrow();
  expect(() => validateTaskInvocation(profile, { context: value.context })).not.toThrow();
  expect(() => validateTaskInvocation(profile, { ...value,
    pastCovariates: Array(16).fill([1, 2]), futureCovariates: Array(8).fill([1, 2]) })).not.toThrow();
  for (const extra of [{ context: Array(17).fill(1) }, { horizon: 9 },
    { pastCovariates: Array(16).fill([1, 2, 3]), futureCovariates: Array(8).fill([1, 2, 3]) }])
    expect(() => validateTaskInvocation(profile, { ...value, ...extra })).toThrow("TASK_INPUT_LIMIT_EXCEEDED");
});

it("requires explicit support for query and contextual embedding controls", () => {
  const legacy = embeddingModes.legacy as unknown as TaskProfile;
  const contextual = embeddingModes.contextual as unknown as TaskProfile;
  const value = { texts: ["first chunk", "second chunk"] };
  expect(Object.keys(legacy.parameters)).toEqual(["normalize"]);
  expect(() => validateTaskInvocation(legacy, value)).not.toThrow();
  for (const extra of [{ inputType: "document" }, { inputType: "query" }, { contextual: false }])
    expect(() => validateTaskInvocation(legacy, { ...value, ...extra })).toThrow();
  expect(() => validateTaskInvocation(contextual, { ...value, inputType: "query" })).not.toThrow();
  expect(() => validateTaskInvocation(contextual, { ...value, inputType: "document", contextual: true })).not.toThrow();
  expect(() => validateTaskInvocation(contextual, { ...value, inputType: "query", contextual: true })).toThrow();
});

it("keeps older embedding registries closed to newly declared controls", () => {
  const contract = TASK_CONTRACTS["embeddings-v1"]!;
  const previous = contract.parameterNames;
  try {
    contract.parameterNames = ["normalize"];
    expect(() => validateTaskProfile(embeddingModes.contextual)).toThrow();
    expect(() => validateTaskProfile(embeddingModes.legacy)).not.toThrow();
  } finally {
    contract.parameterNames = previous;
  }
});

it("enforces a declared zero-shot candidate-label bound", () => {
  const profile = decisionProfile as unknown as TaskProfile;
  const value = {text: "classify", labels: Array.from({length: 26}, (_, i) => `label ${i}`)};
  expect(() => validateTaskInvocation(profile, value)).not.toThrow();
  expect(() => validateTaskInvocation(profile, {...value, labels: [...value.labels, "extra"]})).toThrow("TASK_INPUT_LIMIT_EXCEEDED");
  expect(() => validateTaskInvocation({...profile, limits: {}}, {...value, labels: [...value.labels, "extra"]})).not.toThrow();
});

it("rejects the profile when an older registry does not recognize its label limit", () => {
  const contract = TASK_CONTRACTS["zero-shot-classification-v1"]!;
  const previous = contract.limitNames;
  try {
    contract.limitNames = [];
    expect(() => validateTaskProfile(decisionProfile)).toThrow();
  } finally {
    contract.limitNames = previous;
  }
});

describe("shared cross-language task negotiation fixtures", () => {
  for (const raw of fixtures) {
    it(raw.name, () => {
      const c = raw as unknown as {
        profile: TaskProfile;
        requirements: TaskRequirements;
        requirementsInvalid?: boolean;
        input?: unknown;
        inputValid?: boolean;
        failures: string[];
      };
      if (c.requirementsInvalid) {
        expect(() => validateTaskRequirements(c.requirements)).toThrow();
        return;
      }
      expect(taskRequirementFailures(c.profile, c.requirements)).toEqual(
        c.failures,
      );
      if (c.input) {
        if (c.inputValid)
          expect(() =>
            validateTaskInvocation(c.profile, c.input, c.requirements),
          ).not.toThrow();
        else
          expect(() =>
            validateTaskInvocation(c.profile, c.input, c.requirements),
          ).toThrow();
      }
    });
  }
  it("rejects private provider data in website profiles", () => {
    expect(() =>
      validateTaskProfile({
        ...fixtures[0]!.profile,
        providerOptions: { path: "/private" },
      }),
    ).toThrow();
  });
  it("never fetches provider-selected schema references", () => {
    const p = structuredClone(fixtures[0]!.profile);
    (p.parameters.timestamps as { valueSchema: unknown }).valueSchema = {
      $ref: "https://example.com/loader",
    };
    expect(() => validateTaskProfile(p)).toThrow();
  });
});

import speechReference from "../../../fixtures/task-contracts/speech-reference.json";
it("requires explicit reference speech support and an audio artifact", () => {
  const reference = speechReference.reference as unknown as TaskProfile;
  const legacy = speechReference.legacy as unknown as TaskProfile;
  const value = {text: "Hello.", audioArtifactId: "artifact_" + "a".repeat(24)};
  expect(Object.keys(legacy.parameters)).toEqual(["voice", "language", "speed"]);
  expect(legacy.inputModalities).toEqual(["text"]);
  expect(() => validateTaskInvocation(legacy, {text: "Hello."})).not.toThrow();
  expect(() => validateTaskInvocation(reference, value)).not.toThrow();
  for (const bad of [{text: "Hello."}, {...value, voice: "af_heart"}, {...value, speed: 1.5},
    {...value, language: "zz"}, {...value, text: "a".repeat(301)}, {...value, audioArtifactId: "bad"}])
    expect(() => validateTaskInvocation(reference, bad)).toThrow();
  expect(() => validateTaskInvocation(legacy, value)).toThrow();
});
it("rejects reference speech profiles under older speech contracts", () => {
  const contract = TASK_CONTRACTS["speech-synthesis-v1"]!;
  const previous = contract.parameterNames;
  try {
    contract.parameterNames = ["voice", "language", "speed"];
    expect(() => validateTaskProfile(speechReference.reference)).toThrow();
  } finally { contract.parameterNames = previous; }
});

it("preserves native video frame counts and fixed timing", async () => {
  const { default: raw } = await import("../../../fixtures/task-contracts/video-native-frames.json");
  const profile = raw as unknown as TaskProfile;
  validateTaskProfile(profile);
  for (const frames of [124, 141, 158, 345]) {
    expect(() => validateTaskInvocation(profile, { prompt: "Birds", frames })).not.toThrow();
  }
  for (const extra of [{ frames: 120 }, { frames: 125 }, { frames: 360 }, { frames: 361 }, { fps: 30 }, { guidanceScale: 7.5 }, { negativePrompt: "blur" }]) {
    expect(() => validateTaskInvocation(profile, { prompt: "Birds", ...extra })).toThrow();
  }
});

it("preserves image-video frame limits and required image binding", async () => {
  const { default: raw } = await import("../../../fixtures/task-contracts/video-keyframe-frames.json");
  const profile = raw as unknown as TaskProfile;
  validateTaskProfile(profile);
  const value = { prompt: "Birds", imageArtifactId: "artifact_" + "a".repeat(24) };
  for (const frames of [124, 141, 345]) {
    expect(() => validateTaskInvocation(profile, { ...value, frames })).not.toThrow();
  }
  for (const extra of [{ frames: 129 }, { frames: 360 }, { frames: 361 }, { fps: 30 }]) {
    expect(() => validateTaskInvocation(profile, { ...value, ...extra })).toThrow();
  }
  expect(() => validateTaskInvocation(profile, { prompt: "Birds" })).toThrow();
});
