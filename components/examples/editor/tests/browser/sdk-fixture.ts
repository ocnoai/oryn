import type { Page } from "@playwright/test";
import type {
  InjectedLocalAI,
  LocalAIEvent,
  ModelRequest,
  ModelRequestProgress,
  ModelSession,
} from "@ocno/oryn-sdk";

export interface FixtureInvocation {
  jobId: string;
  sessionId: string;
  input: {
    messages?: { role: string; content: string }[];
    texts?: string[];
    inputType?: string;
    thinking?: boolean;
    maxTokens?: number;
  };
}

export interface SDKFixtureState {
  requests: ModelRequest[];
  invocations: FixtureInvocation[];
  cancelled: string[];
  released: string[];
  listeners: number;
}

interface FixtureControl {
  snapshot(): SDKFixtureState;
  configure(options: FixtureOptions): void;
  delta(jobId: string, text: string): void;
  reasoning(jobId: string, text: string): void;
  finish(jobId: string, text: string): void;
}

export interface FixtureOptions {
  holdText?: boolean;
  deferCancellation?: boolean;
  mode?: "normal" | "reasoning-only" | "reasoning-answer" | "limit" | "unsupported" | "legacy";
}

type FixtureWindow = Window & { __editorSDKFixture: FixtureControl };

/** Installs only the injected Browser boundary. The app still uses its packaged SDK. */
export async function installSDKFixture(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const listeners = new Set<(event: LocalAIEvent) => void>();
    const progressListeners = new Map<
      string,
      (progress: ModelRequestProgress) => void
    >();
    const requests: ModelRequest[] = [];
    const invocations: FixtureInvocation[] = [];
    const cancelled: string[] = [];
    const released: string[] = [];
    const jobs = new Map<string, string>();
    const contracts = new Map<string, string>();
    const reasoningByJob = new Map<string, string>();
    let sessionNumber = 0,
      jobNumber = 0,
      sequence = 0;
    let holdText = false,
      deferCancellation = false;
    let mode: FixtureOptions['mode'] = "normal";
    const success = <T>(value: T) => ({ ok: true as const, value });
    const emit = (
      jobId: string,
      event: LocalAIEvent["event"],
      data: unknown,
    ) => {
      const sessionId = jobs.get(jobId);
      if (!sessionId) throw new Error("Unknown synthetic fixture job.");
      const message = {
        sessionId,
        jobId,
        event,
        data,
        timestamp: Date.now(),
        sequence: ++sequence,
      } as LocalAIEvent;
      for (const listener of listeners) listener(message);
    };
    const finish = (jobId: string, text: string) => {
      const invocation = invocations.find(item => item.jobId === jobId)!;
      const requested = invocation.input.thinking ?? null;
      const value = contracts.get(invocation.sessionId) === "chat-v2" ? {
        text, ...(reasoningByJob.has(jobId) ? { reasoning: reasoningByJob.get(jobId) } : {}),
        finishReason: mode === "limit" ? "length" : "stop",
        usage: { inputTokens: null, outputTokens: null, reasoningTokens: null, answerTokens: null },
        metadata: { thinking: {
          requested, applied: requested !== null && mode !== "unsupported",
          effective: mode === "unsupported" ? true : requested,
          reason: mode === "unsupported" ? "fixed_mode" : requested === null ? "model_default" : "applied",
          source: "engine",
        } },
      } : { text };
      emit(jobId, "job.completed", { result: { kind: "structured", value } });
    };
    const reasoningDelta = (jobId: string, text: string) => {
      reasoningByJob.set(jobId, (reasoningByJob.get(jobId) ?? "") + text);
      emit(jobId, "job.output.delta", { channel: "reasoning", delta: text });
    };
    const model = {
      provider: "huggingface" as const,
      repository: "synthetic/browser-test-model",
      revision: "a".repeat(40),
      sourceUrl: "https://models.invalid/synthetic",
    };
    const bridge: InjectedLocalAI = {
      sdkVersion: "0.1.0",
      getCapabilities: async () => ({
        runtimeAvailable: true,
        runtimeState: "ready",
        sdkVersion: "0.1.0",
        browserApiVersion: "1",
        supportedModelProviders: ["huggingface"],
        taskContracts: mode === "legacy" ? [] : [{ id: "chat-v2", revision: 1 }, { id: "chat-v1", revision: 1 }, { id: "embeddings-v1", revision: 1 }],
        features: {
          taskProfiles: true,
          modelSelection: true,
          modelRequests: true,
          modelSubstitution: true,
          // Oryn's runtime-wide flag is false; chat support is negotiated per model.
          streaming: false,
          artifacts: false,
          artifactUrls: false,
          cancellation: true,
          queueing: true,
          multipleSubmittedJobs: true,
        },
      }),
      requestModel: async (operationId, request) => {
        requests.push(structuredClone(request));
        const embedding = request.requirements?.task === "feature-extraction";
        const contract = embedding ? "embeddings-v1" : request.requirements?.contract?.id ?? "chat-v1";
        const sessionId = `fixture-session-${++sessionNumber}`;
        contracts.set(sessionId, contract);
        const progress = progressListeners.get(operationId);
        progress?.({ stage: "resolving" });
        progress?.({ stage: "awaiting-permission" });
        const session: ModelSession = {
          sessionId,
          role: request.role,
          origin: location.origin,
          resolution: {
            preferredModel: model,
            resolvedModel: model,
            compatibility: "compatible",
            executability: "executable",
            security: "allowed",
            trust: "oryn-verified",
            developerCertified: false,
            substituted: false,
            overriddenByUser: false,
            resolutionReason: "capability-match",
            compatibilityReasons: [],
          },
          capabilities: {
            task: embedding ? "feature-extraction" : "text-generation",
            contextLength: 32768,
          },
          taskProfile: {
            kind: "task-profile",
            schemaVersion: 1,
            profileId: "synthetic-browser-fixture",
            contract: {
              id: contract,
              revision: 1,
            },
            task: embedding ? "feature-extraction" : "text-generation",
            inputModalities: ["text"],
            outputModalities: embedding ? ["embedding"] : ["text"],
            inputSchemaRef: "synthetic",
            resultSchemaRef: "synthetic",
            features: { streaming: "supported" },
            parameters: embedding
              ? {
                  normalize: {
                    inputPointer: "/normalize",
                    valueSchema: { type: "boolean" },
                    handling: "honored",
                    editable: true,
                    affects: ["result"],
                  },
                  inputType: {
                    inputPointer: "/inputType",
                    valueSchema: { enum: ["query", "document"] },
                    handling: "honored",
                    editable: true,
                    affects: ["result"],
                  },
                }
              : {
                  maxTokens: {
                    inputPointer: "/maxTokens",
                    valueSchema: { minimum: 1, maximum: contract === "chat-v2" ? 8192 : 512 },
                    default: 128,
                    handling: "honored",
                    editable: true,
                    affects: ["result"],
                  },
                },
            limits: { contextTokens: 32768 },
            streamingChannels: embedding ? [] : contract === "chat-v2" ? ["text", "reasoning"] : ["text"],
            extensions: {},
            provenance: "runtime-declared",
          },
        };
        progress?.({ stage: "ready", sessionId });
        return success(session);
      },
      cancelModelRequest: async () => undefined,
      onModelRequestProgress: (operationId, listener) => {
        progressListeners.set(operationId, listener);
        return () => {
          progressListeners.delete(operationId);
        };
      },
      releaseModelSession: async (sessionId) => {
        released.push(sessionId);
        return success(undefined);
      },
      invokeJob: async (request) => {
        const jobId = `fixture-job-${++jobNumber}`;
        jobs.set(jobId, request.sessionId);
        const invocation: FixtureInvocation = {
          jobId,
          sessionId: request.sessionId,
          input: structuredClone(request.input) as FixtureInvocation["input"],
        };
        invocations.push(invocation);
        if (invocation.input.texts) {
          emit(jobId, "job.completed", {
            result: {
              kind: "structured",
              value: {
                embeddings: invocation.input.texts.map(() => [1, 0]),
                dimensions: 2,
                normalized: true,
              },
            },
          });
        } else if (!holdText) {
          const filter = invocation.input.messages?.some((message) =>
            message.content.includes("zero-based source indices"),
          );
          if (filter) finish(jobId, "[0]");
          else if (mode === "reasoning-only") {
            reasoningDelta(jobId, "SYNTHETIC_REASONING_ONLY"); finish(jobId, "");
          }
          else {
            if (mode === "reasoning-answer") {
              reasoningDelta(jobId, "SYNTHETIC_REASONING_");
              reasoningDelta(jobId, "SEPARATE");
            }
            // Deliberately emit before acknowledgement to exercise real SDK/app buffering.
            emit(jobId, "job.output.delta", {
              channel: "metadata",
              delta: "NOT-A-TEXT-DELTA",
            });
            emit(jobId, "job.output.delta", {
              channel: "text",
              delta: "A clearer ",
            });
            emit(jobId, "job.output.delta", {
              channel: "text",
              delta: "local sentence.",
            });
            finish(jobId, "A clearer local sentence.");
          }
        }
        return success({
          jobId,
          sessionId: request.sessionId,
          state: "running" as const,
        });
      },
      cancelJob: async (jobId) => {
        cancelled.push(jobId);
        if (!deferCancellation)
          emit(jobId, "job.cancelled", { reason: "website" });
        return success(undefined);
      },
      onJobEvent: (listener) => {
        listeners.add(listener);
        return () => {
          listeners.delete(listener);
        };
      },
      // Text-only tests must never ask for media capabilities.
      beginArtifactCreate: async () => {
        throw new Error("Unexpected media operation in text-only fixture.");
      },
      appendArtifactChunk: async () => {
        throw new Error("Unexpected media operation in text-only fixture.");
      },
      finishArtifactCreate: async () => {
        throw new Error("Unexpected media operation in text-only fixture.");
      },
      abortArtifactCreate: async () => undefined,
      getArtifact: async () => {
        throw new Error("Unexpected media operation in text-only fixture.");
      },
      getArtifactUrl: async () => {
        throw new Error("Unexpected media operation in text-only fixture.");
      },
      saveArtifactAs: async () => {
        throw new Error("Unexpected media operation in text-only fixture.");
      },
      releaseArtifact: async () => success(undefined),
    };
    Object.defineProperty(window, "localAI", {
      value: bridge,
      configurable: true,
    });
    Object.defineProperty(window, "__editorSDKFixture", {
      value: {
        snapshot: () =>
          structuredClone({
            requests,
            invocations,
            cancelled,
            released,
            listeners: listeners.size,
          }),
        configure: (options: FixtureOptions) => {
          if (options.holdText !== undefined) holdText = options.holdText;
          if (options.deferCancellation !== undefined)
            deferCancellation = options.deferCancellation;
          if (options.mode !== undefined) mode = options.mode;
        },
        delta: (jobId: string, text: string) =>
          emit(jobId, "job.output.delta", { channel: "text", delta: text }),
        reasoning: reasoningDelta,
        finish,
      } satisfies FixtureControl,
    });
  });
}

export function sdkSnapshot(page: Page): Promise<SDKFixtureState> {
  return page.evaluate(() =>
    (window as unknown as FixtureWindow).__editorSDKFixture.snapshot(),
  );
}

export async function configureSDK(
  page: Page,
  options: FixtureOptions,
): Promise<void> {
  await page.evaluate(
    (value) =>
      (window as unknown as FixtureWindow).__editorSDKFixture.configure(value),
    options,
  );
}

export async function emitSDKReasoning(page: Page, jobId: string, text: string): Promise<void> {
  await page.evaluate(({ id, value }) => (window as unknown as FixtureWindow).__editorSDKFixture.reasoning(id, value), { id: jobId, value: text });
}

export async function emitSDKText(
  page: Page,
  jobId: string,
  text: string,
  complete = false,
): Promise<void> {
  await page.evaluate(
    ({ id, value, done }) => {
      const fixture = (window as unknown as FixtureWindow).__editorSDKFixture;
      if (done) fixture.finish(id, value);
      else fixture.delta(id, value);
    },
    { id: jobId, value: text, done: complete },
  );
}
