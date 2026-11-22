import { afterEach, describe, expect, it, vi } from "vitest";
import { createLocalAIClient, type InjectedLocalAI } from "@ocno/oryn-sdk";
import type {
  LocalAIClient,
  LocalAIEvent,
  ModelSession,
  RuntimeCapabilities,
} from "@ocno/oryn-sdk";
import {
  buildQuestionPrompt,
  LocalAIService,
  textOf,
  writingPrompt,
  IncompleteGenerationError,
  thinkingNotice,
} from "../src/ai.ts";
import {
  authorizedSources,
  findRelated,
  generateGraph,
} from "../src/retrieval.ts";
import type { EditorDocument } from "../src/model.ts";

function document(id: string, text: string, ownerId?: string): EditorDocument {
  return {
    id,
    title: id,
    content: {
      type: "doc",
      content: [
        {
          type: "paragraph",
          attrs: { blockId: "one" },
          content: [{ type: "text", text }],
        },
      ],
    },
    ideas: {},
    createdAt: 1,
    updatedAt: 1,
    revision: 1,
    protection: ownerId ? "protected" : "open",
    ...(ownerId ? { ownerId } : {}),
    kind: "document",
  };
}

function harness() {
  const listeners = new Set<(event: LocalAIEvent) => void>();
  const sessions = new Map<string, ModelSession>();
  const jobs = new Map<string, string>();
  let sessionSequence = 0,
    jobSequence = 0,
    sequence = 0;
  const capabilities: RuntimeCapabilities = {
    runtimeAvailable: true,
    runtimeState: "ready",
    sdkVersion: "0.1.0",
    supportedModelProviders: ["huggingface"],
    taskContracts: [{ id: "chat-v2", revision: 1 }, { id: "chat-v1", revision: 1 }, { id: "embeddings-v1", revision: 1 }],
    features: {
      taskProfiles: true,
      modelRequests: true,
      modelSubstitution: true,
      streaming: true,
      artifacts: true,
      artifactUrls: true,
      cancellation: true,
      queueing: true,
      multipleSubmittedJobs: true,
    },
  };
  const emit = (
    sessionId: string,
    jobId: string,
    event: LocalAIEvent["event"],
    data: unknown,
  ) => {
    const payload = data as { result?: { kind?: string; value?: Record<string, unknown> } };
    if (event === "job.completed" && sessions.get(sessionId)?.taskProfile?.contract.id === "chat-v2" &&
        payload.result?.kind === "structured" && typeof payload.result.value?.text === "string") {
      const input = [...invoke.mock.calls].reverse().find(([request]) => request.sessionId === sessionId)?.[0].input as { thinking?: boolean } | undefined;
      const requested = input?.thinking ?? null;
      data = { ...payload, result: { ...payload.result, value: {
        finishReason: "stop", usage: { inputTokens: null, outputTokens: null, reasoningTokens: null, answerTokens: null },
        metadata: { thinking: { requested, applied: requested !== null, effective: requested,
          reason: requested === null ? "model_default" : "applied", source: "engine" } },
        ...payload.result.value,
      } } };
    }
    for (const listener of listeners)
      listener({
        sessionId,
        jobId,
        event,
        data,
        timestamp: 1,
        sequence: ++sequence,
      } as LocalAIEvent);
  };
  const model = {
    provider: "huggingface" as const,
    repository: "synthetic/model",
    revision: "test",
    sourceUrl: "https://models.invalid/test",
  };
  const request = vi.fn<LocalAIClient["models"]["request"]>(
    async (request, options) => {
      if (options?.signal?.aborted)
        throw new DOMException("Cancelled", "AbortError");
      options?.onProgress?.({ stage: "awaiting-permission" });
      const embeddings = request.requirements?.task === "feature-extraction";
      const contract = embeddings ? "embeddings-v1" : request.requirements?.contract?.id ?? "chat-v1";
      const session: ModelSession = {
        sessionId: `session-${++sessionSequence}`,
        role: request.role,
        origin: "https://editor.invalid",
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
          contextLength: 32768,
          task: embeddings ? "feature-extraction" : "text-generation",
        },
        taskProfile: {
          kind: "task-profile",
          schemaVersion: 1,
          profileId: "synthetic",
          contract: {
            id: contract,
            revision: 1,
          },
          task: embeddings ? "feature-extraction" : "text-generation",
          inputModalities: ["text"],
          outputModalities: embeddings ? ["embedding"] : ["text"],
          inputSchemaRef: `taskInput.${contract}`,
          resultSchemaRef: `taskResult.${contract}`,
          features: embeddings
            ? { embeddings: "supported" }
            : { chat: "supported", streaming: "supported" },
          parameters: embeddings
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
                  valueSchema: { type: "string", enum: ["query", "document"] },
                  handling: "honored",
                  editable: true,
                  affects: ["result"],
                },
              }
            : {
                maxTokens: {
                  inputPointer: "/maxTokens",
                  valueSchema: { type: "integer", minimum: 1, maximum: contract === "chat-v2" ? 8192 : 512 },
                  default: 128,
                  handling: "honored",
                  editable: true,
                  affects: ["result"],
                },
              },
          limits: embeddings
            ? { batchItems: 64, textCharacters: 8192 }
            : { contextTokens: 32768 },
          streamingChannels: embeddings ? [] : contract === "chat-v2" ? ["text", "reasoning"] : ["text"],
          extensions: {},
          provenance: "runtime-declared",
        },
      };
      sessions.set(session.sessionId, session);
      return session;
    },
  );
  const release = vi.fn(async () => undefined);
  const invoke = vi.fn<LocalAIClient["invoke"]>(async (invocation) => {
    const jobId = `job-${++jobSequence}`;
    jobs.set(jobId, invocation.sessionId);
    const input = invocation.input as {
      messages?: { content: string }[];
      texts?: string[];
    };
    if (input.texts) {
      emit(invocation.sessionId, jobId, "job.completed", {
        result: {
          kind: "structured",
          value: {
            embeddings: input.texts.map(() => [3, 4]),
            dimensions: 2,
            normalized: false,
          },
        },
      });
    } else {
      emit(invocation.sessionId, "another-job", "job.output.delta", {
        channel: "text",
        delta: "wrong-job-secret",
      });
      emit("another-session", jobId, "job.output.delta", {
        channel: "text",
        delta: "wrong-session-secret",
      });
      emit(invocation.sessionId, jobId, "job.output.delta", {
        channel: "metadata",
        delta: "not-text",
      });
      emit(invocation.sessionId, jobId, "job.output.delta", {
        channel: "text",
        delta: "Hello ",
      });
      emit(invocation.sessionId, jobId, "job.output.delta", {
        channel: "text",
        delta: "locally",
      });
      emit(invocation.sessionId, jobId, "job.completed", {
        result: { kind: "structured", value: { text: "Hello locally" } },
      });
    }
    return { jobId, sessionId: invocation.sessionId, state: "running" };
  });
  const cancel = vi.fn(async (jobId: string) => {
    const sessionId = jobs.get(jobId);
    if (sessionId)
      emit(sessionId, jobId, "job.cancelled", { reason: "website" });
  });
  const client = {
    sdkVersion: "0.1.0",
    getCapabilities: vi.fn(async () => capabilities),
    models: { request, releaseSession: release },
    invoke,
    cancel,
    subscribe: (listener: (event: LocalAIEvent) => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    artifacts: {},
  } as unknown as LocalAIClient;
  return {
    client,
    capabilities,
    request,
    release,
    invoke,
    cancel,
    emit,
    listeners,
    sessions,
    jobs,
  };
}

describe("local AI lifecycle", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("does not treat the global streaming flag as a prohibition on a local model", async () => {
    const h = harness();
    // Current Engine reports this global flag false even for streaming chat models.
    h.capabilities.features.streaming = false;
    const service = new LocalAIService(h.client);
    expect((await service.capabilities()).available).toBe(true);
    expect(await service.generate("A synthetic question")).toBe(
      "Hello locally",
    );
    expect(h.request.mock.calls[0][0].requirements?.capabilities).toEqual({
      chat: true,
    });
    expect(
      h.request.mock.calls[0][0].requirements?.requiredFeatures,
    ).toBeUndefined();
    expect(h.release).toHaveBeenCalledOnce();
  });

  it("uses the packaged SDK and a final text result when the selected model does not stream", async () => {
    const h = harness(),
      onText = vi.fn();
    h.capabilities.features.streaming = false;
    const original = h.request.getMockImplementation()!;
    h.request.mockImplementationOnce(async (request, options) => {
      const session = await original(request, options);
      session.taskProfile!.features.streaming = "unsupported";
      session.taskProfile!.streamingChannels = [];
      session.capabilities.capabilities = { chat: true, streaming: false };
      return session;
    });
    h.invoke.mockImplementationOnce(async (request) => {
      h.emit(request.sessionId, "completed-only", "job.completed", {
        result: { kind: "structured", value: { text: "Final local response" } },
      });
      return {
        jobId: "completed-only",
        sessionId: request.sessionId,
        state: "running",
      };
    });
    const unexpectedMedia = async (): Promise<never> => {
      throw new Error(
        "Unexpected media operation in text-only SDK regression test.",
      );
    };
    const bridge: InjectedLocalAI = {
      sdkVersion: "0.1.0",
      getCapabilities: h.client.getCapabilities,
      requestModel: async (
        _operation: string,
        request: Parameters<LocalAIClient["models"]["request"]>[0],
      ) => ({ ok: true, value: await h.client.models.request(request) }),
      cancelModelRequest: async () => undefined,
      onModelRequestProgress: () => () => undefined,
      releaseModelSession: async (id: string) => {
        await h.client.models.releaseSession(id);
        return { ok: true, value: undefined };
      },
      invokeJob: async (request: Parameters<LocalAIClient["invoke"]>[0]) => ({
        ok: true,
        value: await h.client.invoke(request),
      }),
      cancelJob: async (id: string) => {
        await h.client.cancel(id);
        return { ok: true, value: undefined };
      },
      onJobEvent: h.client.subscribe,
      beginArtifactCreate: unexpectedMedia,
      appendArtifactChunk: unexpectedMedia,
      finishArtifactCreate: unexpectedMedia,
      abortArtifactCreate: unexpectedMedia,
      getArtifact: unexpectedMedia,
      getArtifactUrl: unexpectedMedia,
      saveArtifactAs: unexpectedMedia,
      releaseArtifact: unexpectedMedia,
    };
    vi.stubGlobal("window", { localAI: bridge });
    expect(
      await new LocalAIService(createLocalAIClient()).generate(
        "Synthetic question",
        { onText },
      ),
    ).toBe("Final local response");
    expect(onText).toHaveBeenCalledExactlyOnceWith("Final local response");
    expect(h.release).toHaveBeenCalledOnce();
    expect(h.listeners.size).toBe(0);
  });

  it("honors model-specific output bounds nested inside allOf", async () => {
    const h = harness(),
      original = h.request.getMockImplementation()!;
    h.request.mockImplementationOnce(async (request, options) => {
      const session = await original(request, options);
      const parameter = session.taskProfile!.parameters.maxTokens;
      parameter.valueSchema = {
        allOf: [parameter.valueSchema, { maximum: 96 }],
      };
      parameter.default = 64;
      return session;
    });
    await new LocalAIService(h.client).generate("Synthetic question");
    expect(h.invoke.mock.calls[0][0].input).toMatchObject({ maxTokens: 96 });
  });

  it("sends fixed output values and omits unavailable task parameters", async () => {
    const h = harness(),
      original = h.request.getMockImplementation()!;
    h.request.mockImplementationOnce(async (request, options) => {
      const session = await original(request, options);
      const parameter = session.taskProfile!.parameters.maxTokens;
      parameter.handling = "fixed";
      parameter.valueSchema = { allOf: [parameter.valueSchema, { const: 64 }] };
      parameter.default = 64;
      return session;
    });
    const service = new LocalAIService(h.client);
    await service.generate("Synthetic question");
    expect(h.invoke.mock.calls[0][0].input).toMatchObject({ maxTokens: 64 });
    h.request.mockImplementationOnce(async (request, options) => {
      const session = await original(request, options);
      delete session.taskProfile!.parameters.maxTokens;
      return session;
    });
    await service.generate("Another synthetic question");
    expect(h.invoke.mock.calls[1][0].input).not.toHaveProperty("maxTokens");
  });

  it("honors negotiated embedding batch and text-character limits", async () => {
    const h = harness(),
      original = h.request.getMockImplementation()!;
    h.request.mockImplementation(async (request, options) => {
      const session = await original(request, options);
      session.taskProfile!.limits = { batchItems: 2, textCharacters: 10 };
      return session;
    });
    const service = new LocalAIService(h.client);
    expect(
      await service.embeddings(["query", "one", "two", "three", "four"], {
        queryCount: 1,
      }),
    ).toHaveLength(5);
    expect(
      h.invoke.mock.calls.map(
        ([call]) => (call.input as { texts: string[] }).texts.length,
      ),
    ).toEqual([1, 2, 2]);
    await expect(
      service.embeddings(["longer than ten characters"]),
    ).rejects.toThrow(/at most 10 characters/);
    expect(h.invoke).toHaveBeenCalledTimes(3);
    expect(h.release).toHaveBeenCalledTimes(2);
  });

  it("detects unavailable runtime without requesting a model or producing fake text", async () => {
    const h = harness();
    h.capabilities.runtimeAvailable = false;
    const service = new LocalAIService(h.client);
    expect((await service.capabilities()).available).toBe(false);
    await expect(service.generate("Hello")).rejects.toThrow(/compatible Oryn/);
    expect(h.request).not.toHaveBeenCalled();
    expect(h.invoke).not.toHaveBeenCalled();
  });

  it("buffers early events, filters job/session/channel, streams accumulated text and releases", async () => {
    const h = harness(),
      onText = vi.fn(),
      onStatus = vi.fn();
    const service = new LocalAIService(h.client);
    expect(
      await service.generate("A synthetic question", { onText, onStatus }),
    ).toBe("Hello locally");
    expect(onText.mock.calls.map(([text]) => text)).toEqual([
      "Hello ",
      "Hello locally",
    ]);
    expect(onStatus).toHaveBeenCalledWith("Waiting for your Oryn permission…");
    expect(h.request.mock.calls[0][0]).toMatchObject({
      substitution: { automatic: "compatible" },
      requirements: { task: "text-generation" },
    });
    expect(h.request.mock.calls[0][0]).not.toHaveProperty("preferredModel");
    expect(h.release).toHaveBeenCalledWith("session-1");
    expect(h.listeners.size).toBe(0);
  });

  it("returns completed text even when no text deltas arrive", async () => {
    const h = harness(),
      onText = vi.fn();
    h.invoke.mockImplementationOnce(async (request) => {
      h.emit(request.sessionId, "final", "job.completed", {
        result: { kind: "structured", value: { text: "Final only" } },
      });
      return { jobId: "final", sessionId: request.sessionId, state: "running" };
    });
    expect(
      await new LocalAIService(h.client).generate("Question", { onText }),
    ).toBe("Final only");
    expect(onText).toHaveBeenCalledWith("Final only");
    expect(h.release).toHaveBeenCalledOnce();
  });

  it("rejects terminal model failure and invalid results, releasing every session", async () => {
    const h = harness();
    h.invoke.mockImplementationOnce(async (request) => {
      h.emit(request.sessionId, "failed", "job.failed", {
        error: {
          code: "MODEL_OUT_OF_MEMORY",
          message: "Choose a smaller local model.",
        },
      });
      return {
        jobId: "failed",
        sessionId: request.sessionId,
        state: "running",
      };
    });
    const service = new LocalAIService(h.client);
    await expect(service.generate("Question")).rejects.toThrow(
      /smaller local model/,
    );
    h.invoke.mockImplementationOnce(async (request) => {
      h.emit(request.sessionId, "invalid", "job.completed", {
        result: { kind: "structured", value: { unrelated: "not text" } },
      });
      return {
        jobId: "invalid",
        sessionId: request.sessionId,
        state: "running",
      };
    });
    await expect(service.generate("Question")).rejects.toThrow(
      /invalid chat-v2 result/,
    );
    expect(h.release).toHaveBeenCalledTimes(2);
    expect(h.listeners.size).toBe(0);
  });

  it("cancels an abort that occurs before invocation acknowledgement and suppresses late plaintext", async () => {
    const h = harness(),
      controller = new AbortController(),
      onText = vi.fn();
    let acknowledge!: (value: {
      jobId: string;
      sessionId: string;
      state: "running";
    }) => void;
    h.invoke.mockImplementationOnce((request) => {
      h.jobs.set("pending", request.sessionId);
      return new Promise((resolve) => {
        acknowledge = resolve;
      });
    });
    const result = new LocalAIService(h.client).generate(
      "Alice synthetic secret",
      { signal: controller.signal, onText },
    );
    const rejected = expect(result).rejects.toMatchObject({
      code: "REQUEST_ABORTED",
    });
    await vi.waitFor(() => expect(h.invoke).toHaveBeenCalledOnce());
    controller.abort();
    h.emit("session-1", "pending", "job.output.delta", {
      channel: "text",
      delta: "Alice-secret-marker",
    });
    acknowledge({ jobId: "pending", sessionId: "session-1", state: "running" });
    await rejected;
    expect(h.cancel).toHaveBeenCalledWith("pending");
    expect(onText).not.toHaveBeenCalled();
    expect(h.release).toHaveBeenCalledOnce();
    expect(h.listeners.size).toBe(0);
  });

  it("aborts model preparation on dispose and can run a new isolated operation afterward", async () => {
    const h = harness();
    h.request.mockImplementationOnce(
      async (_request, options) =>
        new Promise((_resolve, reject) => {
          options?.signal?.addEventListener(
            "abort",
            () => reject(new DOMException("Cancelled", "AbortError")),
            { once: true },
          );
        }),
    );
    const service = new LocalAIService(h.client),
      result = service.generate("Alice-only");
    const rejected = expect(result).rejects.toThrow(/Cancelled/);
    await vi.waitFor(() => expect(h.request).toHaveBeenCalledOnce());
    await service.dispose();
    await rejected;
    expect(h.invoke).not.toHaveBeenCalled();
    expect(await service.generate("Open-only")).toBe("Hello locally");
    expect(h.release).toHaveBeenCalledOnce();
  });

  it("disposes a running job before a new user can receive its callbacks", async () => {
    const h = harness(),
      onText = vi.fn();
    h.invoke.mockImplementationOnce(async (request) => {
      h.jobs.set("running-private", request.sessionId);
      return {
        jobId: "running-private",
        sessionId: request.sessionId,
        state: "running",
      };
    });
    const service = new LocalAIService(h.client);
    const result = service.generate("Alice-only synthetic question", {
      onText,
    });
    const rejected = expect(result).rejects.toMatchObject({
      code: "REQUEST_ABORTED",
    });
    await vi.waitFor(() => expect(h.invoke).toHaveBeenCalledOnce());
    await service.dispose();
    await rejected;
    h.emit("session-1", "running-private", "job.output.delta", {
      channel: "text",
      delta: "Alice-secret-marker",
    });
    expect(h.cancel).toHaveBeenCalledWith("running-private");
    expect(onText).not.toHaveBeenCalled();
    expect(h.release).toHaveBeenCalledOnce();
    expect(h.listeners.size).toBe(0);
  });

  it("rejects oversize context without invoking or trimming the prompt", async () => {
    const h = harness(),
      original = h.request.getMockImplementation()!;
    h.capabilities.taskContracts = [];
    h.request.mockImplementationOnce(async (request, options) => {
      const session = await original(request, options);
      session.taskProfile!.limits.contextTokens = 700;
      return session;
    });
    await expect(
      new LocalAIService(h.client).generate("x".repeat(200)),
    ).rejects.toThrow(/nothing was truncated/);
    expect(h.invoke).not.toHaveBeenCalled();
    expect(h.release).toHaveBeenCalledOnce();
  });

  it("uses one fresh embedding session for query/document modes and batches, normalizing valid vectors", async () => {
    const h = harness(),
      service = new LocalAIService(h.client);
    const vectors = await service.embeddings(
      Array.from({ length: 67 }, (_, index) => `synthetic ${index}`),
      { queryCount: 1 },
    );
    expect(vectors).toHaveLength(67);
    expect(vectors[0]).toEqual([0.6, 0.8]);
    expect(
      h.invoke.mock.calls.map(
        ([call]) => (call.input as { inputType: string }).inputType,
      ),
    ).toEqual(["query", "document", "document"]);
    expect(h.request).toHaveBeenCalledOnce();
    expect(h.release).toHaveBeenCalledOnce();
  });

  it("rejects malformed or incomplete embedding output", async () => {
    const h = harness();
    h.invoke.mockImplementationOnce(async (request) => {
      h.emit(request.sessionId, "bad-vector", "job.completed", {
        result: {
          kind: "structured",
          value: { embeddings: [[0, 0]], dimensions: 2, normalized: true },
        },
      });
      return {
        jobId: "bad-vector",
        sessionId: request.sessionId,
        state: "running",
      };
    });
    await expect(
      new LocalAIService(h.client).embeddings(["test"]),
    ).rejects.toThrow(/empty embedding/);
    expect(h.release).toHaveBeenCalledOnce();
  });
});

describe("explicit thinking and safe completion", () => {
  it("negotiates v2 and sends Off without making it a model-selection requirement", async () => {
    const h = harness();
    const response = await new LocalAIService(h.client).generateResult("A writing request", { thinking: false });
    expect(h.request.mock.calls[0][0].requirements?.contract).toEqual({ id: "chat-v2", revision: 1 });
    expect(h.request.mock.calls[0][0].requirements).not.toHaveProperty("parameterValues");
    expect(h.invoke.mock.calls[0][0].input).toMatchObject({ thinking: false, maxTokens: 4096 });
    expect(response.metadata.thinking).toMatchObject({ requested: false, applied: true, effective: false, source: "engine" });
  });

  it("continues an unsupported Off request and reports the unapplied fixed mode", async () => {
    const h = harness();
    h.invoke.mockImplementationOnce(async (request) => {
      h.emit(request.sessionId, "fixed", "job.completed", { result: { kind: "structured", value: {
        text: "A complete final answer.",
        metadata: { thinking: { requested: false, applied: false, effective: true, reason: "fixed_mode", source: "engine" } },
      } } });
      return { jobId: "fixed", sessionId: request.sessionId, state: "running" };
    });
    const onResult = vi.fn();
    expect(await new LocalAIService(h.client).generate("Write", { thinking: false, onResult })).toBe("A complete final answer.");
    expect(thinkingNotice(onResult.mock.calls[0][0])).toMatch(/Off was not applied.*fixed mode.*continued/);
    expect(h.invoke).toHaveBeenCalledOnce();
  });

  it("keeps reasoning out of answer callbacks and requires a clean stop before writing", async () => {
    const h = harness(); const onText = vi.fn(), onReasoning = vi.fn();
    h.invoke.mockImplementationOnce(async (request) => {
      h.emit(request.sessionId, "reasoning", "job.output.delta", { channel: "reasoning", delta: "PRIVATE_REASONING" });
      h.emit(request.sessionId, "reasoning", "job.output.delta", { channel: "text", delta: "Final prose." });
      h.emit(request.sessionId, "reasoning", "job.completed", { result: { kind: "structured", value: { text: "Final prose.", reasoning: "PRIVATE_REASONING" } } });
      return { jobId: "reasoning", sessionId: request.sessionId, state: "running" };
    });
    expect(await new LocalAIService(h.client).generate("Write", { thinking: true, onText, onReasoning })).toBe("Final prose.");
    expect(onReasoning).toHaveBeenCalledWith("PRIVATE_REASONING");
    expect(JSON.stringify(onText.mock.calls)).not.toContain("PRIVATE_REASONING");
  });

  it.each([
    { text: "", reasoning: "Reasoning only", finishReason: "stop", message: /reasoning without a final answer/ },
    { text: "An unfinished final answer", reasoning: "", finishReason: "length", message: /generation limit/ },
    { text: "Unverified answer", reasoning: "", finishReason: "unknown", message: /Completion is unconfirmed/ },
  ])("preserves originals for $finishReason / $text", async ({ message, ...value }) => {
    const h = harness();
    h.invoke.mockImplementationOnce(async (request) => {
      h.emit(request.sessionId, "incomplete", "job.completed", { result: { kind: "structured", value } });
      return { jobId: "incomplete", sessionId: request.sessionId, state: "running" };
    });
    await expect(new LocalAIService(h.client).generate("Write", { thinking: false })).rejects.toThrow(message);
    expect(h.release).toHaveBeenCalledOnce();
  });

  it("previews honest legacy output but never auto-writes or transmits an unsupported thinking field", async () => {
    const h = harness(); h.capabilities.taskContracts = [];
    const onText = vi.fn(); const service = new LocalAIService(h.client);
    const result = await service.generateResult("Question", { thinking: false, onText });
    expect(result.finishReason).toBe("unknown");
    expect(result.metadata.thinking).toMatchObject({ requested: false, applied: false, effective: null, reason: "unsupported_peer", source: "sdk_compatibility" });
    expect(h.invoke.mock.calls[0][0].input).not.toHaveProperty("thinking");
    expect(h.invoke.mock.calls[0][0].input).toMatchObject({ maxTokens: 512 });
    expect(onText.mock.calls).toEqual([["Hello locally"]]);
    await expect(service.generate("Write", { thinking: true })).rejects.toBeInstanceOf(IncompleteGenerationError);
  });

  it("never flashes split legacy reasoning markers and withholds tagged legacy output", async () => {
    const h = harness(); h.capabilities.taskContracts = [];
    h.invoke.mockImplementationOnce(async (request) => {
      for (const delta of ["<thi", "nk>private reasoning", "</think>Final"])
        h.emit(request.sessionId, "legacy", "job.output.delta", { channel: "text", delta });
      h.emit(request.sessionId, "legacy", "job.completed", { result: { kind: "structured", value: { text: "<think>private reasoning</think>Final" } } });
      return { jobId: "legacy", sessionId: request.sessionId, state: "running" };
    });
    const onText = vi.fn();
    const result = await new LocalAIService(h.client).generateResult("Question", { onText });
    expect(result.text).toBe("");
    expect(JSON.stringify(onText.mock.calls)).not.toContain("private reasoning");
    expect(result.result).toMatchObject({ value: { text: "<think>private reasoning</think>Final" } });
  });

  it("preserves literal tag examples in a negotiated final-answer channel", async () => {
    const h = harness(); const text = "The literal <think> tag is part of this documentation.";
    h.invoke.mockImplementationOnce(async (request) => {
      h.emit(request.sessionId, "literal", "job.completed", { result: { kind: "structured", value: { text } } });
      return { jobId: "literal", sessionId: request.sessionId, state: "running" };
    });
    expect(await new LocalAIService(h.client).generate("Explain the tag")).toBe(text);
  });

  it("rejects conflicting final reconciliation instead of applying a guessed answer", async () => {
    const h = harness();
    h.invoke.mockImplementationOnce(async (request) => {
      h.emit(request.sessionId, "mismatch", "job.output.delta", { channel: "text", delta: "Streamed final" });
      h.emit(request.sessionId, "mismatch", "job.completed", { result: { kind: "structured", value: { text: "Different final" } } });
      return { jobId: "mismatch", sessionId: request.sessionId, state: "running" };
    });
    await expect(new LocalAIService(h.client).generate("Write")).rejects.toThrow(/did not match its stream/);
  });
});

describe("document boundaries and prompts", () => {
  const open = document("open", "Open writing");
  const alice = document("alice", "Alice-secret-marker", "alice-user");
  const bob = document("bob", "Bob-secret-marker", "bob-user");

  it("restricts sources by destination even when other owners are already decrypted", () => {
    const library = [open, alice, bob, document("other-open", "shared")];
    expect(authorizedSources(open, library).map((source) => source.id)).toEqual(
      ["other-open"],
    );
    expect(
      authorizedSources(alice, library).map((source) => source.id),
    ).toEqual(["open", "other-open"]);
    const aliceSource = document(
      "alice-source",
      "Alice second secret",
      "alice-user",
    );
    expect(
      authorizedSources(alice, [...library, aliceSource]).map(
        (source) => source.id,
      ),
    ).toEqual(["open", "other-open", "alice-source"]);
    expect(
      authorizedSources({ ...alice, ownerId: undefined }, library),
    ).toEqual([]);
  });

  it("sends only authorized source text to embeddings and the optional advanced filter", async () => {
    const embeddings = vi.fn(async (texts: string[]) =>
      texts.map(() => [1, 0]),
    );
    const generate = vi.fn(async () => "[0]");
    const result = await findRelated(
      { embeddings, generate },
      open,
      [open, alice, bob, document("reference", "Open reference")],
      "Open writing",
      { advanced: true },
    );
    expect(result.map((match) => match.documentId)).toEqual(["reference"]);
    const sent = JSON.stringify([embeddings.mock.calls, generate.mock.calls]);
    expect(sent).not.toContain("Alice-secret-marker");
    expect(sent).not.toContain("Bob-secret-marker");
    expect(sent).toContain("Open reference");
  });

  it("does not produce fake local search results when no authorized sources exist", async () => {
    const embeddings = vi.fn(),
      generate = vi.fn();
    expect(
      await findRelated({ embeddings, generate }, open, [alice, bob], "", {}),
    ).toEqual([]);
    expect(embeddings).not.toHaveBeenCalled();
    expect(generate).not.toHaveBeenCalled();
  });

  it("keeps general prompts free of document context and rejects a stale cross-document selection", () => {
    expect(
      buildQuestionPrompt(
        "General question",
        "general",
        alice,
        "Alice-secret-marker",
      ),
    ).toBe("General question");
    expect(() =>
      buildQuestionPrompt("Question", "context", open, "Alice-secret-marker"),
    ).toThrow(/no longer belongs/);
    expect(buildQuestionPrompt("Question", "article", alice, "")).toContain(
      "Alice-secret-marker",
    );
  });

  it("bounds selected-passage context and includes associated ideas as selectable context", () => {
    const doc = document(
      "long",
      `${"before ".repeat(250)}SELECTED${" after".repeat(250)}`,
    );
    const prompt = buildQuestionPrompt("Question", "context", doc, "SELECTED");
    expect(prompt.match(/before/g)).toHaveLength(200);
    expect(prompt.match(/after/g)).toHaveLength(200);
    doc.ideas = { one: "Specific idea" };
    expect(
      buildQuestionPrompt("Question", "context", doc, "Specific idea"),
    ).toContain("Specific idea");
    expect(
      writingPrompt("comments", "Text [expand this]", "", "neighbor"),
    ).toContain("square-bracket");
    expect(
      textOf({
        type: "doc",
        content: [
          { type: "paragraph", content: [{ type: "text", text: "one" }] },
          { type: "paragraph", content: [{ type: "text", text: "two" }] },
        ],
      }),
    ).toBe("one\ntwo");
  });

  it("rejects a stale selection before a source lookup makes any model request", async () => {
    const embeddings = vi.fn(),
      generate = vi.fn();
    await expect(
      findRelated(
        { embeddings, generate },
        open,
        [alice],
        "Alice-secret-marker",
      ),
    ).rejects.toThrow(/no longer belongs/);
    expect(embeddings).not.toHaveBeenCalled();
  });

  it("generates validated topic relations only between the current document ideas", async () => {
    const doc = { ...alice, ideas: { one: "First idea", two: "Second idea" } };
    const embeddings = vi.fn(async () => [
      [1, 0],
      [0.8, 0.2],
    ]);
    const generate = vi.fn(
      async () => '{"edges":[{"from":"one","to":"two","relation":"supports"}]}',
    );
    const graph = await generateGraph({ embeddings, generate }, doc);
    expect(graph.nodes).toHaveLength(2);
    expect(graph.edges[0].relation).toBe("supports");
    expect(JSON.stringify(generate.mock.calls)).not.toContain(
      "Bob-secret-marker",
    );
    generate.mockResolvedValueOnce(
      '{"edges":[{"from":"one","to":"bob-private","relation":"supports"}]}',
    );
    await expect(generateGraph({ embeddings, generate }, doc)).rejects.toThrow(
      /unknown idea/,
    );
    generate.mockResolvedValueOnce(
      '{"edges":[{"from":"one","to":"two","relation":"invented"}]}',
    );
    await expect(generateGraph({ embeddings, generate }, doc)).rejects.toThrow(
      /unknown idea/,
    );
  });
});
