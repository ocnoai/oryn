import { afterEach, describe, expect, it, vi } from "vitest";

import {
  createLocalAIClient,
  invokeAndWait,
  invokeStream,
  LocalAIError,
  type InjectedLocalAI,
  type LocalAIEvent,
  type ModelSession,
} from "../index.js";

function injected(overrides: Partial<InjectedLocalAI> = {}): InjectedLocalAI {
  return {
    sdkVersion: "0.1.0",
    getCapabilities: async () => ({
      runtimeAvailable: true,
      runtimeState: "ready",
      sdkVersion: "0.1.0",
      supportedModelProviders: ["huggingface"],
      features: {
        modelRequests: true,
        modelSubstitution: true,
        streaming: false,
        artifacts: false,
        artifactUrls: false,
        cancellation: true,
        queueing: false,
        multipleSubmittedJobs: false,
      },
    }),
    requestModel: async () => ({
      ok: false,
      error: { code: "USER_DENIED", message: "The user denied this request." },
    }),
    cancelModelRequest: async () => undefined,
    releaseModelSession: async () => ({ ok: true, value: undefined }),
    onModelRequestProgress: () => () => undefined,
    beginArtifactCreate: async () => ({
      ok: true,
      value: { uploadId: "artifact_upload_abcdefghijklmnopqrstuvwxyz" },
    }),
    appendArtifactChunk: async () => ({ ok: true, value: undefined }),
    finishArtifactCreate: async () => ({
      ok: true,
      value: {
        id: "artifact_abcdefghijklmnopqrstuvwxyz",
        mimeType: "application/octet-stream",
        sizeBytes: 0,
      },
    }),
    abortArtifactCreate: async () => undefined,
    getArtifact: async () => ({
      ok: false,
      error: {
        code: "ARTIFACT_NOT_FOUND",
        message: "The artifact is unavailable.",
      },
    }),
    getArtifactUrl: async () => ({
      ok: true,
      value:
        "oryn-artifact://capability/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN",
    }),
    saveArtifactAs: async () => ({ ok: true, value: undefined }),
    releaseArtifact: async () => ({ ok: true, value: undefined }),
    invokeJob: async () => ({
      ok: true,
      value: {
        jobId: "job_example",
        sessionId: "session_one",
        state: "running",
      },
    }),
    cancelJob: async () => ({ ok: true, value: undefined }),
    onJobEvent: () => () => undefined,
    ...overrides,
  };
}

describe("createLocalAIClient", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });
  it.each(['source-checkpoint', 'exact-artifact'] as const)('does not downgrade %s intent in an older Browser', async (intent) => {
    const requestModel = vi.fn(injected().requestModel);
    vi.stubGlobal('window', {localAI: injected({requestModel})});
    const model = {provider:'huggingface' as const,repository:'org/model',revision:'a'.repeat(40)};
    const request = intent === 'source-checkpoint'
      ? {role:'chat',selection:{version:1 as const,intent,sourceCheckpoint:model}}
      : {role:'chat',preferredModel:{...model,files:['weights.gguf']},selection:{version:1 as const,intent}};
    await expect(createLocalAIClient().models.request(request)).rejects.toThrow('does not support model selection');
    expect(requestModel).not.toHaveBeenCalled();
  });

  it("reports an unavailable runtime outside the Local AI Browser", async () => {
    vi.stubGlobal("window", undefined);

    const capabilities = await createLocalAIClient().getCapabilities();

    expect(capabilities.runtimeAvailable).toBe(false);
    expect(capabilities.runtimeState).toBe("unavailable");
  });

  it("delegates to the constrained injected client", async () => {
    const getCapabilities = vi.fn(async () => ({
      runtimeAvailable: true as const,
      runtimeState: "ready" as const,
      sdkVersion: "0.1.0",
      browserApiVersion: "1",
      supportedModelProviders: ["huggingface" as const],
      features: {
        modelRequests: false,
        modelSubstitution: false,
        streaming: false,
        artifacts: false,
        artifactUrls: false,
        cancellation: false,
        queueing: false,
        multipleSubmittedJobs: false,
      },
    }));
    vi.stubGlobal("window", {
      localAI: { sdkVersion: "0.1.0", getCapabilities },
    });

    const capabilities = await createLocalAIClient().getCapabilities();

    expect(getCapabilities).toHaveBeenCalledOnce();
    expect(capabilities.runtimeAvailable).toBe(true);
  });

  it("reports progress, returns a pinned session, and releases it", async () => {
    const progress = vi.fn();
    const releaseModelSession = vi.fn(async () => ({
      ok: true as const,
      value: undefined,
    }));
    const session = {
      sessionId: "session_one",
      role: "chat",
      origin: "https://site.test",
      resolution: {
        preferredModel: {
          provider: "huggingface" as const,
          repository: "org/model",
          revision: "a".repeat(40),
          sourceUrl: `https://huggingface.co/org/model/tree/${"a".repeat(40)}`,
        },
        resolvedModel: {
          provider: "huggingface" as const,
          repository: "org/model",
          revision: "a".repeat(40),
          sourceUrl: `https://huggingface.co/org/model/tree/${"a".repeat(40)}`,
        },
        compatibility: "certified" as const,
        executability: "executable" as const,
        security: "allowed" as const,
        trust: "oryn-verified" as const,
        developerCertified: true,
        substituted: false,
        overriddenByUser: false,
        resolutionReason: "preferred-model" as const,
        compatibilityReasons: [],
      },
      runtime: { runner: "transformers", backend: "cpu", deviceClass: "cpu" },
      capabilities: {
        inputModalities: ["text" as const],
        outputModalities: ["text" as const],
      },
    };
    vi.stubGlobal("window", {
      localAI: injected({
        requestModel: async () => ({ ok: true, value: session }),
        releaseModelSession,
        onModelRequestProgress: (_operationId, listener) => {
          listener({ stage: "resolving" });
          listener({ stage: "ready", sessionId: session.sessionId });
          return () => undefined;
        },
      }),
    });
    const client = createLocalAIClient();
    const result = await client.models.request(
      {
        role: "chat",
        preferredModel: { provider: "huggingface", repository: "org/model" },
      },
      { onProgress: progress },
    );
    expect(result).toEqual(session);
    expect(result.resolution.trust).toBe("oryn-verified");
    expect(progress).toHaveBeenCalledTimes(2);
    await client.models.releaseSession(session.sessionId);
    expect(releaseModelSession).toHaveBeenCalledWith("session_one");
  });

  it("maps sanitized bridge errors and handles AbortSignal", async () => {
    const cancelModelRequest = vi.fn(async () => undefined);
    vi.stubGlobal("window", { localAI: injected({ cancelModelRequest }) });
    await expect(
      createLocalAIClient().models.request({
        role: "chat",
        preferredModel: { provider: "huggingface", repository: "org/model" },
      }),
    ).rejects.toMatchObject({
      code: "USER_DENIED",
      message: "The user denied this request.",
    });

    const controller = new AbortController();
    controller.abort();
    await expect(
      createLocalAIClient().models.request(
        {
          role: "chat",
          preferredModel: { provider: "huggingface", repository: "org/model" },
        },
        { signal: controller.signal },
      ),
    ).rejects.toBeInstanceOf(LocalAIError);

    const pendingController = new AbortController();
    let rejectRequest:
      | ((result: Awaited<ReturnType<InjectedLocalAI["requestModel"]>>) => void)
      | undefined;
    vi.stubGlobal("window", {
      localAI: injected({
        cancelModelRequest,
        requestModel: () =>
          new Promise((resolve) => {
            rejectRequest = resolve;
          }),
      }),
    });
    const pending = createLocalAIClient().models.request(
      {
        role: "chat",
        preferredModel: { provider: "huggingface", repository: "org/model" },
      },
      { signal: pendingController.signal },
    );
    pendingController.abort();
    expect(cancelModelRequest).toHaveBeenCalledOnce();
    rejectRequest?.({
      ok: false,
      error: { code: "REQUEST_ABORTED", message: "The request was aborted." },
    });
    await expect(pending).rejects.toMatchObject({ code: "REQUEST_ABORTED" });
  });

  it("fails privileged calls outside Oryn and exposes no global inventory API", async () => {
    vi.stubGlobal("window", undefined);
    const client = createLocalAIClient();
    await expect(
      client.models.request({
        role: "chat",
        preferredModel: { provider: "huggingface", repository: "org/model" },
      }),
    ).rejects.toMatchObject({ code: "RUNTIME_NOT_AVAILABLE" });
    await expect(
      client.invoke({ sessionId: "session_one", input: {} }),
    ).rejects.toMatchObject({ code: "RUNTIME_NOT_AVAILABLE" });
    await expect(
      client.artifacts.create(new Blob(["input"])),
    ).rejects.toMatchObject({ code: "RUNTIME_NOT_AVAILABLE" });
    expect("listInstalled" in client.models).toBe(false);
    expect("settings" in client).toBe(false);
    expect("activity" in client).toBe(false);
  });

  it("streams large Blob creation in bounded binary chunks and manages metadata", async () => {
    const chunks: ArrayBuffer[] = [];
    const finishArtifactCreate = vi.fn(async () => ({
      ok: true as const,
      value: {
        id: "artifact_abcdefghijklmnopqrstuvwxyz",
        mimeType: "audio/wav",
        sizeBytes: 700_000,
        name: "sample.wav",
      },
    }));
    const getArtifact = vi.fn(async () => ({
      ok: true as const,
      value: {
        id: "artifact_abcdefghijklmnopqrstuvwxyz",
        mimeType: "audio/wav",
        sizeBytes: 700_000,
        name: "sample.wav",
        createdAt: 123,
      },
    }));
    const releaseArtifact = vi.fn(async () => ({
      ok: true as const,
      value: undefined,
    }));
    vi.stubGlobal("window", {
      localAI: injected({
        appendArtifactChunk: async (_uploadId, chunk) => {
          chunks.push(chunk);
          return { ok: true, value: undefined };
        },
        finishArtifactCreate,
        getArtifact,
        releaseArtifact,
      }),
    });
    const client = createLocalAIClient();
    const blob = new Blob([new Uint8Array(700_000)], { type: "audio/wav" });
    const artifact = await client.artifacts.create(blob, {
      name: "sample.wav",
    });

    expect(artifact.sizeBytes).toBe(700_000);
    expect(chunks.length).toBe(3);
    expect(chunks.every((chunk) => chunk.byteLength <= 256 * 1024)).toBe(true);
    expect(chunks.reduce((total, chunk) => total + chunk.byteLength, 0)).toBe(
      700_000,
    );
    expect(JSON.stringify(chunks)).not.toContain("base64");
    await expect(client.artifacts.get(artifact.id)).resolves.toMatchObject({
      createdAt: 123,
    });
    await client.artifacts.release(artifact.id);
    expect(releaseArtifact).toHaveBeenCalledWith(artifact.id);
  });

  it("reads through a capability URL, uploads only explicitly, and keeps Save As paths trusted", async () => {
    const artifactId = "artifact_abcdefghijklmnopqrstuvwxyz";
    const capabilityUrl =
      "oryn-artifact://capability/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN";
    const getArtifactUrl = vi.fn(async () => ({
      ok: true as const,
      value: capabilityUrl,
    }));
    const saveArtifactAs = vi.fn(async () => ({
      ok: true as const,
      value: undefined,
    }));
    const fetchMock = vi.fn(async (input: string | URL | Request) => {
      const url = String(input);
      if (url === capabilityUrl) {
        return new Response(new Uint8Array([1, 2, 3]), {
          headers: { "Content-Type": "audio/wav" },
        });
      }
      if (url === "https://media.test/api/upload")
        return new Response(null, { status: 204 });
      return new Response(null, { status: 404 });
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("window", {
      localAI: injected({ getArtifactUrl, saveArtifactAs }),
    });
    const client = createLocalAIClient();

    expect(await client.artifacts.getUrl(artifactId)).toBe(capabilityUrl);
    const blob = await client.artifacts.readBlob(artifactId);
    expect(blob.size).toBe(3);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0]?.[0])).toBe(capabilityUrl);
    await fetch("https://media.test/api/upload", {
      method: "POST",
      body: blob,
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);

    await expect(
      client.artifacts.saveAs(artifactId, { suggestedFilename: "song.wav" }),
    ).resolves.toBeUndefined();
    expect(saveArtifactAs).toHaveBeenCalledWith(artifactId, {
      suggestedFilename: "song.wav",
    });
    expect(JSON.stringify(saveArtifactAs.mock.calls)).not.toContain("/Users/");
  });

  it("routes owned async jobs, ordered events, helpers, and cancellation", async () => {
    const listeners = new Set<(event: LocalAIEvent) => void>();
    const cancelJob = vi.fn(async () => ({
      ok: true as const,
      value: undefined,
    }));
    const invokeJob = vi.fn(async () => {
      queueMicrotask(() => {
        for (const listener of listeners) {
          listener({
            event: "job.output.delta",
            timestamp: 8,
            jobId: "job_example",
            sessionId: "session_one",
            sequence: 1,
            data: { channel: "text", delta: "hello " },
          });
          listener({
            event: "job.output.delta",
            timestamp: 9,
            jobId: "job_example",
            sessionId: "session_one",
            sequence: 2,
            data: { channel: "text", delta: "world" },
          });
          listener({
            event: "job.completed",
            timestamp: 10,
            jobId: "job_example",
            sessionId: "session_one",
            sequence: 3,
            data: { result: { kind: "structured", value: { text: "done" } } },
          });
        }
      });
      return {
        ok: true as const,
        value: {
          jobId: "job_example",
          sessionId: "session_one",
          state: "running" as const,
        },
      };
    });
    vi.stubGlobal("window", {
      localAI: injected({
        invokeJob,
        cancelJob,
        onJobEvent: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      }),
    });
    const client = createLocalAIClient();
    const request = { sessionId: "session_one", input: { prompt: "hello" } };
    await expect(invokeAndWait(client, request)).resolves.toEqual({
      kind: "structured",
      value: { text: "done" },
    });
    expect(invokeJob).toHaveBeenCalledWith(request);

    const events: LocalAIEvent[] = [];
    for await (const event of invokeStream(client, request)) events.push(event);
    expect(events.map((event) => event.event)).toEqual([
      "job.output.delta",
      "job.output.delta",
      "job.completed",
    ]);
    expect(
      events
        .filter((event) => event.event === "job.output.delta")
        .map((event) => event.data.delta)
        .join(""),
    ).toBe("hello world");
    await client.cancel("job_example");
    expect(cancelJob).toHaveBeenCalledWith("job_example");
  });
});

describe('task contract feature negotiation', () => {
  afterEach(() => vi.unstubAllGlobals());
  it('validates the negotiated contract when the original task tag has different semantics', async () => {
    const ref = {provider: 'huggingface' as const, repository: 'org/pose', revision: 'a'.repeat(40), sourceUrl: 'https://huggingface.co/org/pose'};
    const session: ModelSession = {sessionId: 'session_pose', role: 'pose', origin: 'https://site.test',
      capabilities: {task: 'keypoint-detection'}, resolution: {preferredModel: ref, resolvedModel: ref,
        compatibility: 'compatible', executability: 'executable', security: 'allowed', trust: 'user-approved-unverified',
        developerCertified: false, substituted: false, overriddenByUser: false, resolutionReason: 'preferred-model', compatibilityReasons: []},
      taskProfile: {kind: 'task-profile', schemaVersion: 1, profileId: 'pose-fixture', contract: {id: 'pose-estimation-v1', revision: 1},
        task: 'pose-estimation', inputModalities: ['image'], outputModalities: ['structured'], inputSchemaRef: 'taskInput.pose-estimation-v1',
        resultSchemaRef: 'taskResult.pose-estimation-v1', parameters: {}, features: {}, limits: {}, streamingChannels: [], extensions: {}, provenance: 'adapter-derived'}};
    const invokeJob = vi.fn(injected().invokeJob);
    const releaseModelSession = vi.fn(injected().releaseModelSession);
    vi.stubGlobal('window', {localAI: injected({requestModel: async () => ({ok: true, value: session}), invokeJob, releaseModelSession})});
    const client = createLocalAIClient();
    await client.models.request({role: 'pose', preferredModel: {provider: 'huggingface', repository: 'org/pose'}});
    const imageArtifactId = 'artifact_aaaaaaaaaaaaaaaaaaaaaaaa';
    await expect(client.invoke({sessionId: session.sessionId, input: {imageArtifactId}})).rejects.toThrow();
    expect(invokeJob).not.toHaveBeenCalled();
    await client.invoke({sessionId: session.sessionId, input: {imageArtifactId, boxes: [[0, 0, 1, 1]]}});
    expect(invokeJob).toHaveBeenCalledOnce();
    for (const contract of [{id: 'pose-estimation-v1', revision: 2}, {id: 'pose-estimation-v2', revision: 2}, {id: 'unknown-task-v1', revision: 1}]) {
      session.taskProfile!.contract = contract;
      await expect(client.models.request({role: 'pose', preferredModel: {provider: 'huggingface', repository: 'org/pose'}})).rejects.toThrow('does not support the negotiated task contract revision');
      expect(releaseModelSession).toHaveBeenLastCalledWith(session.sessionId);
    }
    expect(releaseModelSession).toHaveBeenCalledTimes(3);
  });
  it('never silently downgrades a task requirement on an older browser', async () => {
    const requestModel=vi.fn();
    vi.stubGlobal('window',{localAI:injected({requestModel})});
    await expect(createLocalAIClient().models.request({role:'captions',requirements:{task:'automatic-speech-recognition',outputModalities:['text'],contract:{id:'transcription-v1',revision:1},requiredFeatures:['wordTimestamps']}})).rejects.toThrow('does not support task contract negotiation');
    expect(requestModel).not.toHaveBeenCalled();
  });
  it('forwards the unchanged task requirements when the browser advertises support', async () => {
    const base=injected();
    const requestModel=vi.fn(base.requestModel);
    vi.stubGlobal('window',{localAI:injected({requestModel,getCapabilities:async()=>{const c=await base.getCapabilities();return {...c,features:{...c.features,taskProfiles:true}};}})});
    const request={role:'captions',requirements:{task:'automatic-speech-recognition',outputModalities:['text' as const],contract:{id:'transcription-v1',revision:1},requiredFeatures:['wordTimestamps']}};
    await expect(createLocalAIClient().models.request(request)).rejects.toThrow('user denied');
    expect(requestModel.mock.calls[0]?.[1]).toEqual(request);
  });
});
