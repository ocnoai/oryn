# SDK recipes

[SDK overview](../README.md) · [API reference](api-reference.md)

These examples use the website API. They require a compatible Oryn Browser, an executable model, and the user's permission. Functions are intended to be called from your application's user actions; connect their progress and cancellation callbacks to your own UI.

## Choose a model

Prefer a task-only request when the exact model does not matter. If your application has a tested preference, declare it and an explicit substitution policy:

```ts
import type { LocalAIClient } from "@ocno/oryn-sdk";

export async function requestChat(localAI: LocalAIClient, signal: AbortSignal) {
  const session = await localAI.models.request(
    {
      role: "chat",
      preferredModel: {
        provider: "huggingface",
        repository: "Qwen/Qwen3.5-0.8B",
      },
      requirements: {
        task: "text-generation",
        inputModalities: ["text"],
        outputModalities: ["text"],
        capabilities: { chat: true, streaming: true },
      },
      substitution: { automatic: "compatible" },
    },
    { signal, onProgress: (progress) => console.info(progress.stage) },
  );

  console.info({
    model: session.resolution.resolvedModel.repository,
    revision: session.resolution.resolvedModel.revision,
    substituted: session.resolution.substituted,
    trust: session.resolution.trust,
  });
  return session; // The caller owns and must eventually release this session.
}
```

The model name illustrates intent; it is not a guarantee that this checkpoint runs on every installation. Oryn may choose a compatible substitute or reject the request. Only put alternatives in `certifiedModels` when your application has actually tested them.

### Select a checkpoint or an exact artifact

When `features.modelSelection` is available, `source-checkpoint` allows selection of an eligible representation of the same checkpoint. `preferred-variant` accompanies a preferred model. `exact-artifact` additionally pins an immutable revision and a complete selected file list.

```ts
import type { LocalAIClient, ModelRef } from "@ocno/oryn-sdk";

export async function requestExactChat(
  localAI: LocalAIClient,
  model: ModelRef,
  signal: AbortSignal,
) {
  if (!(await localAI.getCapabilities()).features.modelSelection) {
    throw new Error("This browser does not support exact model selection.");
  }
  if (!model.files?.length || !/^(?:[a-f0-9]{40}|[a-f0-9]{64})$/.test(model.revision ?? "")) {
    throw new Error("Supply an immutable model revision and its selected files.");
  }
  return localAI.models.request(
    {
      role: "chat",
      preferredModel: model,
      selection: { version: 1, intent: "exact-artifact" },
      requirements: { task: "text-generation", capabilities: { chat: true } },
      substitution: { automatic: "none" },
    },
    { signal },
  );
}
```

The caller supplies a real model identity and owns the returned session. Inspect the returned resolution even for an exact preference: automatic substitution policy and the person's trusted model choice are separate concerns.

## Require an effective task feature

Use a versioned contract when the feature is essential. Fail visibly on older browsers instead of silently downgrading requirements:

```ts
import type { LocalAIClient } from "@ocno/oryn-sdk";

export async function requestStreamingChat(localAI: LocalAIClient) {
  if (!(await localAI.getCapabilities()).features.taskProfiles) {
    throw new Error("This feature requires task-profile support in Oryn.");
  }
  const session = await localAI.models.request({
    role: "chat",
    requirements: {
      task: "text-generation",
      outputModalities: ["text"],
      contract: { id: "chat-v1", revision: 1 },
      requiredFeatures: ["streaming"],
    },
  });
  console.info(session.taskProfile?.streamingChannels);
  console.info(session.taskProfile?.parameters);
  return session; // Release when the conversation UI is disposed.
}
```

A profile can mark a parameter as fixed or unsupported. Render only controls the profile can honor. Feature states of `unknown` or `requires-input` are not unconditional support.

## Transcribe a WAV file

This recipe expects a PCM WAV file. The header check catches obvious container mismatches; Engine validates the actual audio format. It requests transcription, creates an input artifact, reads a structured text result, and cleans up both resources.

```ts
import {
  createLocalAIClient,
  invokeAndWait,
} from "@ocno/oryn-sdk";

export async function transcribeWav(file: File, signal: AbortSignal): Promise<string> {
  const header = new Uint8Array(await file.slice(0, 12).arrayBuffer());
  const ascii = new TextDecoder("ascii");
  if (
    header.length < 12 ||
    ascii.decode(header.subarray(0, 4)) !== "RIFF" ||
    ascii.decode(header.subarray(8, 12)) !== "WAVE"
  ) {
    throw new Error("Choose a PCM WAV recording.");
  }

  const localAI = createLocalAIClient();
  const capabilities = await localAI.getCapabilities();
  if (!capabilities.runtimeAvailable) throw new Error("Local AI is unavailable.");

  const session = await localAI.models.request(
    {
      role: "transcription",
      preferredModel: { provider: "huggingface", repository: "openai/whisper-small" },
      requirements: {
        task: "automatic-speech-recognition",
        inputModalities: ["audio"],
        outputModalities: ["text"],
      },
    },
    { signal },
  );

  let inputId: string | undefined;
  try {
    const input = await localAI.artifacts.create(file, { lifetime: "session" });
    inputId = input.id;
    const result = await invokeAndWait(localAI, {
      sessionId: session.sessionId,
      input: { audioArtifactId: input.id, language: "auto", timestamps: false },
    }, { signal });

    if (
      result.kind !== "structured" ||
      typeof result.value !== "object" || result.value === null ||
      !("text" in result.value) || typeof result.value.text !== "string"
    ) {
      throw new Error("Unexpected transcription result.");
    }
    return result.value.text;
  } finally {
    try {
      if (inputId) await localAI.artifacts.release(inputId);
    } finally {
      await localAI.models.releaseSession(session.sessionId);
    }
  }
}
```

Timestamps are disabled here. If your product requires word timestamps, negotiate `transcription-v1` with `requiredFeatures: ["wordTimestamps"]` and compatible parameters, then inspect the returned profile. A text-only task match does not promise word timing.

## Use a typed task contract

The SDK exports typed input/output mappings for additional tasks. For zero-shot classification, the mapped output is the **value inside** a structured result:

```ts
import {
  invokeAndWait,
  validateSharedTaskInput,
  type LocalAIClient,
  type SharedTaskInputs,
  type SharedTaskOutputs,
} from "@ocno/oryn-sdk";

type ClassificationResult = {
  kind: "structured";
  value: SharedTaskOutputs["zero-shot-classification-v1"];
};

export async function classify(localAI: LocalAIClient, text: string) {
  if (!(await localAI.getCapabilities()).features.taskProfiles) {
    throw new Error("This feature requires task-profile support in Oryn.");
  }
  const input: SharedTaskInputs["zero-shot-classification-v1"] = {
    text,
    labels: ["support", "sales", "feedback"],
    multiLabel: false,
  };
  validateSharedTaskInput("zero-shot-classification-v1", input);

  const session = await localAI.models.request({
    role: "classification",
    requirements: {
      task: "zero-shot-classification",
      outputModalities: ["structured"],
      contract: { id: "zero-shot-classification-v1", revision: 1 },
    },
  });
  try {
    const result = await invokeAndWait<typeof input, ClassificationResult>(localAI, {
      sessionId: session.sessionId,
      input,
    });
    return result.value.predictions;
  } finally {
    await localAI.models.releaseSession(session.sessionId);
  }
}
```

The generic result annotation is a compile-time expectation. Add runtime checks if your application accepts other result contracts. Other mapped outputs already contain an artifact or multi-result envelope; check the type for the task you are using.

## Display, save, and upload an output

A generated image or audio file can be displayed directly from its artifact URL. The following helper accepts an existing `ArtifactResult` from your chosen generation task and returns a cleanup function. Keep the artifact alive until the media element is finished with it.

```ts
import type { ArtifactResult, LocalAIClient } from "@ocno/oryn-sdk";

export async function displayOutput(
  localAI: LocalAIClient,
  result: ArtifactResult,
  element: HTMLImageElement | HTMLAudioElement | HTMLVideoElement,
): Promise<() => Promise<void>> {
  element.src = await localAI.artifacts.getUrl(result.artifact.id);
  return async () => {
    if (element instanceof HTMLMediaElement) element.pause();
    element.removeAttribute("src");
    if (element instanceof HTMLMediaElement) element.load();
    await localAI.artifacts.release(result.artifact.id);
  };
}

export async function saveOutput(localAI: LocalAIClient, result: ArtifactResult) {
  await localAI.artifacts.saveAs(result.artifact.id, {
    suggestedFilename: result.artifact.name ?? "oryn-output",
  });
}
```

Call `saveOutput` from a Save button. Use a MIME-appropriate filename if the result has no name. For `multi` results, walk the `outputs` array and handle each envelope by its `kind`.

If your application also offers upload, make that a separate user action and clearly identify the destination. This function intentionally reads and uploads the complete artifact:

```ts
import type { ArtifactResult, LocalAIClient } from "@ocno/oryn-sdk";

export async function uploadOutput(localAI: LocalAIClient, result: ArtifactResult) {
  const blob = await localAI.artifacts.readBlob(result.artifact.id);
  const body = new FormData();
  body.append("file", blob, result.artifact.name ?? "oryn-output");
  const response = await fetch("/api/uploads", { method: "POST", body });
  if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
  return response;
}
```

`/api/uploads` is an endpoint your website must implement; it is not an Oryn service. The caller still owns the artifact and should release it when all display/save/upload consumers have finished. `readBlob()` holds the whole output in renderer memory, which matters for large media.

## Build a cancellation UI

For the streaming quickstart, create an `AbortController` per submission and pass its signal to `streamReply`. Wire the Cancel button to `controller.abort()`. For multiple jobs, keep separate controllers so one click does not cancel unrelated work.

Keep queue and model-preparation states visible. Cancelling a request and cancelling an already submitted job are separate operations. A cancellation request can take time to stop the runtime; wait for the terminal outcome before reporting that it has finished. If you manage submission yourself, retain the `jobId` and use `localAI.cancel(jobId)`.

Unsubscribing or breaking out of an async iterator only stops your event consumer. It does not stop inference. A page should clean up event listeners, then release sessions and artifacts according to their actual lifetime.

## Integrate with a framework

The package has no framework dependency. Create a client in a module or component, run feature work only on the client, and let the framework own rendering.

- **React / Vue / Svelte:** keep active controllers and sessions in component-owned state; use lifecycle cleanup for subscriptions and resource release.
- **SSR:** import and client creation are safe, but server execution reports no browser runtime. Detect again after hydration instead of caching a server-side availability result.
- **Rich text:** treat model output as untrusted text. Use escaped text rendering or your application's established sanitization before rendering markup.
- **Fallback:** explain when Local AI is unavailable. If you offer a cloud alternative, make its data destination and user choice clear.

The reference applications in the source distribution show full page UI and progress handling. These recipes focus on the SDK calls that those applications share.
