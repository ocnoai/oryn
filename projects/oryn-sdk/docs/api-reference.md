# SDK API reference

[SDK overview](../README.md) · [Recipes](recipes.md)

This reference describes the exported website client in `@ocno/oryn-sdk` 0.1.0. Runtime and model availability are determined by Oryn Browser, independently of the installed npm package.

## Client and capabilities

### `createLocalAIClient`

```text
createLocalAIClient(options?) → LocalAIClient
```

Creates a lightweight wrapper. Importing the module and creating a client are safe during SSR; they do not start inference, request permission, or download a model.

| Option | Type | Default | Behavior |
| --- | --- | --- | --- |
| `showOrynBrowserAlert` | `boolean` | `false` | Shows a website-owned explanation once per document outside Oryn; does not grant permission |

The client exposes `sdkVersion`, `getCapabilities`, `models`, `artifacts`, `invoke`, `cancel`, and `subscribe`. `SDK_VERSION` is also exported.

### `client.getCapabilities`

```text
client.getCapabilities() → Promise<RuntimeCapabilities>
```

| Field | Meaning |
| --- | --- |
| `runtimeAvailable` | Whether Local AI is available now |
| `runtimeState` | `ready`, `starting`, `unavailable`, or `error` |
| `sdkVersion` | SDK version reported by the capability surface |
| `browserApiVersion?` | Version of the injected browser contract |
| `supportedModelProviders` | Supported model-reference providers; the current type allows `huggingface` |
| `features` | Feature flags listed below |
| `scheduler?` | Generation-lane scope and maximum concurrent running jobs |

The flags are `modelRequests`, `modelSubstitution`, `streaming`, `artifacts`, `artifactUrls`, `cancellation`, `queueing`, and `multipleSubmittedJobs`. Newer peers may also report `modelSelection` and `taskProfiles`; treat absent optional flags as unavailable.

Outside Oryn, this method returns unavailable capabilities rather than throwing solely because the bridge is absent. It is not a list of installed models or an exact hardware report.

## Models and sessions

### `client.models.request`

```text
client.models.request(request, options?) → Promise<ModelSession>
```

A request has a `role` identifying the site's use case, such as `chat` or `transcription`, and one of three forms:

| Form | Required data | Purpose |
| --- | --- | --- |
| Capability request | `requirements.task` and `requirements.outputModalities`; no `preferredModel` | Ask Oryn to resolve an executable match |
| Explicit request | `preferredModel`; optional requirements and `certifiedModels` | Express a model preference and tested alternatives |
| Source-checkpoint request | `selection: { version: 1, intent: "source-checkpoint", sourceCheckpoint }` | Let Oryn select an eligible representation of a checkpoint |

`ModelRef` has `provider: "huggingface"`, `repository: "owner/name"`, optional `revision`, and optional `files`. File pins require an immutable revision; they are model-host relative file names, not local paths. `ResolvedModelRef` includes the resolved revision and `sourceUrl`.

#### Requirements

| Field | Purpose |
| --- | --- |
| `task` | Model task, for example `text-generation` or `automatic-speech-recognition` |
| `inputModalities`, `outputModalities` | Required input/output kinds |
| `capabilities` | Behavioral requirements such as `chat`, `streaming`, or `speechRecognition` |
| `contract` | Versioned task reference `{ id, revision }` |
| `requiredFeatures` | Effective task-profile features the application needs |
| `parameterValues` | Values that must be supported by the negotiated task |
| `architectures` | Architecture constraints |
| `parameters` | Optional `minBillions` / `maxBillions` bounds |
| `minimumContextLength` | Minimum context requirement |
| `memory.maximumRecommendedBytes` | Memory-related model selection constraint |

Task contract negotiation requires `features.taskProfiles`. Selection intents and file pins require `features.modelSelection`. The SDK rejects unsupported explicit requests rather than silently dropping their requirements.

#### Selection and substitution

`selection.version` is `1`. Its intent is `source-checkpoint`, `preferred-variant`, or `exact-artifact`. `preferred-variant` accompanies a `preferredModel`; `exact-artifact` also requires that model's immutable revision and complete selected `files` set.

| `substitution.automatic` | Meaning |
| --- | --- |
| `none` | No automatic replacement of an explicit preferred model |
| `certified-only` | Automatic alternatives are limited to the site's `certifiedModels` |
| `same-checkpoint` | Allow eligible representations of the same source checkpoint |
| `compatible` | Allow a compatible replacement validated by Oryn |

These policies apply to automatic substitution; trusted user choice is separate. The request type constrains which policies apply: capability-only requests accept `compatible`, while source-checkpoint requests accept `none`, `same-checkpoint`, or `compatible`. Always inspect `session.resolution`.

With substitution omitted, an explicit model preference defaults to `none`, a source-checkpoint request defaults to `same-checkpoint`, and a capability-only request uses `compatible` selection.

#### Options and progress

| Option | Type | Behavior |
| --- | --- | --- |
| `signal` | `AbortSignal` | Requests cancellation of model resolution/preparation |
| `onProgress` | `(progress: ModelRequestProgress) => void` | Receives request progress while it is pending |

Stages are `resolving`, `awaiting-permission`, `awaiting-model-choice`, `downloading`, `loading`, and `ready`. Downloading includes `downloadedBytes`, optional `totalBytes`, and the model reference. Loading may include a numeric `progress`. Not every request visits every stage; an installed model may resolve quickly.

#### Returned session

| Field | Meaning |
| --- | --- |
| `sessionId` | Opaque handle used in invocations |
| `role`, `origin` | Role and owning origin |
| `resolution` | Preferred/resolved models, compatibility, executability, trust, and substitution explanation |
| `capabilities` | Effective model capabilities |
| `taskProfile?` | Negotiated task contract, effective features, parameters, and limits |

`resolution.trust` is `oryn-verified` or `user-approved-unverified`. `developerCertified` is a separate declaration about the site's testing. Never present an unverified model as Oryn-certified.

`resolution.selection`, when present, explains whether selection chose the requested artifact, another representation of the same checkpoint, or a different model. Its source/confidence are advisory explanations, not runtime authorization.

### `client.models.releaseSession`

```text
client.models.releaseSession(sessionId) → Promise<void>
```

Releases the site's model session. Do not reuse the ID afterward. Session release is separate from explicit job cancellation and artifact release; clean up each resource you own. A browser restart is not a supported session persistence mechanism.

## Jobs and events

### `client.invoke`

```text
client.invoke<TInput>(request) → Promise<JobReference>
```

`request` contains `{ sessionId, input }`. `input` is the negotiated task's JSON payload, containing artifact IDs for media. The return value contains `jobId`, `sessionId`, initial `state`, and optional `queue` information. It is an acknowledgement of submission, not the result.

Shared-task inputs known to the SDK are validated before bridge invocation. Browser and Engine still enforce authorization, contract compatibility, artifact bindings, and runtime limits.

### `client.subscribe`

```text
client.subscribe(listener) → () => void
```

Registers a job-event listener and returns an unsubscribe function. Outside Oryn it returns a no-op unsubscribe. Subscribe **before** invoking and route by `jobId`; events can arrive before the invocation promise resolves. Helpers handle this early-event buffering for you.

All events have `event`, `timestamp`, `jobId`, `sessionId`, and `sequence`. Sequence values increase within a job. A job has one terminal outcome.

| Event | `data` | Terminal? |
| --- | --- | --- |
| `job.state` | `state`, optional `queue` | No |
| `job.queue.updated` | `lane`, `position`, `jobsAhead`, `reason` | No |
| `job.progress` | Optional `progress` and `stage` | No |
| `job.output.delta` | `channel`, `delta` | No |
| `job.preview` | `artifactId` | No |
| `job.completed` | `result` | Yes |
| `job.failed` | Serialized `error` | Yes |
| `job.cancelled` | Optional `reason` | Yes |

Only consume text deltas when `channel === "text"`; the delta type is `unknown` because channels can carry different data. Render generated text using text nodes or your framework's escaped text rendering.

The current generation lane is global across origins, with one running generation job. Websites only receive their own job events. Use `scheduler` and queue events for UI instead of assuming immediate execution.

### `invokeAndWait`

```text
invokeAndWait<TInput, TResult>(client, request, options?) → Promise<TResult>
```

Subscribes, submits, filters to the new job, and waits for a terminal event. `TResult` extends `InferenceResult` and defaults to it.

| Option | Behavior |
| --- | --- |
| `signal` | Requests job cancellation after the helper has the job reference |
| `onEvent` | Receives events for this job, including the terminal event |

- `job.completed`: resolves with its result.
- `job.failed`: rejects with `LocalAIError`.
- `job.cancelled`: rejects with `LocalAIError` code `REQUEST_ABORTED`.

Keep event callbacks lightweight and non-throwing. The generic result type is a TypeScript expectation, not a runtime validator for arbitrary returned JSON.

### `invokeStream`

```text
invokeStream<TInput>(client, request, options?) → AsyncIterable<LocalAIEvent>
```

Accepts optional `{ signal }`, subscribes before submission, and yields events for this job through its terminal event. **Terminal `job.failed` and `job.cancelled` are yielded, not automatically thrown.** Submission errors can still reject iteration. A signal already aborted when iteration begins causes `REQUEST_ABORTED`.

The iterator starts when consumed. Closing it unsubscribes its listener; it does not cancel the running job. Use the signal while it is active or cancel a known job explicitly.

### `client.cancel`

```text
client.cancel(jobId) → Promise<void>
```

Requests cancellation of an owned queued or running job. A successful call is not proof the runtime has already stopped; use the terminal event to settle the UI. Some runtimes interrupt only at safe execution points.

## Results

`InferenceResult` is a discriminated union:

| Kind | Payload | Typical use |
| --- | --- | --- |
| `structured` | `{ kind: "structured", value: unknown }` | Text, predictions, timestamps, embeddings |
| `artifact` | `{ kind: "artifact", artifact: ArtifactMetadata }` | Generated image/audio/video |
| `multi` | `{ kind: "multi", outputs: InferenceResult[] }` | Multiple images, layers, or media plus metadata |

Check `kind` before accessing fields. Check or validate task-specific structured values at your application boundary. `SharedTaskOutputs` mixes structured **value** types with complete artifact/multi envelopes, depending on the task; do not wrap every mapped type in another `structured` envelope.

## Artifacts

Artifacts are scoped capabilities for binary data. IDs/URLs are opaque and temporary. They do not grant global filesystem access.

| Method | Returns | Notes |
| --- | --- | --- |
| `artifacts.create(input, options?)` | `Promise<ArtifactReference>` | Accepts a `Blob` or `File`; supports `name` and `lifetime: "ephemeral" \| "session"` |
| `artifacts.get(id)` | `Promise<ArtifactMetadata>` | Reads authorized metadata |
| `artifacts.getUrl(id)` | `Promise<string>` | Gets a URL usable by media elements or fetching |
| `artifacts.readBlob(id)` | `Promise<Blob>` | Materializes the complete artifact in renderer memory |
| `artifacts.saveAs(id, options?)` | `Promise<void>` | Opens trusted save UI; optional `suggestedFilename` |
| `artifacts.release(id)` | `Promise<void>` | Releases the website's capability |

`ArtifactReference` contains `id`, `mimeType`, `sizeBytes`, and optional `name`. `ArtifactMetadata` adds `createdAt`, optional `expiresAt`, and optional `url`.

The current SDK copies inputs in 256 KiB chunks. Large media belongs in artifacts rather than base64 strings or large number arrays in invocation JSON. A file extension or a file picker's `accept` attribute is not validation of the actual encoding.

Keep an artifact alive while a media element is using its URL. `readBlob()` can be expensive for large videos; prefer direct media URLs for playback. `saveAs()` does not return the chosen local filesystem path. Uploading a blob with `fetch()` is a separate website decision.

## Task profiles and typed validation

`TaskProfile` includes:

- `contract: { id, revision }`, `task`, `profileId`, and `schemaVersion`;
- input/output modalities and `inputSchemaRef` / `resultSchemaRef` identifiers;
- feature states: `supported`, `unsupported`, `unknown`, or `requires-input`;
- parameters with `inputPointer`, `valueSchema`, optional `default`, and `handling`;
- `limits`, `streamingChannels`, and runtime provenance.

Parameter handling is `honored`, `fixed`, `unsupported`, or `requires-input`. Use this information to build controls that reflect the selected model. Schema references identify negotiated schemas; they are not automatically fetchable website URLs.

### `validateSharedTaskInput`

```text
validateSharedTaskInput(contract, input) → SharedTaskInputs[contract]
```

Checks a supported shared-task input against its generated schema, cross-field constraints, and JSON-size bound. Returns the typed input on success. Failure throws a validation error; do not assume every local failure is a `LocalAIError`. It does not request a model, grant permission, validate artifact ownership, or certify execution.

Exports include `SharedTaskInputs`, `SharedTaskOutputs`, `SharedTaskContract`, `SHARED_TASK_CONTRACTS`, and `SHARED_MODEL_TASK_CONTRACTS`. Additional public types include `TaskContractRef`, `TaskFeatureState`, `TaskParameter`, `TaskProfile`, `TaskRequirements`, `ModelSelection`, and `SelectionExplanation`.

## Errors

`LocalAIError` has `name: "LocalAIError"`, a stable `code`, `message`, and optional sanitized `details`. Construct it from a serialized error, for example when converting a `job.failed` event into a thrown exception.

| Category | Representative codes |
| --- | --- |
| Availability | `RUNTIME_NOT_AVAILABLE`, `RUNTIME_STARTING`, `RUNTIME_DEPENDENCY_UNAVAILABLE`, `RUNTIME_PROTOCOL_ERROR` |
| Permission/origin | `USER_DENIED`, `PERMISSION_DENIED`, `UNAUTHORIZED_ORIGIN`, `UNAUTHORIZED_FRAME` |
| Model selection | `MODEL_NOT_FOUND`, `MODEL_REFERENCE_INVALID`, `NO_EXECUTABLE_MODEL`, `MODEL_NOT_EXECUTABLE` |
| Model safety | `MODEL_SECURITY_BLOCKED`, `CUSTOM_CODE_REQUIRED`, `UNSAFE_MODEL_FORMAT` |
| Resources | `RESOURCE_REQUIREMENTS_UNSATISFIABLE`, `RESOURCE_LIMIT_EXCEEDED`, `DISK_SPACE_INSUFFICIENT`, `MODEL_OUT_OF_MEMORY` |
| Sessions | `SESSION_NOT_FOUND`, `SESSION_ACCESS_DENIED`, `SESSION_RELEASED` |
| Jobs | `JOB_QUEUE_FULL`, `JOB_INPUT_INVALID`, `JOB_EXECUTION_FAILED`, `JOB_CANCELLED`, `REQUEST_ABORTED` |
| Artifacts | `ARTIFACT_NOT_FOUND`, `ARTIFACT_ACCESS_DENIED`, `ARTIFACT_EXPIRED`, `ARTIFACT_INVALID` |
| Task-specific | `AUDIO_FORMAT_UNSUPPORTED`, `ASR_INFERENCE_FAILED`, `MODEL_OUTPUT_FILTERED`, `SONG_GENERATION_FAILED` |

Use the exported `LocalAIErrorCode` union for the complete compile-time list. Keep a fallback for unfamiliar future codes and ordinary JavaScript/network errors. Avoid logging user inputs or complete model outputs simply to diagnose a failure.

## Cancellation and cleanup

Model preparation, job execution, model sessions, and artifacts have separate lifetimes:

1. Abort a pending model request with its `signal`.
2. Cancel a job with its helper's active signal or `cancel(jobId)`.
3. Observe the terminal outcome before declaring the job stopped.
4. Release artifacts after playback, saving, or other consumers finish.
5. Release the session when the feature no longer needs the model.
6. Unsubscribe custom listeners when their owner is disposed.

Cleanup in `finally` prevents a successful or failed invocation from leaving an unused session behind. If cleanup itself fails, decide explicitly how to report that without replacing the original user-facing error. A stream `break`, a tab switch, and a session release should not be used as substitutes for explicit cancellation.
