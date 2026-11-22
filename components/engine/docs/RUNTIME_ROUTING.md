# Runtime routing

`RuntimeRoutingService` is an internal Engine object shared by inspection and execution. It coordinates `RuntimePlanner`, local performance evidence, and bounded failure tracking. It shares the Engine process and is not a network endpoint.

Browser resolves the logical model. Engine ranks allowed execution candidates for that model and task. Candidates from different logical models or tasks are rejected; exact route selection fails visibly when unavailable and has no silent fallback.

The planner enforces signature/authority, ABI, task, selected-file completeness, access/license, security, runtime/backend availability, resource estimates, and failed-conformance gates before preferences. Generated MLX, Diffusers, and GGUF routes and existing signed compatibility routes use this service. Generic Transformers inspection uses the shared descriptor evaluator and its existing loader; it does not yet enumerate alternative foundation variants.

Balanced is the default. Performance, Memory Saver and Quality are trusted settings. The service also accepts an internal soft foundation preference that is not exposed as a Browser setting. Hard gates always win. Throughput is compared only when every candidate has matching, unexpired measurements; otherwise candidates receive equal, explicitly labeled defaults. Unmeasured candidates are not assigned a comparative benchmark ranking.

## Evidence and identity

`RoutingEvidenceStore` accepts bounded local measurement records. Keys bind exact model identity, selected files/build, adapter/backend, task, settings, catalog, hardware class, and OS. Measurements expire within seven days; corrupt, oversized, duplicate, or malformed records are ignored. Measurements can raise memory estimates, but cannot lower an admission gate or confer certification. Automatic calibration is not scheduled by inspection.

Plan IDs include policy version, context, candidate evidence/integrity, scores, and rejection reasons. Model installations retain their eligibility and unverified status through restart. A changed capability catalog or packaged runtime build invalidates stored admission and requires reinspection/reinstallation. Installed files are hashed again before load. Reinstall an older installation without stored hashes to obtain an integrity record.

Three initialization failures quarantine the specific model/files/runtime/adapter/backend combination for one hour in the current Engine process. A build or file-identity change has a different key. Cancellation and input-validation errors do not cause fallback or quarantine. Pre-output initialization failures may try recorded equivalent candidates. Visible output prevents automatic restart on another route. Quarantine is deliberately not a permanent family ban and does not survive Engine restart.

## Execution contracts

MLX chat uses the existing `chat-v1` request and structured text result. It runs with local files, repository code disabled, a bounded context/output, the existing worker supervisor, cancellation, and a separate no-network worker sandbox. MLX-LM's pinned import sets the descriptor limit to 2048/4096; the supervisor permits that bounded ceiling while retaining its other limits.

Image chat accepts an optional top-level `imageArtifactId` to `chat-v1`, attached to the last user message. Only an image-capable adapter accepts it. Browser resolves the artifact under the existing permissions/lease rules. Engine accepts PNG/JPEG/WebP, at most 20 MiB, one frame, at most 4096×4096 pixels; preprocessing reduces the image to at most 1024×1024. URLs and local paths are not accepted as website input. This route uses the packaged Transformers processor/model on MPS.

Generic Diffusers admission validates every declared component and excludes unsafe weights, executable files, alternate exports, and redundant FP16/non-EMA copies when canonical weights exist. The generated table enables Stable Diffusion/SDXL text-to-image and image-to-image. An admitted text-to-image profile with `imageInput` can receive `imageArtifactId` and `strength` through the same bounded image-editing adapter. Video remains a separate acceptance gate.

Encoder embeddings use attention-mask-aware mean pooling and optional L2 normalization; causal embeddings use the last non-padding token. Sequence-classification reranking scores query/document pairs, with sigmoid for one output or positive-class probability for two. These bounded semantics do not promise support for arbitrary Sentence Transformers module graphs or custom ranking heads.


## Process launch and correctness

On macOS, `sandbox_launcher.py` starts the Engine and signed workers as siblings because Seatbelt
cannot be reapplied inside a sandboxed Engine. `runtime_packs/broker.py` exposes only a private,
per-launch authenticated socket. It checks a closed bounded request, active signed pack/version,
command, timeout and symlink-resolved model/input/output scopes. It builds execution arguments
itself. Websites cannot access this channel; workers receive neither its token nor host credentials.
The launcher passes Engine protocol/credential pipes unchanged and supervises shutdown. Worker
resource limits run after exec and before adapter import, avoiding threaded Python pre-exec hooks.

Both sandbox profiles grant only the two Apple Silicon GPU clients needed by Metal. Workers retain
network denial and file restrictions. Temporary hardlink views expose only selected files without
opening the entire Hugging Face blob cache. Cancellation removes those views and terminates workers.

Question answering selects the best bounded span entirely within context tokens and returns
character offsets into that context. Token classification excludes special tokens and provides
labels, scores and text offsets. Multi-label classification uses sigmoid probabilities; ordinary
single-label classification uses softmax. Seq2seq generation checks cancellation between tokens.
DETR's image backbone dependency, `timm==1.0.29`, is pinned once in the shared PyTorch closure.

Diffusers reports `MODEL_OUTPUT_FILTERED` when its own image checker withholds an image. Oryn does
not publish that black placeholder or try an alternate runtime to bypass the filter. The FP16 MPS path remains in use.


## Backend precision and task support

ACE-Step and MFLUX use separate adapter dependency sets. Qwen 3.5 MLX-LM wrappers
expose text-only chat. Non-chat SafeTensors routes are not shadowed by an optional
GGUF export. Image editing and layers use the Diffusers pack and negotiated contracts.
Generated Diffusers inspection preserves Stable Diffusion/SDXL's declared image-editing
capability from the common registry. Their `strength` input is not applied to
instruction-editing families. Native image families use BF16 on MPS/CUDA and FP32
on CPU; routing records the actual precision. Legacy SD/SDXL uses FP16 on MPS.
