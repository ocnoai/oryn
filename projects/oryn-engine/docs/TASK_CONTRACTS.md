# Common task contracts and backend discovery

The Engine uses one discovery and negotiation loop for all supported backends:
Transformers, Diffusers, MLX-LM, llama.cpp, MFLUX, ACE-Step and Kokoro. Existing
in-process Transformers runners also produce common offers. Backend-specific model
loaders and worker protocols remain behind reviewed adapters.

## Ownership and flow

1. Protocol's `schemas/task-contracts/registry.json` owns the 34 existing task
   contracts: identity/revision, inputs, results, parameter semantics, features,
   modalities and units. Utilities in the registry are marked separately; their
   presence does not create a model runtime.
2. The SDK sends `requirements.contract`, `requiredFeatures` and `parameterValues`
   with the existing model request. Browser selects the user's exact model and
   passes only task requirements into Engine inspection, installation and attachment.
3. `ModelStore.inspect` builds provider entries from the active runtime packs and
   trusted compatibility variants. `ExecutionProviderRegistry` visits all entries.
   Existing inspectors and the MFLUX native provider are adapters to this same loop.
   A provider failure adds a bounded diagnostic and does not hide another offer.
4. `task_profiles.profile_for_execution` translates reviewed loader capabilities and
   model metadata into the common effective `TaskProfile`. Every offer is checked
   against the canonical contract and request. One final routing decision evaluates
   the aggregate offers for the requested contract, retaining trust, availability,
   resource and runtime policy checks. Legacy inspectors may still evaluate local
   candidates while constructing their offers.
5. Installation persists private bound offers with exact revision/file identities,
   runtime identity, runtime build digest, registry digest and effective profile.
   Inspection contexts are keyed by the complete public profile so concurrent requests
   cannot replace an installation's private offers.
6. Session attachment checks requirements and pins the model, profile and requested
   controls. Negotiated jobs validate input before queueing, apply defaults once,
   and validate results before completion. Native adapters retain artifact ownership,
   duration, geometry and other semantic checks.
7. Runtime fallback stays within the same model and contract and requires installed
   files. Each active route must accept the already resolved input and requirements;
   fallback cannot silently change a parameter or required feature. Existing rules
   prevent retry after visible output.

## Public contract

```ts
const session = await localAI.models.request({
  role: "captions",
  preferredModel: { provider: "huggingface", repository: "openai/whisper-small" },
  requirements: {
    task: "automatic-speech-recognition",
    contract: { id: "transcription-v1", revision: 1 },
    requiredFeatures: ["wordTimestamps"],
    parameterValues: { language: "sk", timestamps: true },
    inputModalities: ["audio"],
    outputModalities: ["text"],
  },
  substitution: { automatic: "compatible" },
});
// session.taskProfile describes this candidate's effective controls and limits.
```

Feature values are `supported`, `unsupported`, `unknown` or `requires-input`.
Required features must be explicitly supported. Parameters declare an input pointer,
bounded schema, optional default, handling (`honored`, `fixed`, `unsupported` or
`requires-input`) and editability. A backend can narrow the canonical parameter
schema; it cannot widen it. Parameter values requested during negotiation remain
requirements for each invocation; defaults are not permission to change those values.

Browser permits an explicit compatible alternative even when automatic substitution
is disabled. A different model that lacks required timings, modalities, parameter
values or the contract revision cannot satisfy the request. Active sessions retain
their model binding when the user's preference changes.

Transcription timestamps remain seconds relative to the submitted audio artifact.
Whisper word timings require inspected alignment-head metadata; unknown support is
not treated as supported. Qwen ASR's existing adapter fixes `timestamps` to false;
forced alignment remains a separate contract. Caption apps can choose a transcriber
without word timings and then request an alignment session independently.

## Compatibility and privacy

`features.taskProfiles` advertises this additive Protocol-v1 extension. Requests
without an explicit contract keep legacy input defaults. An explicit contract sent
to an older Browser/Engine fails rather than silently dropping its requirements.
Existing process-level shared-task negotiation remains in force for those tasks.

Websites receive only TaskProfile, never private file paths, worker entry points,
runtime plans or device/quantizer controls. `extensions` is reserved and currently
closed. Backend options are translated by reviewed adapters; arbitrary Python imports,
remote code and repository-supplied executable configuration are not accepted.

## Extending support

A checkpoint or quantization already understood by an admitted backend can use the
existing task contract and metadata adapter. A genuinely new architecture still needs
backend implementation and admission evidence. Add a new semantic task to Protocol
first, regenerate TypeScript/Python/SDK artifacts, then implement the backend adapter
and shared positive/negative fixtures. Do not create a contract per repository name.

The common loop is implemented for all seven backends. MFLUX speaks the native
provider inspection ABI; the other backends use Engine adapters. A common discovery
interface does not establish real inference qualification for every backend, model
and quantization combination.

Undeclared native FLUX exports can be resolved from their complete SafeTensors
headers. The signed MFLUX catalog derives component names, dimensions and quantizable
layers from the packaged backend. Discovery checks every selected shard, packed
quantization companion and contiguous transformer block, including pruned depths.
The worker repeats that graph check against installed files. Older output-modulation
biases remain learned parameters during strict loading. This path reuses
`text-to-image-v1` without repository aliases, weight downloads or repository code.
Metadata compatibility does not qualify original-model inference or image quality.

FLUX Swift 1.0.0 saves use the same complete-graph path. The signed catalog derives
tensor-name aliases from the pinned backend mappings. Loading preserves MLX affine
quantization arrays and already-transposed convolution layouts without conversion;
the single T5 relative-position table is shared across the native layer bindings.
Every shard must declare the reviewed serializer, component, bit width, group size
and complete chunk sequence. Unknown or colliding aliases fail. A family name hint
does not bypass complete graph validation for an alternate unindexed layout.

Indexed native Krea saves bind complete transformer, text-encoder and VAE parameter
names against the signed backend graph. Discovery reads three bounded indexes and
three sampled headers; worker preparation checks every installed shard and tensor
shape. Native indexed transformer subdirectories remain distinct from Diffusers
key mappings. Loading restores the deterministic rotary-frequency buffer when it
was omitted and reshapes older VAE channel scales without changing their values.
Stored text-encoder quantization is reconstructed by the pinned backend. Every
learned tensor remains mandatory. This path uses the ordinary native runtime and
does not qualify a fork's block-streaming memory or speed claims.

## Native Wav2Vec2 classification heads

The shared `audio-classification-v1` adapter binds complete native Wav2Vec2
backbones to the reviewed age/gender and dual-hypersphere heads. Every backbone
and head tensor is mandatory, including the age regression head when returning
gender probabilities. The age/gender graph preserves unmasked temporal averaging;
the dual-hypersphere graph preserves masked averaging and both L2 normalizations
from its published implementation. Softmax is applied once to categorical logits.
Continuous emotion or age regression values are not classified as probabilities.

Owned preprocessing accepts bounded mono PCM at 16 kHz after shared resampling.
The profile declares `audioDurationSeconds`: 30 for age/gender and 12 for the
dual-hypersphere graph. Longer clips fail instead of being silently truncated or
combined. Original processor settings, label identities, complete tensor shapes
and finite floating weights are checked without importing repository code.
Generated complete-graph fixtures match the pinned publisher computations on
both CPU and Metal; each backend is compared with its own native reference.
This evidence establishes the binding and metadata coverage, not original-model
classification quality.

## ColQwen3.5 full-width visual retrieval

The shared ColQwen3.5 binding retains the complete hybrid language/vision graph
and learned retrieval projection. Single-head exports preserve their declared
320 or 4096 dimensions. Prefix-MRL exports select 128 dimensions only when the
checkpoint explicitly declares that operation; their complete learned projection
remains required. EVIE-8B uses all 4096 dimensions without prefix truncation.

The profile bounds pages to 144 tokens, queries to 48 tokens each, and batches to
two queries. The native processor uses at most 128 visual tokens and preserves
the released pixel normalization, query whitespace and query augmentation. Full
attention is bidirectional; recurrent attention keeps its native computation.
Every returned token vector is normalized, and scores are recomputed from the
returned vectors using the declared late-interaction rule.

Visual retrieval uses an engine-owned temporary JSON file between the worker and
runner. The worker returns a small descriptor binding its basename, backend,
byte count and SHA-256. The runner verifies the regular file, rejects symlinks,
duplicate keys, nonfinite values and excessive structure, then validates the
task result before returning it. Cancellation and completion both remove the
temporary directory. The worker and broker retain their 1 MiB message limits.
4096-dimensional results have a 12 MiB aggregate bound within the existing
16 MiB engine envelope; narrower outputs retain their 900 KiB bound.

Complete generated CPU/Metal graphs and repeated offline workers verify the
wide projection. Original EVIE metadata matches all 761 native tensors and the
released tokenizer/processor. No original checkpoint inference is claimed.

## Jina v4 visual document retrieval

Jina v4 visual document retrieval uses the complete native Qwen2.5-VL graph,
the original 128-dimensional projection, and the retrieval LoRA set. All three
published task adapters remain mandatory and are validated, including the inactive
text-matching and code parameters. The owned single-image prompt and `Query: `
prefix produce normalized multi-vectors with sum-MaxSim scores. The shared profile
allows two queries, 64 tokens per query, 256 page tokens and 196 visual tokens.
Long queries fail before inference; whitespace is preserved. The native image
processor receives the original image and applies the bounded pixel budget.
The vLLM single-vector export requires a separate semantic binding and is not
covered by this adapter. Repository Python, PEFT configuration expressions and
chat templates are never executed.

## Bounded native forecasting

The shared `time-series-forecasting-v1` contract validates each model's context,
horizon and quantile limits before invocation. Unsupported covariates are rejected.
EXAONE Demand and Yug use complete pinned numerical graphs and Safetensors
checkpoints; discovery reads configuration without importing Torch or executing
repository code.

Yug returns its direct quantile head from the final observed patch, for at most
the native output block (64 steps in the original configuration), with at most
2,048 context values. Partial input patches are left padded and masked out of
normalization and attention. Its point forecast is the native median head.
This binding does not use stochastic autoregressive rollouts. Crossing quantile
outputs fail the shared result validation rather than being reordered.

For Yug, `frequencyMilliseconds` also supplies the native frequency embedding:
the largest exactly dividing fixed unit among week, day, hour, minute and second,
with its integer multiplier. For example, 900,000 means 15 minutes. Absent or
nonintegral-second intervals use the native unknown-frequency bucket at scale one.
Calendar months, quarters and years cannot be inferred from milliseconds.
The existing contract requires timestamps and their interval together.

Aurora retains the full native vision, text, prototype, encoder/decoder and flow
graphs. Its univariate path constructs the native pseudo-image from the context;
text and external images are not inputs to this forecast profile. It supports
2,048 context values and a 256-step horizon. Each invocation uses 100 native flow
samples with a fixed private seed of zero, returning their mean as the point
forecast and empirical deciles as requested quantiles. This is a bounded engine
sampling policy; it does not expose the upstream training or generic generation API.

## OpenCLIP image and label classification

The shared `zero-shot-image-classification-v1` contract also accepts complete
OpenCLIP ViT/text Safetensors exports described by `open_clip_config.json`. The
closed configuration binds both towers, projections and learned logit scale to
the pinned OpenCLIP runtime in the existing PyTorch environment. Native bicubic
resize, center crop, RGB conversion, normalization, GELU, causal text attention,
EOS pooling and normalized image/text similarities are retained. Labels are used
as supplied; scores are the native learned-scale similarities followed by softmax.

This bounded graph uses OpenCLIP's bundled 49,408-token BPE and native 77-token
truncation/EOS behavior. As in the upstream factory for this configuration,
separate Hugging Face tokenizer files do not control this path. HF/timm model
identifiers, tokenizer overrides, custom graph options and pickle checkpoints
are not admitted. Discovery is data-only; complete numeric graph validation runs
before loading. Original tensor-header validation and small CPU/Metal reference
tests are separate from original-model or packaged-device qualification.

## BiRefNet foreground segmentation

The shared `image-segmentation-v1` contract binds complete native BiRefNet Swin-t
and Swin-l graphs, selected from the learned patch projection shape and validated
against every tensor. Native multiscale inputs, shifted-window attention,
deformable convolutions and all decoder/supervision/gradient heads are retained.
Exact published configuration aliases are normalized to this owned graph; other
repository code, pretrained downloads, configurable backbones and pickle weights
are not admitted. Relative-position indices must match native construction and
batch statistics must have valid values and types.

Images follow the published 1,024-square bilinear resize and ImageNet normalization.
The final native logits pass through sigmoid, 8-bit alpha conversion and the
publisher's resize to original image coordinates. The semantic-mask contract then
uses an alpha threshold of 128 for foreground (class 1), with background class 0.
The result is a 16-bit class-ID PNG with labels and pixel counts, through the
existing owned-artifact path. This threshold is the engine's discrete-mask policy;
this contract does not expose an alpha-matting output.

The same foreground contract also supports the complete Apache-licensed ISNetDIS
graph used by RMBG-1.4. All six encoder/decoder side heads remain mandatory. Its
closed metadata aliases select the declared `MyPipe` behavior: tensor bilinear
resize to 1,024 square, retained floating-point pixels, division by 255 and channel
centering by 0.5. This differs from the older example utility's input-byte cast.
The native sigmoid foreground prediction is resized to original coordinates and
normalized by its observed minimum and maximum before 8-bit alpha conversion.
The shared semantic-mask policy again uses threshold 128. Constant predictions
have an undefined native normalization and fail instead of publishing an invented
mask. Model weights retain their publisher's separate license and access terms.

## ONNX vision tagging

Camie-style image tagging uses the existing ONNX foundation and
`image-classification-v1` contract. The closed root declaration selects one
original ONNX graph and its accompanying label metadata. The bounded sidecar
reader admits up to 100,000 unique labels in a 12 MiB document, checks all three
label/category maps for agreement, and preserves ordinary configuration limits.
Graph validation binds the RGB input and all initial/refined/candidate outputs.
No repository code or alternate checkpoint export is selected.

Images are resized with Lanczos to fit the declared square, centered on the
publisher's RGB (124, 116, 104) padding, and normalized with ImageNet statistics.
The refined logits receive independent sigmoid probabilities. The existing
`topK` parameter returns original label IDs and names; no confidence threshold or
category filtering is imposed. Extremely narrow images that would produce a
zero-sized native resize are rejected. Standard ONNX operators run through the
existing isolated CPU worker; CoreML remains unqualified without whole-graph
execution evidence.

## Declarative ONNX image classifiers

The same ONNX foundation accepts `oryn-onnx-image-classifier` configurations with
exactly `model_type`, `profile` and `graph` fields. The first owned profile,
`imagenet-efficientnet-lite4-v1`, binds one standard-operator float32 graph with
`images:0` shaped `[1,224,224,3]` and `Softmax:0` shaped `[1,1000]`. Compatible
exports can declare this profile without adding repository code or environments.
The original bare EfficientNet-Lite4 graph acquires the profile only when its
complete SHA256 and byte size match the reviewed content identity. Discovery
checks immutable inventory metadata; execution hashes the entire installed graph.
Unknown bare graphs and conflicting explicit declarations remain unsupported.

The upstream ONNX Model Zoo label map is preserved byte-for-byte and hashed with
the owned source closure. Pixel processing uses pinned OpenCV RGB decoding,
short-side resize to 256 with linear interpolation, center crop to 224, and
`(pixel - 127) / 128` in NHWC layout. This follows the graph's upstream example;
the inference4j mirror's raw-pixel description is inconsistent with that example.
Only single-frame PNG, JPEG and WebP inputs up to 20 MiB, 16 megapixels and 64:1
aspect ratio are accepted. Native softmax probabilities must be finite, within
`[0,1]` and sum to one; they receive no second softmax. Stable `topK` results
preserve the original 1,000 label IDs. CPU is supported; CoreML remains unqualified.

## PixAI native vision tagging

The PixAI image classifier reuses the pinned Transformers SAM3 ViT backbone with
windowed/global axial rotary attention, tiled absolute positions and the original
learned attention pool and tag head. Original fused QKV matrices are split into
native projections without changing values; the unused original CLS-position row
is retained and validated. Every original tensor must be present, finite and of
the declared shape. Closed configuration and processor aliases bind this complete
graph without loading repository Python. The two native Transformers source files
are included in the signed source closure.

The processor composites non-RGB images over white, converts pixels to floating
point, resizes with native torchvision bilinear antialiasing, centers on zero
padding and normalizes by mean/std 0.5. Input transparency and full original
dimensions reach this processor. Independent sigmoid tag probabilities use the
existing image-classification `topK` result, retaining original tag IDs. Category
threshold presets are validated metadata; the existing top-K contract does not
apply category filtering.

## TIPS contrastive image/text encoders

TIPS zero-shot image classification uses the pinned native Transformers Tipsv2
vision and text models. Closed historical class aliases and exact configuration
semantics select the complete graph. Original fused QKV storage is split into
native projections without changing values; all learned tensors, including
register and mask tokens, must be present and finite. The published temperature
is required explicitly in metadata. Native model/config source files and the
owned binding are included in the signed source closure.

The original SentencePiece artifact supplies lowercased text IDs, without BOS/EOS,
right-padded or truncated to 64 tokens. Empty token sequences fail. Images are
converted to RGB, resized directly to the configured square with PIL bilinear
resampling, and scaled to [0, 1] without channel normalization. The first image
CLS token and valid-token mean text embedding determine normalized similarity,
divided by the declared temperature. Softmax over the caller's candidate labels
produces the existing zero-shot image classification result. No repository Python
or original-checkpoint inference is required for metadata inspection.

## RTMW whole-body keypoints

The RTMW binding uses the complete Apache-2.0 CSPNeXt backbone, top-down and
bottom-up feature pyramid, both head scales, gated attention and SimCC coordinate
heads. All original weights and batch-normalization statistics are mandatory.
Configuration dimensions, processor behavior and published class aliases are
closed; the native source and attribution are covered by the signed capability
closure. Repository configuration never selects runtime Python.

The input image represents one person or an existing person crop. No person
detector or crop selection is implied. Native PIL bicubic preprocessing resizes
the entire image to the declared width/height and normalizes byte-scale RGB
channels without an extra rescaling step. The original minimum of the two axis
maxima supplies confidence; nonpositive predictions are invalid, and valid scores
are clamped using the declared [0, 1] range. Coordinates are scaled to the full
original image dimensions. The existing keypoint contract applies its confidence
threshold and maximum point count, without introducing a skeleton/joint-label
result contract.

## Native GGUF instruction translation

The pinned llama.cpp environment exposes `translation-v1` for native `qwen35`
and `qwen35moe` graphs. Discovery uses architecture and the requested task, so
repository renames do not change support. Installation rechecks the original
container, tokenizer, quantization and complete selected file set. Text translation
uses the language graph without an image projector, as documented by the reviewed
Index translation exports.

The adapter owns the literal translation instruction and non-thinking ChatML
framing. It honors `targetLanguage` (`en`, `zh`, `ja`) and `maxTokens` (1–512,
default 128), uses greedy decoding, and returns the existing structured text result.
There is no streamed text channel in this contract. A conservative UTF-8 byte bound
reserves space for the full prompt and output within at most 4096 context tokens;
oversized input is rejected and native context shifting is disabled. This route
does not execute repository templates or provide glossary, dubbing or image-input
controls. The isolated native worker requires no Transformers or Protocol package.

## Native GPT-OSS MLX text

The shared MLX environment admits GPT-OSS exports with complete native floating,
affine or mixed MXFP4/Q8 graphs. MXFP4 is restricted to native four-bit matrices
with 32-value groups and exponent-byte scales. All expert biases, attention sinks,
output rows and affine quantization companions remain mandatory. Loading checks
complete shapes, exact integer storage types and finite floating values before
strict native binding. The original checkpoint and its repository identity remain
unchanged.

The adapter owns the reviewed text-only Harmony prompt and validates marker/stop
identities against the local tokenizer. Source templates stay inert. Native
reasoning remains inside the bounded generation budget; only a well-formed final
channel is returned. Text is emitted after channel validation, so incremental
reasoning or control headers cannot enter the public text stream. Tool calls and
unsupported channels are rejected. Public context is capped at 8192 tokens and
output at 512 tokens. Padded output-vocabulary rows are retained and an unknown
generated token ID is an error. CPU/Metal numerical graph tests and Metal worker
execution are distinct from original-model output quality or packaged qualification.

Qwen3-VL video questions use the complete native vision, deep-stack merger and
language graphs. Whole-shard numeric validation preserves every learned tensor;
if a tied vocabulary head is stored twice, both copies must agree. Saved video
processor settings preserve normalization and patch geometry. The engine owns
the native resize and eight-frame sampling budget, including for exports whose
processor expects frames to have been resized by its caller. Image contracts
still require their admitted resize behavior.

Owned Qwen3 timestamp framing uses the decoded source times, including the
selected clip's offset, with one vision block per temporal pair. Repository
Jinja and generation defaults do not control the task. Inputs plus generated
tokens must fit the smaller of 4096 tokens and the model context; output is at
most 512 tokens and 16 KiB. The text is a video-question answer: this binding does
not certify temporal-grounding accuracy or parse generated text into intervals.

ONNX token classification shares `token-classification-v1`: each non-structural
input token returns its native label, softmax score and Unicode source offsets.
The input must fit the smaller of 512 tokens, the model context and tokenizer
limit; it is never silently truncated. Output label dimensions must match the
complete declared label map. Unknown-word tokens with real source spans remain
visible; classifier output is not rewritten into grouped entities.

The signed ONNX Runtime environment also admits its built-in
`com.microsoft::MatMulNBits` version 1 for symmetric block-Q4 matrices. The closed
binding requires constant uint8 packed weights, float32 scales, exact K/N/block
geometry, and ordinary float accumulation. Scale metadata may be flat or shaped
by output row and block. Unsupported domains, operators, optional zero-point or
reordering inputs and unreviewed accuracy modes are rejected. This adds CPU
support; CoreML execution remains separately unqualified.

## Native Gemma 4 MLX image and audio

The MLX environment binds complete Gemma 4 dense/shared-KV and MoE graphs to
`chat-v1` for image questions and `multimodal-generation-v1` for one image or
one audio clip with text output. Image chat accepts one user message. Both
routes use an owned Gemma turn frame, a validated contiguous tokenizer, and a
4096-token combined context budget. Image chat allows 512 output tokens; the
multimodal contract retains its 256-token maximum. Audio is mono 16 kHz and is
bounded to eight seconds (one declared processor chunk); audio output and
combined image/audio input are unavailable.

The exact reviewed processor metadata selects native aspect-preserving image
resize, patch/pooling geometry and fixed USM audio features. Audio placeholders
match the actual two-stage convolution output length, including ragged sample
counts. No repository templates or loader code run. Quantized matrices use the
validated affine 4/5/6/8-bit layouts and per-layer overrides. Every learned
weight and clipping scalar must be present, including retained projections in
shared-KV layers. Only native scalar clipping bounds may be infinite. The audio
relative-position helper is rebound to the loaded quantized projection, avoiding
an uninitialized private alias. Complete synthetic graph, incremental-cache,
CPU/Metal and isolated-worker regressions verify these paths. Original tensor
headers and tokenizer/processor checks establish metadata compatibility; they
do not claim original-model inference or packaged-device qualification.

## Native Nemotron RNN-T transcription

The shared PyTorch environment binds `Nemotron3_5AsrForRNNT` to
`transcription-v1`. Complete Conformer, LSTM, prompt-projector and joint-head
weights are required. The closed processor uses 128 Slaney mel filters,
16 kHz audio, native preemphasis and the declared lookahead of three frames.
PCM16 mono/stereo input is resampled within a 30-second bound. Language is
`auto`; timestamps are unavailable. Librosa 0.11.0 and the native generation
sources are included in the signed dependency identity.

Greedy decoding advances on blanks or after ten symbols per frame. Every
returned duration must agree with its token and all valid audio frames must be
consumed. The pinned native MPS stop implementation rolls back a speculative
token while retaining its duration; the binding removes that single sidecar
entry only when the retained path is complete. Repeated nonblank tokens remain
visible. The original tokenizer's appended blank lies beyond the output
vocabulary; the tokenizer is retained intact while decoding uses the model's
numeric blank and pad IDs. Repository generation defaults are never applied.

Complete synthetic CPU/MPS graphs, independent frame-by-frame decoding and
NumPy STFT preprocessing, strict malformed-weight rejection and offline worker
regressions validate this binding. Original config, tokenizer and all 655
tensor headers establish metadata compatibility without original weights or
original-model inference.

## Native ternary Parakeet transcription

The shared PyTorch environment also binds the closed `parakeet_tdt` graph and
`thrush-ternary-v2` layout to `transcription-v1`. The engine expands base-three
bytes with five digits per byte and per-row groups of 128. It validates every
packed matrix, scale, unused padding digit and dense tensor before loading the
complete native Conformer/LSTM/duration-head graph. The six auxiliary VAD
tensors remain mandatory; the bounded transcription contract does not segment
long recordings. Memory estimates include host expansion and a device copy.

The native Parakeet frontend uses the publisher's base-model 128-bin Slaney
configuration and sample-standardized log-mels. The ternary declaration binds
that reviewed frontend when the original export omits processor sidecars;
conflicting sidecars are rejected. Native tokenizer vocabulary, pad and blank
identities are checked. Audio stays within the shared 30-second bound, automatic
language and no-timestamp profile.

An owned greedy TDT loop preserves predicted duration skips, LSTM state across
blank frames, repeated nonblank tokens, and the ten-symbol limit for a single
frame. Repository generation defaults do not alter decoding. Complete CPU/MPS
graphs, an independent scalar unpacker and uncached decoder, NumPy STFT and
normalization references, and repeated offline-worker tests verify the binding.
All 993 original tensor headers match without downloading original weights.


### CTranslate2 Whisper numeric exports

The shared PyTorch environment includes pinned CTranslate2 4.8.3 for a CPU-only
`transcription-v1` binding. Discovery recognizes the closed four-field Whisper
configuration and validates the original `model.bin` version 6 / `WhisperSpec`
revision 3 numeric directory through bounded range reads. This exception admits
that numeric format only; pickle and generic `.bin` checkpoints remain rejected.
All encoder, decoder, projection, normalization and control arrays must match the
complete graph. At load time all learned arrays must be finite FP16 or FP32.

The original `vocabulary.txt` and tokenizer must preserve exact token IDs. A
missing tokenizer suffix is allowed only for the complete 1,501 timestamp tokens;
transcription suppresses these tokens. Native log-mel extraction accepts PCM16
mono/stereo WAV, resamples to 16 kHz, and enforces the existing 30-second bound.
Native language detection precedes owned greedy decoding. Explicit languages and
timestamps are unavailable. CTranslate2 halves its total generation limit, so the
448-position context permits 224 new steps; hitting that bound is rejected rather
than returning a partial transcript. Completed output is bounded to 223 tokens
and 16 KiB. Repository generation settings are not used.

Regression evidence uses complete small native graphs, independent PyTorch
encoder/decoder comparisons for both stored precisions and 80/128 mel inputs,
independent NumPy frontend checks, invalid containers and controls, metadata-only
routing and repeated offline worker requests. This is family implementation and
metadata evidence; it does not qualify inference on the original large checkpoint
or a packaged device runtime.


### Composed NeMo TDT ONNX transcription

The existing ONNX CPU runtime binds `nemo-conformer-tdt` exports as one complete
variant: `nemo128.onnx`, `encoder-model.onnx`, `decoder_joint-model.onnx`, their
exact declared external data, `config.json` and `vocab.txt`. Each member passes
the bounded numeric graph parser and signed native operator admission. All
frontend, encoder, vocabulary, LSTM state and duration-head dimensions must agree.
Partial graph sets, mixed variants and unqualified CoreML execution are rejected.

PCM16 mono/stereo WAV is resampled to 16 kHz and bounded to 320–480,000 samples.
The original frontend runs unchanged. The frozen export uses reflection padding,
a float64 STFT and `floor(samples / 160) + 1` valid frames. The reviewed newer
frontend uses zero padding and masks its terminal frame. The composition honors
either declared output length within that closed range, and checks each encoder
length against eightfold subsampling. Every output and recurrent state must be
finite with the exact dtype and geometry before it can influence decoding.

Owned TDT decoding preserves the last emitted token and updates LSTM state only
on a nonblank prediction. Five duration logits select advances 0–4; blank/zero
advances one frame, and ten nonblank zero-duration emissions also force progress.
The original 30-second frontend yields at most 376 encoder frames and 3,760 decoder
steps. Vocabulary IDs, repeated tokens and source whitespace decoding are retained.
Output is bounded to 16 KiB; language is automatic and timestamps are unavailable.
Repository generation defaults do not control this loop.

Tests compare complete small exported Conformer/LSTM/joint graphs to independent
native PyTorch encoder, recurrent state and full-history decoding references.
Independent NumPy frontend checks include the original 139,764-byte frontend;
that temporary component is deleted immediately after verification. Its successful
execution qualifies only preprocessing, not original ASR inference. Invalid
numeric encoder outputs, including the legacy Torch ONNX export's all-masked
attention failure, are rejected. Original large encoder/decoder weights and
packaged-device ASR execution remain outside this metadata-focused qualification.


### Audio8 bounded streaming transcription

The shared PyTorch worker binds the reviewed `audio8_asr_infinite` version-2 graph to
`transcription-v1`. The native Voxtral realtime encoder, complete Qwen2 decoder,
per-layer delay modulation, maximum-frame projector, frame-length embedding and all
four auxiliary semantic-VAD heads are retained. The last heads are required numeric
weights even though the transcription result does not expose them. The original
938-tensor index and tied decoder projection are validated; repository Python and
chat templates are never executed.

The owned route uses the published 80 ms clock (`frame_len=4`), 480 ms delay,
18 left-pad tokens, 17 right-pad tokens, 52.5 ms look-back and 2.5 ms look-ahead.
CPU float32 128-bin Slaney log-mels preserve the native centered 400-point STFT and
fixed global log-mel floor. Four causal encoder frames are zero-padded to the trained
eight-frame projector width. The text decoder receives the original per-layer
sinusoidal delay plus learned frame-length modulation. Its greedy loop suppresses
EOS, preserves repeated tokens, and ends at the audio-clock boundary. Undefined
padded-vocabulary output IDs and nonfinite outputs fail rather than being remapped.

Inputs use the shared PCM16 mono/stereo/resampling reader and contain 320–480,000
16 kHz samples. The 30-second bound produces at most 409 text-input clock tokens
and 385 generated tokens; output is limited to 16 KiB. English (`en`, default) and
Chinese (`zh`) are explicit controls. Automatic language detection and timestamps
are unsupported. Repository generation defaults cannot change these limits.
Complete float32 CPU/Metal reference graphs and offline-worker tests qualify the
binding; metadata support does not certify original-checkpoint execution or device
fit. Memory estimates include host float32 expansion and a device copy.

### Packed Bonsai Qwen3.5 text generation

The MLX chat binding accepts complete Qwen3.5 affine one-bit/two-bit graphs and
schema-2 `prism_hadamard_qwen35` graphs with explicit Hadamard metadata. All native
language and floating vision tensors remain required, although this binding exposes
text conversation only. One-bit words widen losslessly to the installed native
MLX two-bit format; affine scales and biases stay unchanged. Hadamard graphs retain
normalized float32 input transforms, inverse embedding transforms, grouped GDN
layout and exact per-width signs. Module paths select only enumerated native layers.
Repository runtime modules and chat templates are not executed.

The owned ChatML renderer uses non-thinking generation and omits past reasoning,
matching the publisher templates with `enable_thinking=False` and
`preserve_thinking=False`. Requests end with a user message; system instructions
appear only first. Context is bounded to 4,096 tokens, output to 512 tokens and
64 KiB. Original tokenizer controls and declared stop IDs must agree; selected
padded vocabulary IDs and nonfinite logits fail. Complete graph validation rejects
additional learned branches, missing vision tensors, malformed packed dtypes,
inconsistent transforms, unknown active options and changed source builds.
ConstantKV/crystallized artifacts require a separate attention binding.

Language-only Bolding MTP exports have a separate owned binding. Discovery checks
every backbone and draft tensor against the declared graph, including the eight
8-bit/group-32 draft modules, 2-bit/group-128 backbone, three sign-vector widths,
serialized normalization offsets and convolution layout. The full vocabulary stays
intact; the declared unused tail IDs produce empty decoded pieces. Repository
`model.py` and tokenizer Python modules are never imported. Sign vectors pass the
same tensor preflight as model shards and are independently bound to their hashes;
the original shard index covers the backbone and draft tensors only.

This chat binding uses ordinary autoregressive generation with the same context
and output bounds. The complete native MTP draft graph is retained and tested,
including its shared inverse-transformed embedding, but speculative acceleration
is not advertised. Complete small generated graphs check cached and uncached
backbone/draft logits against independent affine and Sylvester-matrix references.

Validation uses complete generated native graphs, independent unpacked affine and
Sylvester-matrix references, cached decoding, original tensor headers/tokenizers
and repeated offline worker requests. This evidence establishes the family binding
and metadata compatibility; it does not qualify original 27B inference or devices.

### Kolibri mixed-precision chat

The owned Kolibri graph binds every routed expert, shared expert, attention tensor
and sandwich normalization. Routed experts use 4-bit/group-64 affine weights;
attention, shared experts, embeddings and output use 8-bit/group-64. The router
stays unquantized, selects experts using biased logits, and weights their outputs
with the unbiased sigmoid. Router biases retain float32 precision.

Sliding layers use RoPE and rotating caches; full-attention layers retain the
original position-free attention. Original sparse tokenizer IDs are preserved:
valid high IDs remain usable and missing IDs are rejected. The owned non-thinking
ChatML renderer reproduces the original prompt without executing its template.
Context and output limits are 4,096 and 512 tokens. Complete small generated graphs
exercise all five layers in a sliding/full block, mixed precision, independent
PyTorch logits, cache rotation, malformed artifacts and repeated offline workers.
Original header/tokenizer evidence establishes metadata compatibility; original
checkpoint inference and packaged-device qualification remain separate.

### MiniMax Music3

The Diffusers pack binds the complete native Music3 modular graph to
`text-to-music-v1`: Qwen3 language model, RVQ depth decoder, hidden-state condition
encoder, flow transformer, scheduler and stereo vocoder. Modular index entries
select only reviewed bundled classes and local component folders. Repository and
revision hints never initiate downloads. The alternate training-language-model
folder and repository Python are excluded from the selected inference graph.

Inspection checks all component configurations and tensor headers, including
cross-component widths, eight codebooks and the 25 Hz semantic frame timeline.
The runtime preserves the native prompt/lyrics normalization and fixed sampling
recipe. It accepts 1–30 seconds, optional lyrics (default `[Instrumental]`) and
seeds from 0 through 4294967295. Complete components run in float32 on the bound
backend; memory estimates include this expansion. The selected native vocoder
and condition encoder declare 44,100 Hz stereo, which determines the WAV output.

Native generation may end before the duration limit. Output validation checks
the actual PCM channel count, rate, sample count, bounds and complete payload
before staging the artifact. Progress checkpoints cover autoregressive and flow
steps; cancellation prevents publication. Generated complete graphs test native
CPU/MPS parity, chunk overlap, repeatability, conditioning and offline workers.
Original model weights and packaged-device qualification are separate evidence.

### MiniMax H3 video with audio

The Diffusers pack binds the native H3 `t2va` and `fl2va` workflows to
`text-to-video-v1` and `image-text-to-video-v1`. Both retain the complete Qwen3-VL
conditioner, joint video/audio transformer, video VAE and stereo audio VAE. The
separate reference transformer belongs to `ref2va` and is not selected by these
workflows. Component hints are inert provenance; only bundled classes and local
SafeTensors components are loaded.

Inspection verifies every selected tensor shape and shard binding, cross-component
dimensions, native processor settings and tokenizer control IDs. Image conditioning
uses the native vision presentation and sampled VAE latent anchor; plain text uses
the prompt verbatim. Repository Python and chat templates are not executed. The
conditioner reads hidden state 50 from the full language stack. All learned
components compute in float32, including checkpoints stored in FP16 or BF16.

Profiles require 24 fps and frame counts `17*n+5` from 124 through 345, canvases
up to 1024 pixels per side in multiples of 32, and at most 96 MiPixels per output.
Guidance is fixed to 1; negative prompts are unsupported. The shared contract's
360-frame ceiling permits this native geometry; individual adapter profiles keep
their own bounds. Public step counts represent denoiser evaluations, excluding
the scheduler's terminal zero. Image inputs remain leased artifacts.

Output is MP4 containing both video and 32 kHz stereo AAC. The native 40 Hz audio
grid may finish up to half a hop before the video; only excess terminal samples
are trimmed. No audio stretching or resampling is applied. MP4 timing preserves
the exact sample count, and full stream decoding verifies geometry, frame count,
audio duration and finite samples before publication. Cancellation removes
partial output. Complete generated graphs cover CPU/MPS parity, both conditioning
paths, stored precisions, offline workers and shared artifact ownership. Original
checkpoint inference and packaged-device qualification remain separate evidence.

### Text end-of-utterance classification

The shared PyTorch classifier binds the reviewed causal Llama turn graph to
`text-classification-v1`. It retains the complete native backbone and tied output
projection. The two results are the full-vocabulary next-token probability of the
declared end-of-utterance token and its complement (`continue`). No decision
threshold or restricted two-token softmax is substituted.

The owned single-user renderer follows the frozen graph's historical normalization:
remove ASCII punctuation except apostrophes, lowercase, collapse whitespace and
frame with the original chat-start and user controls. Repository chat templates
and Python are not executed. Reserved controls, empty normalized input and more
than 512 tokens fail. The newer Qwen-oriented model card does not define this
older Llama graph's preprocessing.

Closed configuration and tokenizer identities, exact complete numeric weights,
finite full-vocabulary outputs and two-label top-K bounds are checked before
returning predictions. Original tensor headers and tokenizer metadata establish
compatibility without fetching learned weights. Complete generated CPU/Metal
graphs and repeated offline worker tests validate the family implementation.

### Multi-head audio classification

`audio-classification-v1` has an optional `classificationHead` control. It requires
explicit adapter declaration; existing classifier profiles retain their prior
parameters and reject this control. SpeechX declares `language` (default) and
`gender`. A request returns the selected head's original label IDs and softmax
scores. Probabilities from different heads are never combined or renormalized
over only the returned top-K labels. The existing result shape remains unchanged.

The SpeechX binding retains the complete native Whisper-base encoder and both
linear heads. Its closed geometry is built locally, without a base-model fetch or
repository Python. Exact label maps, native tensor shapes and all numeric weights
are mandatory. The original Whisper log-mel frontend pads to 30 seconds, and mean
pooling includes those padded frames, matching the published forward. Shared
PCM16 mono/stereo inputs are resampled to 16 kHz and bounded to 320–480,000 samples;
overlong input fails instead of being truncated. Both heads must remain finite
even when only one is requested.

Original headers, complete generated CPU/Metal graphs, independent full-head
softmax and NumPy frontend references, closed metadata, malformed artifacts and
repeated offline workers validate the binding. Original learned weights and
packaged-device execution are not required for this metadata qualification.

### Legacy flat DDPM image generation

The exact legacy `modeling_ddpm/CustomPipeline` declaration with native UNet2DModel
and DDPMScheduler components maps to the owned unconditional-image binding. Its
repository Python is neither installed nor executed. Closed root UNet/scheduler
metadata selects a local constructor; every native tensor is required. Legacy
attention names map to native query/key/value/output projections without changing
their values. Duplicate aliases, nonnumeric/nonfinite weights and additional
learned branches fail.

The native raster, positional embeddings, attention layout and original DDPM
schedule remain fixed. Requests control only the existing steps/seed parameters;
the engine supplies the seeded generator to all native noise draws. Native
denoiser and scheduler outputs must remain finite before conversion to an image.
Complete generated CPU/Metal graphs and original tensor headers validate loading
and sampling without downloading the original learned weights.

### Sopro reference speech

The shared PyTorch runner binds the original Sopro four-file safetensors layout:
`model.safetensors`, `semantic_encoder.safetensors`, `speaker_encoder.safetensors`,
`vocoder.safetensors`, plus its configuration and SentencePiece tokenizer. Every
component is mandatory and strictly matched to the complete native graph. The
reviewed Sopro 2.2.0 graph subset and TorchAudio 2.11.0 pure tensor math ship with
source provenance and licenses. No repository module, automatic download or
compiled TorchAudio binary participates in execution.

`speech-synthesis-v1` admits `audioArtifactId` only when the adapter explicitly
declares it. Sopro fixes `voice` to `reference`, `speed` to 1, and requires an
owned audio artifact: PCM16 WAV, mono or stereo, 8/16/24/32/44.1/48 kHz, 0.4–10
seconds. The browser exposes a reference file picker and English, Portuguese,
French and German controls. Existing speech profiles retain their parameter set
and reject reference artifacts. Negotiated limits are 300 characters, 10 seconds
of reference audio and 30 seconds of output; text tokenization rejects inputs
above 512 tokens rather than truncating them.

Execution preserves native mel filters, sinc resampling, reference normalization,
semantic encoder, speaker embeddings, cached autoregressive sampling, flow solver,
and vocoder. Generation must emit EOS within its bounded loop; invalid tensors,
missing components, malformed geometry and empty outputs fail. The native
centered ISTFT emits one hop fewer than its input frame count; the adapter retains
that duration and does not append an artificial frame. Output is an authorized
24 kHz mono PCM16 WAV artifact. Metadata coverage and small synthetic CPU/Metal
graph tests do not constitute original-checkpoint inference or packaged-device
qualification.


### OmniVoice automatic speech

The original OmniVoice safe export uses the shared Transformers runtime with the complete bundled Higgs audio codec. The engine owns the reviewed Qwen3/audio graph and the 32-step bidirectional classifier-free decoding loop. All 313 main and 527 codec arrays must match the closed native geometry; no repository Python is imported. Both codec encoder and decoder weights remain mandatory. The main graph supports CPU and Metal; the native codec stays on CPU, as in upstream OmniVoice.

The speech profile declares automatic voice (`voice: auto`), fixed speed 1 and the reviewed language-ID set. It accepts up to 300 characters and 512 text tokens. Native duration estimation is bounded to 745 audio frames (29.8 seconds), leaving room for the native 100ms padding at each edge within the 30-second WAV limit. It preserves inline nonverbal tokenization, all eight audio codebooks, native peak normalization and fades. The upstream optional silence-removal stage is disabled, so timing is preserved without introducing an external media binary. Reference cloning and voice-design instructions are not declared.

The runtime validates the bundled codec configuration and preprocessing metadata, tokenizer controls, all tensor names/shapes/types, codebook offsets and finite values before execution. CPU/Metal miniature full-graph tests and original metadata validation establish this binding; they do not establish original-checkpoint output quality or packaged-device qualification.

### Breeze automatic speech

The original Breeze TTS 2 safe export uses the shared Transformers runtime. The owned binding retains the complete Qwen3 backbone, all 15 Llama depth heads, T5Gemma2 text encoder, original Mimi components, and bundled Qwen audio tokenizer. All 1,115 stored main arrays and 496 codec arrays must match the native geometry; the omitted shared audio embedding is restored from its original tied weight. No repository Python is imported. CPU and Metal execute the same complete graph.

The speech profile accepts text with automatic voice and language, fixed speed 1, and no reference audio. It preserves native BOS/speaker tokenization, repetition penalty, temperature, top-k sampling, cached depth decoding, and chunked codec decoding. Inputs are bounded to 300 characters and 512 tokens. Native EOS is required within 375 audio frames (30 seconds); incomplete generation fails. Output is 24 kHz mono PCM16. Metadata and synthetic graph validation do not establish original-checkpoint quality or packaged-device qualification.

### MolParser Mobile V2 chemical notation

The original MolParser Mobile V2 safe export uses a complete native timm hybrid ResNet/ViT encoder and BART decoder in the shared Transformers environment. All 311 stored arrays remain mandatory; the omitted output projection is restored from the tied decoder embedding. Reviewed `auto_map` entries are normalized to this owned binding. No repository code is imported and no pretrained timm download occurs.

The image-to-text profile preserves RGB conversion, direct OpenCV linear resizing, ImageNet normalization, the 414-token vocabulary, and E-SMILES markup. It emits the original notation as text without interpreting chemical validity. Greedy generation uses native forced EOS, no cache, and at most 256 new tokens plus the starting CLS token, within the original 384-token ceiling. CPU and Metal use the same complete graph. Metadata and synthetic graph tests do not establish accuracy of original learned weights or packaged-device qualification.

### Qwen standalone audio codec

The original Qwen3-TTS-Tokenizer-12Hz export reuses the complete Qwen codec already bound for Breeze. All 496 encoder and decoder tensors remain mandatory. The shared Transformers environment exposes explicit `encode` and `decode` operations without repository code or a new dependency environment.

Encoding accepts mono PCM16 WAV at 24 kHz, from 80 milliseconds to 30 seconds, and returns one row of 16 codebook indices per 80 ms frame. A partial final frame retains native padding; reported duration describes the encoded frame clock. Decoding accepts 1–375 rows, exactly 16 indices per row, each in 0–2047, and returns an owned 24 kHz mono PCM16 WAV. The native chunked decoder and all encoder quantizers are retained. Nonfinite nearest-centroid distances are rejected before token selection, and raw nonfinite decoder samples are rejected before native clipping. The engine validates the final waveform length against the requested code frames.

### RoBERTa-tokenized LayoutLM document QA

The shared PyTorch environment owns the complete LayoutLM graph for the reviewed
Impira document-QA and invoice declarations. The invoice route retains and uses
both span heads and the additional two-class token-selection head; it can return
non-consecutive OCR words within the selected span. `wordStart` and `wordEnd`
bound the selected words, and `box` encloses their page boxes. An empty token
selection is reported as an error rather than a fabricated grounded answer.

Inputs supply the existing page image, question, OCR words and normalized boxes.
The runtime admits at most 512 combined question/document tokens, places
separator tokens at `[1000, 1000, 1000, 1000]`, and preserves the publisher's
15-token document span limit or 512-token invoice span limit. Scores preserve
CLS null probability before excluding CLS from answer selection.

Original `tokenizer_config.json` repeats `add_prefix_space` with conflicting
values. This family excludes that optional file and binds the reviewed native
RoBERTa tokenizer directly to the original `tokenizer.json`, with prefix spaces,
special tokens and a 512-token ceiling owned by the adapter. The generic metadata
parser continues to reject conflicting duplicate settings. No repository code,
OCR service, alternate checkpoint or new runtime dependency is used. Complete
weights, both invoice heads, the retained pooler, legacy integer position buffer,
finite logits and fast-tokenizer word alignment are validated on CPU and Metal.

### MP-SENet speech enhancement

The native `enhance` operation accepts 16kHz mono PCM16 WAV, from 400 samples to
30 seconds. Configuration supplies the complete spectral preprocessing; a
separate processor file is unnecessary. The shared runtime retains all 247
original arrays, dense encoder, sequence-first attention, bidirectional GRUs,
learned magnitude mask and phase heads. It uses the original global energy
normalization and 32,000-sample chunks, merging a final tail shorter than half a
chunk. FFT and inverse FFT run on CPU; the complete neural graph runs on CPU or
Metal. No additional runtime dependency is required.

Centered inverse STFT returns `floor(inputSamples / 100) * 100` samples, so up to
99 final samples are omitted according to the original implementation. The
runtime preserves this behavior, checks the published WAV against that exact
count, and uses the publisher's PCM16 saturation/floor conversion. Silent input
whose native normalization is undefined, nonfinite scores and invalid WAV or
weight geometry are rejected. This route does not advertise codec operations,
voice conversion, arbitrary sample rates or alternate checkpoint conversions.

### RE-USE speech enhancement

The two closed original RE-USE profiles use the existing shared PyTorch environment and native Mamba layers. They accept mono PCM16 WAV at 8 kHz, from 320 samples through two seconds, with `operation: enhance`. The two-second bound limits reference Mamba scan memory. The real-time architecture runs offline at full depth with two-frame lookahead; all original early-exit decoders, base heads and lookahead encoder branches remain mandatory in its complete checkpoint. Streaming latency is not qualified.

The engine preserves log-magnitude compression, original spectral frame suppression, inverse STFT and epsilon padding to the exact input sample count. FFT runs on CPU; the complete neural graph supports CPU and Metal. Original metadata and all tensor shapes are validated without importing repository programs. Only root config and safetensors are selected; no text tokenizer, processor file, CUDA extension or additional dependency is required.

### TIGER dialogue, effects and music separation

The original TIGER-DnR export binds three complete native separation graphs. Its closed profile accepts mono PCM16 WAV at 44.1 kHz, from 2048 samples through 12 seconds, with `operation: separate`. It returns three named WAV artifacts in order: `dialog.wav`, `effect.wav`, `music.wav`. Every output must match the input sample count and declared encoding before any output is staged. Paths, duplicate stems, missing branches, truncated files and nonfinite graph values are rejected.

The native 12-second windows, 4-second hop, zero padding, overlap-add divisor and source selection for each graph are preserved. FFT and complex spectral reconstruction run on CPU; real-valued neural layers support CPU and Metal. Exact CPU adaptive pooling handles nondivisible intermediate bins on Metal. All 2304 original tensor declarations, including the three output heads in each branch, are mandatory. No repository loaders or additional dependencies are used.

VideoLLaMA3 original Qwen2 exports use the existing video-question-answering
contract through the pinned native VideoLlama3 graph. The owned route validates
both original 2B/7B tensor layouts, every vision/projector/language weight, and the
2B tied language head. It reads data-only Qwen BPE files and reviewed processor
metadata; repository Python, processor dispatch and Jinja are never executed.
The shared video decoder samples eight presentation timestamps from local clips
of at most 30 seconds. Frames remain within 512 pixels per side, use original
RGB/bicubic/rescale/normalization and 14-pixel patch ordering, spatial merge 2,
and original temporal compression (including one retained token in an unchanged
frame). Prompts preserve original timestamp framing with no default system turn.
The route bounds context to 4096 tokens and answers to 256 tokens, requires EOS,
and returns the existing text/timestamp/duration result. CPU and Metal share the
same environment. Original eager vision attention incorrectly mixes frames;
the owned graph preserves the frame isolation of the publisher's FlashAttention
path, verified against its SDPA implementation with complete synthetic graphs.

The original LLaVA-Video Qwen2 export is a separate owned video-QA layout in the
same environment. It keeps all 26 exported SigLIP layers and the retained post
norm, uses pre-postnorm features, the complete two-layer projector, bilinear
pooling with ceiling dimensions, and one learned newline after every spatial
row. It never fetches the external SigLIP base referenced in repository config.
Eight bounded RGB frames use the original 384-square processor. The explicit
Qwen conversation includes the publisher's duration/timestamp instruction;
context is bounded to 4096 tokens and answers to 256 with EOS required. Both
original video adapters share the bounded Qwen byte-BPE data loader.

Original Qwen image LoRAs use a shared pinned composition loader. The engine-owned
catalog binds an exact adapter file to an exact base revision; every selected file
retains its source repository, revision and publisher hash. Installation includes
the full base, verifies SafeTensors SHA-256 and metadata Git-blob hashes, and keeps
shared base files until the last installation releases them. Offline workers see
only the selected files. Metadata snapshots embed a separately identified base
snapshot under the original adapter identity and share the bounded read budget.

The native loader validates every A/B pair, rank, target shape and alpha, including
ComfyUI's fused gate/up format. It checks the complete base array set and evaluates
float32 LoRA branches without merging into BF16 weights. Original 6-step Viggle
sampling uses the adapter's scheduler and sigma grid; Lightning uses its fixed
4-step contract. Unsupported control values fail explicitly. The MPS Qwen 2.1
VAE executes its parameter-free grouped average on CPU because the native MPS
grouping corrupts temporal padding; CPU reference comparisons preserve the exact
operation and dtype. Both image base families have complete synthetic CPU/MPS
adapter-on/off regressions. These establish engine behavior, not original model
quality or packaged-device qualification. The two-reference outfit-swap workflow
requires an additional input contract before admission.

The `image-editing-v1` contract accepts `referenceImageArtifactIds` as an optional,
ordered list of up to two images after `imageArtifactId`. A profile must explicitly
declare this control and its `minReferenceImages` / `maxReferenceImages` limits;
older profiles accept only the original image. Browser leases every referenced
artifact under the invoking owner's authority and retains those leases until the
job is terminal. The worker resolves the IDs to verified image bindings in input
order and forwards all images to the native Qwen Image 2.1, Qwen Edit Plus, or
Flux Klein pipeline. Paths, duplicate IDs, unbound IDs, and excess images fail.
Each image retains the existing byte, pixel, and preprocessing bounds. The pinned
Outfit-Swap composition requires exactly two images. Complete miniature native
CPU/Metal graphs test reference ordering, conditioning effects, and offline worker
execution; this evidence does not qualify original large checkpoints or devices.
