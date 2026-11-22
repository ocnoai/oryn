# Adding model support

Use the existing PyTorch, MLX, llama.cpp, or ONNX foundation. A compatible new checkpoint should need data and validation, not a new environment or repository-specific execution branch.

Recognise a model by what its own files declare:
- `config.json` architectures and `model_type`;
- the `model_index.json` pipeline class;
- the GGUF header's architecture, tokenizer and file type;
- the chat template's markers, read as text.

Never recognise a model by its repository name or tags. Per-family knowledge belongs in reviewed, signed data tables, not in family modules. Add code only for a new safe weight container, a new public task contract, or an architecture that no pinned library contains, and make it reusable by architecture or task.

## Choose the smallest change

1. Resolve the repository to an immutable revision. Inspect the actual configuration and selected files. The host's summary can omit quantization, component, and custom-code requirements.
2. Check the generated capability table and the implemented task contract. If both cover the model, use the existing adapter. Record the revision and selected-file hashes in the acceptance manifest.
3. If the model chats and its template markers match no reviewed format, add an entry to `build-inputs/chat-formats.json`. Add a shortened marker-only template and an exact render assertion to `tests/test_chat_formats.py`, then rebuild capabilities and regenerate the Protocol list (`../protocol/scripts/sync-chat-formats.mjs`). Never render the repository template, and never write a per-family prompt function.
4. If the task is missing, implement one reusable task adapter. If the architecture is missing, prefer an upgrade of the existing pinned library. A new foundation needs an explicit product requirement and dependency-cost review.
5. Run the checkpoint and a second compatible checkpoint through unchanged code. Verify task output, offline reload, cancellation, resource admission, and file integrity. Keep failures visible.
6. Review and sign changed runtime manifests and generated capability data in the build process. Package dependencies from hash-locked requirements. Run packaged-runtime verification and the acceptance harness.
7. Publish the exact tested revision, task, adapter/runtime build, device, settings, and results. Static compatibility and a local smoke test do not automatically confer release certification.

## Generate and verify capabilities

`build_capabilities.py` extracts Transformers Auto mappings and MLX-LM model modules from installed build dependencies. Diffusers combines upstream mappings with reviewed native Z-Image, FLUX.2 Klein and Qwen Image task contracts. Upstream availability, Oryn admission and checkpoint execution evidence are separate facts. Every generated entry starts untested.

The output includes source hashes and a signed build envelope. `generated_capabilities.verify_build_inputs()` checks that the packaged Transformers mapping, MLX model modules, and Diffusers mapping match the generated artifact. The Browser assembly and packaged-runtime verifier run this gate. GGUF facts come from the pinned native build with a separate reviewed chat-tuple policy; ONNX retains its reviewed task-specific manifest. Native availability alone never authorizes execution.

For local development, `scripts/build_development_capabilities.py --mlx-package PATH --diffusers-package PATH` creates a temporary development signing key, regenerates capability data, refreshes the development pack/adapter/catalog signatures, retains only the public key, and destroys the private key. Review the resulting diff. This command does not create production release authority. Production releases use the protected release-signing process.

Do not regenerate capability data in response to a website or model download. A library update that changes packaged mapping bytes must produce a reviewed build artifact.

## Acceptance commands

Run `scripts/validate_model_support.py` with the packaged Python interpreter:

```sh
python scripts/validate_model_support.py prepare --state /absolute/test-state --output /absolute/prepare.json
python scripts/validate_model_support.py execute --state /absolute/test-state --output /absolute/execution.json
```

`prepare` explicitly downloads selected model data from the pinned test manifest. `execute` uses installed data offline. `--groups mlx,gguf,embedding,ranking,vision,editing` selects a bounded cohort. The harness does not install libraries, execute repository tests, run conversions, or certify models. Two runs measure reload behavior; MLX and llama.cpp workers currently exit after each invocation, so these are not warm model-residency benchmarks.

The fixture manifest is a test input, not an executable model profile or a source of production runtime authority. Existing signed compatibility manifests and closed capability schemas remain the production admission formats. The manifest cannot supply an arbitrary profile interpreter.

## Security rules

- Repository Python, shell commands, shared libraries, package installers, custom operators, and arbitrary loader options cannot become execution authority.
- A stray training script is excluded from selection. A selected path requiring custom code remains blocked. User trust cannot change this rule.
- Downloads verify byte counts and available SHA-256 identities. Installed files record content hashes and are checked before loading. Sharded indexes cannot name unselected files.
- Runtime code and dependency changes happen in Oryn's build/release workflow, with signatures and revocation controls. MLX execution uses the shared MLX foundation and base runtime.
- Unsafe-weight conversion is not implemented in ordinary inference. Pickle-only checkpoints remain blocked.
- Model license/access acceptance remains Browser-owned. A routing score, performance measurement, or model-trust prompt cannot grant access or waive license terms.

## Required evidence before claiming broader support

Record static support, exact execution results, and device admissibility independently. Do not multiply an architecture-level result across every repository or revision. Do not infer equivalent MLX and GGUF variants from names: switching between repositories still requires signed provenance/equivalence and Browser model-resolution authority.

On Apple Silicon, keep MLX chat and Metal-accelerated GGUF available. Transformers on MPS supplies embeddings, ranking, vision-language input, and other implemented tasks. There is no unconditional runtime preference by operating system.


For the actual macOS process boundary, add `--engine-root /absolute/packaged/engine` and
`--sandbox-profiles /absolute/security` to the execute command. The harness uses the same launcher
as Browser, applies the Engine profile with IP networking additionally denied, and retains the
no-network worker profile. The broker uses its private Unix socket. `HF_HUB_OFFLINE` and
`TRANSFORMERS_OFFLINE` are also set. The default cohort covers 12 checkpoints. The
`--models` option selects an alternative bounded cohort; `--groups` filters its task
groups. Reference inputs are immutable and hash-checked during preparation.

Acceptance records load/reload time, first text delta, execution duration, output checks and
cancellation. Outer-process RSS sampling is explicitly labeled: shared pages can be double-counted,
and it is not peak unified/GPU memory. Do not import this smoke-test record as a comparative routing
benchmark or claim a tested result on another Mac memory class.

## Task onboarding

Use the shared runtime-pack catalog. NLI, image-embedding, segmentation, prompted-mask,
text-guided-detection, caption and VQA cases use `shared_tasks.py` and the existing
Transformers pack. Add a pinned data-only checkpoint to the acceptance manifest; do not
add a model-specific loader, environment or repository exception. Unknown native behavior
requires a reusable family/task adapter and its contract/security tests first.

`build-inputs/llama-chat-policy.json` records reviewed policy independently from the pinned
llama.cpp architecture source. `build_capabilities.py` combines the facts and policy into
signed schema-2 tuples, including withheld capabilities. Update policy only with reviewed
framing, tensor/header and execution evidence. The native binary digest must match during
packaged verification. The local development signing helper destroys its temporary private
key; it is not a production release signer.

Run `scripts/validate_model_support.py prepare` explicitly to inspect/install the pinned
acceptance fixtures, then `execute --sandbox-profiles <Browser packaging profiles>` using
the packaged interpreter. Execution blocks network and records task checks, exact release
manifest, selected-file digest, two reload samples and cancellation. These are fixture
measurements on the actual machine; they do not authorize hidden profiling of user prompts
or imply support on a smaller Mac.


## Runtime and image-task requirements

Test each adapter against the hash-locked shared foundation in its own dependency
search path. Separate adapter dependency sets remain separate from the base runtime.
The base PyTorch closure supplies shared segmentation and tokenizer dependencies.
GLM OCR uses a fixed Text Recognition prompt and bounded native geometry.
Malformed tokenizer metadata remains rejected; choose a valid immutable upstream revision.

For image families, review the entire native component graph and call signature.
Use `image-editing-v1` for instruction editing and `image-layers-v1` for decomposition;
never silently ignore `strength` or repurpose it. The Protocol schema supplies the limits.
All selected shard indexes must match the bounded SafeTensors header inventory exactly.
Record code/schema tests separately from actual checkpoint runs and production clearance.

## Qwen Image 2.1

Browser's `packaging/diffusers-source.json` pins the Diffusers source archive and
reproducible wheel hashes. The pack shares Transformers with the base runtime.
Capability evidence also hashes the native pipeline, transformer and VAE implementations.

`QwenImage21Pipeline` supports text generation of RGBA PNGs and single-reference
instruction editing. Its tokenizer belongs to `processor/`; Oryn constructs the
Qwen3-VL processor before pipeline loading and replaces repository prompt templates
with fixed native framing. Defaults are 512 × 512, 40 steps and true CFG 1.0.
Dimensions must be multiples of 32, with at most 1024² pixels, 50 steps, CFG 10,
and 512 tokens per prompt. Reference images retain alpha and must have an aspect
ratio between 1:2 and 2:1. Multi-reference editing is not exposed yet.

`tests/test_qwen_image_21_native.py` executes small random native checkpoints on
CPU and Metal. `scripts/verify_qwen_image_21_runtime.py` exercises generation and
editing using the packaged interpreter, production worker and network-denying
Seatbelt profile. These checks establish pipeline execution, not full checkpoint
quality or conformance. The official checkpoint remains subject to normal trust,
disk and memory admission; it is not marked Oryn verified from fixture results.
