# Backend providers

Oryn has a common provider ABI and a separate MFLUX provider package. MFLUX
checkpoint support is generated from the pinned backend and validated model metadata.

## Implemented boundary

`src/oryn_engine/providers.py` owns provider discovery, bounded metadata requests,
candidate validation, and execution-plan identity. Providers are declared in signed
runtime-pack manifests. Their catalogs bind their own Python code by digest.

The private ABI extends existing supervised pack workers with `describe`, `inspect`,
and `prepare`; `generate` retains the existing job transport and cancellation.
`inspect` receives repository metadata and a file inventory, and may request named
JSON files or bounded SafeTensors headers. The Engine fetches these; provider workers
remain offline and receive no Hub credentials.

Each candidate supplies its task contract, modalities, exact files, component
representations, controls, generation defaults, memory reservation, and a bounded
provider plan. Installation hashes the selected artifacts, validates their indexes,
and runs preparation against a view containing only those files. Execution checks
the pack version/digest, catalog, model revision, and selected file identities again.
Changed provider installations are reinspected and prepared using the immutable
download cache. Runtime overrides preserve the provider plan and its defaults.

The generic provider handle currently binds the standard text-to-image artifact
contract. Other task envelopes are handled by the corresponding Engine adapters.
The scheduler, routing policy, worker broker, cancellation,
artifact publication, and memory admission remain in use.

## MFLUX integration

`providers/mflux` builds the separate `oryn-mflux-provider` wheel. The runtime ships
MFLUX 0.24.0 at commit `8a7fd0a486635f670ef90e02efe1c051ed3f133c`, with MLX 0.32.0
in the shared foundation. Torch and Transformers are shared with the base runtime.

The catalog builder consumes MFLUX's command capability exporter, native registry,
constructor/generation interfaces, weight definitions, and tokenizer definitions.
It accounts for every registry entry with either a binding or an explicit reason
that another binding is needed. It records upstream source hashes for drift checks.
Inspection and execution consume the same catalog. Execution delegates to the
backend's native command and callback handling, including its low-memory mode.
Decoded tensors are checked for non-finite values before pixel conversion; numerical
failures fail the job instead of publishing a black image as a successful result.

The signed catalog records supported native configurations and explicit gaps.
Binding coverage does not establish that every checkpoint in a supported family has
passed generation.

Repository identity is metadata, not executable selection. Declared base-model or
architecture metadata resolves renamed checkpoints. Name hints can propose a family,
but its required component graph must still validate. Missing or conflicting metadata
does not select an arbitrary class. Repository Python and templates do not become
executable provider code.

Stored bit widths come from component metadata and the backend's supported choices.
Unquantized, 4-bit, 8-bit, and mixed components use the same path. Preparation checks
every selected shard's stored quantization against the inspected plan; the native
loader reconstructs stored per-layer representations. No on-demand conversion is
silently requested. Header inspection currently reports first-shard coverage, rather
than claiming an exhaustive per-layer quantization inventory before download.

Memory is a conservative reservation estimate: selected tensor bytes, loading overlap
for the largest shard, and workspace for the bounded image settings. It is not a
measured universal upper bound. The provider advertises at most 1024 × 1024 pixels
and 50 steps. The same reservation feeds admission and the Browser's needed/available
memory display. Each generation worker releases its weights on exit; a loaded handle
reserves capacity rather than claiming that those weights are permanently resident.

## Updating the provider

1. Pin and review upstream source and build constraints in Browser's
   `packaging/mflux-source.json` and `packaging/mflux-build-constraints.txt`.
   `python scripts/build-mflux-wheel.py` reproduces and verifies the vendored wheel;
   `--archive PATH` can reuse an already downloaded, hash-verified source archive.
2. Assemble the reviewed backend dependencies, then run the source catalog builder
   with the packaged Python interpreter:

   ```sh
   "$RUNTIME_PYTHON" -I providers/mflux/src/oryn_mflux_provider/build_catalog.py \
     --runtime-root "$ENGINE_RUNTIME_ROOT"
   ```

   Review `entries`, `uncovered`, source identities, and changed defaults. `--check`
   checks reproducibility without replacing the catalog.
3. Build `providers/mflux` with `uv build --wheel`, update Browser's provider wheel
   hash, and sign the pack manifest/catalog. Local development uses
   `scripts/build_development_capabilities.py`; release signing remains a separate
   publisher responsibility. Code or catalog changes must produce a new signed build.
4. Run provider tests and `scripts/verify_mflux_provider_runtime.py` with the packaged
   interpreter, Engine directory, and worker sandbox profile. An optional ModelStore
   directory enables an offline generation check of an installed real checkpoint.
5. Assemble and package the Browser. Distribution currently remains bundled with the
   application; no automatic runtime update service is supplied.

## Coverage and limitations

Regression coverage includes three native family descriptors across unquantized,
4-bit and 8-bit exports, renamed repositories, mixed components, missing/invalid
shards, unknown architectures, bounded HTTP header reads, changed plan/code/artifact
identity, installation preparation, restart, runtime replacement, and protocol output.
Packaged verification checks upstream source identities and exercises native MLX
weight application for unquantized, 4-bit and 8-bit layers.

On macOS, the worker permits the Metal compiler helper to issue an extension only
for its OS-owned `com.apple.metalfe` cache. The packaged security check compiles and
runs a custom MLX kernel and checks that network and unrelated file access remain
denied. The checkpoint and provider still execute with the existing worker sandbox.

Diffusers, MLX-LM, Transformers and task-specific audio/ONNX execution use their
corresponding Engine adapters. The provider ABI does not replace those execution
implementations.

GGUF image execution is not supplied by MFLUX or Oryn's llama.cpp chat runtime. A
provider with an appropriate diffusion GGUF loader and matching component layout is
still required. Image editing, fill/depth conditioning, and backend paths dependent on
repository prompt templates also require additional reusable task bindings. The
catalog records those gaps instead of advertising them as text-to-image support.

New compatible checkpoints and supported stored precisions require no Engine code.
An unsupported architecture still requires a backend build implementing it. Native
interface or task-semantic changes may require provider work; the abstraction does
not make unknown architectures executable.
