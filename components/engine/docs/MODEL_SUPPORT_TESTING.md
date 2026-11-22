# Automated model support tests

Layer 1 is implemented by `oryn_engine.metadata_audit`. It runs deterministic Python
checks through Oryn's production `ModelStore.inspect`, task contracts, signed
capability catalog, file selection and provider inspectors. It uses no LLM, model
card prose interpretation, manual report overrides, or inference.

After the root `./scripts/bootstrap`, run from the Engine directory:

```sh
cd components/engine
export PYTHONPATH="$PWD/src:$PWD/providers/mflux/src"
uv run --no-sync python -m oryn_engine.metadata_audit models.csv --output .model-support/my-model-check
```

Try the included example:

```sh
uv run --no-sync python -m oryn_engine.metadata_audit examples/model-support.csv \
  --output .model-support/example-model-check
```

This command uses the existing Engine environment without installing dependencies.
It includes the first-party MFLUX metadata provider from this checkout; the provider
catalog and code are still checked against the signed declaration. Native MFLUX or
other model libraries are not loaded for inference. The Engine unit suite includes
the auditor's regression tests; run just those with:

```sh
uv run --no-sync pytest tests/test_metadata_audit.py
```

## CSV input

The smallest input is one link per line, with or without a `url` header:

```csv
url
https://huggingface.co/openai/whisper-tiny
https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo
```

Optional columns make the intended task and exact artifact explicit:

```csv
url,revision,task,files,expected
https://huggingface.co/openai/whisper-tiny,,automatic-speech-recognition,,metadata-compatible
https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo,fc13ba1986fcb905854dc8de2f52eee395f49b5b,text-to-image,,metadata-compatible
```

- `url`: a repository URL, `/tree/revision` URL, or `owner/repository`. File links,
  subdirectory links, Spaces and datasets are rejected to avoid silently testing
  a different artifact. Query strings and fragments do not select an artifact.
- `revision`: optional branch, tag or immutable commit. A contradictory URL revision
  is an input error. An omitted revision starts at `main` and resolves to a commit.
- `task`: explicit requested task. If omitted, uses the Hub task metadata; an absent
  tag remains `metadata-incomplete`. Unknown tasks remain incompatible with the
  current contract registry; the auditor does not substitute a supported task.
- `files`: optional CSV-quoted JSON array of exact filenames. For example,
  `"[""model.gguf""]"`. An incomplete selection cannot pass.
- `expected`: optional expected status for regression tests, including intentional
  negative cases. Use `--fail-on unexpected` to assert those expectations.

Recognized link headers also include `link`, `model_url`, `hf_url`, `repository`, and
`model`. Use `--url-column your_column` for a different header. Duplicate rows retain
their own outcomes; resolved repository/revision/task identities are counted once
in the summary. `identity` separately fingerprints the selected graph;
`originalIdentity` excludes representation choices for exact benchmark counting.
Rows without a resolved revision are counted separately as `unresolvedRows`, not
invented exact identities.

## Results and repeatability

`--target-platform` accepts `macos-arm64` (the default), `linux-x64-cuda`,
`windows-x64-cuda`, and `mac-nvidia-union`. Profiles use fixed assumed hardware,
intersect backends with signed pack/adapter/foundation declarations, and reject a
foundation lacking the target platform. The current signed foundations declare
macOS only, so a CUDA name does not make a NVIDIA profile pass. These are metadata
projections, with resources `not-evaluated` and no actionable `loadDisposition`.

A union row retains every `profileResults` entry and its `passingProfiles`; it
passes only when at least one profile passes. Reports include separate platform,
NVIDIA, intersection and union metadata counts. Profile configuration and signed
foundation/adapter digests are included in provenance. Physical devices, installed
packages, actual placement and memory remain later-layer checks.

A passing metadata audit records only the first qualification layer. It does not
certify inference quality, physical device support or complete model support.

The output directory contains:

- `results.csv`: one result per input row, resolved revision, status, reason codes,
  selected runtime/files, file identities, memory estimate and snapshot reference.
- `results.json`: those outcomes plus original inspector profiles, task contracts,
  per-provider diagnostics, metadata failures, build/catalog fingerprints and counts.
- `junit.xml`: CI-compatible test results.
- `snapshots/*.json`: content-digested inventory, config and requested header metadata.
- `rows/*.json`: results saved as work completes. An interrupted run can reuse the
  collected snapshots; verdicts are always recomputed against current code.

Statuses have distinct meanings:

| Status | Meaning |
| --- | --- |
| `metadata-compatible` | A complete selected route passes the current metadata rules. |
| `metadata-incompatible` | Collected metadata fails a rule or the requested task has no contract. |
| `metadata-incomplete` | Missing reads, unknown task metadata, unresolved formats or inspection failures prevent a conclusion. |
| `access-required` | Metadata/artifacts require access not established by the recorded check. |
| `invalid-input` | Invalid link, CSV row, revision, file selection syntax or expectation. |
| `error` | Auditor, snapshot integrity or unexpected inspector failure; not a model incompatibility claim. |

`CONFORMANCE_REQUIRED` is preserved separately as `nextLayerReasons`, since it
belongs to execution qualification. Failed alternative providers do not invalidate
a complete selected route, even if both providers requested the same filename.
Failures needed by the selected route do prevent a pass.

```sh
# Recompute with saved metadata and no Hub requests:
uv run --no-sync python -m oryn_engine.metadata_audit models.csv --output .model-support/my-model-check --offline

# Reuse the same frozen input in a separate report, e.g. after a code change:
uv run --no-sync python -m oryn_engine.metadata_audit models.csv --output .model-support/recheck \
  --cache .model-support/my-model-check/snapshots --offline

# Resolve mutable links again and replace their cached metadata:
uv run --no-sync python -m oryn_engine.metadata_audit models.csv --output .model-support/my-model-check --refresh
```

Existing snapshots freeze the resolved revision until `--refresh`. Offline replay
can become incomplete if new inspector code requires metadata absent from the old
snapshot. Never fill such gaps with assumptions. Snapshots record access at
collection time; replay does not establish that access is still available today.

Default authentication is explicitly anonymous, including when a cached Hub login
or `HF_TOKEN` exists. To opt into an existing token environment variable, add
`--token-env HF_TOKEN`. The token value is not written to reports or snapshots.
Authenticated and anonymous snapshots use separate cache identities.

The default `--workers 4` accepts 1–16 concurrent rows. Inventory requests are paced
to one every 0.7 seconds across all workers in the process. HTTP 429
inventory responses share a server-directed cooldown (`Retry-After` or the Hub
`RateLimit` reset) and retry up to three attempts. This pacing covers the anonymous
API quota; other applications or concurrent auditor processes can share that quota.
Quota waiting does not consume the subsequent per-model metadata-read budget.

Each model allows at most
64 uncached metadata reads with a 120-second budget checked before reads; individual
production JSON/header reads have their own time/size limits. ONNX structural scans
add a shared 128-range/30-second budget across all variants of a model. In-flight
requests can finish beyond a scheduling deadline. The scanner reads bounded ranges,
not complete model files; it does not interpret tensor values. No checkpoint is
downloaded to a weight cache, prepared, imported as repository code, or executed.

Exit codes:

- `0`: selected policy passed.
- `1`: metadata assertions failed. Default `--fail-on non-candidate` requires every
  row to be metadata compatible. `--fail-on unexpected` asserts supplied expectations;
  unasserted non-candidates appear as skipped JUnit cases.
- `2`: invalid input, broken snapshot or auditor error, regardless of policy.

For an exploratory inventory, use `--fail-on none`; it records non-candidates without
failing the job. This does not turn them into compatible models. The authoritative
row statuses and unique counts remain unchanged.

## What the first layer establishes

It checks the model's declared architecture, task, serialization, file/component
layout, quantization metadata, access and repository-code requirements using Oryn's
current rules. Structural headers are fetched only when an inspector requests them.
Repository inventories and hashes are declarations from the host, not verified
downloaded bytes. A metadata candidate can still fail loading or produce bad output.

This catalog audit explicitly assumes declared runtime packs/backends are available
and supplies a non-limiting planning memory budget. This keeps missing local packs
or temporary memory pressure out of the compatibility result. It is not a test of
the user's installed application or hardware. The raw inspector's `executable` field
is therefore not execution evidence. Reports always set `executionVerified=false`,
`executedModels=0`, and leave later layers `not-tested`. Resource estimates remain
estimates. Host information is recorded because some production route preferences
are device-dependent; compare runs on the same target platform.

## Execution qualification boundary

This command performs metadata inspection only. It does not download and load each
original checkpoint, validate task quality, exercise runtime lifecycle behavior or
qualify the packaged application on physical devices. Those require separate
execution checks against exact model revisions, selected files and runtime builds.

A family representative establishes an adapter path; it does not certify every
checkpoint, revision or configuration in that family. Keep metadata compatibility
and checkpoint execution results separate.

Hub interface references: [model information API](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.model_info)
and [declarative model card metadata](https://huggingface.co/docs/hub/model-cards).
