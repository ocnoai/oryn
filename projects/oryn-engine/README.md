# Oryn Engine

Supervised Python runtime that owns technical model executability, physical model/cache state, residency, scheduling, resources, and execution for Oryn Browser.

The engine is never exposed directly to websites. Its stdout is reserved for the versioned JSON Lines protocol.

In packaged macOS operation the Browser launches the Engine through an OS filesystem sandbox and
uses the Browser-owned launcher to apply a separate no-network sandbox and resource limits to each runtime-pack worker. The Engine
accepts runtime override paths only when the source Browser explicitly enables development
overrides. Downloaded pack content is verified at extraction and again before execution.

## Model support and routing

See [common task contracts](docs/TASK_CONTRACTS.md) for the all-backend discovery loop,
negotiated inputs/results, effective profiles and model switching.

See [runtime routing](docs/RUNTIME_ROUTING.md) for Mac execution choices, evidence, fallback, and image input; see [model onboarding](docs/MODEL_ONBOARDING.md) for the build and validation workflow. Generated capabilities describe static compatibility; exact checkpoints remain unverified until the required execution evidence is reviewed.

## Develop

Run `./scripts/bootstrap` from the monorepo root first. It regenerates the versioned
Protocol wheel in `vendor/` from the included Oryn Protocol source; the wheel is a
dependency artifact rather than a separately maintained implementation.

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

No package is published by repository scripts.

Model requests support source-checkpoint, preferred-variant and exact-artifact intent
with file-sensitive identities. Recommendation data remains separate from runtime
trust and public task contracts. See the [SDK API reference](../oryn-sdk/docs/api-reference.md).

The development extra pins the reviewed Diffusers wheel in `vendor/` for native image and video tests. Browser's `packaging/diffusers-source.json` records its source revision and reproducible wheel digest. Run `uv sync --extra dev` before the complete test suite.
