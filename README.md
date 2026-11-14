<p align="center">
  <img src="docs/images/oryn-mark.svg" width="72" height="72" alt="Oryn" />
</p>

# Oryn

### Your browser. Your AI. Your call.

Oryn lets websites use AI running on your computer. You control model choice,
permissions and activity in the browser. Developers use a small TypeScript SDK
for text, structured results and media.

**Developer alpha · Apple Silicon macOS**

This source distribution is prepared for [ocnoai/oryn](https://github.com/ocnoai/oryn).
Source builds and local test results do not establish a signed, qualified public
installer. Native dependencies in the current runtime require macOS 26.2; support
for older macOS versions and other operating systems remains unqualified.

[SDK quickstart](projects/oryn-sdk/README.md) · [Architecture](docs/architecture.md) · [API reference](projects/oryn-sdk/docs/api-reference.md) · [Recipes](projects/oryn-sdk/docs/recipes.md)

## What is included

| Component | Responsibility |
| --- | --- |
| [Browser](projects/oryn-browser/README.md) | Browser application, trusted UI, website bridge, permissions, artifacts and Engine supervision |
| [Engine](projects/oryn-engine/README.md) | Model inspection, downloads, resources, residency, scheduling and execution |
| [Protocol](projects/oryn-protocol/README.md) | Versioned Browser–Engine messages, validation and shared task types |
| [SDK](projects/oryn-sdk/README.md) | Framework-independent website client and typed helpers |
| [Examples](projects/oryn-examples/README.md) | Ordinary websites consuming the packaged SDK |

Browser includes a versioned compiled Registry library and baseline model catalog.
Registry source and hosting are maintained separately. No live Registry server is
needed to develop Browser or use its bundled catalog. Model weights are downloaded
separately with the user's approval and are not included in this distribution.

## Develop from source

Use Apple Silicon macOS with a compatible native toolchain, Node.js 22.12 or newer,
pnpm 10.6.5, CPython 3.12.16, `uv` and Git. From the checkout root:

```sh
./scripts/bootstrap
pnpm dev
```

Bootstrap builds the included Protocol and SDK dependency artifacts and installs
the public components. It pins CPython 3.12.16 because source recognition is
qualified against that exact parser version; `uv` downloads it if needed. With
`ORYN_OFFLINE=1`, that interpreter and cached dependencies must already be available.
Bootstrap uses the bundled Registry package directly. Downloads of ordinary
development dependencies require network access.

| Command | Purpose |
| --- | --- |
| `pnpm dev` | Start Oryn with its local Engine and reference websites |
| `pnpm dev:engine` | Run the Engine independently |
| `pnpm typecheck` | Check the TypeScript packages |
| `pnpm test` | Run the component test suites |
| `pnpm build` | Build the included packages |
| `pnpm docs:check` | Check public documentation links and SDK examples |
| `pnpm vendor:verify` | Verify the checked-in dependency artifacts |

The [Browser guide](projects/oryn-browser/README.md) covers packaging and the
limits of development builds. Source checkout tests do not qualify a distributed
installer or a production update feed.

## Add local AI to a website

The SDK is published separately on [npm](https://www.npmjs.com/package/@ocno/oryn-sdk):

```sh
npm install @ocno/oryn-sdk
```

Import the client and use it from a page opened in Oryn:

```ts
import { createLocalAIClient } from "@ocno/oryn-sdk";

const localAI = createLocalAIClient();
const capabilities = await localAI.getCapabilities();
if (!capabilities.runtimeAvailable) {
  throw new Error("Open this page in Oryn with Local AI available.");
}
```

Follow the [SDK quickstart](projects/oryn-sdk/README.md) for model selection,
streaming, cancellation and cleanup. Model availability depends on the exact
checkpoint, supported runtime, available resources and user approval.

## Run the examples

After bootstrap, start an example's development server with its own `pnpm dev`
command, or serve the five static websites together:

```sh
docker compose up --build -d
```

Open them in Oryn: transcription on `http://127.0.0.1:5184`, chat on
`http://127.0.0.1:5185`, song generation on `http://127.0.0.1:5186`, image generation
on `http://127.0.0.1:5187` and video subtitles on `http://127.0.0.1:5189`.
AI runs in Oryn on the visitor's computer. These loopback ports are for local
development; production website integrations require HTTPS.

## Local execution and control

- Inference runs locally. Model discovery and downloads can use the network.
- Websites request capabilities; the person chooses what may run.
- Artifact IDs grant scoped access rather than exposing local filesystem paths.
- The SDK does not automatically upload inputs or generated results. A website's
  own authorized network actions still follow that site's privacy practices.

## Contributing and security

Read [CONTRIBUTING](CONTRIBUTING.md) before changing code and [SECURITY](SECURITY.md)
for private vulnerability reporting and qualification limits.

Original Oryn code and documentation are licensed under the [MIT License](LICENSE).
Third-party code, libraries and model artifacts retain their respective licenses
and attribution notices, including notices shipped inside dependency archives.
