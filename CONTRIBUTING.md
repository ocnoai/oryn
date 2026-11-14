# Contributing to Oryn

Start with `./scripts/bootstrap` and use the root `pnpm typecheck`, `pnpm test`
and `pnpm build` commands for the included components. Run `pnpm docs:check`
after documentation changes. Browser packaging and native qualification are
separate from these source checks.

Use the component guide for the area you change:

- [Browser](projects/oryn-browser/CONTRIBUTING.md)
- [Engine](projects/oryn-engine/CONTRIBUTING.md)
- [Protocol](projects/oryn-protocol/CONTRIBUTING.md)
- [SDK](projects/oryn-sdk/CONTRIBUTING.md)
- [Examples](projects/oryn-examples/CONTRIBUTING.md)

Preserve the boundaries between websites, trusted Browser services and Engine
execution. Protocol changes require coordinated schemas, fixtures and both
language bindings. SDK changes require API/runtime checks and an explicit version
choice before npm publication. Never replace different bytes under an existing
published package version.

Registry is supplied as a versioned compiled dependency with its baseline catalog.
Keep its source-maintenance and artifact-release process separate; do not create
a second editable catalog in Browser. A changed dependency archive must have its
recorded checksum and consumer lockfile updated together.

Use disposable profiles, inert fixtures and synthetic credentials in tests.
Do not commit local credentials, model weights, generated installers or personal
profile data. Report security-sensitive findings through [SECURITY](SECURITY.md).
