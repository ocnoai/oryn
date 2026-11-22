# Oryn Browser

Electron/Chromium browser with trusted Oryn browser services and a constrained website capability bridge.

See [browser behavior, feature limits and release qualification](docs/BROWSER_OPERATIONS.md).

## Ordinary browsing and privacy

Browser includes local HTML opening, private windows, selected Chrome-family/Safari
bookmark/history import, capture-access controls and observed connection/certificate information.
Embedded permissions are limited to qualified fullscreen, geolocation and display capture; embedded
camera/microphone remain denied because Chromium does not enforce delegation before Electron's
handler. Clipboard/pointer-lock/speaker-selection frame support remains unqualified. Local AI is
unavailable in private windows and local files. Ordinary-tab user-agent branding is unchanged.

Private windows use independent in-memory sessions/stores, with explicit saved-file exceptions and
a content-free register-before-write cleanup queue retried after crashes/failures. Import uses a
native sandboxed utility with read-only Chromium copies and selective Safari reads, with no archive
copy or network access. Source fixtures demonstrate isolation; real personal-profile checks remain open.
Capture indicators show permission/access rather than recording; Stop closes through trusted consent
and before-unload, retaining the indicator if closure is canceled. See the operating guide for scope.

Source implementation alone does not qualify a signed release. The bundled MLX runtime requires
macOS 26.2; older macOS versions, Intel and Windows are not qualified support targets. Clean-account
installation, real device/TCC behavior, native accessibility, legacy-profile continuity and signed
A→B updates require separate checks against the exact distributed artifact.
See the [release qualification limits](docs/BROWSER_OPERATIONS.md#release-device-policy-and-qualification-limits).
The first cookie-encryption upgrade permits only the
disclosed one-time reset of cookies that cannot safely migrate; other saved data/preferences stay
preserved and later launches preserve encrypted cookies.

## Develop

Run `./scripts/bootstrap` from the monorepo root before these component commands.
Bootstrap regenerates `vendor/ocno-oryn-protocol-0.1.0.tgz` from the included Oryn
Protocol source. The versioned Registry package and its baseline catalog are bundled
in `vendor/`; developing Browser requires no Registry server or Registry source checkout.

```bash
pnpm install
pnpm typecheck
pnpm lint
pnpm test
pnpm build
```

Source-checkout runtime testing also requires `ORYN_ENGINE_DIR` to identify an Oryn Engine checkout. Packaged builds resolve their bundled engine from application resources.

On macOS, `pnpm dev` and `pnpm test:ui` prepare an isolated Electron copy with extra file-protocol
privileges disabled. The installed Electron bundle is unchanged. Source tests use disposable
`ORYN_PROFILE_ROOT` profiles; these source lanes do not qualify signed release devices or distribution.

Output artifact publication also uses the bundled Python interpreter for a small
Browser-owned, standard-library-only helper (`build/helpers/artifact_io.py` on POSIX,
`artifact_io_windows.py` on Windows). It
runs with `-I -S -B` and securely copies outputs through retained directory/file
handles into private storage. Packaging includes both helpers in `Resources/helpers`.
Large outputs temporarily need disk space for both source and copy; copying uses
1 MiB chunks. See the [publication design and cleanup lifecycle](../../docs/artifact-publication.md)
and [architecture graph](../../docs/images/artifact-publication.svg).
Native macOS and Windows helper verification is still required before release.
Windows currently requires local fixed NTFS storage and an absolute bundled or
configured Python path. Windows Engine confinement is not yet implemented; this
backend is not a substitute for that private-store isolation boundary.

`pnpm security:verify` checks the packaged sandbox resources, patched MLX closure, and—on macOS—the
live runtime-pack filesystem boundary. Development runtime-pack path overrides additionally require
`ORYN_ALLOW_RUNTIME_OVERRIDES=1`.

The package commands also inspect `packaging/.stage`; if it still contains an older dependency
closure they fail and require `pnpm runtime:assemble` before packaging.

## Package for macOS

The initial target is Apple Silicon macOS. Runtime assembly requires compatible Oryn Engine and Oryn Protocol wheels plus the exact tested Engine revision:

```bash
export ORYN_ENGINE_WHEEL=/absolute/path/to/oryn_engine-0.1.0-py3-none-any.whl
export ORYN_PROTOCOL_WHEEL=/absolute/path/to/oryn_protocol-0.1.0-py3-none-any.whl
export ORYN_ENGINE_REVISION=<tested-git-commit>
pnpm runtime:assemble
pnpm package:mac:dev
```

`pnpm package:mac:dev` prepares the separate local-development flavor,
`Oryn Browser Local.app`, under a fresh
`apps/browser/release-local-development-<timestamp>/mac-arm64/` directory.
Its application ID is `ai.ocno.oryn.browser.localdevelopment`; the ASAR package metadata carries
`orynBuildFlavor=local-development`. Its default profile is
`<appData>/Oryn Browser Local` (`~/Library/Application Support/Oryn Browser Local` on macOS),
and application updates are always disabled. Explicit `ORYN_PROFILE_ROOT` and smoke-profile
overrides remain available for isolated qualification; use disposable directories for those checks.

This flavor preserves the exact trusted staged Engine payload, including Python and native vendor
signatures, so signed capability manifests retain their pinned native-binary hashes. Every upstream
Engine Mach-O's supported arm64 slice must already be non-hardened, have no Team ID or entitlements, and use only ad-hoc or
linker-ad-hoc flags (`0x2` or `0x20002`); a sealed inventory checks every Engine native byte.
Only Electron, the main app and other helpers are re-signed ad-hoc/non-hardened, with the two JIT
support keys (`allow-jit` and `allow-unsigned-executable-memory`). No device or
`disable-library-validation` entitlement is allowed. All nine release fuses, renderer/Engine
sandboxing, ordinary-page guards, ingress and private cleanup remain enforced.

Run the full native ABI and packaged application checks against each development artifact.
Check Engine Ready, the public bridge, website isolation, private-window cleanup, local-file
restrictions and clean shutdown using a disposable profile. Local development checks do not
establish release distribution, physical device behavior, Intel support or Gatekeeper qualification.

`pnpm package:mac:local` retains the strict release-policy lane. It assembles a hardened ad-hoc
`Oryn Browser.app` under a fresh
`apps/browser/release-hardened-<timestamp>/mac-arm64/` directory
without Developer ID signing or notarization. This ad-hoc lane supplies no Apple Team ID and does not
establish hardened native-launch or distribution qualification; use the Developer ID release lane
for a signed release candidate.
`pnpm package:mac:inspect-test` prepares a separately labelled inspector-enabled test variant; it
does not remove hardened-runtime/library-validation requirements or qualify the release artifact.
Set `ORYN_PACKAGED_APP` to the exact bundle when running `pnpm test:packaged-runtime` or
`pnpm test:packaged`. For Playwright UI qualification, point it at the labelled inspector clone
and run `pnpm test:ui`; the exact release artifact keeps its inspector fuse disabled.

To inspect a build's disk footprint, use Finder Get Info or run:

```sh
du -sh "/path/to/Oryn Browser Local.app" "/path/to/Oryn Browser Local.app/Contents/Resources"
```

Directory size does not establish native launch or release qualification.

By default, `pnpm package:mac:release` requires a Developer ID Application identity, Apple notarization
credentials, `ORYN_APPLE_TEAM_ID`, a production `ORYN_UPDATE_BASE_URL`, and
`ORYN_UPDATE_CHANNEL=stable|beta`. It enables hardened runtime, creates DMG/ZIP artifacts,
notarizes, staples, and generates size-qualified update metadata. Stable percentage metadata is
rejected unless the separately qualified `ORYN_STAGED_ROLLOUT_QUALIFIED=1` gate is present. Verify
both the app and the actual distributed ZIP with:

```bash
ORYN_APPLE_TEAM_ID=<expected-team> node scripts/verify-macos-release.mjs \
  "apps/browser/release-candidates/<version>/mac-arm64/Oryn Browser.app" \
  "apps/browser/release-candidates/<version>/Oryn-Browser-<version>-arm64.zip"
```

For a signed manual installer before update hosting is available, run
`ORYN_RELEASE_UPDATES=0 pnpm package:mac:release` with the same signing/notarization credentials and
`ORYN_APPLE_TEAM_ID`. This mode needs no update URL or channel, disables application update checks,
and records the final DMG/ZIP sizes and hashes without creating update feed metadata. Signing,
notarization, hardened runtime and final app/ZIP verification remain required. Installing a later
build with updates enabled is a manual operation.

The packaged runtime is CPython 3.12.16 with lock-pinned Oryn Engine, Oryn Protocol, PyTorch,
Transformers, and the shared MLX 0.32.4 foundation. It launches with Python isolated mode, ignores
source-checkout Engine overrides, and fails closed if its macOS Engine or runtime-pack sandbox
profile is missing. Model weights are never included and remain downloaded on demand into the
application-support profile.

The runtime uses Python 3.12. MFLUX includes Pillow 13 in its own pack; the base runtime uses
Pillow 12.3 with reviewed image preprocessing and mask output. Speech and music packs retain
their upstream-compatible Transformers and PyTorch closures independently of the base engines.

Packaging never publishes. Separate candidate and publication workflows enforce signing,
artifact checks, monotonic versions, channel isolation and reviewed qualification evidence.
See [the release runbook](docs/APP_UPDATES.md#candidate-building-and-publication) for runner,
hosting and credential configuration. The production host and signed distribution qualification
remain pending.

## Application updates

Updater-enabled signed builds use one build-fixed HTTPS directory with `latest-mac.yml` for Stable
and `beta-mac.yml` for Beta. Checks may run automatically, but every application ZIP requires the
person to choose Download for that exact version. Restart is also explicit. Development and local
unsigned builds make no update request.

Browser main owns the feed, local rollout decision, disk admission, cancellation, and safe restart.
Websites and the public SDK receive no updater capability. See [the update runbook](docs/APP_UPDATES.md)
for feed layout, release ordering, privacy controls, pausing, and recovery. A real signed A→B update
against the approved host remains a required release-qualification gate.

## Common task negotiation

Browser forwards versioned task requirements through inspection, installation and
session attachment, validates returned TaskProfiles and validates negotiated input
before artifact leases. User overrides can select any compatible model; a hard
contract mismatch cannot be overridden. Public sessions expose only the effective
task profile, while Engine retains private runtime offers and selection details.

For packaged contract checks, set `ORYN_PACKAGED_APP`, `ORYN_SMOKE_PROFILE_ROOT` and
`ORYN_SMOKE_SITE_ROOT`, then run `node scripts/run-electron-smoke.mjs` with
`ORYN_TASK_CONTRACT_SMOKE=1` and one of `ORYN_CHAT_SMOKE=1`,
`ORYN_TRANSCRIPTION_SMOKE=1` or `ORYN_SONG_SMOKE=1`. The matching reference app must
be built with the contract-aware SDK. This focused mode reports
`scope: "task-contracts"`; it does not replace the full Browser UI smoke suite.

Model requests support source-checkpoint, preferred-variant and exact-artifact intent
with file-sensitive identities. Recommendation data remains separate from runtime
trust and public task contracts. See the [SDK API reference](../sdk/docs/api-reference.md).

Trusted alternatives can be saved as exact future role preferences after local reinspection. This does not change active sessions or authorize downloads. Registry endpoint/key configuration is build-owned; the source checkout does not provision production hosting.
