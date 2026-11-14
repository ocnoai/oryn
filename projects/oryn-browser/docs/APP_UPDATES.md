# Application updates — implementation contract

Browser owns application update checks, download consent and installation handoff.
Source and unsigned development builds intentionally keep updates unavailable.
A signed build requires configured hosting, protected signing/notarization credentials,
exact artifact verification and a real signed A→B qualification before distribution.
This source distribution does not provide an approved download or update feed.

One signed application release contains Browser, Electron/Chromium, Engine and the reviewed bundled runtime closure. The updater does not update model weights, install model-specific environments or expose a website/SDK update API. Initial distribution targets macOS arm64 outside the Mac App Store. Each release's declared minimum OS must match every shipped native dependency.

The bundled MLX runtime closure requires macOS 26.2; macOS 14 support is not qualified.
Release verification rejects native dependencies above the application's declared minimum OS.
The updater also rejects an authenticated app minimum newer than the running OS before installation.

## Platform support and payload identity

The implementation targets macOS arm64 ZIPs, requires an Apple Team ID and uses native macOS
installation. Windows packaging, publisher verification, installation, Engine confinement and
native qualification are unavailable. Keep an unsupported platform's update path disabled.

Installation must require an authenticated application version that matches the exact release
the person approved and is strictly newer than the installed version. An older,
legitimately signed release must not become installable just because feed YAML
advertises a newer version. A filename or checksum from that same feed does not
authenticate the version-to-payload relationship.

Consent, version/channel policy and candidate invalidation are shared. The macOS implementation
uses native signed-bundle inspection, specified below, to bind both authenticated version fields,
application/publisher and architecture to the exact ZIP. It preserves the existing fixed YAML feed
and uses schema v2, native payload inspection, per-offer approval IDs and retained-descriptor
handoff, without an additional signed sidecar or metadata-signing service. Schema 1 cannot authorize installation. Real signed A→B and private-host qualification
remain open; source regression tests alone cannot close the exact-version binding requirement.

Qualify the actual installation path separately on disposable macOS and Windows
environments with locally signed test releases: newer valid release accepted;
older signed payload advertised as newer rejected; approved B with a different signed C payload,
same release version with a different build, altered payload, wrong publisher/application/platform/
architecture and stale authorization rejected. Include cache reuse, cancellation, restart and channel changes. Unit
tests or qualification on one OS do not establish the other OS's behavior. Keep
an unsupported platform's update path unavailable until its adapter is complete.

## Selected dependencies and release format

### Updater architecture

The application-update path is:

```text
AppUpdateService (consent, version/channel policy, disk admission, safe restart)
  → ElectronUpdaterAdapter (restricted update session, exact payload authorization)
  → electron-updater 6.8.9 / MacUpdater (download, cache and local staging feed)
  → Electron autoUpdater / Squirrel.Mac (native macOS installation handoff)
```

`ElectronUpdaterAdapter` defaults to `electronUpdater.autoUpdater`, explicitly
resets `allowDowngrade` after channel changes, and uses `monitorMacInstall` around
`quitAndInstall`. The installed dependency's `MacUpdater` itself obtains
`require('electron').autoUpdater`, creates a private loopback feed, and delegates
native staging and installation to it. The native handoff observer already exists
in Oryn to handle failure and recover browsing/Engine operation.

The adapter verifies the signed payload's embedded release/build identity before installation
and retains the verified archive descriptor through the native handoff. Signed A→B qualification
remains a separate release requirement; implementation and source tests alone do not establish it.

Source entry points: [service](../apps/browser/src/main/app-updates/app-update-service.ts),
[adapter](../apps/browser/src/main/app-updates/electron-updater-adapter.ts),
[handoff observer](../apps/browser/src/main/app-updates/mac-install-handoff.ts),
and [pinned MacUpdater source](https://github.com/electron-userland/electron-builder/blob/electron-updater@6.8.9/packages/electron-updater/src/MacUpdater.ts).
Output artifact storage has a separate [file-copy boundary](../../../docs/artifact-publication.md).

### Pinned dependencies

The current baseline is **electron-updater 6.8.9**, whose exact dependency is **builder-util-runtime 9.7.0**, with **electron-builder 26.0.12** and **Electron 45.1.0**. Keep the updater and Electron versions exactly pinned and commit the resolved lockfile. Re-review relevant source, advisories and the network tests whenever these versions change; do not silently follow a range or the 7.x prerelease documentation.

The updater uses an isolated Electron network session, supports disabling its logger, and exposes a cancellation token. Its channel setter enables downgrades, and its default metadata requests and logging use a local staging identifier. See the versioned [AppUpdater source](https://github.com/electron-userland/electron-builder/blob/electron-updater@6.8.9/packages/electron-updater/src/AppUpdater.ts) and [HTTP executor](https://github.com/electron-userland/electron-builder/blob/electron-updater@6.8.9/packages/electron-updater/src/electronHttpExecutor.ts).

Use one immutable, signed-build-configured generic-provider base:

```text
https://registry.oryn.ocno.ai/
  latest-mac.yml
  beta-mac.yml
  Oryn-Browser-<version>-arm64.zip
  Oryn-Browser-<version>-arm64.dmg
```

The configured root uses no additional `/oryn/macos/arm64/` prefix. A configured
hostname alone does not establish hosting readiness or enable an updater-disabled
manual build. Verify the deployed TLS and hosted bytes separately; placeholders
must fail production validation. Stable maps to updater channel `latest`; Beta maps to `beta`. There are no `/stable/`
or `/beta/` subdirectories and no user-supplied feed URL. Read generated `app-update.yml` normally;
do not call the public `setFeedURL` merely to switch channels. The library's private native Squirrel
loopback feed remains a separate installation detail.

Only one compatible arm64 ZIP may be selected from a candidate's files. The default update-enabled
release lane continues to build ZIP and DMG. For manual installation, a separately qualified signed,
notarized ZIP is sufficient; DMG is optional in the manual ZIP-only lane. Both paths require final
archive/extracted-app checks and clean-account installation. This distinction does not change the
updater's required ZIP payload. Disable differential downloads (`disableDifferentialDownload = true`),
so consent, byte limits, free-space admission and cancellation concern one complete ZIP, without
blockmap/fallback requests. Optimizing transfer size requires later evidence and must not weaken
these controls.

## Main-process initialization and channel invariant

Create one updater in Browser main, after Electron is ready and before any update check. Configuration order is part of the contract:

```ts
updater.logger = null;
updater.autoDownload = false;
updater.autoInstallOnAppQuit = false;
updater.disableDifferentialDownload = true;
updater.requestHeaders = {
  'x-user-staging-id': 'oryn-no-staging-id',
  'Accept-Encoding': 'identity',
};
// Install the session policy below before making any request.
updater.channel = selectedChannel === 'stable' ? 'latest' : 'beta';
updater.allowDowngrade = false; // MUST immediately follow every channel assignment.
updater.allowPrerelease = selectedChannel === 'beta';
```

This is initialization guidance, not a complete service. The constant header is identical on every installation and replaces the unique default before it reaches the HTTP executor; the network hook removes it entirely. No caller can replace `requestHeaders` later. A single main-owned channel-setting method performs the assignment/reset synchronously, with no `await` or check between them. Assert `allowDowngrade === false` immediately before every check/download/install and independently require candidate semver to exceed the installed version. Reject stale operations/candidates when changing channels. Test stable→beta→stable, including an installed beta newer than the available stable version; switching back waits, never downgrades.

## Network boundary, redirects and privacy

Use the supported `updater.netSession` session, which this version's executor uses for metadata and payload requests. Browser main owns its `webRequest` handlers exclusively. Install them before the first check; never share this session with browsing, sidebar apps or a renderer. Electron keeps only the last handler for an event, so adding another listener must not silently replace this policy. These interception points are documented in [Electron WebRequest](https://www.electronjs.org/docs/latest/api/web-request).

- `onBeforeRequest`, without a restrictive URL filter: default-deny. Allow GET only to the exact configured HTTPS origin (including port) and the canonical selected channel filename. Permit only the updater's bounded timestamp-based `noCache` query on metadata. Permit the exact validated candidate ZIP URL only while its current per-release download authorization is active. Reject URL credentials, fragments, encoded path separators/traversal, foreign origins, unapproved paths/schemes and arbitrary queries. A string-prefix host comparison is insufficient.
- `onBeforeSendHeaders`: remove `x-user-staging-id` case-insensitively, along with Cookie, Authorization, Proxy-Authorization, Referer and Origin. Allow only the closed ordinary transport headers needed for these public GETs; no model-host/private release credentials or user-agent identifier. Force identity encoding. The session must not share browser cookies or accept an HTTP/proxy authentication challenge by borrowing browser/OS credentials.
- `onHeadersReceived`: **reject every 3xx response**, even same-origin redirects, before any follow-up request. Also remove Set-Cookie, reject non-identity content encoding, and enforce the response-size rules below. Independently applying the URL gate to every new request protects against redirect handling changes. Hosting must serve final objects directly at these URLs; a redirecting storage/CDN configuration is unsupported.
- Reject non-200 payload/metadata responses and missing, duplicate/conflicting or invalid Content-Length. Metadata is at most 256 KiB and payload length must equal the consented `files[].size`. Do not accept chunked or content-encoded responses as a fallback. Keep bounded request deadlines and post-parse schema/count/string limits. The ordinary HTTP framing plus final file-size/hash checks must be exercised with truncated, excess and misleading bodies; do not rely solely on an honest server header.
- Disable raw updater logging with `logger = null` before checking. Oryn logs only its own closed event schema: phase, validated version/channel, byte counters and an allowlisted error code. Never forward dependency message/stack strings, URLs, headers, metadata blobs, UUIDs or local paths. Disable dependency debug namespaces in packaged execution; exclude `.updaterId`, policy snapshots and request headers from crash/support collection.

The local `.updaterId` used by the maintained library may remain in its documented user-data location solely for native local cohort calculation; it must not be copied into Browser policy or diagnostics. Stable percentage rollout starts **disabled by a build/release qualification flag**. It can be enabled only after actual Electron server-side capture proves absence of a unique header/value on checks, payloads, retries and denied redirects, and logs/support output are also clean. Disabling percentages alone does not remove the default ID, so **all production checks**, including beta and 100% stable, require the privacy/network controls to pass.

Before this gate passes, release CI publishes beta plus full stable offers with no `stagingPercentage`; a client receiving a partial percentage while percentage support is disabled rejects that candidate instead of treating it as a full rollout. Once qualified, wrap the supported `isUserWithinRollout` callback: validate an integer percentage in 0–100, handle 0 as nobody and 100 as everybody, and otherwise invoke the captured native local-cohort predicate. Beta metadata must omit the percentage; it is always eligible subject to version/compatibility. Manual checks use the same policy. Eligibility never grants download consent.

The remote policy excludes the updater's native Squirrel handoff: after explicit safe restart, the library serves the already downloaded ZIP from a random-port, loopback-only server to its native installer. This is neither an alternate external feed nor permission to fetch remote resources outside the guarded session. Tests must prove no native remote URL is supplied and no installer is triggered merely by downloading.

## Release size and disk admission

**Release CI augments and validates the builder-generated metadata.** Do not make client HEAD requests or guess a size. After signing/notarization/stapling and final ZIP creation, the release script measures each final artifact and writes `files[].size` as a positive safe integer byte count, preserving/rechecking URL and SHA-512. Builder 26's general metadata builder does not itself guarantee that field; inspect its [published metadata builder](https://unpkg.com/app-builder-lib@26.0.12/out/publish/updateInfoBuilder.js).

The current parser and release scripts require this closed schema-v2 object in the same channel
YAML, within the existing 256 KiB bound:

```yaml
oryn:
  schemaVersion: 2
  platform: darwin
  arch: arm64
  unpackedSize: <positive-integer-byte-count>
  bundleVersion: <exact-final-signed-CFBundleVersion>
```

`bundleVersion` is nonempty ASCII of at most 64 bytes without whitespace/control characters,
copied from the final signed app's
`CFBundleVersion`. Top-level `version` must equal its `CFBundleShortVersionString` exactly. Expected
application identifier and publisher come from the signed running build, never the feed.

`unpackedSize` is a conservative footprint measured by CI from the final packaged app/ZIP: account for per-file allocation rounding and native resources, and do not follow archive symlinks outside the bundle. CI validates its relation to archive entry sizes and the packaged app's allocated size. The v2 client validates the closed schema and exactly one matching ZIP before offering Download. Missing/invalid/inconsistent fields produce “Update information is incomplete” and no payload request. Bind download consent to channel, both version fields, platform/architecture, canonical ZIP URL, SHA-512, download size and unpacked size; a changed tuple needs a new offer/approval. A schema v1 record cannot satisfy the v2 installation gate.

These are release-pipeline declarations delivered through authenticated HTTPS, **not independently signed metadata** in updater 6.8.9. They authorize no code. Protect publication and retain expected app/publisher signature verification in the native installer. Do not claim feed-compromise/freeze protection or that a same-feed digest authenticates a publisher. Runtime header/file-size/hash checks detect inconsistent delivery; they do not make a compromised publisher trustworthy.

### Native payload binding

After the existing downloader verifies ZIP size and SHA-512, a Browser-owned verifier safely inspects
that exact file in bounded private scratch. Validate archive paths and links before extraction; bound
entry count, decompressed size, scratch space and processing time. Run no code from the archive.
Authenticate the extracted app with strict native signature/designated-requirement, hardened runtime
and Gatekeeper's Notarized Developer ID assessment for Oryn's signed-build-fixed app identity and
publisher. Consumer verification uses system tools and does not require Xcode/stapler. Final release
packaging retains its independent stapled-ticket check. Only then read
`CFBundleIdentifier`, `CFBundleShortVersionString`, `CFBundleVersion` and supported Mach-O architecture.
Require the expected identity, exact approved version/build/platform/architecture and a release
version strictly newer than the installed version. Its authenticated `LSMinimumSystemVersion` must
also be compatible with the current macOS version. Same-version rebuilds are not an update path.

The verification result belongs to the current consent generation and exact ZIP size/digest/file
identity. Cache reuse must pass the same gate. Before native handoff, revalidate the binding and
ensure Squirrel receives those same verified bytes, not a subsequently replaced path. Missing
evidence, candidate changes or cancellation invalidate the result and prevent installation. Clean
scratch on success, cancellation and failure; export no extracted content or scratch paths. Keep
the maintained updater's native publisher verification and installer. No new remote request, feed,
metadata-signing key or installer is authorized by this gate.

`verified-update-archive.ts` makes a private, unlinked archive copy and retains its read-only
descriptor after verification. ZIP preflight bounds paths, symlinks, entry count and decompressed
size; extraction uses system ditto confined to its scratch directory. No archive program is run.
`verified-mac-handoff.ts` intercepts only pinned MacUpdater 6.8.9's payload response while preserving
its random loopback feed/authentication. It serves the retained verified descriptor instead of
reopening the mutable updater-cache pathname. Cancellation or a changed generation closes that
authority. Re-review this narrow dependency hook before changing electron-updater.

Each checked offer receives a fresh Browser-local approval ID. Download and restart carry it
through trusted UI/preload/IPC, so an old dialog cannot approve another archive/build of the same
release version. The ID is not sent to the feed or persisted as telemetry.

An offer remains an advertised release until this native verification succeeds. Neither successful
download nor an updater `update-downloaded` event alone proves publisher/version binding. The real
qualification suite must exercise cache replacement, B/C mismatch, same-version/different-build,
wrong-app/same-publisher, wrong publisher/architecture and stale results after cancel or channel changes.

Before the first payload request, query available space on every distinct filesystem used by the updater cache, verification scratch, native temporary staging and installed app destination. Use the actual resolved locations, `statfs` available-block counts, overflow-safe arithmetic and no model deletion to manufacture space. Initial conservative requirement on each participating filesystem is:

```text
requiredFreeBytes = 3 * zipDownloadBytes + 3 * unpackedBytes + 1 GiB
```

This conservative bound allows three ZIP copies and three allocated application copies across
cache, retained verification authority and native staging/backup. Browser queries the cache, verification root, system temporary and installed-app
destination volumes before transfer/restart; the verifier also checks its scratch admission. It
releases verification extraction before native staging. Confirm measured native peak usage during
real signed qualification. If a staging location/free-space query cannot be established or space is
insufficient, block the transfer with actionable UI. Recheck before restart and during long transfers;
handle ENOSPC safely because the check cannot reserve space against other applications. Display both
download bytes and required temporary free space before consent. Cache hits still undergo native
payload verification and do not waive install-space admission.

## Mandatory cancellation and safe restart

Pass an operation-owned `CancellationToken` from electron-updater to `downloadUpdate(token)`. **Cancel download is required**, including for a multi-gigabyte transfer. On cancel, mark the generation canceled, revoke its network authorization, cancel the token, await request/file closure and remove only the operation's incomplete temporary files through the maintained cache path. Ignore racing progress/completion; cancellation cannot become `ready` or start Squirrel. Retry is a new deliberate action for the currently validated candidate; do not silently restart the payload. No resume promise is made in this version.

Keep `autoInstallOnAppQuit = false`. In the delivered macOS implementation, checksum completion
precedes the maintained updater's native signature/staging checks initiated by `quitAndInstall`.
The v2 gate additionally authenticates the payload and exact approved version before
installation authorization, and revalidates its bytes at handoff; it does not replace those native
installer checks. Label UI accurately and never infer verification from `update-downloaded` alone.
Use the existing supervisor through a coordinated restart path: active/queued work defers; page close cancellation keeps services running; only accepted safe closing reaches native installation. Actual install acceptance must reject unsigned/wrong-publisher/wrong-app payloads. Prove native errors cannot leave a usable old app or Engine irrecoverably half shut down.

## Policy ownership and user controls

App updates are **Browser Core policy**, not Local AI policy. Add a versioned `browser-core/app-updates.json` through `ProfileDataPaths` and an `AppUpdatePolicyStore` in `apps/browser/src/main/data/`, reusing the atomic JSON-store pattern. Its bounded schema covers `schemaVersion`, `automaticChecks`, `channel`, channel/version skips and reminder state. Validate it fail-closed without preventing ordinary browsing. Never store feeds, keys, installer paths, raw cohort IDs or an auto-download switch in this file. It must not share `local-ai/settings.json` or `OrynSettingsService` ownership.

Place controls in **general Browser Settings → Updates / About**, plus the native Check for Updates menu. Show channel/version, plain-text notes, download and temporary-space sizes; offer Download update, Remind me later and Skip this version. Show Cancel while downloading, Retry after a failure/cancel, and Restart to update/Later after successful download checks. Exact-version skips persist and can be reversed; newer releases can still be offered. A downloaded release never restarts automatically.

## Qualification and release operation

Do not publish until the Browser tests, runtime/security gates and real signed A→B update pass. Network capture must use the actual pinned updater/Electron, not a fake adapter: test startup/manual checks, both channels, cohort inclusion/exclusion, changed channels, first/retried/canceled payload, and redirect targets recording **zero requests**. Test a redirect to the same origin/path too, missing/forged lengths, compressed/chunked metadata, stable after beta, low space with no payload contact, and consent invalidation. Tests may use isolated fixture configuration; no test feed/certificate bypass can ship.

Use immutable artifacts and publish validated channel metadata last. Beta publication must not rewrite `latest-mac.yml`; inspect builder-generated outputs and publish only the intended channel. Qualify beta before stable. Stable percentages require the privacy gate above. Pause future offers and fix through a higher-version release; a feed change cannot retract every cached payload or guarantee remote rollback. Preserve a signed manual installer and document one-time bootstrap for existing unsigned/updater-less installs. Signing identity changes, storage migrations and skipped releases require qualification.

Source inspection does not substitute for implementation tests or signing and hosting qualification.

## Maintainer release runbook

1. Increment the application version monotonically and assemble the exact reviewed Engine/runtime
   closure. Run tests, typecheck, lint, build, security verification, packaged-runtime checks, and
   representative packaged AI workloads.
2. For an update-enabled candidate, configure protected signing/notarization credentials plus
   `ORYN_APPLE_TEAM_ID`, `ORYN_UPDATE_BASE_URL`, and `ORYN_UPDATE_CHANNEL`. Beta must not set a percentage. Stable may set
   `ORYN_UPDATE_ROLLOUT_PERCENTAGE=0..100` only together with
   `ORYN_STAGED_ROLLOUT_QUALIFIED=1` after the packaged privacy qualification has actually passed.
3. Run `pnpm package:mac:release`. The release script passes an ephemeral builder configuration,
   embeds the fixed update authority, creates signed/notarized DMG and ZIP artifacts, measures the
   final ZIP and allocated app footprint, and augments only the selected channel metadata.
4. Run `ORYN_APPLE_TEAM_ID=<team> node scripts/verify-macos-release.mjs <app> <zip>`. Inspect the
   metadata for one ZIP, SHA-512, positive measured sizes, the Oryn footprint object, the intended
   channel, and absence of a beta percentage. Verify both version fields against the final signed
   app and exercise the native-payload gate. Schema v1 cannot authorize installation.
5. Stage those immutable versioned artifacts and metadata privately at the approved isolated feed.
   Fetch the staged objects and compare hashes, Content-Length, identity encoding, direct 200
   responses and cache policy. Do not alter a user-facing channel to obtain qualification evidence.
6. On an isolated Mac profile/machine, prove signed A → newer signed B, captured requests without a
   persistent staging value, explicit download and restart, preserved profile/model data, Engine
   health, exact version/build and wrong-app/publisher rejection, interruption, and higher-version
   recovery. Complete the required artifact-bound evidence before public publication.
7. Publish immutable ZIP/DMG objects first, verify their final HTTPS bytes, then atomically replace
   only the selected channel YAML. Never replace bytes under an existing version. Qualify the final
   stable-version artifact privately before promoting those same bytes to Stable; enable a
   percentage later only if its separate privacy gate passed.

### Existing strict signing and manual-install lane

The strict signer already verifies the original reviewed runtime, records its native/source inputs,
signs nested code and seals a separate Developer-ID/publisher-pinned original-to-final hash bundle.
It preserves the original RSA capabilities, signs only the enclosing app after binding, and reruns
the byte/runtime gates. Reuse this implementation; the exact-byte Engine preservation rule for
`Oryn Browser Local.app` remains separate. See [the signer](../scripts/sign-macos.mjs),
[binding builder](../scripts/macos-native-release-bindings.mjs).

The existing manual release mode, `ORYN_RELEASE_UPDATES=0 pnpm package:mac:release`, needs no update
host/channel, leaves application updates disabled and generates no channel feed metadata. Its
default packaging still creates ZIP and DMG. A separately qualified manual ZIP-only finalization
path is also permitted; a manual artifact must record which targets it actually contains. A ZIP
alone must not be presented as output from the unchanged default two-target command. In either
case verify the final extracted app/signature/stapled ticket/runtime, exact archive hashes and
installation on a clean account without Gatekeeper overrides. Packaging does not authorize publishing.

Signing and notarization qualify only the exact bytes checked. Validate the final
ZIP and extracted app, clean-account installation, supported OS versions and legacy
profile continuity separately. A new namespaced Keychain test does not establish
migration of an existing profile, and a manual installer does not establish a
working update-enabled distribution or signed A→B update.

### Pausing offers and manual recovery

To pause a release, stop publishing new offers by atomically restoring metadata that points to the
last safe immutable version, then publish a higher-version fix. This cannot retract a ZIP already
cached on a device and must not be described as a kill switch. Keep a qualified signed manual installer available
for updater bootstrap, read-only/translocated installations, signing-identity transitions, and feed
migration recovery. Never publish from pull-request or ordinary local-package jobs; these scripts do
not publish as part of packaging. Publication is a separate qualified workflow described below.

## Lifecycle and release hardening

Channel changes invalidate both service and adapter checks. An old request must drain before a new
channel check starts. Stable independently rejects prerelease versions even if its feed is wrong.
Ready downloads survive manual checks, wake events and timers; download consent is never silently
reissued. Cancellation remains authoritative when a cache hit or late completion resolves, and
the transfer must close before retry/channel switching. Every required disk must return a valid
nonnegative available-byte count; zero space is a failure, never omitted. Cache admission includes
macOS's updater cache volume rather than substituting the Browser profile volume.

Restart is single-flight and blocks other update actions during preparation and installation.
Native errors or a ten-minute handoff timeout remove only the quit callbacks added by pinned
MacUpdater 6.8.9, unfreeze tabs, restore close guards and restart Engine. A failed native handoff
requires restarting Oryn before another installation attempt because native staging exposes no
cancellation API. A separate recovery failure is shown if Engine cannot restart. Ordinary browsing
is unfrozen in either case. The handoff observer uses native events only; electron-updater still
owns download, signature staging and installation. Requalify this listener ownership when upgrading
the dependency. Synthetic event tests do not establish real wrong-publisher rejection.

New profiles inherit the signed build's `orynUpdate.channel`. Existing saved channel choices remain
unchanged. Beta packages require `X.Y.Z-beta.N`; stable packages require `X.Y.Z`. Returning from beta
to stable waits for a newer stable version. Unsigned/updater-less users must manually install the
first qualified update-enabled build from its ZIP or DMG; no migration can add an updater remotely
to an app that has none. A signed manual build with updates disabled still requires that bootstrap.

### Candidate building and publication

`package:mac:release` now creates a fresh `apps/browser/release-candidates/<version>/` directory
(override with `ORYN_RELEASE_OUTPUT`). Existing output is rejected. The command disables builder
publication and cross-channel metadata generation, embeds the initial channel, requires final app
and ZIP signature/notarization checks, recomputes every artifact's SHA-512 and size, and writes an
artifact-bound `release-manifest.json`. The original unsigned `release/` remains local-only.

`.github/workflows/release-candidate.yml` runs only by dispatch on `main`, in the protected
`oryn-release` environment on a dedicated `[self-hosted, macOS, ARM64, oryn-release]` runner.
It takes immutable Engine/Protocol/Examples SHAs, runs source gates, assembles the runtime, builds
and verifies the signed candidate, and retains immutable workflow artifacts and source revisions.
Configure the three `ORYN_*_REPOSITORY` variables, a read-only `ORYN_SOURCE_READ_TOKEN`,
`ORYN_UPDATE_BASE_URL`, `ORYN_APPLE_TEAM_ID`, signing credentials and notarization credentials/profile.
Use fresh release runners or clean their retained runtime/candidate directories between jobs.

`.github/workflows/release-publish.yml` accepts artifacts only from successful main-branch candidate
runs. Both user-facing channels require a reviewed `docs/release-qualification/<version>.json`
record matching the exact version, signing team and ZIP SHA-512. Each required check has
`{ "passed": true, "evidence": "<retained report or capture reference>" }`. The checks are
`signedUpgrade`, `dataPreserved`, `engineHealthy`, `wrongPublisherRejected`, `interruptionRecovery`,
`networkPrivacy`, `channelIsolation`, and `securityReview`. These are reviewed attestations linked
to evidence, not a cryptographic proof or a substitute for performing the tests. No passing
production qualification record is included in the repository.

The publisher supports a static-host filesystem backend: `ORYN_RELEASE_PUBLISH_DIR` must be a
mounted directory served at the exact configured HTTPS base. The directory must support atomic
hard links/renames and exclude dotfiles from HTTP serving. It creates immutable artifacts without
overwriting existing bytes, fetches and hashes their final HTTPS URLs, and atomically replaces only
the selected channel YAML last. A shared filesystem lock and workflow concurrency serialize
publication. Retrying an identical publication rechecks the hosted bytes. Changed bytes under the
same version, downgrades, redirects, compression and incorrect lengths fail. YAML should use
`Cache-Control: no-cache`; versioned artifacts should use immutable caching. Configure protected
`oryn-publish-beta` / `oryn-publish-stable` environments with the host variables. An object-storage
host needs a conditional-write backend; a bulk directory sync is not an equivalent substitute.

```bash
pnpm release:publish <candidate-directory> beta docs/release-qualification/<version>.json
pnpm release:publish <candidate-directory> stable docs/release-qualification/<version>.json
```

For stable promotion, first qualify the final **stable-version artifact** on an isolated beta feed,
then publish those same ZIP/DMG bytes to stable. A prerelease ZIP is never relabeled as stable.
Private staging for qualification must not change a user-facing channel before its evidence passes.
The approved `https://registry.oryn.ocno.ai/` staging base still needs that isolated hosting
arrangement; no server deployment or production channel was provisioned by these scripts.

### Signed qualification harness

**Controller limitation:** `qualify-app-update.mjs` uses Playwright's Electron
launcher, whose pinned implementation requires the Node inspector. Strict release fuses disable
`EnableNodeCliInspectArguments`, so this controller cannot currently launch unchanged release bytes.
Use native/manual UI control on a dedicated disposable macOS account/machine, or implement and
review a compatible controller before relying on this script. Do not change release fuses or treat
an inspect-test app as qualification of the release. Stage beta metadata and immutable artifacts at
the signed build's feed authority without changing a user-facing channel.

The script's intended flow copies A to a disposable writable app location, uses a separate profile, checks the
exact beta offer, explicitly downloads/restarts, verifies replacement by signed B, and checks
Engine health, a real bookmark, update preferences, and profile/model-directory sentinel hashes.
It retains events and a partial qualification record. It does not download representative models
or claim quality/workload qualification from sentinels. Native relaunch is why a disposable OS account
is required in addition to a temporary profile. The harness is not run by ordinary CI or packaging.
Complete real wrong-publisher, interruption, channel, receiving-server privacy captures and the
security review separately; the publisher rejects the harness's intentionally incomplete record.

Signing and notarization credentials must be protected and provisioned for each release.
Working isolated update hosting, signed A→B evidence and real publisher/privacy
qualification remain release gates. Local regression tests and workflow definitions do not close them.
