# Browser operation and qualification

The source includes ordinary site security, local HTML opening, selected browser-data
import and private windows. See [release qualification limits](#release-device-policy-and-qualification-limits)
for the distinction between source fixtures, native checks and distributed artifacts.
Final installation, native device access, legacy profiles, real-account continuity,
accessibility and signed updates require their own qualification. Synthetic fixtures
alone do not qualify an everyday-browser or public signed release.

## Closing, downloads and updates

Tab, workspace, window and application closure use Chromium's native before-unload path. Keep page
open preserves the live document and its draft. Discard closes it. An unresponsive close offers
continued waiting or an explicit force close after ten seconds. Sites that do not expose unsaved
state cannot be detected reliably.

Bulk close is sequential: accepted earlier closures remain closed if a later page refuses. The
workspace itself is retained after refusal. Moving tabs to Main and deleting a workspace preserves
the live WebContents, document authority and in-memory work. Main cannot be deleted. Closing the
last tab creates New Tab in the same valid workspace; invalid organization references recover to Main.

Active or queued Local AI work, artifact writes, live file downloads and model-page downloads block
quit. Browser-owned IPC stops admitting new actions during application shutdown. A downloaded update
does not authorize installation. Restart and install requires separate user action and closed website
and sidebar pages. This conservative restriction avoids destructive or synthetic unload preflight.
If a page must stay open, close checks preserve it and the update remains pending.

The first upgrade that enables cookie encryption preserves cookies through safe migration wherever
possible. Cookies that cannot safely migrate may be reset and require sign-in once, disclosed
in release notes before the upgrade. Other browser data,
model weights, saved user data including generated outputs, and permission preferences stay
preserved. Subsequent launches/upgrades preserve encrypted cookies; plaintext cookie fallback copies
and disabling encryption are not alternatives. Real-profile cookie/Gmail continuity remains a gate.
For signed releases, electron-builder automatically includes
[build/release-notes.md](../apps/browser/build/release-notes.md) in update metadata. The update dialog
displays these notes before download approval and installation.

Downloads update while their panel is open. Pause/resume depends on Chromium and the server; cancellation
stops the live transfer. A live interruption remains pausable/cancellable until Chromium reports its
terminal outcome. After process restart unfinished records are interrupted, not active transfers;
retry starts a new request. Removal deletes the record, never its file. Choose download folder changes
the default destination. Save dialogs still permit choosing a different file destination. Terminal
records are capped at 1,000; progress persistence is coalesced to 250 ms.

On macOS, completion is followed by a quarantine check before Open becomes available. The pinned
Electron 45.1.0 direct/redirect and normal/private fixtures did not add quarantine, so Browser adds
the missing attribute through a held file descriptor and preserves an existing native tag. A failed
check keeps Open disabled; Retry file check explicitly reviews the current file again. Restart
rechecks completed records. Installer/script or executable-looking files require a native Open/Cancel
confirmation. File identity is checked again after the answer, and the OS receives a native file
reference bound to the held inode. Removing a download record still leaves its file intact. These
checks do not scan files or declare them malware-free.

## Permissions and stored data

Permission requests and checks share browser-derived session, committed-document, requesting-frame
and ancestor-chain identity. Unknown, ambiguous, opaque, detached or unresolvable frames and workers
deny. Top-level prompts offer Block, Allow for this page and Always allow where appropriate. Page
grants expire on document replacement or destruction; navigation invalidates pending answers.
Camera and microphone are distinct grants. Ordinary site consent never grants Local AI authority.

Embedded permissions are limited to the subset qualified in Electron 45.1.0: fullscreen,
geolocation and display capture, with Chromium delegation enforced through the full ancestor chain.
Fullscreen needs Chromium activation/delegation; location and display capture need trusted consent
that names both top-level and embedded sites. Embedded camera/microphone remain denied because the
runtime forwards those requests before enforcing delegation. Embedded clipboard read/write,
pointer lock and speaker selection remain unqualified and denied; notifications, MIDI and idle
detection remain denied by policy. Same-origin ancestry cannot bypass the frame qualification gate
or inherit authority through a foreign ancestor. No frame gains the Local AI bridge.

Remembered embedded decisions include partition, top-level site, requesting site, permission and
ancestor-origin path. Live page grants also bind exact frame/document ancestry. A remembered normal
site grant does not become an embedded or sidebar grant. Site controls expose embedded decisions and
revocation; clearing either participating site removes its matching pairs. Mutations remain bound
to the reviewed tab/document/origin/partition rather than a subsequently active tab.

Site controls bind ordinary settings to the displayed context. Popup and sound settings apply to
ordinary tabs for that origin; popup allowance still respects Chromium opener/noopener rules.
Site clearing removes the selected site's cookies, storage and permission decisions. Domain cookies
may be shared by subdomains. All-site clearing explicitly selects cookies/storage, cache or
permissions; history separately offers last hour/day/all time. These actions do not remove models,
AI Activity or retained artifacts. Sidebar apps use independent partitions and permission records.
Old origin-only records migrate to ordinary browsing only. JavaScript/cookie blocking controls remain
unimplemented. macOS device denial/restriction receives a trusted explanation and matching privacy
settings link. Actual device/TCC consent and helper necessity still need native qualification.

Screen sharing uses the system source picker on macOS 15+, never automatically chooses a display,
never remembers a source grant and denies its unsupported fallback. Capture chrome means “this page
can use your camera, microphone or screen”, not that a stream is recording now. It tracks access
including remembered-grant paths, background tabs and sidebar pages. Stop and close page asks for
trusted confirmation, revokes live and remembered access, prevents immediate reacquisition, and
uses the ordinary page-close path. A canceled close keeps the document and indicator visible; a
click or permission-store update is not termination evidence. Use macOS's own indicator to inspect
actual device capture until a qualified native stream-state API exists.

Connection details come from navigation-bound Network/Page/Audits observations, including observed
cross-process frames; response bodies are not collected. Secure requires attributed TLS/certificate
evidence and functioning issue observation. HTTP is Not secure (loopback is Local); local files and
Oryn pages have separate labels. Stale, incomplete, detached or unavailable evidence is Unavailable,
never inferred from an HTTPS prefix. Site controls show observed mixed-content outcomes and bounded,
read-only certificate details/export. Certificate errors stay blocked; the matching trusted error
page may show their certificate, with no trust action or bypass. Debugger conflicts make details
unavailable. Successful trusted-TLS/mixed-content matrix and DevTools/packaged coexistence remain
qualification gates until their individual evidence records close them.

One native per-session request gate conservatively warns for secure-to-non-loopback-HTTP document
navigations because a reliable native distinction between forms and other document navigation is
not available. It covers frames/popups and works while CDP is unavailable. Go back cancels the
original request; Send anyway releases that same paused native request, preserving method/body.
Oryn does not read, copy or reconstruct the body. A changed frame/document or destroyed page
invalidates the answer. This does not enable insecure subresources or certificate bypasses. The
same session boundary retains artifact-capability authorization through one request listener.
HTTP/proxy credentials are collected in trusted chrome and bound to the requesting window/document;
client certificates require a browser-owned choice. Authentication is not persisted by Oryn.

## Navigation, organization and bookmarks

Command+L focuses the address field; Command+T opens a tab; Command+N opens a window; Command+D bookmarks;
Command+W closes; Command+Shift+T reopens the last closed tab; Command+F finds text. Standard numeric and
relative tab shortcuts follow pinned tabs, configured island order, then ungrouped tabs. Tab context
menus offer a keyboard reorder alternative and moving a live tab into a new same-profile window.
Normal windows share ordinary cookies and profile stores; Workspaces remain organization only.
New Private Window (Command+Shift+N) creates the separate temporary context described below.

HTTP(S), recognized Oryn routes, localhost, IPv6 and Unicode hosts have deliberate routing. Colon
queries such as `site:example.com cats` search. Google, DuckDuckGo and Bing are selectable. Local
history/bookmark/tab suggestions are opt-in and produce no remote suggestion requests. Cross-class
navigation from a model workspace creates a fresh ordinary website view; websites cannot gain an
internal workspace principal by navigating to its URL. Reviewed mailto, tel, Zoom, Slack and Teams
links from verified normal top-level HTTP(S) tab/sidebar documents use a native confirmation naming
the site and installed handler. Cancel, missing handlers and stale documents launch nothing.
Subframe, unknown-initiator, private, local and internal requests remain denied. Nonweb window.open
popups are denied because the pinned public popup API does not identify their initiating frame.
Other HTTP(S) sidebar links open Oryn tabs. Unsupported schemes are not launched.

File → Open File… (Command+O), OS Open With/Finder delivery, native HTML drops and trusted address-bar
file URLs/absolute paths can open HTML, HTM, XHTML and SHTML. Local history/bookmark/restore actions
remain browser-owned entry points. Registration uses an alternate Viewer association rather than
taking over on installation. Website navigations, redirects and popups cannot open file URLs. Local
files have sandboxed renderer behavior, no Local AI preload or persistent device grants, and cannot
read other local files through fetch/XHR/iframe content. Camera, microphone, location and notifications
are denied. Following HTTP(S) from a local file creates a fresh ordinary website view. Missing files
show the normal error page. Native Finder/default-browser delivery remains a separate manual gate.

File → Save Page As… (Command+S) and the page context menu use the native Save dialog and Electron's
HTML-only saver. MHTML is not offered. The context menu also exposes source-document-bound editing,
image/link, spelling, save and print actions according to native edit flags; a later active-tab
change cannot redirect a selected action. Private pages may be saved explicitly, while their
dictionary menu cannot persist a word into the system dictionary. Saved HTML reopens with the same
local-file restrictions above and cannot inherit the original website's Local AI or device grants.
HTML-only saving does not promise a snapshot of JavaScript-generated state or offline copies of
images/styles; the native round-trip fixture verifies the static document content.

Ordinary tabs use the standard Electron user agent. Sidebar compatibility handling strips
the Electron token. Oryn detection uses bridge capability presence; there is no Chrome shim
or Google-branded client hint.

History records completed documents, including reloads. Same-document URL changes record one visit
per changed URL, including pushState/replaceState/hash/back-forward; unchanged URLs do not create a
second same-document entry. Title changes update the latest matching visit. History is separate from
AI Activity. Per-site zoom is persisted separately. Find reports match count and current match.

Bookmark folders copy recursively with independent IDs. Selecting a parent and a child does not copy
the child twice. Restoring a child restores its required ancestors. Graph operations reject cycles,
missing parents, depth above eight, more than 20,000 nodes, unsafe URLs and oversized fields.
The existing Netscape HTML bookmark import still accepts at most 2 MiB, previews the count and
requires confirmation. It assigns fresh IDs and preserves duplicates without changing existing
entries. Active content is ignored and imported URLs must be HTTP(S); bookmark export escapes
contents. Browser-owned explicit local-file bookmarks are separate from external import admission.

## Import from other browsers

Settings → Import browser data supports selected bookmarks/history from Chrome, Chromium, Brave,
Edge, Vivaldi and Arc profiles, plus a supported default search-provider mapping. Safari uses the
user's exported browsing-data ZIP rather than Full Disk Access. Older Safari can use bookmark HTML
without claiming history import. The export may contain unencrypted secrets: Oryn warns about safe
delete-after-use and never opens/extracts password, payment or extension members.

Choose a browser/profile and data types, inspect exact counts/skipped reasons, then confirm within
five minutes in that same trusted window/document. Browser-profile imports deduplicate on re-import;
bookmarks enter a named imported folder and history retains the newest entries within its cap.
Cancel changes no store. No passwords, cookies, sessions, autofill, payment data, extensions or
arbitrary settings are imported. Source files remain read-only and unchanged.

Selected Chromium files and SQLite WAL/SHM sidecars are copied as a coherent bounded snapshot.
An Electron utility process parses copies only after a native macOS sandbox denies network,
unrelated filesystem access, writes to selected copies and child processes. Native CPU/file limits,
a wall deadline and a measured RSS watchdog provide resource bounds; ignored Node heap flags are
not claimed as a boundary. Safari member count, sizes, compression ratio, CRC and paths are checked,
and only selected bookmark/history member streams are opened. Safari stays at the selected export
location: its confined worker reads that exact file read-only, validates its identity and keeps
selected member buffers in memory. Oryn never copies the ZIP into its storage or opens, decompresses,
parses or persists forbidden member streams. Main validates results before canonical store mutation.
Chromium copies are deleted after worker exit; dedicated import scratch cleanup retries
leftovers at startup. Synthetic real-Electron/native-sandbox tests have demonstrated those denials,
source-hash preservation and watchdog termination. Real personal-profile imports and full UI/native
picker qualification remain separate gates. Import is unavailable in private windows.

## Private windows

Each private window has its own nonpersistent Chromium partition and memory-backed browsing stores;
it shares no cookies, site storage, grants or auth cache with normal windows or other private windows.
Tabs cannot move across that boundary. Ordinary permission, capture, download, authentication and
connection protections still apply. A distinct Private label is visible. Local AI reports unavailable
through a capability-only bridge: private pages cannot reach Engine, model workspaces, sessions,
jobs, artifacts or AI Activity. Sidebar apps and Workspace switching are unavailable; Tab Islands
remain in memory. Multiple persistent browser profiles remain deferred.

Private browsing history, restore/recently-closed state, download lists, zoom, permissions, favicon
cache and suggestion data stay out of normal persistent stores. Suggestions may read existing
normal bookmarks/history. Explicitly saved downloads, bookmarks and exports persist by user choice;
private mode does not hide files saved to disk or activity from websites, networks, employers or ISPs.
OS links open in normal windows, and private windows are never restored after quit/crash.

A versioned content-free cleanup journal durably registers an opaque private-session namespace
before temporary disk writes. Temporary targets derive beneath a fixed private-temp root; no URLs,
titles, cookies, credentials, arbitrary paths or user-selected saves enter the journal. Close honors
before-unload and active-download decisions. Accepted close stops work, releases handles, clears
session storage/cache/auth/connections and removes its registered temporary namespace. Failed cleanup
retains the journal entry and shows a trusted warning. Startup acquires profile ownership and retries
prior-run entries before admitting private windows; an unsafe journal or failed recovery prevents
new private windows. Deletion/acknowledgement is crash-safe and repeatable, and cannot target another
live window or a saved file. This promises successful cleanup and retry, not zero temporary residue
before cleanup succeeds or forensic secure erasure. Normal Quit waits for approved private-window
closures and cleanup before exiting; interrupted or failed cleanup still uses next-startup recovery.
Private-window cleanup must be checked across ordinary closure, crashes and interrupted downloads.
Real-account and signed-release qualification remain open.

## Persistence and recovery

Writes use a flushed temporary file and atomic rename; supported Unix filesystems also flush the
directory. History and download progress are batched for up to 250 ms; a sudden process loss can omit that most recent batch. Accepted window/application closure flushes history; quit and accepted update restart flush downloads as well. Malformed data is quarantined as `.corrupt-<timestamp>` and replaced with defaults. Security
stores recover empty, never by reusing damaged grants. Permission/credential data are not inferred from
workspace restoration. I/O permission failures are surfaced rather than mistaken for corrupt JSON.

Restore includes tab URLs, order, pin/island/workspace organization and per-window active selection.
Inactive tabs load lazily. It does not persist page forms, scroll/back-forward stacks, model prompts,
conversations, sessions or capability URLs. There is no automatic memory discard of live tabs.
Renderer crashes revoke document authority and show trusted Reload recovery; reload starts a new
document. Export model results/conversations or Save As for artifacts is explicit. Exporting structured
results releases the temporary export artifact after the save operation.

For diagnosis, quit first and copy the profile before recovery experiments. Use a fresh isolated
`ORYN_PROFILE_ROOT` for reproduction; never point a test at the working profile. Retain quarantined
files locally because they may contain browsing data. When sharing diagnostics include versions,
error codes, scenario and synthetic fixture output; remove credentials, prompts, capability URLs,
history and unnecessary local paths. Settings exports diagnostic JSON only after a requested Save.
Format 2 adds build flavor and the compiled-main SHA-256, aggregate renderer-crash count and bounded
in-memory Engine failure count/reason/exit code to the existing versions, Engine state and counts.
It excludes raw errors, stderr, stacks, URLs, prompts, credentials, environment variables and profile
paths. Cancel writes no diagnostic file; there are no crash dumps or automatic uploads.

## Local-development macOS build

This build is for local Apple Silicon development. It does not qualify a signed release.
From the Browser workspace (`projects/oryn-browser/`), after the documented runtime-assembly
prerequisites, run:

```sh
pnpm package:mac:dev
```

The command uses a fresh `apps/browser/release-local-development-<timestamp>/mac-arm64/` output
and names the bundle `Oryn Browser Local.app`. Its application ID is
`ai.ocno.oryn.browser.localdevelopment`, its ASAR package metadata contains
`orynBuildFlavor=local-development`, and its default profile is `<appData>/Oryn Browser Local`
(`~/Library/Application Support/Oryn Browser Local` on macOS). Application updates are always
disabled, including when an update feed is configured. Explicit `ORYN_PROFILE_ROOT` and smoke-profile
overrides retain their existing qualification purpose; use separate disposable directories.

The dedicated `package-macos-development.mjs`, `sign-macos-development.mjs` and
`verify-macos-development.mjs` scripts preserve the exact trusted staged Engine payload, including
Python/native vendor signatures and pinned hashes. Every upstream Engine Mach-O's supported arm64 slice must be
non-hardened, without Team ID or entitlements, with only `0x2` ad-hoc or `0x20002` linker-ad-hoc flags.
A sealed source inventory compares every Engine native byte. Other Electron/main/helper binaries
are re-signed non-hardened/ad-hoc with exactly the two JIT support entitlements, `allow-jit` and
`allow-unsigned-executable-memory`; device and disable-library-validation entitlements are forbidden.
All nine release fuses remain enforced, including disabled Node inspection and extra file privileges.
Renderer/Engine sandboxes, ingress, ordinary-page guards and private cleanup remain unchanged.

Run the full native ABI and packaged application checks against each development artifact.
Verify Engine Ready, the public bridge, the absence of Node access from websites, private-window
isolation and cleanup, local-file restrictions and clean shutdown using a disposable profile.
Preserve staged Engine bytes and capability identities; do not rewrite pinned hashes or bypass
verification to make a package launch.

`package:mac:local` and `package:mac:inspect-test` retain their hardened policies and
Apple Team-ID execution requirement. The development artifact has its own identity/profile and
qualification; its successful launch does not qualify release distribution, hardened device behavior
or camera/microphone/location parity. Intel, Gatekeeper, final archive/install, legacy-profile,
real-account, physical-device, accessibility and signed-update qualification remain separate gates.

To inspect local bundle size, use Finder Get Info or run:

```sh
du -sh "/path/to/Oryn Browser Local.app" "/path/to/Oryn Browser Local.app/Contents/Resources"
```

Bundle size does not establish native signing or launch qualification.

## Release device policy and qualification limits

Release packaging uses reviewed per-executable device entitlements and branded macOS usage text.
Only tested necessary Electron helpers may receive capture authority; Engine, Python, runtime packs
and scripts never do. RunAsNode, Node environment/inspect arguments and extra file-protocol privileges
are disabled; ASAR integrity, ASAR-only loading and cookie encryption are enforced before signing.
The inspector test clone preserves application payload and differs only in the inspect fuse plus
regenerated signatures/seals. It is separate evidence from the exact release artifact.

With extra file-protocol privileges disabled, packaged trusted chrome loads its ES modules through
the browser-owned standard secure `oryn://browser/index.html` route and closed bundled asset names.
The native request boundary restricts that route to registered trusted WebContents and rejects
embedded-document access. Exact expected renderer URL, managed WebContents and main-frame IPC
checks remain required; possession of the route alone grants nothing. Website/local-file content
never becomes trusted chrome, and no file privilege fuse is restored to make modules load.

Developer ID and hardened-runtime signing, accepted notarization and stapling are
checked on exact candidate bytes. A separately sealed publisher-pinned native-release
binding preserves the original signed capabilities and maps reviewed native inputs
to the final signed files. It does not regenerate capability keys or disable hashes.

Verify the final archive and extracted app as well as the bundle itself. Real-Keychain
smoke tests use new namespaced items and do not establish legacy-profile migration.
Clean-account installation, supported OS versions, real device/TCC behavior, account
continuity, native accessibility and signed A→B updates remain separate gates.
Signing or notarization alone does not establish those properties.

The bundled runtime's selected native dependencies can require a newer macOS version than
individual app helpers. The current MLX runtime closure requires macOS 26.2; macOS 14 support
is not qualified. Native release validation compares every selected arm64 Mach-O minimum with
the declared application minimum. A valid signature or notarization ticket does not establish
compatibility with an older operating system.

## Implementation and qualification

The source implements the external handoff, native menu/save, download-safety and diagnostic
behavior described above. Exact update payload verification is implemented in Browser main and
the release lane; see [APP_UPDATES](APP_UPDATES.md). Source tests do not establish clean-account
installation, real-account continuity, physical-device access, VoiceOver or a signed A→B update.
Those checks remain open until performed against the exact distributed artifact.

## Repeatable validation

From the Browser repository:

```sh
pnpm security:verify
pnpm typecheck
pnpm lint
pnpm test
pnpm test:ui
```

The UI lane launches real Electron with disposable profiles and loopback fixtures. It covers ordinary
browsing independently of Engine availability. Playwright drives actual rendered controls; direct
APIs supply fixtures, native-dialog responses, deliberate crashes and corroborating document/store
observations. Native permission/close button interaction, system file pickers, VoiceOver and native
WebContentsView hit testing are distinct manual gates. A renderer screenshot or axe scan does not
establish those behaviors. The suite fails on zero tests, retains failure traces, uses no automatic
retries, and writes `apps/browser/test-results/electron-results.json`.

Run the same inventory against an isolated local package after packaging:

```sh
pnpm --dir apps/browser exec env ORYN_PACKAGED_APP="/path/to/Oryn Browser Local.app" playwright test
```

PR and scheduled CI run types, lint, unit/mounted tests, build and the Electron suite. Scheduled CI
repeats the deterministic inventory three times to expose timing failures; broader Engine/device/race
qualification remains separate. Release qualification additionally requires packaged critical flows, keyboard/VoiceOver,
native overlays, representative websites/devices and signed update/distribution checks.
Never replace an installed application or publish a release to produce test evidence.

The release owner checks Electron security advisories and supported release branches weekly and before
each release candidate, pins a patched Electron release, updates the lockfile, then runs security,
unit, source UI, packaged/runtime and signed-update checks as applicable. Record Electron and Chromium
versions and executable/app.asar hashes with the results. A dependency update alone does not qualify
a package or change the supported Apple Silicon/macOS platform claim.
