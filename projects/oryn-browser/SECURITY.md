# Security

Do not publish security-sensitive details in public issues. Follow the [repository security reporting guidance](../../SECURITY.md).

Packaged macOS Engine launch is fail-closed behind the checked-in `engine.sb` profile; runtime-pack
workers receive the stricter no-network `runtime-pack.sb` profile and bounded process resources.
Trusted Browser UI is authorized by exact main-frame URL as well as WebContents identity. Chromium
permission request/check handlers use one closed permission allowlist, and loopback Local AI grants
are one-time only. Run `pnpm security:verify` before packaging. See the public [artifact boundary](../../docs/artifact-publication.md) and
[browser operation guide](docs/BROWSER_OPERATIONS.md) for implementation limits.

Application updates are controlled only by Browser main and exact trusted Browser UI main frames.
The signed build fixes one production HTTPS feed root; websites, sidebars, workspaces, model metadata,
deep links, and renderer input cannot provide a feed, payload path, or installer argument. The
dedicated updater session admits only the selected channel metadata and an explicitly authorized ZIP,
strips staging/credential headers, rejects redirects and non-identity or length-invalid responses,
and disables dependency logging. The updater's local cohort ID is not transmitted or exported.

Automatic payload downloads, differential downloads, downgrade installation, and install-on-quit
are disabled. Each exact candidate requires user approval, conservative free-space checks, and a
cancellable transfer; restart requires a second action and protected-work checks. Release artifacts
still require the expected Developer ID team, notarization/stapling, native publisher verification,
and a signed A→B qualification. See `docs/APP_UPDATES.md` for the complete boundary and residual
hosting/signing gates.
