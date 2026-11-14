# Security

Do not post vulnerability details, exploit material or credentials in public
issues or pull requests.

Use the repository's **Security → Report a vulnerability** option when GitHub
private vulnerability reporting is enabled. The intended repository is
[ocnoai/oryn](https://github.com/ocnoai/oryn/security). If that option is unavailable,
contact the repository owner privately before sending sensitive details. This
source export does not enable GitHub account features or establish a public
security email address.

Include the source revision or package version, affected component, expected
boundary and a minimal reproduction using synthetic data. Do not probe a deployed
service, other users' data or production credentials to demonstrate a finding.

Oryn is a developer alpha. Source tests and local fixtures are scoped evidence;
they are not a complete security audit or qualification of a distributed app.
Packaged macOS operation requires the checked-in Engine/worker sandboxes and
verified signed artifacts. Windows process confinement and other unqualified
platform paths must not be treated as equivalent protection.

See the public [architecture](docs/architecture.md), [artifact boundary](docs/artifact-publication.md),
[Browser security notes](projects/oryn-browser/SECURITY.md) and
[Engine security notes](projects/oryn-engine/SECURITY.md). Release signing keys,
credentials and live hosting configuration are supplied separately.
