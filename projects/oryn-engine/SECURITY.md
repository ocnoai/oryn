# Security

Do not publish security-sensitive details in public issues. Follow the [repository security reporting guidance](../../SECURITY.md).

The packaged macOS Engine receives a constructed environment and runs inside the Browser-owned
default-deny profile. The Browser-owned launcher starts Engine and runtime-pack workers as separately
sandboxed children; nested sandbox reinitialization is prohibited. Workers use a no-network profile, invocation-scoped
model/artifact/scratch roots, and apply CPU, file-size, descriptor, and core-dump limits. Downloaded
pack trees are content-digested after verified extraction and rechecked before activation and use.
Development runtime overrides require the Browser's explicit opt-in and are unavailable in packaged
launch. See the public [artifact boundary](../../docs/artifact-publication.md)
and [runtime routing](docs/RUNTIME_ROUTING.md) for implementation limits.
