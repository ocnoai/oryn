# Packaged SDK dependency

The root `./scripts/bootstrap` builds `ocno-oryn-sdk-0.1.0.tgz` from the included
Oryn SDK source and refreshes the Examples consumer lockfile. The examples use
this package boundary rather than importing Browser or Engine internals.

Original SDK code is MIT-licensed. Preserve package licenses and attribution
notices when redistributing it; do not replace different bytes under an already
published npm version. See the root [dependency notices](../../../THIRD_PARTY_NOTICES.md).
