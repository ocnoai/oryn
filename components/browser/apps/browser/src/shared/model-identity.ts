/** Pure renderer-safe identity: provider, repository, exact revision and selected files. */
export function modelIdentityKey(model: {
  provider: string;
  repository: string;
  revision?: string;
  files?: string[];
}): string {
  return JSON.stringify([
    model.provider,
    model.repository,
    model.revision ?? null,
    model.files ? [...model.files].sort() : null,
  ]);
}
