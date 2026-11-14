import { readFileSync } from 'node:fs';
import { join } from 'node:path';

export type BrowserBuildFlavor = 'release' | 'local-development';

/** Read the packaged marker from the integrity-protected application metadata. */
export function readBrowserBuildFlavor(appPath: string): BrowserBuildFlavor {
  const metadata: unknown = JSON.parse(readFileSync(join(appPath, 'package.json'), 'utf8'));
  if (typeof metadata !== 'object' || metadata === null || Array.isArray(metadata)) {
    throw new Error('Invalid Browser build metadata.');
  }
  const flavor = (metadata as Record<string, unknown>).orynBuildFlavor;
  if (flavor === undefined) return 'release';
  if (flavor === 'local-development') return flavor;
  throw new Error('Unsupported Browser build flavor.');
}

export function browserApplicationName(flavor: BrowserBuildFlavor): string {
  return flavor === 'local-development' ? 'Oryn Browser Local' : 'Oryn Browser';
}
