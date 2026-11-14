import { createRequire } from 'node:module';
import { pathToFileURL, URL } from 'node:url';

// Build-time dependencies belong to the Browser application workspace. Scripts
// remain at the repository root and must not rely on transitive pnpm hoisting.
const appRequire = createRequire(new URL('../apps/browser/package.json', import.meta.url));
export const { FuseState, FuseV1Options, FuseVersion, flipFuses, getCurrentFuseWire } =
  await import(pathToFileURL(appRequire.resolve('@electron/fuses')).href);
export const { signAsync } = appRequire('@electron/osx-sign');
