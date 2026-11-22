import { randomUUID } from 'node:crypto';
import { lstatSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { isAbsolute, join, relative } from 'node:path';
import type { Cookie, Cookies, SafeStorage } from 'electron';

const STATE_NAME = 'keychain-persistence-smoke.json';
const COOKIE_NAME = 'oryn_keychain_persistence_smoke';
const COOKIE_PATH = '/__oryn_keychain_persistence__/';

export interface KeychainSmokeIdentity {
  applicationName: string;
  keychainScope: 'standard-app-item' | 'namespaced-test-item';
}

export function keychainPersistenceSmokeIdentity(
  applicationName: string,
  environment: Readonly<Record<string, string | undefined>>,
): KeychainSmokeIdentity {
  const namespace = environment.ORYN_KEYCHAIN_SMOKE_NAMESPACE;
  if (
    environment.LOCAL_AI_SMOKE_TEST !== '1' ||
    environment.ORYN_KEYCHAIN_PERSISTENCE_SMOKE !== '1' ||
    namespace === undefined
  ) {
    return { applicationName, keychainScope: 'standard-app-item' };
  }
  if (
    namespace.length !== 36 ||
    !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/iu.test(namespace)
  ) {
    throw new Error('ORYN_KEYCHAIN_SMOKE_NAMESPACE must be a canonical RFC UUID.');
  }
  return {
    applicationName: `Oryn Browser Release Verification ${namespace.toLowerCase()}`,
    keychainScope: 'namespaced-test-item',
  };
}

interface MarkerState extends KeychainSmokeIdentity {
  version: 1;
  marker: string;
  ciphertext: string;
  expirationDate: number;
}

export interface KeychainPersistenceSmokeResult extends KeychainSmokeIdentity {
  passed: true;
  phase: 'initialized' | 'verified';
  persistedAcrossLaunches: boolean;
  markerDecrypted: true;
  persistentHttpOnlyCookie: true;
  fixtureOrigin: string;
}

interface ProbeOptions {
  identity: KeychainSmokeIdentity;
  profileRoot: string;
  fixtureURL: string;
  mockKeychain: boolean;
  protection: Pick<SafeStorage, 'isEncryptionAvailable' | 'encryptString' | 'decryptString'>;
  websiteSession: {
    isPersistent(): boolean;
    getStoragePath(): string | null;
    cookies: Pick<Cookies, 'get' | 'set' | 'flushStore'>;
  };
}

/** Opt-in release smoke only. All stored values are generated, non-secret fixtures. */
export async function runKeychainPersistenceSmoke({
  identity,
  profileRoot,
  fixtureURL,
  mockKeychain,
  protection,
  websiteSession,
}: ProbeOptions): Promise<KeychainPersistenceSmokeResult> {
  if (mockKeychain) throw new Error('Keychain persistence smoke requires the real Keychain.');
  const fixture = new URL(fixtureURL);
  if (fixture.protocol !== 'http:' || fixture.hostname !== '127.0.0.1') {
    throw new Error('Keychain persistence smoke requires an HTTP fixture on 127.0.0.1.');
  }
  const storagePath = websiteSession.getStoragePath();
  if (!websiteSession.isPersistent() || !storagePath) {
    throw new Error('Keychain persistence smoke requires a persistent website session.');
  }
  const storageRelative = relative(realpathSync(profileRoot), realpathSync(storagePath));
  if (isAbsolute(storageRelative) || storageRelative.split(/[\\/]/u).includes('..')) {
    throw new Error('Keychain persistence smoke cookie storage is outside its smoke profile.');
  }

  const statePath = join(profileRoot, STATE_NAME);
  const state = readState(statePath);
  if (
    state &&
    (state.applicationName !== identity.applicationName ||
      state.keychainScope !== identity.keychainScope)
  ) {
    throw new Error(
      'Keychain persistence smoke marker belongs to a different application name or scope.',
    );
  }
  if (!protection.isEncryptionAvailable()) {
    throw new Error('Keychain persistence smoke secure storage is unavailable.');
  }
  // Exercise the same synchronous encryption format currently used for Oryn's
  // saved HF credentials, without reading or writing an actual credential.
  const marker = state?.marker ?? `oryn-keychain-smoke-${randomUUID()}`;
  const ciphertext = state
    ? Buffer.from(state.ciphertext, 'base64')
    : protection.encryptString(marker);
  if (!ciphertext.length || protection.decryptString(ciphertext) !== marker) {
    throw new Error('Keychain persistence smoke marker decryption did not match.');
  }
  const expirationDate = state?.expirationDate ?? Math.floor(Date.now() / 1000) + 7 * 86_400;
  // Cookies are scoped by host/path, never port. The reserved path keeps this
  // fake marker away from normal fixture document requests.
  let cookies = await websiteSession.cookies.get({ name: COOKIE_NAME, domain: '127.0.0.1' });
  if (!state) {
    if (cookies.length) {
      throw new Error('Keychain persistence smoke found a cookie without its marker file.');
    }
    await websiteSession.cookies.set({
      url: new URL(COOKIE_PATH, fixture.origin).href,
      name: COOKIE_NAME,
      value: marker,
      path: COOKIE_PATH,
      httpOnly: true,
      secure: false,
      sameSite: 'strict',
      expirationDate,
    });
    await websiteSession.cookies.flushStore();
    cookies = await websiteSession.cookies.get({ name: COOKIE_NAME, domain: '127.0.0.1' });
  }
  if (cookies.length !== 1 || !cookies[0] || !matchesCookie(cookies[0], marker, expirationDate)) {
    throw new Error('Keychain persistence smoke persisted cookie is missing or does not match.');
  }
  if (!state) {
    writeFileSync(
      statePath,
      JSON.stringify({
        version: 1,
        marker,
        ciphertext: ciphertext.toString('base64'),
        expirationDate,
        ...identity,
      }),
      { mode: 0o600, flag: 'wx' },
    );
  }
  return {
    ...identity,
    passed: true,
    phase: state ? 'verified' : 'initialized',
    persistedAcrossLaunches: Boolean(state),
    markerDecrypted: true,
    persistentHttpOnlyCookie: true,
    fixtureOrigin: fixture.origin,
  };
}

function readState(path: string): MarkerState | undefined {
  let stat;
  try {
    stat = lstatSync(path);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
    throw error;
  }
  if (!stat.isFile() || (stat.mode & 0o777) !== 0o600 || stat.size > 16_384) {
    throw new Error('Keychain persistence smoke marker file is invalid.');
  }
  const state: unknown = JSON.parse(readFileSync(path, 'utf8'));
  if (
    typeof state !== 'object' ||
    state === null ||
    !('version' in state) ||
    state.version !== 1 ||
    !('applicationName' in state) ||
    typeof state.applicationName !== 'string' ||
    !('keychainScope' in state) ||
    (state.keychainScope !== 'standard-app-item' &&
      state.keychainScope !== 'namespaced-test-item') ||
    !('marker' in state) ||
    typeof state.marker !== 'string' ||
    !/^oryn-keychain-smoke-[0-9a-f-]{36}$/u.test(state.marker) ||
    !('ciphertext' in state) ||
    typeof state.ciphertext !== 'string' ||
    !state.ciphertext.length ||
    Buffer.from(state.ciphertext, 'base64').toString('base64') !== state.ciphertext ||
    !('expirationDate' in state) ||
    typeof state.expirationDate !== 'number' ||
    !Number.isInteger(state.expirationDate)
  ) {
    throw new Error('Keychain persistence smoke marker file is invalid.');
  }
  return state as MarkerState;
}

function matchesCookie(cookie: Cookie, marker: string, expirationDate: number): boolean {
  return (
    cookie.value === marker &&
    cookie.domain === '127.0.0.1' &&
    cookie.hostOnly === true &&
    cookie.path === COOKIE_PATH &&
    cookie.httpOnly === true &&
    cookie.secure === false &&
    cookie.session === false &&
    cookie.sameSite === 'strict' &&
    cookie.expirationDate === expirationDate &&
    expirationDate > Date.now() / 1000
  );
}
