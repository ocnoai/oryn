import { chmodSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { Cookie, CookiesSetDetails } from 'electron';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  keychainPersistenceSmokeIdentity,
  runKeychainPersistenceSmoke,
} from '../smoke-keychain-persistence';

const roots: string[] = [];
const temporaryRoot = () => {
  const root = mkdtempSync(join(tmpdir(), 'oryn-keychain-probe-test-'));
  roots.push(root);
  return root;
};

function fixture(profileRoot = temporaryRoot(), persisted: Cookie[] = []) {
  let cookies = structuredClone(persisted);
  const protection = {
    isEncryptionAvailable: vi.fn(() => true),
    encryptString: vi.fn((value: string) => Buffer.from(`sealed:${value}`)),
    decryptString: vi.fn((value: Buffer) => value.toString().replace(/^sealed:/u, '')),
  };
  const cookieStore = {
    get: vi.fn(async () => structuredClone(cookies)),
    set: vi.fn(async (details: CookiesSetDetails) => {
      cookies = [
        {
          name: details.name ?? '',
          value: details.value ?? '',
          domain: new URL(details.url).hostname,
          hostOnly: details.domain === undefined,
          path: details.path,
          httpOnly: details.httpOnly ?? false,
          secure: details.secure ?? false,
          session: details.expirationDate === undefined,
          expirationDate: details.expirationDate,
          sameSite: details.sameSite ?? 'unspecified',
        },
      ];
    }),
    flushStore: vi.fn(async () => undefined),
  };
  return {
    options: {
      identity: keychainPersistenceSmokeIdentity('Oryn Browser', {}),
      profileRoot,
      fixtureURL: 'http://127.0.0.1:8123/',
      mockKeychain: false,
      protection,
      websiteSession: {
        isPersistent: () => true,
        getStoragePath: () => profileRoot,
        cookies: cookieStore,
      },
    },
    snapshot: () => structuredClone(cookies),
    statePath: join(profileRoot, 'keychain-persistence-smoke.json'),
  };
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe('opt-in Keychain persistence smoke', () => {
  it('verifies the original marker and persistent cookie on a later launch with a different port', async () => {
    const first = fixture();
    await expect(runKeychainPersistenceSmoke(first.options)).resolves.toEqual({
      passed: true,
      phase: 'initialized',
      persistedAcrossLaunches: false,
      markerDecrypted: true,
      persistentHttpOnlyCookie: true,
      fixtureOrigin: 'http://127.0.0.1:8123',
      applicationName: 'Oryn Browser',
      keychainScope: 'standard-app-item',
    });
    expect(statSync(first.statePath).mode & 0o777).toBe(0o600);
    const savedBytes = readFileSync(first.statePath);
    const state = JSON.parse(savedBytes.toString()) as { marker: string; ciphertext: string };
    expect(state.marker).toMatch(/^oryn-keychain-smoke-/u);
    expect(Buffer.from(state.ciphertext, 'base64').toString()).toBe(`sealed:${state.marker}`);
    expect(first.options.websiteSession.cookies.flushStore).toHaveBeenCalledOnce();
    expect(first.options.websiteSession.cookies.set).toHaveBeenCalledWith(
      expect.objectContaining({
        url: 'http://127.0.0.1:8123/__oryn_keychain_persistence__/',
        path: '/__oryn_keychain_persistence__/',
        httpOnly: true,
        sameSite: 'strict',
      }),
    );
    expect(first.options.websiteSession.cookies.set.mock.calls[0]?.[0]).not.toHaveProperty(
      'domain',
    );

    // New adapters simulate a separate process: only the persisted file and
    // cookie snapshot survive, not the original JavaScript objects.
    const next = fixture(first.options.profileRoot, first.snapshot());
    next.options.fixtureURL = 'http://127.0.0.1:9345/another-fixture/';
    await expect(runKeychainPersistenceSmoke(next.options)).resolves.toMatchObject({
      phase: 'verified',
      persistedAcrossLaunches: true,
      fixtureOrigin: 'http://127.0.0.1:9345',
    });
    expect(next.options.protection.encryptString).not.toHaveBeenCalled();
    expect(next.options.protection.decryptString).toHaveBeenCalledWith(
      Buffer.from(state.ciphertext, 'base64'),
    );
    expect(next.options.websiteSession.cookies.set).not.toHaveBeenCalled();
    expect(next.options.websiteSession.cookies.get).toHaveBeenCalledWith({
      name: 'oryn_keychain_persistence_smoke',
      domain: '127.0.0.1',
    });
    expect(readFileSync(first.statePath)).toEqual(savedBytes);
  });

  it('fails closed when a later launch has lost the cookie, without recreating it', async () => {
    const first = fixture();
    await runKeychainPersistenceSmoke(first.options);
    const savedBytes = readFileSync(first.statePath);
    const next = fixture(first.options.profileRoot);
    await expect(runKeychainPersistenceSmoke(next.options)).rejects.toThrow(
      'persisted cookie is missing or does not match',
    );
    expect(next.options.websiteSession.cookies.set).not.toHaveBeenCalled();
    expect(readFileSync(first.statePath)).toEqual(savedBytes);
  });

  it('fails closed on decryption failure or mismatch, preserving both persisted fixtures', async () => {
    const first = fixture();
    await runKeychainPersistenceSmoke(first.options);
    const savedBytes = readFileSync(first.statePath);
    for (const decrypt of [
      () => 'a different marker',
      () => {
        throw new Error('key unavailable');
      },
    ]) {
      const next = fixture(first.options.profileRoot, first.snapshot());
      next.options.protection.decryptString.mockImplementation(decrypt);
      await expect(runKeychainPersistenceSmoke(next.options)).rejects.toThrow();
      expect(next.options.websiteSession.cookies.set).not.toHaveBeenCalled();
      expect(next.options.protection.encryptString).not.toHaveBeenCalled();
      expect(readFileSync(first.statePath)).toEqual(savedBytes);
    }
  });

  it.each([
    { value: 'replaced' },
    { httpOnly: false },
    { hostOnly: false },
    { session: true },
    { domain: 'localhost' },
    { path: '/' },
    { expirationDate: 1 },
  ])('rejects a changed persisted cookie %j', async (change) => {
    const first = fixture();
    await runKeychainPersistenceSmoke(first.options);
    const [cookie] = first.snapshot();
    if (!cookie) throw new Error('Fixture cookie was not initialized.');
    const next = fixture(first.options.profileRoot, [{ ...cookie, ...change }]);
    await expect(runKeychainPersistenceSmoke(next.options)).rejects.toThrow(
      'persisted cookie is missing or does not match',
    );
    expect(next.options.websiteSession.cookies.set).not.toHaveBeenCalled();
  });

  it('does not replace a malformed marker file or an orphaned cookie', async () => {
    const first = fixture();
    await runKeychainPersistenceSmoke(first.options);
    const persisted = first.snapshot();
    writeFileSync(first.statePath, '{broken');
    const next = fixture(first.options.profileRoot, persisted);
    await expect(runKeychainPersistenceSmoke(next.options)).rejects.toThrow();
    expect(readFileSync(first.statePath, 'utf8')).toBe('{broken');
    expect(next.options.websiteSession.cookies.set).not.toHaveBeenCalled();
    rmSync(first.statePath);
    await expect(runKeychainPersistenceSmoke(next.options)).rejects.toThrow(
      'cookie without its marker file',
    );
    expect(next.options.websiteSession.cookies.set).not.toHaveBeenCalled();
  });

  it('binds persisted smoke state to the same application name and Keychain scope', async () => {
    const first = fixture();
    first.options.identity = keychainPersistenceSmokeIdentity('Oryn Browser', {
      LOCAL_AI_SMOKE_TEST: '1',
      ORYN_KEYCHAIN_PERSISTENCE_SMOKE: '1',
      ORYN_KEYCHAIN_SMOKE_NAMESPACE: '75e109cb-e0fe-42da-88e9-2ee9205f71a0',
    });
    await expect(runKeychainPersistenceSmoke(first.options)).resolves.toMatchObject({
      applicationName: 'Oryn Browser Release Verification 75e109cb-e0fe-42da-88e9-2ee9205f71a0',
      keychainScope: 'namespaced-test-item',
      phase: 'initialized',
    });
    const next = fixture(first.options.profileRoot, first.snapshot());
    const originalBytes = readFileSync(first.statePath);
    await expect(runKeychainPersistenceSmoke(next.options)).rejects.toThrow(
      'different application name or scope',
    );
    expect(next.options.protection.isEncryptionAvailable).not.toHaveBeenCalled();
    expect(next.options.websiteSession.cookies.get).not.toHaveBeenCalled();
    expect(readFileSync(first.statePath)).toEqual(originalBytes);
    next.options.identity = first.options.identity;
    await expect(runKeychainPersistenceSmoke(next.options)).resolves.toMatchObject({
      keychainScope: 'namespaced-test-item',
      phase: 'verified',
      persistedAcrossLaunches: true,
    });
  });

  it('rejects a marker file whose permissions were broadened', async () => {
    const first = fixture();
    await runKeychainPersistenceSmoke(first.options);
    chmodSync(first.statePath, 0o644);
    const next = fixture(first.options.profileRoot, first.snapshot());
    await expect(runKeychainPersistenceSmoke(next.options)).rejects.toThrow(
      'marker file is invalid',
    );
    expect(next.options.protection.decryptString).not.toHaveBeenCalled();
  });

  it('rejects mock Keychain, off-profile storage, private sessions and non-loopback fixtures before access', async () => {
    const configurations = [
      (probe: ReturnType<typeof fixture>) => {
        probe.options.mockKeychain = true;
      },
      (probe: ReturnType<typeof fixture>) => {
        probe.options.websiteSession.getStoragePath = () => temporaryRoot();
      },
      (probe: ReturnType<typeof fixture>) => {
        probe.options.websiteSession.isPersistent = () => false;
      },
      (probe: ReturnType<typeof fixture>) => {
        probe.options.fixtureURL = 'https://example.com/';
      },
    ];
    for (const configure of configurations) {
      const probe = fixture();
      configure(probe);
      await expect(runKeychainPersistenceSmoke(probe.options)).rejects.toThrow();
      expect(probe.options.protection.isEncryptionAvailable).not.toHaveBeenCalled();
      expect(probe.options.websiteSession.cookies.get).not.toHaveBeenCalled();
      expect(probe.options.websiteSession.cookies.set).not.toHaveBeenCalled();
    }
  });

  it('does not write a marker if secure storage is unavailable or the cookie flush fails', async () => {
    const unavailable = fixture();
    unavailable.options.protection.isEncryptionAvailable.mockReturnValue(false);
    await expect(runKeychainPersistenceSmoke(unavailable.options)).rejects.toThrow('unavailable');
    expect(unavailable.options.websiteSession.cookies.set).not.toHaveBeenCalled();
    const failedFlush = fixture();
    failedFlush.options.websiteSession.cookies.flushStore.mockRejectedValue(new Error('disk full'));
    await expect(runKeychainPersistenceSmoke(failedFlush.options)).rejects.toThrow('disk full');
    expect(() => readFileSync(failedFlush.statePath)).toThrow();
  });
});

describe('Keychain smoke application naming', () => {
  const namespace = '75E109CB-E0FE-42DA-88E9-2EE9205F71A0';

  it('keeps ordinary release and non-persistence smoke names unchanged even when the namespace is set', () => {
    for (const environment of [
      { ORYN_KEYCHAIN_SMOKE_NAMESPACE: namespace },
      { LOCAL_AI_SMOKE_TEST: '1', ORYN_KEYCHAIN_SMOKE_NAMESPACE: namespace },
      { ORYN_KEYCHAIN_PERSISTENCE_SMOKE: '1', ORYN_KEYCHAIN_SMOKE_NAMESPACE: namespace },
      { LOCAL_AI_SMOKE_TEST: '1', ORYN_KEYCHAIN_PERSISTENCE_SMOKE: '1' },
    ]) {
      expect(keychainPersistenceSmokeIdentity('Oryn Browser', environment)).toEqual({
        applicationName: 'Oryn Browser',
        keychainScope: 'standard-app-item',
      });
    }
  });

  it('uses the same canonical UUID namespace across enabled smoke launches', () => {
    const identity = keychainPersistenceSmokeIdentity('Oryn Browser', {
      LOCAL_AI_SMOKE_TEST: '1',
      ORYN_KEYCHAIN_PERSISTENCE_SMOKE: '1',
      ORYN_KEYCHAIN_SMOKE_NAMESPACE: namespace,
    });
    expect(identity).toEqual({
      applicationName: 'Oryn Browser Release Verification 75e109cb-e0fe-42da-88e9-2ee9205f71a0',
      keychainScope: 'namespaced-test-item',
    });
  });

  it.each([
    '',
    'not-a-uuid',
    '../Oryn Browser',
    '75e109cbe0fe42da88e92ee9205f71a0',
    '75e109cb-e0fe-02da-88e9-2ee9205f71a0',
    '75e109cb-e0fe-42da-08e9-2ee9205f71a0',
    '75e109cb-e0fe-42da-88e9-2ee9205f71a0\n',
  ])('rejects a noncanonical namespace %j before sessions initialize', (invalid) => {
    expect(() =>
      keychainPersistenceSmokeIdentity('Oryn Browser', {
        LOCAL_AI_SMOKE_TEST: '1',
        ORYN_KEYCHAIN_PERSISTENCE_SMOKE: '1',
        ORYN_KEYCHAIN_SMOKE_NAMESPACE: invalid,
      }),
    ).toThrow('must be a canonical RFC UUID');
  });
});
