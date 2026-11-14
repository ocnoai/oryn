import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  HuggingFaceCredentialStore,
  type CredentialProtection,
} from '../data/hugging-face-credential-store';

class FakeProtection implements CredentialProtection {
  available = true;

  isAvailable(): boolean {
    return this.available;
  }

  encrypt(value: string): Buffer {
    return Buffer.from(`sealed:${value}`, 'utf8');
  }

  decrypt(value: Buffer): string {
    return value.toString('utf8').replace(/^sealed:/u, '');
  }
}

describe('HuggingFaceCredentialStore', () => {
  it('persists only protected token bytes and returns status without the secret', () => {
    const protection = new FakeProtection();
    const store = new HuggingFaceCredentialStore(
      join(mkdtempSync(join(tmpdir(), 'oryn-hf-credential-')), 'credential.json'),
      protection,
    );
    const token = `hf_${'a'.repeat(32)}`;

    expect(store.status()).toEqual({ configured: false, secureStorageAvailable: true });
    expect(store.save(token)).toEqual({ configured: true, secureStorageAvailable: true });
    expect(store.token()).toBe(token);
    expect(store.clear()).toEqual({ configured: false, secureStorageAvailable: true });
    expect(store.token()).toBeUndefined();
  });

  it('rejects malformed tokens and unavailable secure storage', () => {
    const protection = new FakeProtection();
    const store = new HuggingFaceCredentialStore(
      join(mkdtempSync(join(tmpdir(), 'oryn-hf-credential-')), 'credential.json'),
      protection,
    );

    expect(() => store.save('not-a-token')).toThrow('HUGGING_FACE_TOKEN_INVALID');
    protection.available = false;
    expect(() => store.save(`hf_${'b'.repeat(32)}`)).toThrow('SECURE_STORAGE_UNAVAILABLE');
  });
});
