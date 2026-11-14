import { JsonFileStore, isRecord } from './json-file-store';

interface PersistedHuggingFaceCredential {
  version: 1;
  encryptedToken?: string;
}

export interface CredentialProtection {
  isAvailable(): boolean;
  encrypt(value: string): Buffer;
  decrypt(value: Buffer): string;
}

export interface HuggingFaceCredentialStatus {
  configured: boolean;
  secureStorageAvailable: boolean;
}

const TOKEN_PATTERN = /^hf_[A-Za-z0-9]{20,512}$/u;

export class HuggingFaceCredentialStore {
  private readonly store: JsonFileStore<PersistedHuggingFaceCredential>;

  constructor(
    filePath: string,
    private readonly protection: CredentialProtection,
  ) {
    this.store = new JsonFileStore(
      filePath,
      () => ({ version: 1 }),
      isPersistedHuggingFaceCredential,
    );
  }

  status(): HuggingFaceCredentialStatus {
    return {
      configured: Boolean(this.store.read().encryptedToken),
      secureStorageAvailable: this.protection.isAvailable(),
    };
  }

  save(token: string): HuggingFaceCredentialStatus {
    if (!TOKEN_PATTERN.test(token)) throw new Error('HUGGING_FACE_TOKEN_INVALID');
    if (!this.protection.isAvailable()) throw new Error('SECURE_STORAGE_UNAVAILABLE');
    const encryptedToken = this.protection.encrypt(token).toString('base64');
    this.store.write({ version: 1, encryptedToken });
    return this.status();
  }

  clear(): HuggingFaceCredentialStatus {
    this.store.write({ version: 1 });
    return this.status();
  }

  token(): string | undefined {
    const encryptedToken = this.store.read().encryptedToken;
    if (!encryptedToken) return undefined;
    if (!this.protection.isAvailable()) throw new Error('SECURE_STORAGE_UNAVAILABLE');
    const token = this.protection.decrypt(Buffer.from(encryptedToken, 'base64'));
    if (!TOKEN_PATTERN.test(token)) throw new Error('HUGGING_FACE_TOKEN_INVALID');
    return token;
  }
}

function isPersistedHuggingFaceCredential(value: unknown): value is PersistedHuggingFaceCredential {
  return (
    isRecord(value) &&
    value.version === 1 &&
    (value.encryptedToken === undefined ||
      (typeof value.encryptedToken === 'string' && value.encryptedToken.length <= 4096))
  );
}
