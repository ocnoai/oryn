import { IntegrityError } from "./model";

export interface KdfParameters {
  algorithm: "argon2id";
  version: 19;
  memoryKiB: number;
  iterations: number;
  parallelism: number;
}

export const DEFAULT_KDF: KdfParameters = Object.freeze({
  algorithm: "argon2id",
  version: 19,
  memoryKiB: 65_536,
  iterations: 3,
  parallelism: 1,
});

export type PasswordDeriver = (
  password: string,
  salt: Uint8Array,
  parameters: KdfParameters,
  signal?: AbortSignal,
) => Promise<Uint8Array>;

export interface CipherBox {
  iv: string;
  ciphertext: string;
}

export function assertKdf(value: unknown): asserts value is KdfParameters {
  const p = value as KdfParameters | null;
  if (
    !p ||
    p.algorithm !== "argon2id" ||
    p.version !== 19 ||
    !Number.isInteger(p.memoryKiB) ||
    p.memoryKiB < 19_456 ||
    p.memoryKiB > 131_072 ||
    !Number.isInteger(p.iterations) ||
    p.iterations < 2 ||
    p.iterations > 10 ||
    !Number.isInteger(p.parallelism) ||
    p.parallelism < 1 ||
    p.parallelism > 4
  ) {
    throw new IntegrityError(
      "Unsupported or unsafe password derivation parameters.",
    );
  }
}

export function toBase64(bytes: Uint8Array): string {
  let text = "";
  for (let offset = 0; offset < bytes.length; offset += 16_384) {
    text += String.fromCharCode(...bytes.subarray(offset, offset + 16_384));
  }
  return btoa(text);
}

export function fromBase64(
  value: unknown,
  maximum = 8 * 1024 * 1024,
): Uint8Array {
  if (
    typeof value !== "string" ||
    value.length > Math.ceil(maximum / 3) * 4 ||
    value.length % 4 !== 0 ||
    !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(
      value,
    )
  ) {
    throw new IntegrityError("Invalid encrypted record encoding.");
  }
  let decoded: string;
  try {
    decoded = atob(value);
  } catch {
    throw new IntegrityError("Invalid encrypted record encoding.");
  }
  if (decoded.length > maximum)
    throw new IntegrityError("Encrypted record is too large.");
  return Uint8Array.from(decoded, (char) => char.charCodeAt(0));
}

export function randomBytes(length: number): Uint8Array<ArrayBuffer> {
  return crypto.getRandomValues(new Uint8Array(length));
}

function buffer(bytes: Uint8Array): ArrayBuffer {
  return new Uint8Array(bytes).buffer;
}

export async function importAesKey(bytes: Uint8Array): Promise<CryptoKey> {
  if (bytes.length !== 32) throw new IntegrityError("Invalid encryption key.");
  return crypto.subtle.importKey(
    "raw",
    buffer(bytes),
    { name: "AES-GCM" },
    false,
    ["encrypt", "decrypt"],
  );
}

export async function seal(
  key: CryptoKey,
  plaintext: Uint8Array,
  aad: readonly unknown[],
): Promise<CipherBox> {
  const iv = randomBytes(12);
  const ciphertext = await crypto.subtle.encrypt(
    {
      name: "AES-GCM",
      iv: buffer(iv),
      additionalData: new TextEncoder().encode(JSON.stringify(aad)),
      tagLength: 128,
    },
    key,
    buffer(plaintext),
  );
  return { iv: toBase64(iv), ciphertext: toBase64(new Uint8Array(ciphertext)) };
}

export function assertCipherBox(
  value: unknown,
  maximum = 8 * 1024 * 1024,
): asserts value is CipherBox {
  if (!value || typeof value !== "object")
    throw new IntegrityError("Missing encrypted data.");
  const box = value as CipherBox;
  if (
    fromBase64(box.iv, 12).length !== 12 ||
    fromBase64(box.ciphertext, maximum).length < 16
  ) {
    throw new IntegrityError("Invalid encrypted data.");
  }
}

export async function unseal(
  key: CryptoKey,
  box: CipherBox,
  aad: readonly unknown[],
  maximum?: number,
): Promise<Uint8Array> {
  assertCipherBox(box, maximum);
  try {
    return new Uint8Array(
      await crypto.subtle.decrypt(
        {
          name: "AES-GCM",
          iv: buffer(fromBase64(box.iv, 12)),
          additionalData: new TextEncoder().encode(JSON.stringify(aad)),
          tagLength: 128,
        },
        key,
        buffer(fromBase64(box.ciphertext, maximum)),
      ),
    );
  } catch {
    throw new IntegrityError();
  }
}

/** One short-lived worker per request: no KDF/password state is retained after completion. */
export const derivePasswordKey: PasswordDeriver = (
  password,
  salt,
  parameters,
  signal,
) => {
  assertKdf(parameters);
  if (password.length > 1024 || salt.length !== 16)
    return Promise.reject(new IntegrityError("Invalid password or salt."));
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Unlock cancelled.", "AbortError"));
      return;
    }
    const worker = new Worker(new URL("./crypto-worker.ts", import.meta.url), {
      type: "module",
    });
    const cleanup = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      worker.terminate();
    };
    const abort = () => {
      cleanup();
      reject(new DOMException("Unlock cancelled.", "AbortError"));
    };
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error("Password derivation timed out."));
    }, 60_000);
    signal?.addEventListener("abort", abort, { once: true });
    worker.onmessage = (
      event: MessageEvent<{ result?: Uint8Array; error?: string }>,
    ) => {
      cleanup();
      if (
        event.data.result instanceof Uint8Array &&
        event.data.result.length === 32
      )
        resolve(event.data.result);
      else reject(new IntegrityError("Password derivation failed."));
    };
    worker.onerror = () => {
      cleanup();
      reject(new Error("The local encryption worker could not start."));
    };
    worker.postMessage({ password, salt, parameters });
  });
};
