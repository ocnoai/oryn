import { describe, expect, it } from "vitest";
import {
  assertKdf,
  DEFAULT_KDF,
  fromBase64,
  importAesKey,
  randomBytes,
  seal,
  toBase64,
  unseal,
} from "../src/crypto";
import { IntegrityError } from "../src/model";

describe("authenticated local encryption", () => {
  it("opens the published AES-256-GCM all-zero-key/IV block vector", async () => {
    // NIST SP 800-38D AES-256-GCM test case: 128-bit zero plaintext, no AAD.
    const bytes = Uint8Array.from(
      "cea7403d4d606b6e074ec5d3baf39d18d0d1c8a799996bf0265b98b5d48ab919".match(
        /../g,
      )!,
      (value) => parseInt(value, 16),
    );
    const key = await importAesKey(new Uint8Array(32));
    // The application's seal/unseal always binds JSON-encoded AAD; verify the
    // primitive's reference vector directly, then exercise that binding below.
    const plaintext = await crypto.subtle.decrypt(
      { name: "AES-GCM", iv: new Uint8Array(12), tagLength: 128 },
      key,
      bytes,
    );
    expect(new Uint8Array(plaintext)).toEqual(new Uint8Array(16));
    expect(key.extractable).toBe(false);
    await expect(crypto.subtle.exportKey("raw", key)).rejects.toThrow();
  });

  it("authenticates ciphertext and the owner/document/revision identity", async () => {
    const key = await importAesKey(randomBytes(32));
    const aad = ["format", "alice", "document-a", 3];
    const box = await seal(key, new TextEncoder().encode("private text"), aad);
    expect(new TextDecoder().decode(await unseal(key, box, aad))).toBe(
      "private text",
    );
    await expect(
      unseal(key, box, ["format", "bob", "document-a", 3]),
    ).rejects.toBeInstanceOf(IntegrityError);
    await expect(
      unseal(key, box, ["format", "alice", "document-a", 4]),
    ).rejects.toBeInstanceOf(IntegrityError);
    const damaged = fromBase64(box.ciphertext);
    damaged[0] ^= 1;
    await expect(
      unseal(key, { ...box, ciphertext: toBase64(damaged) }, aad),
    ).rejects.toBeInstanceOf(IntegrityError);
  });

  it("uses a fresh 96-bit IV for repeated writes and rejects malformed envelopes", async () => {
    const key = await importAesKey(randomBytes(32));
    const first = await seal(key, new Uint8Array([1]), ["record"]);
    const second = await seal(key, new Uint8Array([1]), ["record"]);
    expect(fromBase64(first.iv)).toHaveLength(12);
    expect(first.iv).not.toBe(second.iv);
    await expect(
      unseal(key, { ...first, iv: toBase64(new Uint8Array(11)) }, ["record"]),
    ).rejects.toThrow();
    expect(() => fromBase64("not base64")).toThrow(IntegrityError);
    expect(() => fromBase64(toBase64(new Uint8Array(20)), 10)).toThrow(
      IntegrityError,
    );
  });

  it("bounds imported Argon2 work before computation and forbids weak/unknown KDFs", () => {
    expect(() => assertKdf(DEFAULT_KDF)).not.toThrow();
    for (const patch of [
      { algorithm: "sha256" },
      { version: 16 },
      { memoryKiB: 1024 },
      { memoryKiB: 1_000_000 },
      { iterations: 1 },
      { iterations: 100 },
      { parallelism: 32 },
    ])
      expect(() => assertKdf({ ...DEFAULT_KDF, ...patch })).toThrow(
        IntegrityError,
      );
  });
});
