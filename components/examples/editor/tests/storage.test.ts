import "fake-indexeddb/auto";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  EditorStorage,
  ConflictError,
  IntegrityError,
  LockedError,
} from "../src/storage";
import type { EditorDocument } from "../src/model";
import type { PasswordDeriver } from "../src/crypto";

// Fixtures replace only expensive derivation. Persistence and authenticated
// encryption use real IndexedDB transactions and the platform Web Crypto API.
const fixtureDeriver: PasswordDeriver = async (password, salt) => {
  const input = new TextEncoder().encode(
    `test-fixture-only:${password}:${Array.from(salt).join(",")}`,
  );
  return new Uint8Array(await crypto.subtle.digest("SHA-256", input));
};
const stores: EditorStorage[] = [];
async function open(name = crypto.randomUUID()) {
  const storage = await EditorStorage.open(name, {
    derivePasswordKey: fixtureDeriver,
  });
  stores.push(storage);
  return storage;
}
const alicePassword = "Alice strong test passphrase";
const bobPassword = "Bob different test passphrase";
function withText(doc: EditorDocument, text: string): EditorDocument {
  return {
    ...doc,
    title: text,
    content: {
      type: "doc",
      content: [
        {
          type: "paragraph",
          attrs: { blockId: "block-one" },
          content: [{ type: "text", text }],
        },
      ],
    },
    ideas: { "block-one": `${text}-idea` },
  };
}
async function rawDatabase(
  name: string,
): Promise<{ users: any[]; documents: any[] }> {
  return new Promise((resolve, reject) => {
    const opening = indexedDB.open(name);
    opening.onerror = () => reject(opening.error);
    opening.onsuccess = () => {
      const db = opening.result;
      const tx = db.transaction(["users", "documents"], "readonly");
      const users = tx.objectStore("users").getAll();
      const documents = tx.objectStore("documents").getAll();
      tx.oncomplete = () => {
        db.close();
        resolve({ users: users.result, documents: documents.result });
      };
      tx.onabort = () => {
        db.close();
        reject(tx.error);
      };
    };
  });
}
async function replaceRawDocument(name: string, value: unknown): Promise<void> {
  return new Promise((resolve, reject) => {
    const opening = indexedDB.open(name);
    opening.onerror = () => reject(opening.error);
    opening.onsuccess = () => {
      const db = opening.result;
      const tx = db.transaction("documents", "readwrite");
      tx.objectStore("documents").put(value);
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onabort = () => {
        db.close();
        reject(tx.error);
      };
    };
  });
}

afterEach(() => {
  for (const storage of stores.splice(0)) storage.dispose();
  vi.restoreAllMocks();
});

describe("local Open and owner-Protected storage", () => {
  it("works without an account and never lets a stale save resurrect protected or deleted Open content", async () => {
    const storage = await open();
    const openDoc = await storage.saveDocument(
      withText(await storage.createDocument("open"), "Shared text"),
    );
    expect(storage.activeUser).toBeNull();
    expect((await storage.listDocuments())[0].title).toBe("Shared text");
    await expect(storage.createDocument("protected")).rejects.toBeInstanceOf(
      LockedError,
    );
    await storage.createUser("Alice", alicePassword);
    const protectedDoc = await storage.setProtection(
      openDoc.id,
      "protected",
      openDoc.revision,
    );
    expect(protectedDoc.revision).toBe(openDoc.revision + 1);
    await expect(
      storage.saveDocument(withText(openDoc, "Stale plaintext")),
    ).rejects.toBeInstanceOf(ConflictError);
    storage.lock();
    expect(await storage.listDocuments()).toEqual([]);
    const shared = await storage.createDocument("open");
    await storage.deleteDocument(shared.id, shared.revision);
    await expect(storage.saveDocument(shared)).rejects.toBeInstanceOf(
      ConflictError,
    );
  });

  it("keeps Alice/Bob independent, including titles, ideas and derivatives on disk", async () => {
    const name = crypto.randomUUID();
    const storage = await open(name);
    await storage.saveDocument(
      withText(await storage.createDocument("open"), "Shared-visible"),
    );
    await storage.createUser("Alice", alicePassword);
    const alice = storage.activeUser!;
    const aliceDoc = withText(
      await storage.createDocument("protected"),
      "Alice-secret",
    );
    aliceDoc.graph = {
      nodes: [{ id: "block-one", label: "Alice-secret-graph" }],
      edges: [],
    };
    await storage.saveDocument(aliceDoc);
    await storage.createUser("Bob", bobPassword);
    const bob = storage.activeUser!;
    await storage.saveDocument(
      withText(await storage.createDocument("protected"), "Bob-secret"),
    );
    expect(
      (await storage.listDocuments()).map((doc) => doc.title).sort(),
    ).toEqual(["Bob-secret", "Shared-visible"]);
    const raw = JSON.stringify(await rawDatabase(name));
    expect(raw).toContain("Shared-visible");
    expect(raw).not.toContain("Alice-secret");
    expect(raw).not.toContain("Bob-secret");
    expect(raw).not.toContain(alicePassword);
    expect(raw).not.toContain(bobPassword);
    await expect(
      storage.unlockUser(alice.id, bobPassword),
    ).rejects.toBeInstanceOf(IntegrityError);
    expect(storage.activeUser).toBeNull();
    expect((await storage.listDocuments()).map((doc) => doc.title)).toEqual([
      "Shared-visible",
    ]);
    await storage.unlockUser(alice.id, alicePassword);
    expect(
      (await storage.listDocuments()).map((doc) => doc.title).sort(),
    ).toEqual(["Alice-secret", "Shared-visible"]);
    expect(alice.id).not.toBe(bob.id);
  });

  it("locks cooperative tabs synchronously when a local identity switches, leaving Open work usable", async () => {
    const name = crypto.randomUUID();
    const first = await open(name);
    const second = await open(name);
    await first.createDocument("open");
    await first.createUser("Alice", alicePassword);
    const alice = first.activeUser!;
    await first.createDocument("protected");
    const epoch = first.epoch;
    await second.unlockUser(alice.id, alicePassword);
    expect(first.activeUser).toBeNull();
    expect(first.epoch).toBeGreaterThan(epoch);
    expect(await first.listDocuments()).toHaveLength(1);
    expect(await second.listDocuments()).toHaveLength(2);
    first.lock();
    expect(second.activeUser).toBeNull();
  });

  it("does not commit delayed private encryption after Lock", async () => {
    const storage = await open();
    await storage.createUser("Alice", alicePassword);
    const doc = await storage.createDocument("protected");
    let entered!: () => void;
    const started = new Promise<void>((resolve) => {
      entered = resolve;
    });
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const encrypt = crypto.subtle.encrypt.bind(crypto.subtle);
    vi.spyOn(crypto.subtle, "encrypt").mockImplementation(async (...args) => {
      entered();
      await gate;
      return encrypt(...args);
    });
    const pending = storage.saveDocument(
      withText(doc, "Never committed private text"),
    );
    const rejected = expect(pending).rejects.toBeInstanceOf(LockedError);
    await started;
    storage.lock();
    release();
    await rejected;
    expect(storage.activeUser).toBeNull();
  });

  it("rejects an Open clone whose source becomes Protected before its commit", async () => {
    const name = crypto.randomUUID();
    const owner = await open(name);
    const guest = await open(name);
    const source = await owner.saveDocument(
      withText(await owner.createDocument("open"), "Formerly shared"),
    );
    await owner.createUser("Alice", alicePassword);
    let entered!: () => void;
    const started = new Promise<void>((resolve) => {
      entered = resolve;
    });
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    // Pause at the persistence boundary, not by replacing its CAS implementation.
    const internal = guest as any;
    const commit = internal.commit.bind(guest);
    vi.spyOn(internal, "commit").mockImplementation(
      async (...args: unknown[]) => {
        entered();
        await gate;
        return commit(...args);
      },
    );
    const pending = guest.cloneDocument(source.id);
    const rejected = expect(pending).rejects.toBeInstanceOf(ConflictError);
    await started;
    await owner.setProtection(source.id, "protected", source.revision);
    release();
    await rejected;
    expect(await guest.listDocuments()).toEqual([]);
  });

  it("does not finish a delayed Protected-to-Open disclosure after Lock", async () => {
    const storage = await open();
    await storage.createUser("Alice", alicePassword);
    const doc = await storage.saveDocument(
      withText(await storage.createDocument("protected"), "Stay private"),
    );
    let entered!: () => void;
    const started = new Promise<void>((resolve) => {
      entered = resolve;
    });
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const decrypt = crypto.subtle.decrypt.bind(crypto.subtle);
    vi.spyOn(crypto.subtle, "decrypt").mockImplementation(async (...args) => {
      entered();
      await gate;
      return decrypt(...args);
    });
    const pending = storage.setProtection(doc.id, "open", doc.revision);
    const rejected = expect(pending).rejects.toBeInstanceOf(LockedError);
    await started;
    storage.lock();
    release();
    await rejected;
    expect(await storage.listDocuments()).toEqual([]);
  });

  it("preserves protection for clones and makes disclosure explicit through setProtection only", async () => {
    const storage = await open();
    await storage.createUser("Alice", alicePassword);
    const alice = storage.activeUser!;
    const original = await storage.saveDocument(
      withText(await storage.createDocument("protected"), "Private clone"),
    );
    const clone = await storage.cloneDocument(original.id);
    expect(clone.id).not.toBe(original.id);
    expect(clone.ownerId).toBe(alice.id);
    expect(clone.protection).toBe("protected");
    const openDoc = await storage.setProtection(
      original.id,
      "open",
      original.revision,
    );
    expect(openDoc.ownerId).toBeUndefined();
    expect(openDoc.protection).toBe("open");
    storage.lock();
    expect((await storage.listDocuments()).map((doc) => doc.id)).toEqual([
      original.id,
    ]);
  });

  it("rejects damaged ciphertext and cross-document envelope substitution", async () => {
    const name = crypto.randomUUID();
    const storage = await open(name);
    await storage.createUser("Alice", alicePassword);
    await storage.createDocument("protected");
    await storage.createDocument("protected");
    const raw = await rawDatabase(name);
    const first = raw.documents[0];
    const second = raw.documents[1];
    await replaceRawDocument(name, {
      ...first,
      keyEnvelope: second.keyEnvelope,
    });
    await expect(storage.listDocuments()).rejects.toBeInstanceOf(
      IntegrityError,
    );
    await replaceRawDocument(name, {
      ...first,
      data: {
        ...first.data,
        ciphertext: `${first.data.ciphertext[0] === "A" ? "B" : "A"}${first.data.ciphertext.slice(1)}`,
      },
    });
    await expect(storage.listDocuments()).rejects.toBeInstanceOf(
      IntegrityError,
    );
  });

  it("exports only the active owner and authenticates complete encrypted backups before restore", async () => {
    const storage = await open();
    await storage.createUser("Bob", bobPassword);
    const bob = storage.activeUser!;
    await storage.saveDocument(
      withText(await storage.createDocument("protected"), "Bob-secret"),
    );
    await storage.createDocument("open");
    await storage.createUser("Alice", alicePassword);
    await storage.saveDocument(
      withText(await storage.createDocument("protected"), "Alice-secret"),
    );
    const backup = await storage.exportBackup();
    expect(backup).not.toContain("Alice-secret");
    expect(backup).not.toContain("Bob-secret");
    expect(backup).not.toContain(bob.id);
    expect(JSON.parse(backup).documents).toHaveLength(1);
    const restored = await open();
    await expect(
      restored.restoreBackup(backup, bobPassword),
    ).rejects.toBeInstanceOf(IntegrityError);
    expect(await restored.users()).toEqual([]);
    await restored.restoreBackup(backup, alicePassword);
    expect((await restored.listDocuments()).map((doc) => doc.title)).toEqual([
      "Alice-secret",
    ]);
    const truncated = JSON.parse(backup);
    truncated.documents = [];
    const emptyTarget = await open();
    await expect(
      emptyTarget.restoreBackup(JSON.stringify(truncated), alicePassword),
    ).rejects.toBeInstanceOf(IntegrityError);
    expect(await emptyTarget.users()).toEqual([]);
    const damaged = JSON.parse(backup);
    damaged.documents[0].ownerId = bob.id;
    await expect(
      restored.restoreBackup(JSON.stringify(damaged), alicePassword),
    ).rejects.toBeInstanceOf(IntegrityError);
    expect((await restored.listDocuments()).map((doc) => doc.title)).toEqual([
      "Alice-secret",
    ]);
  });

  it("changes the password without changing data keys, and older backups cannot reset the current password", async () => {
    const storage = await open();
    await storage.createUser("Alice", alicePassword);
    const alice = storage.activeUser!;
    await storage.saveDocument(
      withText(await storage.createDocument("protected"), "Alice-secret"),
    );
    const oldBackup = await storage.exportBackup();
    const nextPassword = "A newer different passphrase";
    await expect(
      storage.changePassword("incorrect password", nextPassword),
    ).rejects.toBeInstanceOf(IntegrityError);
    await storage.changePassword(alicePassword, nextPassword);
    expect(storage.activeUser).toBeNull();
    await expect(
      storage.unlockUser(alice.id, alicePassword),
    ).rejects.toBeInstanceOf(IntegrityError);
    await storage.unlockUser(alice.id, nextPassword);
    await storage.restoreBackup(oldBackup, alicePassword);
    expect((await storage.listDocuments())[0].title).toBe("Alice-secret");
    storage.lock();
    await expect(
      storage.unlockUser(alice.id, alicePassword),
    ).rejects.toBeInstanceOf(IntegrityError);
    await storage.unlockUser(alice.id, nextPassword);
    expect((await storage.listDocuments())[0].title).toBe("Alice-secret");
  });
});
