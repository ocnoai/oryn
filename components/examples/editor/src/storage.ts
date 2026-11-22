import {
  ConflictError,
  IntegrityError,
  LockedError,
  MAX_DOCUMENT_BYTES,
  newDocument,
  validateDocument,
  validId,
  type DocumentKind,
  type EditorDocument,
  type LocalUser,
  type Protection,
} from "./model";
import {
  assertCipherBox,
  assertKdf,
  DEFAULT_KDF,
  derivePasswordKey,
  fromBase64,
  importAesKey,
  randomBytes,
  seal,
  toBase64,
  unseal,
  type CipherBox,
  type KdfParameters,
  type PasswordDeriver,
} from "./crypto";

export { ConflictError, IntegrityError, LockedError } from "./model";

const FORMAT = "oryn-editor-v1";
// With random 96-bit IVs, this cap bounds a single key's collision probability.
// Counters are preserved when restoring into the existing identity. Random IVs
// also avoid deterministic counter reuse across independent restored copies.
const MAX_KEY_USES = 1_000_000;
const MAX_BACKUP_BYTES = 50 * 1024 * 1024;
const MAX_BACKUP_DOCUMENTS = 1000;

interface UserRecord extends LocalUser {
  keyId: string;
  passwordVersion: number;
  salt: string;
  kdf: KdfParameters;
  keyEnvelope: CipherBox;
  wraps: number;
}

interface BaseRecord {
  id: string;
  revision: number;
  classification: number;
}
interface OpenRecord extends BaseRecord {
  protection: "open";
  document: EditorDocument;
}
interface ProtectedRecord extends BaseRecord {
  protection: "protected";
  ownerId: string;
  keyId: string;
  keyEnvelope: CipherBox;
  data: CipherBox;
  writes: number;
}
interface DeletedRecord extends BaseRecord {
  protection: "deleted";
  ownerId?: string;
}
type DocumentRecord = OpenRecord | ProtectedRecord | DeletedRecord;
interface Control {
  id: "session";
  revision: number;
  ticket?: string;
  userId?: string;
}
interface Session {
  user: LocalUser;
  keyId: string;
  key: CryptoKey;
  ticket: string;
  epoch: number;
}
interface Notice {
  source: string;
  type: "lock" | "change";
}
interface Backup {
  format: typeof FORMAT;
  version: 1;
  user: UserRecord;
  documents: ProtectedRecord[];
  manifest: CipherBox;
}

export interface StorageOptions {
  /** Deterministic tests can inject a derivation function. Production uses the Argon2id worker. */
  derivePasswordKey?: PasswordDeriver;
}

function request<T>(operation: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    operation.onsuccess = () => resolve(operation.result);
    operation.onerror = () =>
      reject(operation.error ?? new Error("Local storage request failed."));
  });
}

function userAad(user: UserRecord): readonly unknown[] {
  return [
    FORMAT,
    "user-key",
    user.id,
    user.keyId,
    user.passwordVersion,
    user.kdf.algorithm,
    user.kdf.version,
    user.kdf.memoryKiB,
    user.kdf.iterations,
    user.kdf.parallelism,
    user.salt,
  ];
}
function keyAad(row: ProtectedRecord): readonly unknown[] {
  return [FORMAT, "document-key", row.ownerId, row.id, row.keyId];
}
function documentAad(row: ProtectedRecord): readonly unknown[] {
  return [
    FORMAT,
    "document",
    row.ownerId,
    row.id,
    row.keyId,
    row.revision,
    row.classification,
  ];
}

function onlyFields(value: object, fields: readonly string[]): boolean {
  return Object.keys(value).every((key) => fields.includes(key));
}

function backupAad(backup: Pick<Backup, "user">): readonly unknown[] {
  return [
    FORMAT,
    "backup-manifest",
    backup.user.id,
    backup.user.keyId,
    backup.user.passwordVersion,
  ];
}

async function backupDigest(
  backup: Pick<Backup, "user" | "documents">,
): Promise<Uint8Array> {
  return new Uint8Array(
    await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(
        JSON.stringify({ user: backup.user, documents: backup.documents }),
      ),
    ),
  );
}

function assertUser(value: unknown): asserts value is UserRecord {
  const user = value as UserRecord | null;
  if (
    !user ||
    !validId(user.id) ||
    !validId(user.keyId) ||
    typeof user.name !== "string" ||
    !user.name.trim() ||
    user.name.length > 100 ||
    !Number.isSafeInteger(user.passwordVersion) ||
    user.passwordVersion < 1 ||
    !Number.isSafeInteger(user.wraps) ||
    user.wraps < 0 ||
    user.wraps > MAX_KEY_USES ||
    !onlyFields(user, [
      "id",
      "name",
      "keyId",
      "passwordVersion",
      "salt",
      "kdf",
      "keyEnvelope",
      "wraps",
    ])
  ) {
    throw new IntegrityError("Invalid local user record.");
  }
  assertKdf(user.kdf);
  if (fromBase64(user.salt, 16).length !== 16)
    throw new IntegrityError("Invalid password salt.");
  assertCipherBox(user.keyEnvelope, 48);
  if (fromBase64(user.keyEnvelope.ciphertext, 48).length !== 48)
    throw new IntegrityError("Invalid user key envelope.");
}

function assertRecord(value: unknown): asserts value is DocumentRecord {
  const row = value as DocumentRecord | null;
  if (
    !row ||
    !validId(row.id) ||
    !Number.isSafeInteger(row.revision) ||
    row.revision < 1 ||
    !Number.isSafeInteger(row.classification) ||
    row.classification < 1
  ) {
    throw new IntegrityError("Invalid document record.");
  }
  if (row.protection === "open") {
    if (
      !onlyFields(row, [
        "id",
        "revision",
        "classification",
        "protection",
        "document",
      ])
    )
      throw new IntegrityError("Unknown Open record fields.");
    const doc = validateDocument(row.document);
    if (
      doc.id !== row.id ||
      doc.revision !== row.revision ||
      doc.protection !== "open"
    ) {
      throw new IntegrityError("Document identity mismatch.");
    }
  } else if (row.protection === "protected") {
    if (
      !validId(row.ownerId) ||
      !validId(row.keyId) ||
      !Number.isSafeInteger(row.writes) ||
      row.writes < 1 ||
      row.writes > MAX_KEY_USES ||
      !onlyFields(row, [
        "id",
        "revision",
        "classification",
        "protection",
        "ownerId",
        "keyId",
        "keyEnvelope",
        "data",
        "writes",
      ])
    )
      throw new IntegrityError("Invalid protected record.");
    assertCipherBox(row.keyEnvelope, 48);
    if (fromBase64(row.keyEnvelope.ciphertext, 48).length !== 48)
      throw new IntegrityError("Invalid document key envelope.");
    assertCipherBox(row.data, MAX_DOCUMENT_BYTES + 16);
  } else if (
    row.protection !== "deleted" ||
    (row.ownerId !== undefined && !validId(row.ownerId)) ||
    !onlyFields(row, [
      "id",
      "revision",
      "classification",
      "protection",
      "ownerId",
    ])
  ) {
    throw new IntegrityError("Unknown document protection.");
  }
}

function passwordPolicy(password: string): void {
  if (
    typeof password !== "string" ||
    password.length < 12 ||
    password.length > 1024
  ) {
    throw new Error(
      "Use a password or passphrase of at least 12 characters (maximum 1024).",
    );
  }
}

/** Local persistence only. No passwords, document content or encryption keys are sent over a network. */
export class EditorStorage {
  private static peers = new Map<string, Set<EditorStorage>>();
  private readonly listeners = new Set<() => void>();
  private readonly source = crypto.randomUUID();
  private readonly deriver: PasswordDeriver;
  private readonly pendingTransactions = new Set<IDBTransaction>();
  private readonly derivations = new Set<AbortController>();
  private channel: BroadcastChannel | null = null;
  private session: Session | null = null;
  private generation = 0;
  private disposed = false;
  private pendingLock: Promise<void> = Promise.resolve();

  private constructor(
    private readonly db: IDBDatabase,
    private readonly name: string,
    options: StorageOptions,
  ) {
    this.deriver = options.derivePasswordKey ?? derivePasswordKey;
    const peers = EditorStorage.peers.get(name) ?? new Set<EditorStorage>();
    peers.add(this);
    EditorStorage.peers.set(name, peers);
    if (typeof BroadcastChannel !== "undefined") {
      this.channel = new BroadcastChannel(`oryn-editor:${name}`);
      this.channel.onmessage = (event: MessageEvent<Notice>) => {
        if (!event.data || event.data.source === this.source) return;
        // In-process peers already received the synchronous notification.
        if (
          [...(EditorStorage.peers.get(name) ?? [])].some(
            (peer) => peer.source === event.data.source,
          )
        )
          return;
        if (event.data.type === "lock") this.clearSession();
        else if (event.data.type === "change") this.notify();
      };
    }
    db.onversionchange = () => this.dispose();
  }

  static async open(
    name = "oryn-editor",
    options: StorageOptions = {},
  ): Promise<EditorStorage> {
    const opening = indexedDB.open(name, 1);
    opening.onupgradeneeded = () => {
      const db = opening.result;
      db.createObjectStore("users", { keyPath: "id" });
      db.createObjectStore("documents", { keyPath: "id" });
      db.createObjectStore("control", { keyPath: "id" });
    };
    return new EditorStorage(await request(opening), name, options);
  }

  get activeUser(): LocalUser | null {
    return this.session ? { ...this.session.user } : null;
  }
  get epoch(): number {
    return this.generation;
  }
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify(): void {
    for (const listener of this.listeners) listener();
  }
  private broadcast(type: Notice["type"]): void {
    const notice: Notice = { source: this.source, type };
    for (const peer of EditorStorage.peers.get(this.name) ?? []) {
      if (peer !== this) {
        if (type === "lock") peer.clearSession();
        else peer.notify();
      }
    }
    this.channel?.postMessage(notice);
    this.notify();
  }

  private clearSession(): void {
    this.generation += 1;
    this.session = null;
    for (const controller of this.derivations) controller.abort();
    this.derivations.clear();
    for (const tx of this.pendingTransactions) {
      try {
        tx.abort();
      } catch {
        /* The transaction has already completed. */
      }
    }
    this.notify();
  }

  lock(): void {
    if (this.disposed) return;
    this.clearSession();
    this.broadcast("lock");
    this.pendingLock = this.transaction(
      ["control"],
      "readwrite",
      async (tx) => {
        const store = tx.objectStore("control");
        const current = await request<Control | undefined>(
          store.get("session"),
        );
        store.put({
          id: "session",
          revision: (current?.revision ?? 0) + 1,
        } satisfies Control);
      },
    );
    // Synchronous lock must clear memory even when storage is unavailable; a subsequent unlock sees the failure.
    void this.pendingLock.catch(() => {});
  }

  private assertEpoch(epoch: number): void {
    if (this.disposed || this.generation !== epoch)
      throw new LockedError(
        "The local session changed. Try again after unlocking.",
      );
  }
  private owner(ownerId?: string): Session {
    if (!this.session || (ownerId && this.session.user.id !== ownerId))
      throw new LockedError();
    this.assertEpoch(this.session.epoch);
    return this.session;
  }
  private assertSession(session: Session): void {
    this.assertEpoch(session.epoch);
    if (this.session !== session) throw new LockedError();
  }

  private async transaction<T>(
    stores: string[],
    mode: IDBTransactionMode,
    task: (tx: IDBTransaction) => Promise<T>,
    sensitive = false,
  ): Promise<T> {
    if (this.disposed) throw new Error("Local storage is closed.");
    const tx = this.db.transaction(stores, mode);
    if (sensitive) this.pendingTransactions.add(tx);
    const completion = new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onabort = () =>
        reject(
          tx.error ?? new Error("Local storage operation was interrupted."),
        );
      tx.onerror = () => {
        /* onabort reports the final transaction failure. */
      };
    });
    void completion.catch(() => {});
    try {
      const result = await task(tx);
      await completion;
      return result;
    } catch (error) {
      try {
        tx.abort();
      } catch {
        /* Already completed or aborted. */
      }
      await completion.catch(() => {});
      throw error;
    } finally {
      this.pendingTransactions.delete(tx);
    }
  }

  private async checkOwner(
    tx: IDBTransaction,
    session: Session,
  ): Promise<UserRecord> {
    this.assertSession(session);
    const [control, user] = await Promise.all([
      request<Control | undefined>(tx.objectStore("control").get("session")),
      request<UserRecord | undefined>(
        tx.objectStore("users").get(session.user.id),
      ),
    ]);
    this.assertSession(session);
    if (
      control?.ticket !== session.ticket ||
      control.userId !== session.user.id ||
      !user ||
      user.keyId !== session.keyId
    ) {
      throw new LockedError(
        "This local user was locked in another tab. Unlock again.",
      );
    }
    assertUser(user);
    return user;
  }

  private async derive(
    password: string,
    user: UserRecord,
    epoch: number,
  ): Promise<CryptoKey> {
    assertUser(user);
    const controller = new AbortController();
    this.derivations.add(controller);
    let bytes: Uint8Array | undefined;
    try {
      bytes = await this.deriver(
        password,
        fromBase64(user.salt, 16),
        user.kdf,
        controller.signal,
      );
      this.assertEpoch(epoch);
      return await importAesKey(bytes);
    } finally {
      bytes?.fill(0);
      this.derivations.delete(controller);
    }
  }

  private async userKey(
    password: string,
    user: UserRecord,
    epoch: number,
  ): Promise<CryptoKey> {
    const passwordKey = await this.derive(password, user, epoch);
    const bytes = await unseal(
      passwordKey,
      user.keyEnvelope,
      userAad(user),
      48,
    );
    try {
      this.assertEpoch(epoch);
      return await importAesKey(bytes);
    } finally {
      bytes.fill(0);
    }
  }

  private async activate(
    user: UserRecord,
    key: CryptoKey,
    epoch: number,
  ): Promise<void> {
    this.assertEpoch(epoch);
    const ticket = crypto.randomUUID();
    await this.transaction(
      ["users", "control"],
      "readwrite",
      async (tx) => {
        const [stored, control] = await Promise.all([
          request<UserRecord | undefined>(tx.objectStore("users").get(user.id)),
          request<Control | undefined>(
            tx.objectStore("control").get("session"),
          ),
        ]);
        this.assertEpoch(epoch);
        if (
          !stored ||
          stored.keyId !== user.keyId ||
          stored.passwordVersion !== user.passwordVersion ||
          stored.keyEnvelope.ciphertext !== user.keyEnvelope.ciphertext
        )
          throw new ConflictError("The local user changed. Unlock again.");
        tx.objectStore("control").put({
          id: "session",
          revision: (control?.revision ?? 0) + 1,
          ticket,
          userId: user.id,
        } satisfies Control);
      },
      true,
    );
    this.assertEpoch(epoch);
    this.session = {
      user: { id: user.id, name: user.name },
      keyId: user.keyId,
      key,
      ticket,
      epoch,
    };
    // A new identity invalidates cooperative tabs; usable keys never cross this channel.
    this.broadcast("lock");
  }

  async users(): Promise<LocalUser[]> {
    const rows = await this.transaction(["users"], "readonly", (tx) =>
      request<UserRecord[]>(tx.objectStore("users").getAll()),
    );
    return rows.map((row) => {
      assertUser(row);
      return { id: row.id, name: row.name };
    });
  }

  async createUser(name: string, password: string): Promise<void> {
    passwordPolicy(password);
    name = name.trim();
    if (!name || name.length > 100)
      throw new Error("Enter a local user name of 1–100 characters.");
    this.lock();
    await this.pendingLock;
    const epoch = this.epoch;
    const raw = randomBytes(32);
    const user: UserRecord = {
      id: crypto.randomUUID(),
      name,
      keyId: crypto.randomUUID(),
      passwordVersion: 1,
      salt: toBase64(randomBytes(16)),
      kdf: { ...DEFAULT_KDF },
      keyEnvelope: { iv: "", ciphertext: "" },
      wraps: 0,
    };
    let key: CryptoKey;
    try {
      // derive() validates an existing envelope; the new record is not encrypted yet.
      const controller = new AbortController();
      this.derivations.add(controller);
      let derived: Uint8Array | undefined;
      try {
        derived = await this.deriver(
          password,
          fromBase64(user.salt, 16),
          user.kdf,
          controller.signal,
        );
        this.assertEpoch(epoch);
        const passwordKey = await importAesKey(derived);
        user.keyEnvelope = await seal(passwordKey, raw, userAad(user));
        key = await importAesKey(raw);
      } finally {
        derived?.fill(0);
        this.derivations.delete(controller);
      }
    } finally {
      raw.fill(0);
    }
    await this.transaction(
      ["users"],
      "readwrite",
      async (tx) => {
        const users = tx.objectStore("users");
        const existing = await request<UserRecord[]>(users.getAll());
        this.assertEpoch(epoch);
        if (
          existing.some(
            (row) => row.name.toLocaleLowerCase() === name.toLocaleLowerCase(),
          )
        )
          throw new Error("A local user with that name already exists.");
        users.add(user);
      },
      true,
    );
    await this.activate(user, key!, epoch);
  }

  async unlockUser(id: string, password: string): Promise<void> {
    if (!validId(id) || typeof password !== "string" || password.length > 1024)
      throw new IntegrityError();
    this.lock();
    await this.pendingLock;
    const epoch = this.epoch;
    const user = await this.transaction(["users"], "readonly", (tx) =>
      request<UserRecord | undefined>(tx.objectStore("users").get(id)),
    );
    if (!user) throw new IntegrityError("Local user not found.");
    const key = await this.userKey(password, user, epoch);
    await this.activate(user, key, epoch);
  }

  private async row(id: string): Promise<DocumentRecord> {
    if (!validId(id)) throw new IntegrityError("Invalid document ID.");
    const row = await this.transaction(["documents"], "readonly", (tx) =>
      request<DocumentRecord | undefined>(tx.objectStore("documents").get(id)),
    );
    if (!row || row.protection === "deleted")
      throw new ConflictError(
        "The document was deleted or is no longer available.",
      );
    assertRecord(row);
    return row;
  }

  private async documentKey(
    row: ProtectedRecord,
    session: Session,
  ): Promise<CryptoKey> {
    this.assertSession(session);
    if (row.ownerId !== session.user.id) throw new LockedError();
    const bytes = await unseal(session.key, row.keyEnvelope, keyAad(row), 48);
    try {
      this.assertSession(session);
      return await importAesKey(bytes);
    } finally {
      bytes.fill(0);
    }
  }

  private async decode(
    row: DocumentRecord,
    session: Session | null = this.session,
  ): Promise<EditorDocument> {
    assertRecord(row);
    if (row.protection === "deleted")
      throw new ConflictError("This document was deleted.");
    if (row.protection === "open") return validateDocument(row.document);
    if (!session || row.ownerId !== session.user.id) throw new LockedError();
    const key = await this.documentKey(row, session);
    const plaintext = await unseal(
      key,
      row.data,
      documentAad(row),
      MAX_DOCUMENT_BYTES + 16,
    );
    let doc: EditorDocument;
    try {
      doc = validateDocument(
        JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(plaintext)),
      );
    } catch (error) {
      throw error instanceof IntegrityError
        ? error
        : new IntegrityError("Invalid encrypted document.");
    } finally {
      plaintext.fill(0);
    }
    this.assertSession(session);
    if (
      doc.id !== row.id ||
      doc.ownerId !== row.ownerId ||
      doc.revision !== row.revision ||
      doc.protection !== "protected"
    ) {
      throw new IntegrityError("Encrypted document identity mismatch.");
    }
    return doc;
  }

  async listDocuments(): Promise<EditorDocument[]> {
    const epoch = this.epoch;
    const session = this.session;
    const rows = await this.transaction(["documents"], "readonly", (tx) =>
      request<DocumentRecord[]>(tx.objectStore("documents").getAll()),
    );
    const documents: EditorDocument[] = [];
    for (const row of rows) {
      if (
        row.protection === "open" ||
        (row.protection === "protected" && session?.user.id === row.ownerId)
      ) {
        documents.push(await this.decode(row, session));
      }
    }
    this.assertEpoch(epoch);
    if (session) this.assertSession(session);
    return documents.sort((a, b) => b.updatedAt - a.updatedAt);
  }

  private async encode(
    doc: EditorDocument,
    classification: number,
    session?: Session,
    previous?: ProtectedRecord,
  ): Promise<{ row: OpenRecord | ProtectedRecord; wrapped: boolean }> {
    doc = validateDocument(doc);
    if (doc.protection === "open")
      return {
        row: {
          id: doc.id,
          revision: doc.revision,
          classification,
          protection: "open",
          document: doc,
        },
        wrapped: false,
      };
    session ??= this.owner(doc.ownerId);
    if (doc.ownerId !== session.user.id) throw new LockedError();
    let key: CryptoKey;
    let row: ProtectedRecord;
    let wrapped = false;
    if (previous && previous.writes < MAX_KEY_USES) {
      key = await this.documentKey(previous, session);
      row = {
        ...previous,
        revision: doc.revision,
        classification,
        writes: previous.writes + 1,
      };
    } else {
      const raw = randomBytes(32);
      row = {
        id: doc.id,
        revision: doc.revision,
        classification,
        protection: "protected",
        ownerId: session.user.id,
        keyId: crypto.randomUUID(),
        keyEnvelope: { iv: "", ciphertext: "" },
        data: { iv: "", ciphertext: "" },
        writes: 1,
      };
      try {
        key = await importAesKey(raw);
        row.keyEnvelope = await seal(session.key, raw, keyAad(row));
      } finally {
        raw.fill(0);
      }
      wrapped = true;
    }
    const bytes = new TextEncoder().encode(JSON.stringify(doc));
    try {
      row.data = await seal(key, bytes, documentAad(row));
    } finally {
      bytes.fill(0);
    }
    this.assertSession(session);
    return { row, wrapped };
  }

  private async commit(
    row: OpenRecord | ProtectedRecord,
    expected: DocumentRecord | null,
    session: Session | null,
    wrapped: boolean,
    source?: DocumentRecord,
  ): Promise<void> {
    await this.transaction(
      ["documents", "users", "control"],
      "readwrite",
      async (tx) => {
        const store = tx.objectStore("documents");
        const [current, user, sourceNow] = await Promise.all([
          request<DocumentRecord | undefined>(store.get(row.id)),
          session ? this.checkOwner(tx, session) : Promise.resolve(null),
          source
            ? request<DocumentRecord | undefined>(store.get(source.id))
            : Promise.resolve(null),
        ]);
        if (
          expected
            ? !current ||
              current.revision !== expected.revision ||
              current.classification !== expected.classification ||
              current.protection !== expected.protection ||
              (current.protection === "protected" &&
                expected.protection === "protected" &&
                current.ownerId !== expected.ownerId)
            : !!current
        )
          throw new ConflictError();
        if (
          source &&
          (!sourceNow ||
            sourceNow.revision !== source.revision ||
            sourceNow.classification !== source.classification ||
            sourceNow.protection !== source.protection)
        )
          throw new ConflictError();
        if (session) this.assertSession(session);
        if (wrapped && user) {
          if (user.wraps >= MAX_KEY_USES)
            throw new Error(
              "This user key reached its encryption limit. Rotate keys before continuing.",
            );
          tx.objectStore("users").put({ ...user, wraps: user.wraps + 1 });
        }
        if (expected) store.put(row);
        else store.add(row);
      },
      !!session,
    );
    if (session) this.assertSession(session);
    this.broadcast("change");
  }

  async createDocument(
    protection: Protection,
    kind: DocumentKind = "document",
  ): Promise<EditorDocument> {
    if (
      (protection !== "open" && protection !== "protected") ||
      (kind !== "document" && kind !== "reference")
    )
      throw new IntegrityError("Invalid document type.");
    const session = protection === "protected" ? this.owner() : null;
    const doc = newDocument(protection, kind, session?.user.id);
    const encoded = await this.encode(doc, 1, session ?? undefined);
    await this.commit(encoded.row, null, session, encoded.wrapped);
    return doc;
  }

  async saveDocument(value: EditorDocument): Promise<EditorDocument> {
    const doc = validateDocument(value);
    const row = await this.row(doc.id);
    if (
      row.revision !== doc.revision ||
      row.protection !== doc.protection ||
      (row.protection === "protected" && row.ownerId !== doc.ownerId)
    )
      throw new ConflictError();
    const session =
      row.protection === "protected" ? this.owner(row.ownerId) : null;
    const next = validateDocument({
      ...doc,
      revision: doc.revision + 1,
      updatedAt: Date.now(),
    });
    const encoded = await this.encode(
      next,
      row.classification,
      session ?? undefined,
      row.protection === "protected" ? row : undefined,
    );
    await this.commit(encoded.row, row, session, encoded.wrapped);
    return next;
  }

  async deleteDocument(id: string, expectedRevision: number): Promise<void> {
    const row = await this.row(id);
    if (row.revision !== expectedRevision) throw new ConflictError();
    const session =
      row.protection === "protected" ? this.owner(row.ownerId) : null;
    await this.transaction(
      ["documents", "users", "control"],
      "readwrite",
      async (tx) => {
        const store = tx.objectStore("documents");
        const [current] = await Promise.all([
          request<DocumentRecord | undefined>(store.get(id)),
          session ? this.checkOwner(tx, session) : Promise.resolve(null),
        ]);
        if (
          !current ||
          current.revision !== row.revision ||
          current.protection !== row.protection
        )
          throw new ConflictError();
        if (session) this.assertSession(session);
        // Keep a tombstone so no stale save/import can resurrect deleted plaintext.
        store.put({
          id,
          revision: row.revision + 1,
          classification: row.classification + 1,
          protection: "deleted",
          ...(row.protection === "protected" ? { ownerId: row.ownerId } : {}),
        } satisfies DeletedRecord);
      },
      !!session,
    );
    this.broadcast("change");
  }

  async cloneDocument(id: string): Promise<EditorDocument> {
    const row = await this.row(id);
    const session =
      row.protection === "protected" ? this.owner(row.ownerId) : null;
    const original = await this.decode(row, session);
    const now = Date.now();
    const clone = validateDocument({
      ...original,
      id: crypto.randomUUID(),
      title: original.title ? `${original.title} (copy)` : "",
      createdAt: now,
      updatedAt: now,
      revision: 1,
    });
    const encoded = await this.encode(clone, 1, session ?? undefined);
    // Compare the source inside the destination transaction: a concurrent
    // protection change must not leave an Open clone of formerly Open text.
    await this.commit(encoded.row, null, session, encoded.wrapped, row);
    return clone;
  }

  async setProtection(
    id: string,
    target: Protection,
    expectedRevision: number,
  ): Promise<EditorDocument> {
    if (target !== "open" && target !== "protected")
      throw new IntegrityError("Invalid protection state.");
    const row = await this.row(id);
    if (row.revision !== expectedRevision) throw new ConflictError();
    if (row.protection === target) return this.decode(row);
    const session = this.owner(
      row.protection === "protected" ? row.ownerId : undefined,
    );
    const doc = await this.decode(row, session);
    const next: EditorDocument = {
      ...doc,
      protection: target,
      revision: doc.revision + 1,
      updatedAt: Date.now(),
    };
    if (target === "protected") next.ownerId = session.user.id;
    else delete next.ownerId;
    const encoded = await this.encode(next, row.classification + 1, session);
    await this.commit(encoded.row, row, session, encoded.wrapped);
    // Peers must drop formerly Open context; the initiating owner remains
    // unlocked. Its caller must discard work bound to the previous revision.
    this.broadcast("lock");
    return next;
  }

  async changePassword(
    oldPassword: string,
    newPassword: string,
  ): Promise<void> {
    passwordPolicy(newPassword);
    const session = this.owner();
    const user = await this.transaction(["users"], "readonly", (tx) =>
      request<UserRecord>(tx.objectStore("users").get(session.user.id)),
    );
    assertUser(user);
    const oldKey = await this.derive(oldPassword, user, session.epoch);
    const raw = await unseal(oldKey, user.keyEnvelope, userAad(user), 48);
    const next: UserRecord = {
      ...user,
      passwordVersion: user.passwordVersion + 1,
      salt: toBase64(randomBytes(16)),
      kdf: { ...DEFAULT_KDF },
    };
    try {
      const newKey = await this.derive(newPassword, next, session.epoch);
      next.keyEnvelope = await seal(newKey, raw, userAad(next));
    } finally {
      raw.fill(0);
    }
    await this.transaction(
      ["users", "control"],
      "readwrite",
      async (tx) => {
        const current = await this.checkOwner(tx, session);
        if (current.passwordVersion !== user.passwordVersion)
          throw new ConflictError("The password was changed elsewhere.");
        tx.objectStore("users").put({ ...next, wraps: current.wraps });
      },
      true,
    );
    this.lock();
    await this.pendingLock;
  }

  async exportBackup(): Promise<string> {
    const session = this.owner();
    const snapshot = await this.transaction(
      ["users", "documents", "control"],
      "readwrite",
      async (tx): Promise<Pick<Backup, "user" | "documents">> => {
        const [user, rows] = await Promise.all([
          this.checkOwner(tx, session),
          request<DocumentRecord[]>(tx.objectStore("documents").getAll()),
        ]);
        if (user.wraps >= MAX_KEY_USES)
          throw new Error("User key encryption limit reached.");
        const nextUser = { ...user, wraps: user.wraps + 1 };
        tx.objectStore("users").put(nextUser);
        const documents = rows.filter(
          (row): row is ProtectedRecord =>
            row.protection === "protected" && row.ownerId === session.user.id,
        );
        for (const row of documents) assertRecord(row);
        return { user: nextUser, documents };
      },
      true,
    );
    if (snapshot.documents.length > MAX_BACKUP_DOCUMENTS)
      throw new Error("This backup exceeds the example’s 1000-document limit.");
    const backup: Backup = {
      format: FORMAT,
      version: 1,
      ...snapshot,
      manifest: await seal(
        session.key,
        await backupDigest(snapshot),
        backupAad(snapshot),
      ),
    };
    this.assertSession(session);
    const text = JSON.stringify(backup);
    if (new TextEncoder().encode(text).length > MAX_BACKUP_BYTES)
      throw new Error("This backup exceeds the 50 MiB limit.");
    return text;
  }

  async restoreBackup(text: string, password: string): Promise<void> {
    if (
      typeof text !== "string" ||
      text.length > MAX_BACKUP_BYTES ||
      new TextEncoder().encode(text).length > MAX_BACKUP_BYTES
    )
      throw new IntegrityError("Backup exceeds 50 MiB.");
    let backup: Backup;
    try {
      backup = JSON.parse(text) as Backup;
    } catch {
      throw new IntegrityError("Invalid backup JSON.");
    }
    if (
      backup?.format !== FORMAT ||
      backup.version !== 1 ||
      !Array.isArray(backup.documents) ||
      backup.documents.length > MAX_BACKUP_DOCUMENTS ||
      !onlyFields(backup, [
        "format",
        "version",
        "user",
        "documents",
        "manifest",
      ])
    )
      throw new IntegrityError("Unsupported backup format.");
    assertUser(backup.user);
    assertCipherBox(backup.manifest, 48);
    const ids = new Set<string>();
    for (const row of backup.documents) {
      assertRecord(row);
      if (
        row.protection !== "protected" ||
        row.ownerId !== backup.user.id ||
        ids.has(row.id)
      )
        throw new IntegrityError(
          "Invalid backup ownership or duplicate documents.",
        );
      ids.add(row.id);
    }
    const previousSession = this.session;
    this.lock();
    await this.pendingLock;
    const epoch = this.epoch;
    const backupKey = await this.userKey(password, backup.user, epoch);
    const expectedDigest = await backupDigest(backup);
    const authenticatedDigest = await unseal(
      backupKey,
      backup.manifest,
      backupAad(backup),
      48,
    );
    if (
      authenticatedDigest.length !== expectedDigest.length ||
      authenticatedDigest.some((byte, index) => byte !== expectedDigest[index])
    ) {
      throw new IntegrityError(
        "The backup manifest was changed or documents are missing.",
      );
    }
    const existing = await this.transaction(
      ["users", "documents", "control"],
      "readonly",
      async (tx) => ({
        user: await request<UserRecord | undefined>(
          tx.objectStore("users").get(backup.user.id),
        ),
        rows: await request<DocumentRecord[]>(
          tx.objectStore("documents").getAll(),
        ),
        control: await request<Control | undefined>(
          tx.objectStore("control").get("session"),
        ),
      }),
    );
    let user = backup.user;
    if (existing.user) {
      assertUser(existing.user);
      const currentKey =
        previousSession?.user.id === existing.user.id &&
        previousSession.keyId === existing.user.keyId
          ? previousSession.key
          : await this.userKey(password, existing.user, epoch);
      // Equal IDs or aliases never authorize replacing somebody else's random key.
      const challenge = randomBytes(32);
      const box = await seal(currentKey, challenge, [
        FORMAT,
        "restore-key-check",
      ]);
      const decoded = await unseal(
        backupKey,
        box,
        [FORMAT, "restore-key-check"],
        48,
      );
      try {
        if (
          decoded.length !== challenge.length ||
          decoded.some((byte, index) => byte !== challenge[index])
        )
          throw new IntegrityError("Backup belongs to a different key.");
      } finally {
        challenge.fill(0);
        decoded.fill(0);
      }
      user = existing.user; // Do not roll an existing user's password back to the backup's envelope.
    }
    this.assertEpoch(epoch);
    // A temporary restore session is local only; committed session tickets still enforce writes.
    const restoreSession: Session = {
      user: { id: user.id, name: user.name },
      keyId: user.keyId,
      key: backupKey,
      ticket: "",
      epoch,
    };
    this.session = restoreSession;
    const prepared: ProtectedRecord[] = [];
    try {
      for (const row of backup.documents) {
        const prior = existing.rows.find((item) => item.id === row.id);
        if (
          prior &&
          !(
            (prior.protection === "protected" ||
              prior.protection === "deleted") &&
            prior.ownerId === user.id
          )
        ) {
          throw new ConflictError(
            "A backup document ID belongs to Open work or another user.",
          );
        }
        const doc = await this.decode(row, restoreSession);
        doc.revision = Math.max(row.revision, prior?.revision ?? 0) + 1;
        const encoded = await this.encode(
          doc,
          Math.max(row.classification, prior?.classification ?? 0) + 1,
          restoreSession,
        );
        prepared.push(encoded.row as ProtectedRecord);
      }
      await this.transaction(
        ["users", "documents", "control"],
        "readwrite",
        async (tx) => {
          const [currentUser, currentRows, control] = await Promise.all([
            request<UserRecord | undefined>(
              tx.objectStore("users").get(user.id),
            ),
            request<DocumentRecord[]>(tx.objectStore("documents").getAll()),
            request<Control | undefined>(
              tx.objectStore("control").get("session"),
            ),
          ]);
          this.assertSession(restoreSession);
          if (
            control?.revision !== existing.control?.revision ||
            control?.ticket !== existing.control?.ticket
          )
            throw new LockedError("A local user switched during restore.");
          if (
            existing.user
              ? !currentUser ||
                currentUser.passwordVersion !== existing.user.passwordVersion ||
                currentUser.keyId !== existing.user.keyId
              : !!currentUser
          ) {
            throw new ConflictError("The local user changed during restore.");
          }
          for (const next of prepared) {
            const before = existing.rows.find((item) => item.id === next.id);
            const now = currentRows.find((item) => item.id === next.id);
            if (
              before
                ? !now ||
                  now.revision !== before.revision ||
                  now.protection !== before.protection
                : !!now
            )
              throw new ConflictError("A document changed during restore.");
          }
          const wraps =
            Math.max(user.wraps, currentUser?.wraps ?? 0) + prepared.length;
          if (wraps > MAX_KEY_USES)
            throw new Error("User key encryption limit reached.");
          user = { ...user, wraps };
          tx.objectStore("users").put(user);
          for (const row of prepared) tx.objectStore("documents").put(row);
        },
        true,
      );
      this.session = null;
      await this.activate(user, backupKey, epoch);
      this.broadcast("change");
    } catch (error) {
      this.clearSession();
      throw error;
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.clearSession();
    this.disposed = true;
    this.channel?.close();
    this.channel = null;
    EditorStorage.peers.get(this.name)?.delete(this);
    if (EditorStorage.peers.get(this.name)?.size === 0)
      EditorStorage.peers.delete(this.name);
    this.listeners.clear();
    this.db.close();
  }
}
