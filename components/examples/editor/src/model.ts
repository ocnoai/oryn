export type Protection = "open" | "protected";
export type DocumentKind = "document" | "reference";

export interface RichNode {
  type: string;
  attrs?: Record<string, any>;
  content?: RichNode[];
  text?: string;
  marks?: { type: string; attrs?: Record<string, any> }[];
}

export interface EditorDocument {
  id: string;
  title: string;
  content: RichNode;
  ideas: Record<string, string>;
  createdAt: number;
  updatedAt: number;
  revision: number;
  protection: Protection;
  ownerId?: string;
  kind: DocumentKind;
  sourceUrl?: string;
  graph?: {
    nodes: { id: string; label: string }[];
    edges: { from: string; to: string; relation: string }[];
  };
}

export interface LocalUser {
  id: string;
  name: string;
}

export class ConflictError extends Error {
  constructor(
    message = "This document changed elsewhere. Reload it before saving.",
  ) {
    super(message);
    this.name = "ConflictError";
  }
}

export class LockedError extends Error {
  constructor(message = "Unlock the document owner to continue.") {
    super(message);
    this.name = "LockedError";
  }
}

export class IntegrityError extends Error {
  constructor(
    message = "The password is incorrect or the encrypted data is damaged.",
  ) {
    super(message);
    this.name = "IntegrityError";
  }
}

export const MAX_DOCUMENT_BYTES = 5 * 1024 * 1024;

function object(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function boundedString(value: unknown, maximum: number): value is string {
  return typeof value === "string" && value.length <= maximum;
}

export function validId(value: unknown): value is string {
  return typeof value === "string" && /^[a-zA-Z0-9_-]{1,100}$/.test(value);
}

function validateNode(
  value: unknown,
  budget: { nodes: number },
  depth = 0,
): void {
  if (
    !object(value) ||
    depth > 64 ||
    ++budget.nodes > 50_000 ||
    !boundedString(value.type, 100) ||
    !value.type ||
    (value.text !== undefined &&
      !boundedString(value.text, MAX_DOCUMENT_BYTES)) ||
    (value.attrs !== undefined && !object(value.attrs)) ||
    (value.content !== undefined && !Array.isArray(value.content)) ||
    (value.marks !== undefined && !Array.isArray(value.marks))
  ) {
    throw new IntegrityError("Invalid or oversized rich-text document.");
  }
  for (const mark of (value.marks ?? []) as unknown[]) {
    if (
      !object(mark) ||
      !boundedString(mark.type, 100) ||
      (mark.attrs !== undefined && !object(mark.attrs))
    ) {
      throw new IntegrityError("Invalid rich-text formatting.");
    }
  }
  for (const child of (value.content ?? []) as unknown[])
    validateNode(child, budget, depth + 1);
}

/** Validates persistence/import boundaries; rendering still needs a safe editor schema. */
export function validateDocument(value: unknown): EditorDocument {
  let encoded: string;
  try {
    encoded = JSON.stringify(value);
  } catch {
    throw new IntegrityError("Invalid document.");
  }
  if (
    !encoded ||
    new TextEncoder().encode(encoded).length > MAX_DOCUMENT_BYTES ||
    !object(value) ||
    !validId(value.id) ||
    !boundedString(value.title, 10_000) ||
    !Number.isSafeInteger(value.revision) ||
    (value.revision as number) < 1 ||
    !Number.isSafeInteger(value.createdAt) ||
    !Number.isSafeInteger(value.updatedAt) ||
    (value.protection !== "open" && value.protection !== "protected") ||
    (value.kind !== "document" && value.kind !== "reference") ||
    !object(value.ideas) ||
    (value.sourceUrl !== undefined &&
      !boundedString(value.sourceUrl, 10_000)) ||
    (value.protection === "protected"
      ? !validId(value.ownerId)
      : value.ownerId !== undefined)
  ) {
    throw new IntegrityError("Invalid or oversized document.");
  }
  validateNode(value.content, { nodes: 0 });
  if ((value.content as RichNode).type !== "doc")
    throw new IntegrityError("Invalid document root.");
  for (const [id, idea] of Object.entries(value.ideas)) {
    if (!validId(id) || !boundedString(idea, 100_000))
      throw new IntegrityError("Invalid document ideas.");
  }
  if (value.graph !== undefined) {
    if (
      !object(value.graph) ||
      !Array.isArray(value.graph.nodes) ||
      !Array.isArray(value.graph.edges) ||
      value.graph.nodes.length > 5000 ||
      value.graph.edges.length > 20_000
    ) {
      throw new IntegrityError("Invalid topic graph.");
    }
    for (const node of value.graph.nodes) {
      if (
        !object(node) ||
        !validId(node.id) ||
        !boundedString(node.label, 100_000)
      )
        throw new IntegrityError("Invalid graph node.");
    }
    for (const edge of value.graph.edges) {
      if (
        !object(edge) ||
        !validId(edge.from) ||
        !validId(edge.to) ||
        !boundedString(edge.relation, 100)
      ) {
        throw new IntegrityError("Invalid graph edge.");
      }
    }
  }
  // Drop unknown top-level fields: accidentally attached UI state must not get persisted.
  const result: EditorDocument = {
    id: value.id,
    title: value.title,
    content: value.content as unknown as RichNode,
    ideas: value.ideas as Record<string, string>,
    createdAt: value.createdAt as number,
    updatedAt: value.updatedAt as number,
    revision: value.revision as number,
    protection: value.protection,
    kind: value.kind,
  };
  if (value.ownerId !== undefined) result.ownerId = value.ownerId as string;
  if (value.sourceUrl !== undefined)
    result.sourceUrl = value.sourceUrl as string;
  if (value.graph !== undefined)
    result.graph = value.graph as unknown as EditorDocument["graph"];
  return structuredClone(result);
}

export function newDocument(
  protection: Protection,
  kind: DocumentKind,
  ownerId?: string,
): EditorDocument {
  const now = Date.now();
  return {
    id: crypto.randomUUID(),
    title: "",
    content: {
      type: "doc",
      content: [{ type: "paragraph", attrs: { blockId: crypto.randomUUID() } }],
    },
    ideas: {},
    createdAt: now,
    updatedAt: now,
    revision: 1,
    protection,
    kind,
    ...(protection === "protected" ? { ownerId } : {}),
  };
}
