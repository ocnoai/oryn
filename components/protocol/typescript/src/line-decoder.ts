import { MAX_MESSAGE_BYTES } from "./types.js";

export class ProtocolLineError extends Error {
  constructor(
    message: string,
    readonly code:
      | "PROTOCOL_MESSAGE_TOO_LARGE"
      | "PROTOCOL_INVALID_UTF8"
      | "PROTOCOL_INCOMPLETE_LINE"
      | "PROTOCOL_INVALID_JSON"
      | "PROTOCOL_COMPLEXITY_LIMIT",
  ) {
    super(message);
    this.name = "ProtocolLineError";
  }
}

export function parseProtocolJSONLine(line: string): unknown {
  let value: unknown;
  try {
    value = JSON.parse(line) as unknown;
  } catch {
    throw new ProtocolLineError(
      "Engine emitted invalid JSON.",
      "PROTOCOL_INVALID_JSON",
    );
  }
  assertBoundedComplexity(value);
  return value;
}

function assertBoundedComplexity(value: unknown): void {
  const stack: Array<{ value: unknown; depth: number }> = [{ value, depth: 0 }];
  let nodes = 0;
  while (stack.length > 0) {
    const item = stack.pop()!;
    nodes += 1;
    if (nodes > 100_000 || item.depth > 64) complexityError();
    if (Array.isArray(item.value)) {
      if (item.value.length > 50_000) complexityError();
      for (const child of item.value) stack.push({ value: child, depth: item.depth + 1 });
    } else if (item.value !== null && typeof item.value === "object") {
      const entries = Object.entries(item.value);
      if (entries.length > 10_000) complexityError();
      for (const [, child] of entries) stack.push({ value: child, depth: item.depth + 1 });
    }
  }
}

function complexityError(): never {
  throw new ProtocolLineError(
    "Engine protocol message exceeded structural complexity limits.",
    "PROTOCOL_COMPLEXITY_LIMIT",
  );
}

export class ProtocolLineDecoder {
  private buffered = Buffer.alloc(0);
  private readonly decoder = new TextDecoder("utf-8", { fatal: true });

  constructor(private readonly maxMessageBytes = MAX_MESSAGE_BYTES) {}

  push(chunk: Buffer): string[] {
    this.buffered = Buffer.concat([this.buffered, chunk]);
    const lines: string[] = [];

    while (true) {
      const newlineIndex = this.buffered.indexOf(0x0a);
      if (newlineIndex === -1) {
        break;
      }

      let lineBytes = this.buffered.subarray(0, newlineIndex);
      this.buffered = this.buffered.subarray(newlineIndex + 1);

      if (lineBytes.length > this.maxMessageBytes) {
        throw new ProtocolLineError(
          "Engine protocol line exceeded the negotiated size limit.",
          "PROTOCOL_MESSAGE_TOO_LARGE",
        );
      }

      if (lineBytes.at(-1) === 0x0d) {
        lineBytes = lineBytes.subarray(0, -1);
      }

      try {
        lines.push(this.decoder.decode(lineBytes));
      } catch {
        throw new ProtocolLineError(
          "Engine emitted invalid UTF-8.",
          "PROTOCOL_INVALID_UTF8",
        );
      }
    }

    if (this.buffered.length > this.maxMessageBytes) {
      throw new ProtocolLineError(
        "Engine protocol line exceeded the negotiated size limit.",
        "PROTOCOL_MESSAGE_TOO_LARGE",
      );
    }

    return lines;
  }

  finish(): void {
    if (this.buffered.length > 0) {
      throw new ProtocolLineError(
        "Engine stdout ended in the middle of a protocol line.",
        "PROTOCOL_INCOMPLETE_LINE",
      );
    }
  }
}
