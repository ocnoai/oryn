import { argon2id } from "hash-wasm";
import type { KdfParameters } from "./crypto";

const worker = self as unknown as {
  onmessage:
    | ((
        event: MessageEvent<{
          password: string;
          salt: Uint8Array;
          parameters: KdfParameters;
        }>,
      ) => void)
    | null;
  postMessage: (message: unknown, transfer?: Transferable[]) => void;
};

worker.onmessage = async (event) => {
  const { password, salt, parameters } = event.data;
  try {
    if (
      !parameters ||
      parameters.algorithm !== "argon2id" ||
      parameters.version !== 19 ||
      !Number.isInteger(parameters.memoryKiB) ||
      parameters.memoryKiB < 19_456 ||
      parameters.memoryKiB > 131_072 ||
      !Number.isInteger(parameters.iterations) ||
      parameters.iterations < 2 ||
      parameters.iterations > 10 ||
      !Number.isInteger(parameters.parallelism) ||
      parameters.parallelism < 1 ||
      parameters.parallelism > 4
    ) {
      throw new Error("Unsupported password derivation parameters.");
    }
    if (
      typeof password !== "string" ||
      password.length > 1024 ||
      !(salt instanceof Uint8Array) ||
      salt.length !== 16
    ) {
      throw new Error("Invalid derivation input.");
    }
    const result = await argon2id({
      password,
      salt,
      parallelism: parameters.parallelism,
      iterations: parameters.iterations,
      memorySize: parameters.memoryKiB,
      hashLength: 32,
      outputType: "binary",
    });
    worker.postMessage({ result }, [result.buffer]);
  } catch {
    worker.postMessage({ error: "Password derivation failed." });
  }
};
