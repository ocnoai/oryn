import type { InjectedLocalAI } from "./types.js";

declare global {
  interface Window {
    readonly localAI?: InjectedLocalAI;
  }
}

export {};
