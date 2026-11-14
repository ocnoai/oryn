// @vitest-environment happy-dom

import { Window as HappyWindow } from "happy-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createLocalAIClient, type InjectedLocalAI } from "../index.js";

function bridge(): InjectedLocalAI {
  return {
    sdkVersion: "0.1.0",
    getCapabilities: async () => {
      throw new Error("The discovery check must not call capabilities.");
    },
    requestModel: async () => {
      throw new Error("unused");
    },
    cancelModelRequest: async () => undefined,
    releaseModelSession: async () => ({ ok: true, value: undefined }),
    onModelRequestProgress: () => () => undefined,
    beginArtifactCreate: async () => ({ ok: true, value: { uploadId: "unused" } }),
    appendArtifactChunk: async () => ({ ok: true, value: undefined }),
    finishArtifactCreate: async () => {
      throw new Error("unused");
    },
    abortArtifactCreate: async () => undefined,
    getArtifact: async () => {
      throw new Error("unused");
    },
    getArtifactUrl: async () => {
      throw new Error("unused");
    },
    saveArtifactAs: async () => ({ ok: true, value: undefined }),
    releaseArtifact: async () => ({ ok: true, value: undefined }),
    invokeJob: async () => {
      throw new Error("unused");
    },
    cancelJob: async () => ({ ok: true, value: undefined }),
    onJobEvent: () => () => undefined,
  };
}

function alertHost(): HTMLElement | null {
  return document.querySelector<HTMLElement>("[data-oryn-browser-alert]");
}

async function mounted(): Promise<void> {
  await new Promise<void>((resolve) => queueMicrotask(resolve));
}

describe("opt-in Oryn Browser discovery alert", () => {
  beforeEach(() => {
    const isolatedWindow = new HappyWindow({ url: "https://example.test/" });
    vi.stubGlobal("window", isolatedWindow);
    vi.stubGlobal("document", isolatedWindow.document);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("has no default or explicit-false DOM side effect", async () => {
    createLocalAIClient();
    createLocalAIClient({ showOrynBrowserAlert: false });
    await mounted();
    expect(alertHost()).toBeNull();
  });

  it("suppresses the alert for a callable injected bridge without probing runtime state", async () => {
    (window as Window & { localAI?: InjectedLocalAI }).localAI = bridge();
    createLocalAIClient({ showOrynBrowserAlert: true });
    await mounted();
    expect(alertHost()).toBeNull();
  });

  it("renders once, isolates presentation, and uses the exact safe destination", async () => {
    createLocalAIClient({ showOrynBrowserAlert: true });
    createLocalAIClient({ showOrynBrowserAlert: true });
    await mounted();
    const host = alertHost();
    expect(host).not.toBeNull();
    expect(document.querySelectorAll("[data-oryn-browser-alert]")).toHaveLength(1);
    expect(host?.shadowRoot?.textContent).toContain("This feature needs Oryn Browser");
    const link = host?.shadowRoot?.querySelector("a");
    expect(link?.href).toBe("https://oryn.ocno.ai/");
    expect(link?.target).toBe("_blank");
    expect(link?.rel).toContain("noopener");
    expect(link?.rel).toContain("noreferrer");
  });

  it("dismisses with Escape and restores focus", async () => {
    const trigger = document.createElement("button");
    document.body.append(trigger);
    trigger.focus();
    createLocalAIClient({ showOrynBrowserAlert: true });
    await mounted();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    expect(alertHost()).toBeNull();
    expect(document.activeElement).toBe(trigger);
    trigger.remove();
  });

  it("treats a malformed spoof as untrusted and never invokes it", async () => {
    const getCapabilities = () => {
      throw new Error("must not be called");
    };
    Object.defineProperty(window, "localAI", {
      configurable: true,
      value: { getCapabilities },
    });
    createLocalAIClient({ showOrynBrowserAlert: true });
    await mounted();
    expect(alertHost()).not.toBeNull();
  });
});
