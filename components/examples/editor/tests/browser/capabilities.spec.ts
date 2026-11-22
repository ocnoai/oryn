import { expect, test, type Page } from "@playwright/test";
import type { InjectedLocalAI } from "@ocno/oryn-sdk";
import { configureSDK, installSDKFixture, sdkSnapshot } from "./sdk-fixture";

interface CapabilityControl {
  available: boolean;
  hold: boolean;
  calls: number;
  release(): void;
}
type CapabilityWindow = Window & {
  __editorCapabilityFixture: CapabilityControl;
};

async function installCapabilityFixture(
  page: Page,
  available = false,
  hold = false,
) {
  let fixtureSource = "";
  // Compose one preload so the capability override precedes App's first probe,
  // regardless of Playwright's ordering of separately registered init scripts.
  await installSDKFixture({
    addInitScript: async (script: unknown) => {
      if (typeof script !== "function")
        throw new Error("Expected the SDK fixture bootstrap.");
      fixtureSource = `(${script.toString()})();`;
    },
  } as unknown as Page);
  const controlSource = (initial: { available: boolean; hold: boolean }) => {
    const bridge = (window as unknown as { localAI: InjectedLocalAI }).localAI;
    const original = bridge.getCapabilities.bind(bridge);
    const pending: Array<() => void> = [];
    const control: CapabilityControl = {
      ...initial,
      calls: 0,
      release() {
        control.hold = false;
        for (const resolve of pending.splice(0)) resolve();
      },
    };
    bridge.getCapabilities = async () => {
      control.calls++;
      if (control.hold)
        await new Promise<void>((resolve) => pending.push(resolve));
      const capabilities = await original();
      if (control.available) return capabilities;
      return {
        ...capabilities,
        runtimeAvailable: false,
        runtimeState: "starting",
        supportedModelProviders: [],
        features: Object.fromEntries(
          Object.keys(capabilities.features).map((key) => [key, false]),
        ) as typeof capabilities.features,
      };
    };
    Object.defineProperty(window, "__editorCapabilityFixture", {
      value: control,
    });
  };
  // Only the injected Browser capability response changes. The SDK, model
  // requests, invocation, streaming and App code remain the production paths.
  await page.addInitScript({
    content: `${fixtureSource}\n(${controlSource.toString()})(${JSON.stringify({ available, hold })});`,
  });
}

async function openDraft(page: Page) {
  await page.getByRole("button", { name: "New document" }).click();
  await page
    .locator(".new-popover")
    .getByRole("button", { name: "Open document", exact: false })
    .click();
  await page
    .getByRole("textbox", { name: "Document title", exact: true })
    .fill("Capability recovery fixture");
  const editor = page.getByRole("textbox", {
    name: "Document text",
    exact: true,
  });
  await editor.fill(
    "Synthetic writing remains available during runtime recovery.",
  );
  await expect(page.getByText("Saved locally", { exact: true })).toBeVisible();
  await page
    .getByLabel("Ask local AI", { exact: true })
    .fill("Give one clearer sentence.");
  return editor;
}

test("automatically recovers after an initially unavailable runtime without requesting a model prematurely", async ({
  page,
}) => {
  await installCapabilityFixture(page);
  await page.goto("/");
  await openDraft(page);
  await expect(
    page.getByRole("button", { name: "Send question", exact: true }),
  ).toBeDisabled();
  const firstCalls = await page.evaluate(
    () =>
      (window as unknown as CapabilityWindow).__editorCapabilityFixture.calls,
  );
  expect((await sdkSnapshot(page)).requests).toHaveLength(0);
  await page.evaluate(() => {
    (
      window as unknown as CapabilityWindow
    ).__editorCapabilityFixture.available = true;
  });
  await expect(
    page.getByRole("button", { name: "Send question", exact: true }),
  ).toBeEnabled();
  expect(
    await page.evaluate(
      () =>
        (window as unknown as CapabilityWindow).__editorCapabilityFixture.calls,
    ),
  ).toBeGreaterThan(firstCalls);
  expect((await sdkSnapshot(page)).requests).toHaveLength(0);
  await page
    .getByRole("button", { name: "Send question", exact: true })
    .click();
  await expect(page.locator(".ai-answer")).toHaveText(
    "A clearer local sentence.",
  );
});

test("manual Retry recovers after the bounded automatic retry burst ends", async ({
  page,
}) => {
  await installCapabilityFixture(page);
  await page.goto("/");
  await openDraft(page);
  await expect
    .poll(
      () =>
        page.evaluate(
          () =>
            (window as unknown as CapabilityWindow).__editorCapabilityFixture
              .calls,
        ),
      { timeout: 25000 },
    )
    .toBeGreaterThanOrEqual(6);
  const retry = page.getByRole("button", {
    name: "Retry local AI connection",
    exact: true,
  });
  await expect(retry).toBeEnabled();
  await page.evaluate(() => {
    (
      window as unknown as CapabilityWindow
    ).__editorCapabilityFixture.available = true;
  });
  await retry.click();
  await expect(
    page.getByRole("button", { name: "Send question", exact: true }),
  ).toBeEnabled();
  await expect(page.locator(".runtime-message")).toHaveCount(0);
  expect((await sdkSnapshot(page)).requests).toHaveLength(0);
});

test("focus, pageshow and visible-page events recheck the live Browser capabilities", async ({
  page,
}) => {
  await installCapabilityFixture(page);
  await page.goto("/");
  const editor = await openDraft(page);
  await page.evaluate(() => {
    (
      window as unknown as CapabilityWindow
    ).__editorCapabilityFixture.available = true;
    window.dispatchEvent(new Event("focus"));
  });
  await expect(
    page.getByRole("button", { name: "Send question", exact: true }),
  ).toBeEnabled();
  await page.evaluate(() => {
    (
      window as unknown as CapabilityWindow
    ).__editorCapabilityFixture.available = false;
    window.dispatchEvent(new PageTransitionEvent("pageshow"));
  });
  await expect(
    page.getByRole("button", { name: "Send question", exact: true }),
  ).toBeDisabled();
  await page.evaluate(() => {
    (
      window as unknown as CapabilityWindow
    ).__editorCapabilityFixture.available = true;
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await expect(
    page.getByRole("button", { name: "Send question", exact: true }),
  ).toBeEnabled();
  await expect(editor).toHaveText(
    "Synthetic writing remains available during runtime recovery.",
  );
  expect((await sdkSnapshot(page)).requests).toHaveLength(0);
});

test("a hung capability check times out visibly and later recovery keeps editing usable", async ({
  page,
}) => {
  await installCapabilityFixture(page, false, true);
  await page.goto("/");
  const editor = await openDraft(page);
  await expect(
    page
      .getByRole("status")
      .filter({ hasText: "The local AI connection did not respond." }),
  ).toBeVisible({ timeout: 12000 });
  await editor.fill(
    "Synthetic changes made while the capability check is pending.",
  );
  await page.evaluate(() => {
    const control = (window as unknown as CapabilityWindow)
      .__editorCapabilityFixture;
    control.available = true;
    control.release();
    window.dispatchEvent(new Event("focus"));
  });
  await expect(
    page.getByRole("button", { name: "Send question", exact: true }),
  ).toBeEnabled();
  await expect(editor).toHaveText(
    "Synthetic changes made while the capability check is pending.",
  );
});

test("editing a running answer keeps its cancellation explanation visible", async ({
  page,
}) => {
  await installCapabilityFixture(page, true);
  await page.goto("/");
  const editor = await openDraft(page);
  await configureSDK(page, { holdText: true });
  await page
    .getByRole("button", { name: "Send question", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Send question", exact: true }),
  ).toHaveCount(0);
  await expect
    .poll(async () => (await sdkSnapshot(page)).invocations.length)
    .toBe(1);
  await editor.fill("Synthetic owner edit cancels the stale AI request.");
  await expect(page.locator(".answer-card")).toHaveCount(0);
  await expect(
    page
      .getByRole("status")
      .filter({
        hasText: "Cancellation requested because the document changed.",
      }),
  ).toBeVisible();
  await expect
    .poll(async () => (await sdkSnapshot(page)).cancelled.length)
    .toBe(1);
});
