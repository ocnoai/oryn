import { expect, test, type Page } from "@playwright/test";

async function profile(page: Page) {
  await page.locator(".profile-button").click();
}
async function startPrivate(page: Page) {
  await page.goto("/");
  await page
    .getByRole("button", { name: /Or start with a protected page/ })
    .click();
  await page.getByLabel("Local name", { exact: true }).fill("Alice");
  await page
    .getByLabel("Password", { exact: true })
    .fill("Alice synthetic private passphrase");
  await page.getByRole("button", { name: "Create & unlock" }).click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect(page.locator(".protection-badge")).toContainText("PROTECTED");
  await page
    .getByRole("textbox", { name: "Document title", exact: true })
    .fill("Alice backup marker");
  await page
    .getByRole("textbox", { name: "Document text", exact: true })
    .fill("A synthetic encrypted body marker.");
  await expect(page.getByText("Saved locally", { exact: true })).toBeVisible();
}

test("Encrypted backup, password change and owner-authenticated restore work through the UI", async ({
  page,
}, testInfo) => {
  page.on("dialog", (dialog) => dialog.accept());
  await startPrivate(page);
  await profile(page);
  const downloaded = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Download my encrypted backup", exact: true })
    .click();
  const backup = await downloaded;
  const file = testInfo.outputPath("encrypted-backup.json");
  await backup.saveAs(file);
  const stream = await backup.createReadStream();
  let data = "";
  for await (const chunk of stream!) data += chunk.toString();
  expect(data).not.toContain("Alice backup marker");
  expect(data).not.toContain("synthetic encrypted body");
  await page
    .getByRole("button", { name: "Change my password", exact: true })
    .click();
  await page
    .getByLabel("Current password", { exact: true })
    .fill("Alice synthetic private passphrase");
  await page
    .getByLabel("New private passphrase", { exact: true })
    .fill("Alice changed private passphrase");
  await page
    .getByRole("button", { name: "Change password", exact: true })
    .click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect(page.locator("body")).not.toContainText("Alice backup marker");
  await profile(page);
  await page
    .getByLabel("Local user", { exact: true })
    .selectOption({ label: "Alice" });
  await page
    .getByLabel("Password", { exact: true })
    .fill("Alice changed private passphrase");
  await page
    .getByRole("button", { name: "Unlock locally", exact: true })
    .click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await page.getByRole("button", { name: /Alice backup marker/ }).click();
  await page
    .getByRole("textbox", { name: "Document text", exact: true })
    .fill("A newer synthetic change to replace.");
  await expect(page.getByText("Saved locally", { exact: true })).toBeVisible();
  await profile(page);
  await page
    .getByRole("button", { name: "Restore an encrypted backup…", exact: true })
    .click();
  await page
    .locator('input[type=file][accept=".json,application/json"]')
    .setInputFiles(file);
  await page
    .getByLabel("Backup password", { exact: true })
    .fill("Alice synthetic private passphrase");
  await page
    .getByRole("button", { name: "Validate & restore", exact: true })
    .click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await page.getByRole("button", { name: /Alice backup marker/ }).click();
  await expect(
    page.getByRole("textbox", { name: "Document text", exact: true }),
  ).toContainText("A synthetic encrypted body marker.");
});

test("Production asset cache supports password unlock and editing offline", async ({
  page,
  context,
}) => {
  await startPrivate(page);
  await page.evaluate(async () => {
    await navigator.serviceWorker.ready;
    if (!navigator.serviceWorker.controller)
      await new Promise<void>((resolve) =>
        navigator.serviceWorker.addEventListener(
          "controllerchange",
          () => resolve(),
          { once: true },
        ),
      );
  });
  await page
    .getByRole("button", { name: "Lock protected documents", exact: true })
    .click();
  await context.setOffline(true);
  await page.reload();
  await profile(page);
  await page
    .getByLabel("Local user", { exact: true })
    .selectOption({ label: "Alice" });
  await page
    .getByLabel("Password", { exact: true })
    .fill("Alice synthetic private passphrase");
  await page
    .getByRole("button", { name: "Unlock locally", exact: true })
    .click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await page.getByRole("button", { name: /Alice backup marker/ }).click();
  await page
    .getByRole("textbox", { name: "Document text", exact: true })
    .fill("Offline synthetic change.");
  await expect(page.getByText("Saved locally", { exact: true })).toBeVisible();
  await context.setOffline(false);
});

test("Failed persistence remains visible and Retry saves the current draft", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByRole("button", { name: "Start a document", exact: true })
    .click();
  await page.evaluate(() => {
    const original = IDBObjectStore.prototype.put;
    let remaining = 1;
    IDBObjectStore.prototype.put = function (
      ...args: Parameters<IDBObjectStore["put"]>
    ) {
      if (this.name === "documents" && remaining-- > 0)
        throw new DOMException("Synthetic quota failure", "QuotaExceededError");
      return original.apply(this, args);
    };
  });
  await page
    .getByRole("textbox", { name: "Document title", exact: true })
    .fill("Unsaved synthetic marker");
  await expect(page.getByText("Save failed", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("textbox", { name: "Document title", exact: true }),
  ).toHaveValue("Unsaved synthetic marker");
  await page.getByRole("button", { name: "Retry", exact: true }).click();
  await expect(page.getByText("Saved locally", { exact: true })).toBeVisible();
  await page.reload();
  await expect(
    page.getByRole("button", { name: /Unsaved synthetic marker/ }),
  ).toBeVisible();
});
