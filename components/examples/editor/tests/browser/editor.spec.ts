import { expect, test, type Page } from "@playwright/test";

async function newOpen(page: Page) {
  await page.getByRole("button", { name: "New document" }).click();
  await page
    .getByRole("button", { name: "Open document", exact: false })
    .click();
  await expect(
    page.getByRole("textbox", { name: "Document title", exact: true }),
  ).toBeVisible();
}
async function fillDocument(page: Page, title: string, text: string) {
  await page
    .getByRole("textbox", { name: "Document title", exact: true })
    .fill(title);
  const editor = page.getByRole("textbox", {
    name: "Document text",
    exact: true,
  });
  await editor.fill(text);
  await expect(page.getByText("Saved locally", { exact: true })).toBeVisible();
}
async function createUser(page: Page, name: string, password: string) {
  await page.getByRole("button", { name: "Local users", exact: false }).click();
  await page
    .getByRole("button", { name: "Create local user", exact: true })
    .click();
  await page.getByLabel("Local name", { exact: true }).fill(name);
  await page.getByLabel("Password", { exact: true }).fill(password);
  await page.getByRole("button", { name: "Create & unlock" }).click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect(
    page.getByRole("button", { name: "Lock protected documents", exact: true }),
  ).toBeVisible();
}

test("Open writing, formatting, ideas, clone, search, reload and safe HTML export", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Make room/ })).toBeVisible();
  await expect(
    page.getByText(/Editing, protection and export work without AI/),
  ).toBeVisible();
  await newOpen(page);
  await fillDocument(
    page,
    "The shared garden",
    "A synthetic shared paragraph for everyone.",
  );
  const editor = page.getByRole("textbox", {
    name: "Document text",
    exact: true,
  });
  await editor.focus();
  await page.keyboard.press("Meta+a");
  await page.getByRole("button", { name: "Bold", exact: true }).click();
  await expect(editor.locator("strong")).toContainText(
    "A synthetic shared paragraph",
  );
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  await expect(editor.locator("strong")).toHaveCount(0);
  const idea = page.locator(".rich-editor-idea textarea").first();
  await idea.fill("An idea attached to this paragraph");
  await expect(page.getByText("Saved locally", { exact: true })).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: /The shared garden/ }).click();
  await expect(editor).toContainText("A synthetic shared paragraph");
  await expect(page.locator(".rich-editor-idea textarea").first()).toHaveValue(
    "An idea attached to this paragraph",
  );
  await page
    .getByRole("button", { name: "Document actions", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Clone document", exact: true })
    .click();
  await expect(
    page.getByRole("textbox", { name: "Document title", exact: true }),
  ).toHaveValue("The shared garden (copy)");
  await page
    .getByRole("textbox", { name: "Search documents", exact: true })
    .fill("not-existing");
  await expect(page.getByText("No matching shared documents.")).toBeVisible();
  await page
    .getByRole("textbox", { name: "Search documents", exact: true })
    .fill("");
  await page.getByRole("button", { name: "Export", exact: true }).click();
  await expect(page.locator(".export-preview")).not.toContainText(
    "An idea attached",
  );
  await page.getByLabel("Include ideas", { exact: true }).check();
  await expect(page.locator(".export-preview")).toContainText(
    "An idea attached",
  );
  const download = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Download HTML", exact: true })
    .click();
  expect((await download).suggestedFilename()).toMatch(/\.html$/);
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();
  await page.screenshot({
    path: "test-results/desktop-editor.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("Alice and Bob are equal; lock clears private search, titles and text; Open stays shared", async ({
  page,
}) => {
  await page.goto("/");
  await newOpen(page);
  await fillDocument(
    page,
    "Shared sunshine",
    "Everyone may see this Open document.",
  );
  await createUser(page, "Alice", "Alice private long passphrase");
  await page.getByRole("button", { name: "New document" }).click();
  await page.getByRole("button", { name: /^Protected document Only/ }).click();
  await fillDocument(
    page,
    "Alice-secret-title",
    "Alice-secret-body and a confidential sentence.",
  );
  await page
    .getByRole("textbox", { name: "Search documents", exact: true })
    .fill("Alice-secret");
  const second = await page.context().newPage();
  await second.goto("/");
  await expect(second.locator("body")).not.toContainText("Alice-secret");
  await expect(
    second.getByRole("button", { name: /Shared sunshine/ }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Lock protected documents", exact: true })
    .click();
  await expect(
    page.getByRole("textbox", { name: "Search documents", exact: true }),
  ).toHaveValue("");
  await expect(page.locator("body")).not.toContainText("Alice-secret");
  await createUser(page, "Bob", "Bob private long passphrase");
  await expect(page.locator("body")).not.toContainText("Alice-secret");
  await page.getByRole("button", { name: "New document" }).click();
  await page.getByRole("button", { name: /^Protected document Only/ }).click();
  await fillDocument(page, "Bob-secret-title", "Bob-secret-body only.");
  await page.getByRole("button", { name: /Shared sunshine/ }).click();
  await expect(
    page.getByRole("textbox", { name: "Document text", exact: true }),
  ).toContainText("Everyone may see");
  await page
    .getByRole("button", { name: "Lock protected documents", exact: true })
    .click();
  await page.getByRole("button", { name: "Local users", exact: false }).click();
  await page
    .getByLabel("Local user", { exact: true })
    .selectOption({ label: "Alice" });
  await page
    .getByLabel("Password", { exact: true })
    .fill("This is the wrong passphrase");
  await page
    .getByRole("button", { name: "Unlock locally", exact: true })
    .click();
  await expect(page.getByRole("alert")).toContainText(/incorrect|damaged/i);
  await expect(page.getByRole("dialog")).toBeVisible();
  await page
    .getByLabel("Password", { exact: true })
    .fill("Alice private long passphrase");
  await page
    .getByRole("button", { name: "Unlock locally", exact: true })
    .click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect(
    page.getByRole("button", { name: /Alice-secret-title/ }),
  ).toBeVisible();
  await expect(page.locator("body")).not.toContainText("Bob-secret");
  // Inspect only this isolated test origin's raw IndexedDB and synthetic markers.
  const raw = await page.evaluate(async () => {
    const db = await new Promise<IDBDatabase>((resolve, reject) => {
      const r = indexedDB.open("oryn-editor");
      r.onsuccess = () => resolve(r.result);
      r.onerror = () => reject(r.error);
    });
    const rows = await new Promise<unknown[]>((resolve, reject) => {
      const r = db.transaction("documents").objectStore("documents").getAll();
      r.onsuccess = () => resolve(r.result);
      r.onerror = () => reject(r.error);
    });
    db.close();
    return JSON.stringify(rows);
  });
  expect(raw).toContain("Shared sunshine");
  expect(raw).not.toContain("Alice-secret");
  expect(raw).not.toContain("Bob-secret");
});

test("Protection changes and cross-tab locks preserve the owner boundary", async ({
  page,
}) => {
  page.on("dialog", (dialog) => dialog.accept());
  await page.goto("/");
  await createUser(page, "Alice", "A long private passphrase for Alice");
  await newOpen(page);
  await fillDocument(
    page,
    "Convert-me",
    "A formerly shared synthetic paragraph.",
  );
  await page
    .getByRole("button", { name: "Document actions", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Protect document…", exact: true })
    .click();
  await expect(page.locator(".protection-badge")).toContainText("PROTECTED");
  await expect(
    page.getByRole("textbox", { name: "Document text", exact: true }),
  ).toContainText("formerly shared");
  const second = await page.context().newPage();
  await second.goto("/");
  await expect(second.locator("body")).not.toContainText("Convert-me");
  await second
    .getByRole("button", { name: "Local users", exact: false })
    .click();
  await second
    .getByLabel("Local user", { exact: true })
    .selectOption({ label: "Alice" });
  await second
    .getByLabel("Password", { exact: true })
    .fill("A long private passphrase for Alice");
  await second
    .getByRole("button", { name: "Unlock locally", exact: true })
    .click();
  await expect(second.getByRole("dialog")).not.toBeVisible();
  await expect(page.locator("body")).not.toContainText("Convert-me");
  await second.getByRole("button", { name: /Convert-me/ }).click();
  await expect(
    second.getByRole("textbox", { name: "Document text", exact: true }),
  ).toContainText("formerly shared");
});

test("Mobile panels remain reachable and no workspace data is requested over the network", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const requests: string[] = [];
  page.on("request", (request) => requests.push(request.url()));
  await page.goto("/");
  await page
    .getByRole("button", { name: "Start a document", exact: true })
    .click();
  await fillDocument(
    page,
    "Mobile synthetic title",
    "Mobile synthetic marker only.",
  );
  await page.getByRole("button", { name: "Assistant", exact: true }).click();
  await expect(
    page.getByRole("complementary", { name: "Local AI assistant" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Documents", exact: true }).click();
  await expect(
    page.getByRole("textbox", { name: "Search documents", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Write", exact: true }).click();
  await expect(
    page.getByRole("textbox", { name: "Document title", exact: true }),
  ).toHaveValue("Mobile synthetic title");
  expect(
    requests.every((url) => url.startsWith("http://127.0.0.1:5192/")),
  ).toBe(true);
  expect(
    requests.some((url) => /Mobile%20|synthetic%20|marker/i.test(url)),
  ).toBe(false);
  await page.screenshot({
    path: "test-results/mobile-editor.png",
    fullPage: true,
  });
});
