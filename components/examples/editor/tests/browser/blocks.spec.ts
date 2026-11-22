import { expect, test, type Locator, type Page } from "@playwright/test";

async function newOpen(page: Page, title: string) {
  await page.goto("/");
  await page.getByRole("button", { name: "New document" }).click();
  await page
    .getByRole("button", { name: "Open document", exact: false })
    .click();
  await page
    .getByRole("textbox", { name: "Document title", exact: true })
    .fill(title);
  const editor = page.getByRole("textbox", {
    name: "Document text",
    exact: true,
  });
  await expect(editor).toBeVisible();
  return editor;
}

async function selectText(page: Page, editor: Locator) {
  await editor.focus();
  await page.keyboard.press("ControlOrMeta+a");
}

async function blockActions(page: Page, index: number) {
  const controls = page.locator(".rich-editor-block-controls").nth(index);
  const details = controls.locator("details");
  if ((await details.getAttribute("open")) === null)
    await details.locator("summary").click();
  return controls;
}

test("formatting changes real document markup and autosave leaves the insertion position intact", async ({
  page,
}) => {
  const editor = await newOpen(page, "Formatting exercise");
  await editor.fill("A paragraph with deliberate formatting.");
  await selectText(page, editor);
  await page.getByRole("button", { name: "Bold", exact: true }).click();
  await page.getByRole("button", { name: "Italic", exact: true }).click();
  await page.getByRole("button", { name: "Underline", exact: true }).click();
  await expect(editor.locator("strong")).toHaveText(
    "A paragraph with deliberate formatting.",
  );
  await expect(editor.locator("em")).toHaveText(
    "A paragraph with deliberate formatting.",
  );
  await expect(editor.locator("u")).toHaveText(
    "A paragraph with deliberate formatting.",
  );
  await page.getByLabel("Font family", { exact: true }).selectOption("Arial");
  await page.getByLabel("Font size", { exact: true }).selectOption("20");
  await expect(editor.locator('span[style*="font-family"]')).toHaveCSS(
    "font-family",
    "Arial",
  );
  await expect(editor.locator('span[style*="font-size"]')).toHaveCSS(
    "font-size",
    "20px",
  );
  await page.getByRole("button", { name: "Align center", exact: true }).click();
  await expect(editor.locator("p")).toHaveCSS("text-align", "center");

  await page
    .getByLabel("Block type", { exact: true })
    .selectOption("heading-2");
  await expect(editor.locator("h2")).toHaveText(
    "A paragraph with deliberate formatting.",
  );
  // A completed autosave also separates the user's structural undo groups.
  await expect(page.getByText("Saved locally", { exact: true })).toBeVisible();
  await page
    .getByLabel("Block type", { exact: true })
    .selectOption("codeBlock");
  await expect(editor.locator("pre code")).toHaveText(
    "A paragraph with deliberate formatting.",
  );
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  await expect(editor.locator("h2")).toHaveText(
    "A paragraph with deliberate formatting.",
  );

  await page
    .getByLabel("Block type", { exact: true })
    .selectOption("paragraph");
  await selectText(page, editor);
  await page.keyboard.press("ArrowRight");
  await page.keyboard.type(" First addition.");
  await expect(page.getByText("Saved locally", { exact: true })).toBeVisible();
  await page.keyboard.type(" Second addition.");
  await expect(editor).toHaveText(
    "A paragraph with deliberate formatting. First addition. Second addition.",
  );
});

test("slash and plus menus work from the keyboard and preserve the chosen block structure", async ({
  page,
}) => {
  const editor = await newOpen(page, "Keyboard blocks");
  await editor.focus();
  await page.keyboard.type("/");
  await expect(
    page.getByRole("menu", { name: "Block commands" }),
  ).toBeVisible();
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Enter");
  await expect(
    page.getByRole("menu", { name: "Block commands" }),
  ).not.toBeVisible();
  await page.keyboard.type("A keyboard-created heading");
  await expect(editor.locator("h2")).toHaveText("A keyboard-created heading");
  await expect(editor).not.toContainText("/");

  await page
    .getByRole("button", { name: "Insert block after 1", exact: true })
    .click();
  await expect(
    page.getByRole("menuitem", { name: "Text", exact: true }),
  ).toBeFocused();
  for (let step = 0; step < 4; step++) await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("menu", { name: "Block commands" }),
  ).not.toBeVisible();
  await page.keyboard.type("A list item created without a mouse menu");
  await expect(editor.locator("ul li")).toContainText(
    "A list item created without a mouse menu",
  );

  await page
    .getByRole("button", { name: "Insert block after 2", exact: true })
    .click();
  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("menu", { name: "Block commands" }),
  ).not.toBeVisible();
  await expect(editor).toBeFocused();
  await expect(editor.locator("h2")).toHaveText("A keyboard-created heading");
  await expect(editor.locator("ul")).toHaveCount(1);
});

test("block moves and delete/undo keep each idea with its own text, including after reload", async ({
  page,
}) => {
  const editor = await newOpen(page, "Stable ideas exercise");
  await editor.fill("First synthetic block");
  await editor.focus();
  await page.keyboard.press("End");
  await page.keyboard.press("Enter");
  await page.keyboard.type("Second synthetic block");
  const paragraphs = editor.locator(":scope > p");
  await expect(paragraphs).toHaveCount(2);
  const firstId = await paragraphs.nth(0).getAttribute("data-block-id");
  const secondId = await paragraphs.nth(1).getAttribute("data-block-id");
  expect(firstId).toBeTruthy();
  expect(secondId).toBeTruthy();
  expect(firstId).not.toBe(secondId);
  const firstIdea = page.locator(`textarea[data-idea-id="${firstId}"]`);
  const secondIdea = page.locator(`textarea[data-idea-id="${secondId}"]`);
  await firstIdea.fill("Idea belonging only to the first block");
  await secondIdea.fill("Idea belonging only to the second block");

  const secondActions = await blockActions(page, 1);
  await secondActions
    .getByRole("button", { name: "Move up", exact: true })
    .click();
  await expect(paragraphs).toHaveText([
    "Second synthetic block",
    "First synthetic block",
  ]);
  await expect(page.locator(".rich-editor-idea textarea").nth(0)).toHaveValue(
    "Idea belonging only to the second block",
  );
  await expect(page.locator(".rich-editor-idea textarea").nth(1)).toHaveValue(
    "Idea belonging only to the first block",
  );
  await expect(firstIdea).toHaveValue("Idea belonging only to the first block");

  const movedActions = await blockActions(page, 0);
  await movedActions
    .getByRole("button", { name: "Delete block", exact: true })
    .click();
  await expect(paragraphs).toHaveText(["First synthetic block"]);
  await expect(secondIdea).toHaveCount(0);
  await expect(firstIdea).toHaveValue("Idea belonging only to the first block");
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  await expect(paragraphs).toHaveCount(2);
  await expect(secondIdea).toHaveValue(
    "Idea belonging only to the second block",
  );
  await expect(firstIdea).toHaveValue("Idea belonging only to the first block");

  await expect(page.getByText("Saved locally", { exact: true })).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: /Stable ideas exercise/ }).click();
  await expect(firstIdea).toHaveValue("Idea belonging only to the first block");
  await expect(secondIdea).toHaveValue(
    "Idea belonging only to the second block",
  );
  await expect(editor.locator(`[data-block-id="${firstId}"]`)).toHaveText(
    "First synthetic block",
  );
  await expect(editor.locator(`[data-block-id="${secondId}"]`)).toHaveText(
    "Second synthetic block",
  );
});

test("rich paste keeps safe text and links without executable HTML or remote-media requests", async ({
  page,
}) => {
  const externalRequests: string[] = [];
  const errors: string[] = [];
  page.on("request", (request) => {
    if (!request.url().startsWith("http://127.0.0.1:5192/"))
      externalRequests.push(request.url());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  const editor = await newOpen(page, "Untrusted paste exercise");
  await editor.focus();
  await editor.evaluate((element) => {
    const transfer = new DataTransfer();
    transfer.setData(
      "text/html",
      '<p><strong>Safe pasted words</strong> <a href="https://example.test/read">safe source</a> <a href="javascript:window.__editorPasteExecuted=true">unsafe source</a><img src="https://untrusted.invalid/paste-probe/image" onerror="window.__editorPasteExecuted=true"><iframe src="https://untrusted.invalid/paste-probe/frame"></iframe><script>window.__editorPasteExecuted=true</script><span style="background-image:url(https://untrusted.invalid/paste-probe/style)"> inert styled text</span></p>',
    );
    transfer.setData(
      "text/plain",
      "Safe pasted words safe source unsafe source inert styled text",
    );
    element.dispatchEvent(
      new ClipboardEvent("paste", {
        clipboardData: transfer,
        bubbles: true,
        cancelable: true,
      }),
    );
  });
  await expect(editor).toContainText("Safe pasted words");
  await expect(editor.locator("strong")).toHaveText("Safe pasted words");
  await expect(
    editor.locator('a[href="https://example.test/read"]'),
  ).toHaveText("safe source");
  await expect(editor.locator('a[href^="javascript:"]')).toHaveCount(0);
  await expect(
    editor.locator("img, iframe, script, object, embed, video, audio"),
  ).toHaveCount(0);
  await expect(editor.locator('[style*="url("]')).toHaveCount(0);
  expect(
    await page.evaluate(
      () =>
        (window as unknown as Record<string, unknown>).__editorPasteExecuted,
    ),
  ).toBeUndefined();

  const beforeDrop = await editor.innerText();
  await editor.evaluate((element) => {
    const transfer = new DataTransfer();
    transfer.setData(
      "text/html",
      '<img src="https://untrusted.invalid/paste-probe/dropped-image">',
    );
    element.dispatchEvent(
      new DragEvent("drop", {
        dataTransfer: transfer,
        bubbles: true,
        cancelable: true,
      }),
    );
  });
  await expect(editor).toHaveText(beforeDrop);
  expect(externalRequests).toEqual([]);
  expect(errors).toEqual([]);
});
