import { expect, test, type Page } from "@playwright/test";
import {
  configureSDK,
  emitSDKText,
  emitSDKReasoning,
  installSDKFixture,
  sdkSnapshot,
} from "./sdk-fixture";

async function newDocument(
  page: Page,
  protection: "Open" | "Protected",
  title: string,
  text: string,
): Promise<void> {
  await page.getByRole("button", { name: "New document" }).click();
  await page
    .locator(".new-popover")
    .getByRole("button", { name: `${protection} document`, exact: false })
    .click();
  await page
    .getByRole("textbox", { name: "Document title", exact: true })
    .fill(title);
  await page
    .getByRole("textbox", { name: "Document text", exact: true })
    .fill(text);
  await expect(page.getByText("Saved locally", { exact: true })).toBeVisible();
}

async function createUser(page: Page, name: string): Promise<void> {
  await page.getByRole("button", { name: "Local users", exact: false }).click();
  await page
    .getByRole("button", { name: "Create local user", exact: true })
    .click();
  await page.getByLabel("Local name", { exact: true }).fill(name);
  await page
    .getByLabel("Password", { exact: true })
    .fill(`${name} synthetic private passphrase`);
  await page.getByRole("button", { name: "Create & unlock" }).click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect(
    page.getByRole("button", { name: "Lock protected documents", exact: true }),
  ).toBeVisible();
}

async function ask(page: Page, question: string): Promise<void> {
  await page.getByRole("button", { name: "Ask AI", exact: true }).click();
  await page.getByLabel("Ask local AI", { exact: true }).fill(question);
  await expect(
    page.getByRole("button", { name: "Send question", exact: true }),
  ).toBeEnabled();
  await page
    .getByRole("button", { name: "Send question", exact: true })
    .click();
}

async function lastTextJob(page: Page, expectedCount: number): Promise<string> {
  await expect
    .poll(
      async () =>
        (await sdkSnapshot(page)).invocations.filter(
          (job) => job.input.messages,
        ).length,
    )
    .toBe(expectedCount);
  return (await sdkSnapshot(page)).invocations
    .filter((job) => job.input.messages)
    .at(-1)!.jobId;
}

test.beforeEach(async ({ page }) => {
  // This fixture replaces the Browser injection only; production SDK code is unmodified.
  await installSDKFixture(page);
  await page.goto("/");
});

test("packaged SDK streams a Context answer, reports completion and releases the model session", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await newDocument(
    page,
    "Open",
    "Context fixture",
    "The synthetic garden grows slowly and welcomes patient writers.",
  );
  await page
    .getByRole("textbox", { name: "Document text", exact: true })
    .focus();
  await ask(page, "Give one clearer sentence about this passage.");
  await expect(page.locator(".ai-answer")).toHaveText(
    "A clearer local sentence.",
  );
  await expect(
    page.getByText("Completed locally", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Insert at cursor ↗", exact: true }),
  ).toBeEnabled();
  await expect
    .poll(async () => (await sdkSnapshot(page)).released.length)
    .toBe(1);
  const state = await sdkSnapshot(page);
  expect(state.requests).toHaveLength(1);
  expect(state.requests[0]).toMatchObject({
    requirements: {
      task: "text-generation",
      contract: { id: "chat-v2", revision: 1 },
    },
  });
  const prompt = state.invocations[0].input.messages![0].content;
  expect(prompt).toContain("selectedPassage");
  expect(prompt).toContain("The synthetic garden grows slowly");
  expect(prompt).toContain("Give one clearer sentence");
  await expect(page.locator(".ai-answer")).not.toContainText(
    "NOT-A-TEXT-DELTA",
  );
  expect(state.listeners).toBe(0);
  expect(errors).toEqual([]);
});

test("editing clears completed or partial answers and rejects late model output", async ({
  page,
}) => {
  await newDocument(
    page,
    "Open",
    "Editable fixture",
    "A synthetic first sentence.",
  );
  await ask(page, "Offer a clearer sentence.");
  await expect(page.locator(".ai-answer")).toHaveText(
    "A clearer local sentence.",
  );
  const editor = page.getByRole("textbox", {
    name: "Document text",
    exact: true,
  });
  await editor.fill("A synthetic replacement written by the owner.");
  await expect(page.locator(".ai-answer")).toHaveCount(0);
  await expect(
    page.getByText("Completed locally", { exact: true }),
  ).not.toBeVisible();
  await expect(
    page.getByRole("button", { name: "Insert at cursor ↗", exact: true }),
  ).toHaveCount(0);

  await configureSDK(page, { holdText: true, deferCancellation: true });
  await ask(page, "Hold this synthetic response until the test releases it.");
  const jobId = await lastTextJob(page, 2);
  await emitSDKText(page, jobId, "SYNTHETIC_PARTIAL_BEFORE_EDIT");
  await expect(page.locator(".ai-answer")).toHaveText(
    "SYNTHETIC_PARTIAL_BEFORE_EDIT",
  );
  await editor.fill(
    "A new revision must never receive the previous model response.",
  );
  await expect(page.locator(".ai-answer")).toHaveCount(0);
  await expect
    .poll(async () => (await sdkSnapshot(page)).cancelled)
    .toContain(jobId);
  await emitSDKText(page, jobId, "LATE_OUTPUT_AFTER_EDIT");
  await emitSDKText(page, jobId, "LATE_OUTPUT_AFTER_EDIT", true);
  await expect
    .poll(async () => (await sdkSnapshot(page)).released.length)
    .toBe(2);
  await expect(page.locator("body")).not.toContainText(
    "LATE_OUTPUT_AFTER_EDIT",
  );
  await expect(editor).toHaveText(
    "A new revision must never receive the previous model response.",
  );
  await expect(
    page.getByRole("button", { name: "Insert at cursor ↗", exact: true }),
  ).toHaveCount(0);
});

test("Open source search excludes unlocked Alice documents before embeddings and advanced filtering", async ({
  page,
}) => {
  await createUser(page, "Alice");
  await newDocument(
    page,
    "Protected",
    "ALICE_PRIVATE_SOURCE_TITLE",
    "ALICE_PRIVATE_SOURCE_BODY never belongs in Open retrieval.",
  );
  await newDocument(
    page,
    "Open",
    "Shared source fixture",
    "The community garden uses patient planting and shared watering.",
  );
  await newDocument(
    page,
    "Open",
    "Shared destination fixture",
    "We want to write a public paragraph about a community garden.",
  );
  await page
    .getByRole("button", { name: "Related sources", exact: true })
    .click();
  await page.getByText("Source filters", { exact: true }).click();
  await page
    .getByLabel("Filter matches with local AI", { exact: true })
    .check();
  await page
    .getByRole("button", {
      name: "Find sources for selection / draft",
      exact: true,
    })
    .click();
  await expect(page.locator(".source-card")).toHaveCount(1);
  await expect(page.locator(".source-card")).toContainText(
    "Shared source fixture",
  );
  await expect(page.locator(".related-panel")).not.toContainText(
    "ALICE_PRIVATE_SOURCE",
  );
  await expect
    .poll(async () => (await sdkSnapshot(page)).released.length)
    .toBe(2);
  const state = await sdkSnapshot(page);
  expect(state.requests.map((request) => request.requirements?.task)).toEqual([
    "feature-extraction",
    "text-generation",
  ]);
  const inputs = JSON.stringify(state.invocations.map((job) => job.input));
  expect(inputs).toContain("community garden");
  expect(inputs).not.toContain("ALICE_PRIVATE_SOURCE_BODY");
  expect(inputs).not.toContain("ALICE_PRIVATE_SOURCE_TITLE");
  expect(state.listeners).toBe(0);
});

test("locking Alice clears private AI state and late output cannot enter Bob or Open content", async ({
  page,
}) => {
  await newDocument(
    page,
    "Open",
    "Shared after lock",
    "This shared document stays useful without any local login.",
  );
  await createUser(page, "Alice");
  await newDocument(
    page,
    "Protected",
    "ALICE_LOCK_PRIVATE_TITLE",
    "ALICE_LOCK_PRIVATE_BODY belongs only to this local owner.",
  );
  await configureSDK(page, { holdText: true, deferCancellation: true });
  await ask(page, "Explain this protected synthetic passage.");
  const jobId = await lastTextJob(page, 1);
  await emitSDKText(page, jobId, "ALICE_PRIVATE_AI_PARTIAL");
  await emitSDKReasoning(page, jobId, "ALICE_PRIVATE_REASONING");
  await expect(page.locator(".ai-answer")).toHaveText(
    "ALICE_PRIVATE_AI_PARTIAL",
  );
  await page
    .getByRole("textbox", { name: "Search documents", exact: true })
    .fill("ALICE_LOCK_PRIVATE");
  await page
    .getByRole("button", { name: "Lock protected documents", exact: true })
    .click();
  await expect(
    page.getByRole("textbox", { name: "Search documents", exact: true }),
  ).toHaveValue("");
  await expect(page.locator("body")).not.toContainText("ALICE_LOCK_PRIVATE");
  await expect(page.locator("body")).not.toContainText(
    "ALICE_PRIVATE_AI_PARTIAL",
  );
  await expect(page.locator(".model-reasoning")).toHaveCount(0);
  await expect
    .poll(async () => (await sdkSnapshot(page)).cancelled)
    .toContain(jobId);

  // A new owner and Open destination must remain independent of the old job.
  await createUser(page, "Bob");
  await page.getByRole("button", { name: /Shared after lock/ }).click();
  await emitSDKText(page, jobId, "ALICE_PRIVATE_AI_LATE");
  await emitSDKText(page, jobId, "ALICE_PRIVATE_AI_LATE", true);
  await expect
    .poll(async () => (await sdkSnapshot(page)).released.length)
    .toBe(1);
  await expect(page.locator("body")).not.toContainText("ALICE_PRIVATE_AI_LATE");
  await expect(page.locator(".ai-answer")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Insert at cursor ↗", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("textbox", { name: "Document text", exact: true }),
  ).toHaveText("This shared document stays useful without any local login.");
});

test("writing actions preserve originals and provide undo for generated text and ideas", async ({
  page,
}) => {
  page.on("dialog", (dialog) => dialog.accept());
  await newDocument(
    page,
    "Open",
    "Writing actions",
    "Original synthetic text with [make this clearer].",
  );
  const editor = page.getByRole("textbox", {
    name: "Document text",
    exact: true,
  });
  const idea = page.getByRole("textbox", {
    name: "Idea for block 1",
    exact: true,
  });
  await idea.fill("The original paragraph idea");
  await page
    .getByRole("button", { name: "Generate idea", exact: true })
    .click();
  await expect(idea).toHaveValue("A clearer local sentence.");
  await page
    .getByRole("button", { name: "Undo generated idea", exact: true })
    .click();
  await expect(idea).toHaveValue("The original paragraph idea");
  await page
    .getByRole("button", { name: "Generate text", exact: true })
    .click();
  await expect(editor).toHaveText("A clearer local sentence.");
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  await expect(editor).toHaveText(
    "Original synthetic text with [make this clearer].",
  );
  await editor.focus();
  await page
    .getByRole("button", { name: "Implement [comments]", exact: true })
    .click();
  await expect(editor.locator(":scope > p")).toHaveText([
    "Original synthetic text with [make this clearer].",
    "A clearer local sentence.",
  ]);
  await page
    .getByLabel("Writing recommendation", { exact: true })
    .fill("Use shorter sentences.");
  await page
    .getByRole("button", { name: "Rewrite article", exact: true })
    .click();
  await expect(editor.locator(":scope > p")).toHaveText([
    "Original synthetic text with [make this clearer].",
    "A clearer local sentence.",
    "A clearer local sentence.",
    "A clearer local sentence.",
  ]);
});

test("thinking Off is the writing default and mode changes apply only to the next request", async ({ page }) => {
  await newDocument(page, "Open", "Thinking controls", "Original writing remains separate from the model reasoning.");
  await expect(page.getByLabel("Thinking mode", { exact: true })).toHaveValue("off");
  await configureSDK(page, { holdText: true });
  await ask(page, "Hold the first request.");
  const first = await lastTextJob(page, 1);
  await page.getByLabel("Thinking mode", { exact: true }).selectOption("on");
  expect((await sdkSnapshot(page)).invocations[0].input.thinking).toBe(false);
  await emitSDKText(page, first, "A complete first answer.", true);
  await expect(page.locator(".thinking-notice")).toContainText("Thinking Off applied");

  await configureSDK(page, { holdText: false, mode: "reasoning-answer" });
  await ask(page, "Use the next request's thinking setting.");
  await expect(page.locator(".ai-answer")).toHaveText("A clearer local sentence.");
  expect((await sdkSnapshot(page)).invocations[1].input.thinking).toBe(true);
  await page.locator(".model-reasoning summary").click();
  await expect(page.locator(".model-reasoning p")).toHaveText("SYNTHETIC_REASONING_SEPARATE");
  await expect(page.getByRole("textbox", { name: "Document text", exact: true })).not.toContainText("SYNTHETIC_REASONING");
  await expect(page.locator(".ai-answer")).not.toContainText("SYNTHETIC_REASONING");

  await page.getByLabel("Thinking mode", { exact: true }).selectOption("default");
  await configureSDK(page, { mode: "normal" });
  await ask(page, "Use the model default.");
  await lastTextJob(page, 3);
  expect((await sdkSnapshot(page)).invocations[2].input).not.toHaveProperty("thinking");
});

test("unsupported Off continues writing with a truthful fixed-mode notice", async ({ page }) => {
  page.on("dialog", dialog => dialog.accept());
  await newDocument(page, "Open", "Best effort", "Original prose.");
  await page.getByLabel("Idea for block 1", { exact: true }).fill("Original idea");
  await configureSDK(page, { mode: "unsupported" });
  await page.getByRole("button", { name: "Generate idea", exact: true }).click();
  await expect(page.getByLabel("Idea for block 1", { exact: true })).toHaveValue("A clearer local sentence.");
  await expect(page.locator(".thinking-notice")).toContainText("Thinking Off was not applied");
  await expect(page.locator(".thinking-notice")).toContainText("fixed mode");
  expect((await sdkSnapshot(page)).invocations).toHaveLength(1);
});

test("reasoning-only output never replaces prose or becomes an idea", async ({ page }) => {
  page.on("dialog", dialog => dialog.accept());
  await newDocument(page, "Open", "Reasoning boundary", "Original text must survive.");
  const idea = page.getByLabel("Idea for block 1", { exact: true });
  await idea.fill("Original idea must survive.");
  await configureSDK(page, { mode: "reasoning-only" });
  await page.getByRole("button", { name: "Generate text", exact: true }).click();
  await expect(page.locator(".job-status")).toContainText("reasoning without a final answer");
  await expect(page.getByRole("textbox", { name: "Document text", exact: true })).toHaveText("Original text must survive.");
  await expect(idea).toHaveValue("Original idea must survive.");
  await expect(page.locator(".ai-answer")).not.toContainText("SYNTHETIC_REASONING_ONLY");
  await expect(page.getByRole("button", { name: "Use incomplete preview…", exact: true })).toHaveCount(0);
});

test("limited writing actions preserve every original and require deliberate preview use", async ({ page }) => {
  page.on("dialog", dialog => dialog.accept());
  await newDocument(page, "Open", "Incomplete writing", "Original paragraph with [a comment].");
  const idea = page.getByLabel("Idea for block 1", { exact: true });
  await idea.fill("Original idea.");
  await page.getByLabel("Writing recommendation", { exact: true }).fill("Use shorter prose.");
  await configureSDK(page, { mode: "limit" });
  for (const action of ["Generate idea", "Generate text", "Implement [comments]", "Rewrite article"]) {
    await page.getByRole("button", { name: action, exact: true }).click();
    await expect(page.locator(".job-status")).toContainText("generation limit");
    await expect(page.getByRole("textbox", { name: "Document text", exact: true })).toHaveText("Original paragraph with [a comment].");
    await expect(idea).toHaveValue("Original idea.");
    await expect(page.getByRole("button", { name: "Use incomplete preview…", exact: true })).toBeEnabled();
  }
});

test("legacy peers continue generation with unknown completion and no automatic edit", async ({ page }) => {
  page.on("dialog", dialog => dialog.accept());
  await newDocument(page, "Open", "Legacy compatibility", "Original legacy writing.");
  await page.getByLabel("Idea for block 1", { exact: true }).fill("Original legacy idea.");
  await configureSDK(page, { mode: "legacy" });
  await page.getByRole("button", { name: "Generate idea", exact: true }).click();
  await expect(page.locator(".job-status")).toContainText("Completion is unconfirmed");
  await expect(page.getByLabel("Idea for block 1", { exact: true })).toHaveValue("Original legacy idea.");
  await expect(page.locator(".thinking-notice")).toContainText("this Oryn version cannot apply the control");
  expect((await sdkSnapshot(page)).invocations[0].input).not.toHaveProperty("thinking");
  expect((await sdkSnapshot(page)).invocations[0].input.maxTokens).toBeLessThanOrEqual(512);
});

test("Stop clears reasoning and control metadata and rejects late reasoning", async ({ page }) => {
  await newDocument(page, "Open", "Cancel channels", "Only current output belongs in this workspace.");
  await configureSDK(page, { holdText: true, deferCancellation: true });
  await ask(page, "Wait for cancellation.");
  const jobId = await lastTextJob(page, 1);
  await emitSDKReasoning(page, jobId, "CANCEL_REASONING_MARKER");
  await expect(page.locator(".model-reasoning")).toHaveCount(1);
  await page.getByRole("button", { name: "■ Stop", exact: true }).click();
  await expect(page.locator(".model-reasoning")).toHaveCount(0);
  await expect(page.locator(".thinking-notice")).toHaveCount(0);
  await emitSDKReasoning(page, jobId, "LATE_REASONING_MARKER");
  await emitSDKText(page, jobId, "A late answer.", true);
  await expect(page.locator("body")).not.toContainText("LATE_REASONING_MARKER");
});
