import { describe, expect, it } from "vitest";
import {
  documentBlocks,
  nodeText,
  normalizeContent,
  plainParagraph,
  pruneIdeas,
  reconcileIdeas,
  safeLink,
  selectionContext,
} from "../src/editor-utils";
import type { RichNode } from "../src/model";
import { getSchema } from "@tiptap/core";
import StarterKit from "@tiptap/starter-kit";
import { AllSelection, EditorState, TextSelection } from "@tiptap/pm/state";
import { wrapIn } from "@tiptap/pm/commands";
import {
  collapseAllSelection,
  createStableBlockPlugin,
  StableBlocks,
} from "../src/RichEditor";

describe("editor links and imported content", () => {
  it("permits deliberate web/email links but rejects executable, embedded, relative and credential URLs", () => {
    expect(safeLink("https://example.test/article")).toBe(
      "https://example.test/article",
    );
    expect(safeLink("mailto:writer@example.test")).toBe(
      "mailto:writer@example.test",
    );
    for (const input of [
      "javascript:alert(1)",
      "java\nscript:alert(1)",
      "data:text/html,<script>x</script>",
      "file:///secret",
      "//example.test",
      "/relative",
      "https://user:password@example.test",
      "https://example.test/with space",
    ])
      expect(safeLink(input)).toBeNull();
  });

  it("keeps supported formatting while removing dangerous URLs, remote media and arbitrary CSS", () => {
    const content: RichNode = {
      type: "doc",
      content: [
        {
          type: "paragraph",
          attrs: {
            blockId: "block-a",
            onclick: "steal()",
            style: "background:url(https://example.test/track)",
            indent: 99,
          },
          content: [
            {
              type: "text",
              text: "<script>is inert text</script>",
              marks: [
                { type: "bold" },
                { type: "link", attrs: { href: "javascript:alert(1)" } },
                {
                  type: "textStyle",
                  attrs: {
                    color: "#123456",
                    fontFamily: "url(evil)",
                    fontSize: "999px",
                    backgroundImage: "url(evil)",
                  },
                },
              ],
            },
            { type: "image", attrs: { src: "https://example.test/tracker" } },
          ],
        },
      ],
    };
    const normalized = normalizeContent(content);
    expect(normalized.content?.[0].attrs).toEqual({
      blockId: "block-a",
      indent: 8,
    });
    expect(normalized.content?.[0].content).toEqual([
      {
        type: "text",
        text: "<script>is inert text</script>",
        marks: [
          { type: "bold" },
          { type: "textStyle", attrs: { color: "#123456" } },
        ],
      },
    ]);
  });

  it("assigns distinct identities without changing valid existing block identities", () => {
    let counter = 0;
    const result = normalizeContent(
      {
        type: "doc",
        content: [
          plainParagraph("first", "stable-a"),
          plainParagraph("duplicate", "stable-a"),
          { type: "paragraph" },
          plainParagraph("third", "stable-c"),
        ],
      },
      () => `generated-${++counter}`,
    );
    expect(documentBlocks(result).map((block) => block.id)).toEqual([
      "stable-a",
      "generated-1",
      "generated-2",
      "stable-c",
    ]);
    expect(
      documentBlocks(normalizeContent(result)).map((block) => block.id),
    ).toEqual(["stable-a", "generated-1", "generated-2", "stable-c"]);
  });

  it("keeps plain generation as text rather than interpreting model-generated HTML", () => {
    const result = plainParagraph(
      '<img src="https://example.test/track">\nhello',
      "a",
    );
    expect(result.content?.map((node) => node.type)).toEqual([
      "text",
      "hardBreak",
      "text",
    ]);
    expect(nodeText(result)).toBe(
      '<img src="https://example.test/track">\nhello',
    );
  });
});

describe("block ideas and bounded selection context", () => {
  it("retains idea ownership across reordering and removes orphan/empty notes", () => {
    const ideas = {
      a: "first idea",
      b: "second idea",
      c: "deleted note",
      empty: "  ",
    };
    expect(pruneIdeas(ideas, ["b", "a", "empty"])).toEqual({
      b: "second idea",
      a: "first idea",
    });
    expect(ideas.c).toBe("deleted note");
  });

  it("restores a deleted block idea on text undo without resurrecting an externally cleared live idea", () => {
    expect(
      reconcileIdeas(
        { a: "current" },
        { b: "restored", a: "old" },
        new Set(["a"]),
        ["a", "b"],
      ),
    ).toEqual({ a: "current", b: "restored" });
    expect(
      reconcileIdeas({}, { a: "cleared by another tab" }, new Set(["a"]), [
        "a",
      ]),
    ).toEqual({});
  });

  it("bounds surrounding context and never substitutes a different selected subject", () => {
    const result = selectionContext(
      {
        type: "doc",
        content: [
          plainParagraph("one two three four five", "a"),
          plainParagraph("selected block text", "b"),
          plainParagraph("six seven eight nine ten", "c"),
        ],
      },
      "only this passage",
      "b",
      2,
    );
    expect(result).toBe(
      "Before (context only): four five\n\nSelected text: only this passage\n\nAfter (context only): six seven",
    );
    expect(
      selectionContext(
        { type: "doc", content: [plainParagraph("private text", "a")] },
        "explicit selection",
        "missing",
      ),
    ).toBe("explicit selection");
    expect(
      selectionContext(
        {
          type: "doc",
          content: [plainParagraph("before one subject after two", "a")],
        },
        "subject",
        "a",
        1,
      ),
    ).toBe(
      "Before (context only): one\n\nSelected text: subject\n\nAfter (context only): after",
    );
    expect(
      selectionContext(
        { type: "doc", content: [plainParagraph("before subject after", "a")] },
        "subject",
        "a",
        0,
      ),
    ).toBe("Selected text: subject");
    expect(
      selectionContext(
        {
          type: "doc",
          content: [
            plainParagraph("before first", "a"),
            plainParagraph("second after", "b"),
          ],
        },
        "first\nsecond",
        "a",
        1,
      ),
    ).toBe(
      "Before (context only): before\n\nSelected text: first\nsecond\n\nAfter (context only): after",
    );
  });

  it("produces a usable empty paragraph from an unsupported or empty document", () => {
    expect(normalizeContent({ type: "doc" }, () => "empty").content).toEqual([
      { type: "paragraph", attrs: { blockId: "empty" } },
    ]);
    expect(
      normalizeContent(
        { type: "image", attrs: { src: "https://example.test/image" } },
        () => "empty",
      ).content,
    ).toEqual([{ type: "paragraph", attrs: { blockId: "empty" } }]);
  });

  it("repairs imported block nesting to a renderable inert schema", () => {
    let index = 0;
    const normalized = normalizeContent(
      {
        type: "doc",
        content: [
          {
            type: "paragraph",
            content: [
              {
                type: "heading",
                content: [{ type: "text", text: "nested heading" }],
              },
            ],
          },
          { type: "bulletList" },
          {
            type: "codeBlock",
            content: [
              {
                type: "text",
                text: "safe",
                marks: [
                  { type: "link", attrs: { href: "https://example.test" } },
                ],
              },
              { type: "hardBreak" },
            ],
          },
        ],
      },
      () => `block-${++index}`,
    );
    expect(normalized.content?.[0].content).toEqual([
      { type: "text", text: "nested heading" },
    ]);
    expect(normalized.content?.[1].content).toEqual([
      { type: "listItem", content: [{ type: "paragraph" }] },
    ]);
    expect(normalized.content?.[2].content).toEqual([
      { type: "text", text: "safe\n" },
    ]);
  });
});

describe("actual editor block transactions", () => {
  const schema = getSchema([StarterKit, StableBlocks]);
  const stateFor = (content: RichNode) =>
    EditorState.create({
      schema,
      doc: schema.nodeFromJSON(content),
      plugins: [createStableBlockPlugin()],
    });

  it("splitting keeps the original ID only on the first resulting block", () => {
    let state = stateFor({
      type: "doc",
      content: [plainParagraph("hello", "original")],
    });
    state = state.applyTransaction(state.tr.split(3)).state;
    const ids = documentBlocks(state.doc.toJSON() as RichNode).map(
      (block) => block.id,
    );
    expect(ids).toHaveLength(2);
    expect(ids[0]).toBe("original");
    expect(ids[1]).not.toBe("original");
  });

  it("wrapping a paragraph keeps its idea identity on the new top-level block", () => {
    let state = stateFor({
      type: "doc",
      content: [plainParagraph("hello", "original")],
    });
    state = state.apply(
      state.tr.setSelection(TextSelection.create(state.doc, 2)),
    );
    expect(
      wrapIn(schema.nodes.blockquote)(state, (transaction) => {
        state = state.applyTransaction(transaction).state;
      }),
    ).toBe(true);
    expect(state.doc.firstChild?.type.name).toBe("blockquote");
    expect(state.doc.firstChild?.attrs.blockId).toBe("original");
  });

  it("moving a complete block never changes identities or attaches an idea to its neighbor", () => {
    let state = stateFor({
      type: "doc",
      content: [plainParagraph("first", "a"), plainParagraph("second", "b")],
    });
    const first = state.doc.firstChild!;
    const end = state.doc.content.size;
    const transaction = state.tr.delete(0, first.nodeSize);
    transaction.insert(transaction.mapping.map(end), first);
    state = state.applyTransaction(transaction).state;
    const ids = documentBlocks(state.doc.toJSON() as RichNode).map(
      (block) => block.id,
    );
    expect(ids).toEqual(["b", "a"]);
    expect(pruneIdeas({ a: "first idea", b: "second idea" }, ids)).toEqual({
      b: "second idea",
      a: "first idea",
    });
  });

  it("collapses document-wide selection before typing so existing text cannot be replaced accidentally", () => {
    let state = stateFor({
      type: "doc",
      content: [plainParagraph("Keep the original.", "a")],
    });
    state = state.apply(state.tr.setSelection(new AllSelection(state.doc)));
    const collapse = collapseAllSelection(state, "right");
    expect(collapse).not.toBeNull();
    state = state.apply(collapse!);
    expect(state.selection.empty).toBe(true);
    state = state.applyTransaction(state.tr.insertText(" Append.")).state;
    expect(state.doc.textContent).toBe("Keep the original. Append.");
    expect(collapseAllSelection(state, "left")).toBeNull();

    state = state.apply(state.tr.setSelection(new AllSelection(state.doc)));
    state = state.apply(collapseAllSelection(state, "left")!);
    state = state.applyTransaction(state.tr.insertText("Prepend. ")).state;
    expect(state.doc.textContent).toBe("Prepend. Keep the original. Append.");
  });
});
