import { describe, expect, it } from "vitest";
import { articlePlainText, escapeHTML, renderNode } from "../src/export";
import { newDocument } from "../src/model";

describe("safe article export", () => {
  it("preserves paragraph separation and requires explicit idea inclusion", () => {
    const doc = newDocument("open", "document");
    doc.title = "Synthetic draft";
    doc.content.content = [
      {
        type: "paragraph",
        attrs: { blockId: "one" },
        content: [{ type: "text", text: "First paragraph." }],
      },
      {
        type: "paragraph",
        attrs: { blockId: "two" },
        content: [{ type: "text", text: "Second paragraph." }],
      },
    ];
    doc.ideas.one = "Private drafting thought";
    expect(articlePlainText(doc)).toBe(
      "Synthetic draft\n\nFirst paragraph.\n\nSecond paragraph.",
    );
    expect(articlePlainText(doc, true)).toContain(
      "First paragraph.\nIdea: Private drafting thought\n\nSecond",
    );
  });
  it("escapes text and rejects executable or automatic resource markup", () => {
    expect(escapeHTML('<script>"&')).toBe("&lt;script&gt;&quot;&amp;");
    expect(
      renderNode({
        type: "text",
        text: "<img src=https://tracker.invalid>",
        marks: [{ type: "link", attrs: { href: "javascript:alert(1)" } }],
      }),
    ).toBe("&lt;img src=https://tracker.invalid&gt;");
    expect(
      renderNode({ type: "image", attrs: { src: "https://tracker.invalid" } }),
    ).not.toContain("src=");
    expect(
      renderNode({
        type: "text",
        text: "Link",
        marks: [
          { type: "link", attrs: { href: "https://example.invalid/?a=1&b=2" } },
        ],
      }),
    ).toContain('rel="noreferrer noopener"');
  });
});
