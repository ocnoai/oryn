import DOMPurify from "dompurify";
import type { EditorDocument, RichNode } from "./model";
import { textOf } from "./ai";

export function escapeHTML(value: string): string {
  return value.replace(
    /[&<>"']/g,
    (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        ch
      ]!,
  );
}

const safeLink = (url: unknown): string =>
  typeof url === "string" && /^(https?:\/\/|mailto:)/i.test(url) ? url : "";
const styleValue = (value: unknown): string =>
  typeof value === "string" && /^[\w\s#(),.%'-]{1,80}$/.test(value)
    ? value
    : "";

function inline(node: RichNode): string {
  let text = node.type === "hardBreak" ? "<br>" : escapeHTML(node.text ?? "");
  for (const mark of node.marks ?? []) {
    const tag = (
      {
        bold: "strong",
        italic: "em",
        underline: "u",
        strike: "s",
        code: "code",
        subscript: "sub",
        superscript: "sup",
      } as Record<string, string>
    )[mark.type];
    if (tag) text = `<${tag}>${text}</${tag}>`;
    if (mark.type === "link") {
      const href = safeLink(mark.attrs?.href);
      if (href)
        text = `<a href="${escapeHTML(href)}" rel="noreferrer noopener">${text}</a>`;
    }
    if (mark.type === "highlight") text = `<mark>${text}</mark>`;
    if (mark.type === "textStyle") {
      const styles = [
        "color",
        "fontFamily",
        "fontSize",
        "backgroundColor",
      ].flatMap((key) => {
        const value = styleValue(mark.attrs?.[key]);
        return value
          ? [`${key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}:${value}`]
          : [];
      });
      if (styles.length)
        text = `<span style="${escapeHTML(styles.join(";"))}">${text}</span>`;
    }
  }
  return text;
}

export function renderNode(node: RichNode): string {
  if (node.text !== undefined || node.type === "hardBreak") return inline(node);
  const children = (node.content ?? []).map(renderNode).join("");
  const tag =
    (
      {
        doc: "",
        paragraph: "p",
        blockquote: "blockquote",
        codeBlock: "pre",
        bulletList: "ul",
        orderedList: "ol",
        listItem: "li",
        horizontalRule: "hr",
      } as Record<string, string>
    )[node.type] ??
    (node.type === "heading"
      ? `h${Math.min(3, Math.max(1, Number(node.attrs?.level) || 1))}`
      : "p");
  if (!tag) return children;
  if (tag === "hr") return "<hr>";
  const align = node.attrs?.textAlign;
  const attr = ["left", "center", "right", "justify"].includes(String(align))
    ? ` style="text-align:${align}"`
    : "";
  return `<${tag}${attr}>${children || "<br>"}</${tag}>`;
}

export function articleHTML(doc: EditorDocument, includeIdeas = false): string {
  const body = (doc.content.content ?? [])
    .map((node) => {
      const idea = doc.ideas[String(node.attrs?.blockId ?? "")];
      return includeIdeas && idea?.trim()
        ? `<section class="with-idea"><div>${renderNode(node)}</div><aside>${escapeHTML(idea)}</aside></section>`
        : renderNode(node);
    })
    .join("");
  return DOMPurify.sanitize(
    `<h1>${escapeHTML(doc.title || "Untitled")}</h1>${body}`,
    {
      ALLOWED_TAGS: [
        "h1",
        "h2",
        "h3",
        "p",
        "br",
        "strong",
        "em",
        "u",
        "s",
        "code",
        "sub",
        "sup",
        "a",
        "mark",
        "span",
        "blockquote",
        "pre",
        "ul",
        "ol",
        "li",
        "hr",
        "section",
        "div",
        "aside",
      ],
      ALLOWED_ATTR: ["href", "rel", "style", "class"],
    },
  );
}

export function standaloneHTML(
  doc: EditorDocument,
  includeIdeas = false,
): string {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="referrer" content="no-referrer"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escapeHTML(doc.title || "Untitled")}</title><style>body{max-width:760px;margin:60px auto;padding:0 24px;color:#222;font:18px/1.7 Georgia,serif}h1,h2,h3{line-height:1.25}a{color:#275b4a}pre{white-space:pre-wrap;background:#f5f3ee;padding:16px}blockquote{border-left:3px solid #aaa;padding-left:20px}.with-idea{display:grid;grid-template-columns:3fr 1fr;gap:30px}.with-idea aside{font:13px/1.5 system-ui;color:#666;border-left:1px solid #ddd;padding:12px}@media print{body{margin:0;max-width:none}a{color:inherit}}</style></head><body>${articleHTML(doc, includeIdeas)}</body></html>`;
}

export function downloadFile(
  text: string,
  filename: string,
  type = "application/json",
): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename.replace(/[/\\\u0000-\u001f]/g, "_");
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function articlePlainText(
  doc: EditorDocument,
  includeIdeas = false,
): string {
  const blocks = (doc.content.content ?? []).map((node) => {
    const idea = doc.ideas[String(node.attrs?.blockId ?? "")];
    return [textOf(node), includeIdeas && idea?.trim() ? `Idea: ${idea}` : ""]
      .filter(Boolean)
      .join("\n");
  });
  return [doc.title || "Untitled", ...blocks].join("\n\n");
}

export async function copyArticle(
  doc: EditorDocument,
  includeIdeas: boolean,
): Promise<void> {
  const html = articleHTML(doc, includeIdeas);
  const text = articlePlainText(doc, includeIdeas);
  if (!navigator.clipboard?.write || typeof ClipboardItem === "undefined") {
    throw new Error(
      "Rich clipboard is unavailable here. Download HTML instead.",
    );
  }
  await navigator.clipboard.write([
    new ClipboardItem({
      "text/html": new Blob([html], { type: "text/html" }),
      "text/plain": new Blob([text], { type: "text/plain" }),
    }),
  ]);
}

export function printArticle(
  doc: EditorDocument,
  includeIdeas: boolean,
): () => void {
  const frame = document.createElement("iframe");
  frame.title = "Article print preview";
  frame.className = "print-frame";
  // A same-origin generated document only; article markup has already been sanitized.
  frame.srcdoc = standaloneHTML(doc, includeIdeas);
  frame.onload = () => {
    frame.contentWindow?.focus();
    frame.contentWindow?.print();
  };
  document.body.append(frame);
  return () => frame.remove();
}
