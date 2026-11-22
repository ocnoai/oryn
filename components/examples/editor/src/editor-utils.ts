import type { RichNode } from "./model";

export const FONT_FAMILIES = [
  "Default",
  "Arial",
  "Courier New",
  "Georgia",
  "Times New Roman",
  "Trebuchet MS",
  "Verdana",
] as const;
export const FONT_SIZES = [
  "12",
  "13",
  "14",
  "15",
  "16",
  "17",
  "18",
  "20",
  "24",
  "32",
  "40",
  "48",
] as const;
export const BLOCK_TYPES = [
  "paragraph",
  "heading",
  "blockquote",
  "codeBlock",
  "bulletList",
  "orderedList",
  "listItem",
] as const;

/** Only explicit, user-activated HTTP(S) and email links are supported. */
export function safeLink(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed || /[\u0000-\u0020\u007f]/u.test(trimmed)) return null;
  try {
    const url = new URL(trimmed);
    if (!["https:", "http:", "mailto:"].includes(url.protocol)) return null;
    if (url.username || url.password) return null;
    if (url.protocol !== "mailto:" && !url.hostname) return null;
    return url.href;
  } catch {
    return null;
  }
}

export function blockId(value: unknown): string | null {
  return typeof value === "string" && /^[a-zA-Z0-9_-]{1,100}$/u.test(value)
    ? value
    : null;
}

export function nodeText(node: RichNode): string {
  if (node.type === "text") return node.text ?? "";
  if (node.type === "hardBreak") return "\n";
  const separator = [
    "doc",
    "bulletList",
    "orderedList",
    "blockquote",
    "listItem",
  ].includes(node.type)
    ? "\n"
    : "";
  return (node.content ?? []).map(nodeText).join(separator);
}

export function documentBlocks(
  content: RichNode,
): Array<{ id: string; text: string }> {
  return (content.content ?? []).flatMap((node) => {
    const id = blockId(node.attrs?.blockId);
    return id ? [{ id, text: nodeText(node) }] : [];
  });
}

export function pruneIdeas(
  ideas: Record<string, string>,
  ids: readonly string[],
): Record<string, string> {
  const result: Record<string, string> = {};
  for (const id of ids) {
    if (
      Object.hasOwn(ideas, id) &&
      typeof ideas[id] === "string" &&
      ideas[id].trim()
    )
      result[id] = ideas[id];
  }
  return result;
}

/** Restore a deleted block's note on undo, never resurrect a cleared live note. */
export function reconcileIdeas(
  ideas: Record<string, string>,
  undoCache: Record<string, string>,
  previousIds: ReadonlySet<string>,
  ids: readonly string[],
): Record<string, string> {
  const result = { ...ideas };
  for (const id of ids)
    if (
      !previousIds.has(id) &&
      !Object.hasOwn(result, id) &&
      Object.hasOwn(undoCache, id)
    )
      result[id] = undoCache[id];
  return pruneIdeas(result, ids);
}

/** Context is local text only; a bounded window cannot grow with a whole article. */
export function selectionContext(
  content: RichNode,
  selected: string,
  selectedBlockId: string | null,
  radius = 200,
): string {
  const blocks = documentBlocks(content);
  const index = blocks.findIndex((block) => block.id === selectedBlockId);
  if (index < 0) return selected;
  const allText = blocks.map((block) => block.text).join("\n");
  const start = blocks
    .slice(0, index)
    .reduce((length, block) => length + block.text.length + 1, 0);
  const offset = selected ? allText.indexOf(selected, start) : -1;
  const found = offset >= start && offset <= start + blocks[index].text.length;
  const preceding = found
    ? allText.slice(0, offset)
    : blocks
        .slice(0, index)
        .map((block) => block.text)
        .join(" ");
  const following = found
    ? allText.slice(offset + selected.length)
    : blocks
        .slice(index + 1)
        .map((block) => block.text)
        .join(" ");
  const limit = Math.max(0, Math.min(200, Math.floor(radius)));
  const beforeWords = preceding.split(/\s+/u).filter(Boolean);
  const before = limit ? beforeWords.slice(-limit).join(" ") : "";
  const after = following
    .split(/\s+/u)
    .filter(Boolean)
    .slice(0, limit)
    .join(" ");
  return [
    before && `Before (context only): ${before}`,
    `Selected text: ${selected}`,
    after && `After (context only): ${after}`,
  ]
    .filter(Boolean)
    .join("\n\n");
}

function color(value: unknown): string | undefined {
  return typeof value === "string" &&
    /^#[\da-f]{3}(?:[\da-f]{3})?$/iu.test(value)
    ? value
    : undefined;
}

/** Reduce imported JSON to the supported inert editor schema before rendering. */
export function normalizeContent(
  input: RichNode,
  makeId: () => string = () => crypto.randomUUID(),
): RichNode {
  const allowed = new Set<string>([...BLOCK_TYPES, "text", "hardBreak", "doc"]);
  const marks = new Set([
    "bold",
    "italic",
    "underline",
    "strike",
    "code",
    "subscript",
    "superscript",
  ]);
  const clean = (node: RichNode, depth = 0): RichNode | null => {
    if (
      !node ||
      typeof node !== "object" ||
      typeof node.type !== "string" ||
      !allowed.has(node.type) ||
      depth > 100
    )
      return null;
    if (node.type === "text") {
      if (typeof node.text !== "string" || !node.text) return null;
      return {
        type: "text",
        text: node.text,
        ...(node.marks?.length
          ? {
              marks: node.marks.flatMap<NonNullable<RichNode["marks"]>[number]>(
                (mark) => {
                  if (marks.has(mark.type)) return [{ type: mark.type }];
                  if (mark.type === "link") {
                    const href = safeLink(mark.attrs?.href);
                    return href
                      ? [
                          {
                            type: "link",
                            attrs: {
                              href,
                              target: "_blank",
                              rel: "noopener noreferrer",
                            },
                          },
                        ]
                      : [];
                  }
                  if (mark.type === "highlight") {
                    const value = color(mark.attrs?.color);
                    return [
                      {
                        type: "highlight",
                        ...(value ? { attrs: { color: value } } : {}),
                      },
                    ];
                  }
                  if (mark.type === "textStyle") {
                    const attrs: Record<string, unknown> = {};
                    const value = color(mark.attrs?.color);
                    if (value) attrs.color = value;
                    if (
                      FONT_FAMILIES.slice(1).some(
                        (font) => font === mark.attrs?.fontFamily,
                      )
                    )
                      attrs.fontFamily = mark.attrs?.fontFamily;
                    const size = mark.attrs?.fontSize;
                    if (
                      typeof size === "string" &&
                      FONT_SIZES.some((value) => `${value}px` === size)
                    )
                      attrs.fontSize = size;
                    return Object.keys(attrs).length
                      ? [{ type: "textStyle", attrs }]
                      : [];
                  }
                  return [];
                },
              ),
            }
          : {}),
      };
    }
    const attrs: Record<string, unknown> = {};
    const id = blockId(node.attrs?.blockId);
    if (id) attrs.blockId = id;
    if (node.type === "heading")
      attrs.level = [1, 2, 3].includes(Number(node.attrs?.level))
        ? Number(node.attrs?.level)
        : 1;
    if (
      ["left", "center", "right", "justify"].includes(
        String(node.attrs?.textAlign),
      )
    )
      attrs.textAlign = node.attrs?.textAlign;
    if (Number.isInteger(node.attrs?.indent))
      attrs.indent = Math.max(0, Math.min(8, Number(node.attrs?.indent)));
    if (node.type === "orderedList" && Number.isInteger(node.attrs?.start))
      attrs.start = Math.max(1, Math.min(9999, Number(node.attrs?.start)));
    const children = (Array.isArray(node.content) ? node.content : [])
      .map((child) => clean(child, depth + 1))
      .filter((child): child is RichNode => Boolean(child));
    let content = children;
    const inline = (nodes: RichNode[]) =>
      nodes.flatMap((child) =>
        child.type === "text" || child.type === "hardBreak"
          ? [child]
          : nodeText(child)
            ? [{ type: "text", text: nodeText(child) }]
            : [],
      );
    const blocks = (nodes: RichNode[]) =>
      nodes.map((child) =>
        child.type === "text" ||
        child.type === "hardBreak" ||
        child.type === "listItem" ||
        child.type === "doc"
          ? { type: "paragraph", content: inline([child]) }
          : child,
      );
    if (node.type === "hardBreak") content = [];
    else if (node.type === "paragraph" || node.type === "heading")
      content = inline(children);
    else if (node.type === "codeBlock") {
      const text = children.map(nodeText).join("");
      content = text ? [{ type: "text", text }] : [];
    } else if (node.type === "bulletList" || node.type === "orderedList") {
      content = children.map((child) =>
        child.type === "listItem"
          ? child
          : {
              type: "listItem",
              content: [{ type: "paragraph", content: inline([child]) }],
            },
      );
      if (!content.length)
        content = [{ type: "listItem", content: [{ type: "paragraph" }] }];
    } else if (node.type === "listItem") {
      content = blocks(children);
      if (content[0]?.type !== "paragraph")
        content.unshift({ type: "paragraph" });
    } else if (node.type === "blockquote" || node.type === "doc") {
      content = blocks(children);
      if (!content.length) content = [{ type: "paragraph" }];
    }
    return {
      type: node.type,
      ...(Object.keys(attrs).length ? { attrs } : {}),
      ...(content.length ? { content } : {}),
    };
  };
  const nodes = (input.type === "doc" ? (input.content ?? []) : [input])
    .map((node) => clean(node))
    .filter((node): node is RichNode => Boolean(node));
  const ids = new Set<string>();
  const top = (nodes.length ? nodes : [{ type: "paragraph" }]).map((node) => {
    const normalized = ["text", "hardBreak", "listItem", "doc"].includes(
      node.type,
    )
      ? {
          type: "paragraph",
          ...(nodeText(node)
            ? { content: [{ type: "text", text: nodeText(node) }] }
            : {}),
        }
      : node;
    let id = blockId(normalized.attrs?.blockId);
    if (!id || ids.has(id)) id = makeId();
    ids.add(id);
    return { ...normalized, attrs: { ...normalized.attrs, blockId: id } };
  });
  return { type: "doc", content: top };
}

export function plainParagraph(
  text: string,
  id: string = crypto.randomUUID(),
): RichNode {
  return {
    type: "paragraph",
    attrs: { blockId: id },
    ...(text
      ? {
          content: text
            .split("\n")
            .flatMap((line, index) => [
              ...(index ? [{ type: "hardBreak" }] : []),
              ...(line ? [{ type: "text", text: line }] : []),
            ]),
        }
      : {}),
  };
}
