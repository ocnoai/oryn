import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { EditorContent, useEditor, type Editor } from "@tiptap/react";
import { Extension, type JSONContent } from "@tiptap/core";
import {
  AllSelection,
  Plugin,
  Selection,
  type EditorState,
} from "@tiptap/pm/state";
import { Mapping } from "@tiptap/pm/transform";
import { closeHistory } from "@tiptap/pm/history";
import StarterKit from "@tiptap/starter-kit";
import { TextStyle, FontSize } from "@tiptap/extension-text-style";
import Color from "@tiptap/extension-color";
import Highlight from "@tiptap/extension-highlight";
import FontFamily from "@tiptap/extension-font-family";
import TextAlign from "@tiptap/extension-text-align";
import Subscript from "@tiptap/extension-subscript";
import Superscript from "@tiptap/extension-superscript";
import Link from "@tiptap/extension-link";
import Underline from "@tiptap/extension-underline";
import DOMPurify from "dompurify";
import type { RichNode } from "./model";
import {
  BLOCK_TYPES,
  FONT_FAMILIES,
  FONT_SIZES,
  blockId,
  documentBlocks,
  normalizeContent,
  plainParagraph,
  pruneIdeas,
  reconcileIdeas,
  safeLink,
  selectionContext,
} from "./editor-utils";

export type { RichNode } from "./model";
export interface RichSelection {
  text: string;
  context: string;
  blockId: string | null;
  kind?: "text" | "idea";
}
export interface RichEditorHandle {
  getContent(): RichNode;
  getText(): string;
  replaceBlockText(blockId: string, text: string): void;
  insertAfterBlock(blockId: string, text: string): void;
  insertText(text: string): void;
  getBlocks(): Array<{ id: string; text: string }>;
  undo(): void;
}
interface Props {
  documentId: string;
  content: RichNode;
  ideas: Record<string, string>;
  ideasVisible: boolean;
  ideasPrimary: boolean;
  onChange(content: RichNode, ideas: Record<string, string>): void;
  onSelection(selection: RichSelection): void;
  onReady?(api: RichEditorHandle | null): void;
}
interface BlockLayout {
  id: string;
  text: string;
  pos: number;
  size: number;
  top: number;
  height: number;
}
interface BlockMenu {
  blockId: string;
  mode: "insert" | "replace";
  top: number;
  index: number;
  slashPos?: number;
}
const COMMANDS = [
  { label: "Text", type: "paragraph" },
  { label: "Heading 1", type: "heading", level: 1 },
  { label: "Heading 2", type: "heading", level: 2 },
  { label: "Heading 3", type: "heading", level: 3 },
  { label: "Bulleted list", type: "bulletList" },
  { label: "Numbered list", type: "orderedList" },
  { label: "Quote", type: "blockquote" },
  { label: "Code block", type: "codeBlock" },
] as const;
const PASTE_TAGS = [
  "p",
  "br",
  "div",
  "span",
  "strong",
  "b",
  "em",
  "i",
  "u",
  "s",
  "strike",
  "del",
  "code",
  "pre",
  "blockquote",
  "ul",
  "ol",
  "li",
  "h1",
  "h2",
  "h3",
  "a",
  "sub",
  "sup",
  "mark",
];

/** IDs belong to top-level blocks, even when a paragraph is wrapped in a list. */
export const StableBlocks = Extension.create({
  name: "stableBlocks",
  addGlobalAttributes() {
    return [
      {
        types: [...BLOCK_TYPES],
        attributes: {
          blockId: {
            default: null,
            parseHTML: (element) =>
              blockId(element.getAttribute("data-block-id")),
            renderHTML: (attrs) =>
              blockId(attrs.blockId) ? { "data-block-id": attrs.blockId } : {},
          },
          indent: {
            default: 0,
            parseHTML: () => 0,
            renderHTML: (attrs) =>
              Number(attrs.indent) > 0
                ? {
                    style: `margin-left: ${Math.min(8, Number(attrs.indent)) * 24}px`,
                  }
                : {},
          },
        },
      },
    ];
  },
  addProseMirrorPlugins() {
    return [createStableBlockPlugin()];
  },
});

export function createStableBlockPlugin() {
  return new Plugin({
    appendTransaction(transactions, oldState, state) {
      if (!transactions.some((transaction) => transaction.docChanged))
        return null;
      const mapping = new Mapping(
        transactions.flatMap((transaction) => transaction.mapping.maps),
      );
      const previous = new Map<number, string>();
      oldState.doc.forEach((node, pos) => {
        const id = blockId(node.attrs.blockId);
        if (!id) return;
        const mapped = Math.min(
          Math.max(0, mapping.map(pos + 1)),
          state.doc.content.size,
        );
        const index = state.doc.resolve(mapped).index(0);
        if (!previous.has(index)) previous.set(index, id);
      });
      const existing = new Set<string>();
      state.doc.forEach((node) => {
        const id = blockId(node.attrs.blockId);
        if (id) existing.add(id);
      });
      const used = new Set<string>();
      const transaction = state.tr;
      state.doc.forEach((node, pos, index) => {
        let id = blockId(node.attrs.blockId);
        if (!id || used.has(id)) {
          const oldId = previous.get(index);
          id =
            oldId && !used.has(oldId) && !existing.has(oldId)
              ? oldId
              : crypto.randomUUID();
          transaction.setNodeMarkup(pos, undefined, {
            ...node.attrs,
            blockId: id,
          });
        }
        used.add(id);
      });
      return transaction.docChanged ? transaction : null;
    },
  });
}

function findBlock(editor: Editor, id: string) {
  let result: {
    pos: number;
    size: number;
    node: typeof editor.state.doc;
  } | null = null;
  editor.state.doc.forEach((node, pos) => {
    if (node.attrs.blockId === id) result = { pos, size: node.nodeSize, node };
  });
  return result as {
    pos: number;
    size: number;
    node: typeof editor.state.doc;
  } | null;
}

function selectedBlock(editor: Editor): {
  id: string | null;
  pos: number;
  text: string;
} {
  const { $from } = editor.state.selection;
  const node = $from.depth > 0 ? $from.node(1) : editor.state.doc.firstChild;
  return {
    id: blockId(node?.attrs.blockId),
    pos: $from.depth > 0 ? $from.before(1) : 0,
    text: node?.textContent ?? "",
  };
}

/** A menu action is one undoable intention, independent of nearby typing. */
function standaloneChange(editor: Editor, change: () => void) {
  editor.view.dispatch(closeHistory(editor.state.tr));
  try {
    change();
  } finally {
    editor.view.dispatch(closeHistory(editor.state.tr));
  }
}

/** Keep browser and model selection in sync before an immediately typed character. */
export function collapseAllSelection(
  state: EditorState,
  direction: "left" | "right",
) {
  if (!(state.selection instanceof AllSelection)) return null;
  return state.tr.setSelection(
    direction === "right"
      ? Selection.atEnd(state.doc)
      : Selection.atStart(state.doc),
  );
}

/** Mount with key=documentId. Parent autosave echoes never reset the selection. */
export function RichEditor(props: Props) {
  const propsRef = useRef(props);
  propsRef.current = props;
  const ideasRef = useRef({ ...props.ideas });
  const ideaUndoCache = useRef({ ...props.ideas });
  const stageRef = useRef<HTMLDivElement>(null);
  const [layouts, setLayouts] = useState<BlockLayout[]>([]);
  const [menu, setMenu] = useState<BlockMenu | null>(null);
  const menuRef = useRef(menu);
  menuRef.current = menu;
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [drop, setDrop] = useState<{
    id: string;
    after: boolean;
    top: number;
  } | null>(null);
  const [notice, setNotice] = useState("");
  const [focusVersion, requestFocus] = useState(0);
  const [ideasVersion, rerender] = useState(0);
  const frames = useRef<number | null>(null);
  const keyboardRef = useRef<(event: KeyboardEvent) => boolean>(() => false);
  const initialContent = useRef(normalizeContent(props.content));
  const previousBlockIds = useRef(
    new Set(documentBlocks(initialContent.current).map((block) => block.id)),
  );
  const emittedContent = useRef(new WeakSet<object>());
  const lastIncomingContent = useRef(props.content);

  const publish = useCallback((editor: Editor) => {
    const content = editor.getJSON() as RichNode;
    const ids = documentBlocks(content).map((block) => block.id);
    const current = reconcileIdeas(
      ideasRef.current,
      ideaUndoCache.current,
      previousBlockIds.current,
      ids,
    );
    previousBlockIds.current = new Set(ids);
    ideasRef.current = current;
    emittedContent.current.add(content);
    propsRef.current.onChange(content, current);
  }, []);

  const selectionChanged = useCallback((editor: Editor) => {
    const block = selectedBlock(editor);
    const { from, to, empty } = editor.state.selection;
    const text = empty
      ? block.text
      : editor.state.doc.textBetween(from, to, "\n");
    setSelectedId(block.id);
    propsRef.current.onSelection({
      text,
      blockId: block.id,
      kind: "text",
      context: selectionContext(editor.getJSON() as RichNode, text, block.id),
    });
  }, []);

  const editor = useEditor(
    {
      extensions: [
        StarterKit.configure({
          heading: { levels: [1, 2, 3] },
          link: false,
          underline: false,
          horizontalRule: false,
          trailingNode: false,
        }),
        StableBlocks,
        TextStyle,
        FontSize,
        Color,
        FontFamily,
        Highlight.configure({ multicolor: true }),
        TextAlign.configure({ types: ["heading", "paragraph"] }),
        Subscript,
        Superscript,
        Underline,
        Link.configure({
          autolink: false,
          linkOnPaste: false,
          openOnClick: false,
          isAllowedUri: (value) => Boolean(safeLink(value)),
          HTMLAttributes: { target: "_blank", rel: "noopener noreferrer" },
        }),
      ],
      content: initialContent.current as JSONContent,
      immediatelyRender: false,
      shouldRerenderOnTransaction: true,
      editorProps: {
        attributes: {
          class: "rich-text",
          role: "textbox",
          "aria-label": "Document text",
          "aria-multiline": "true",
          spellcheck: "true",
        },
        transformPastedHTML: (html) =>
          DOMPurify.sanitize(html, {
            ALLOWED_TAGS: PASTE_TAGS,
            ALLOWED_ATTR: ["href", "title"],
            ALLOW_DATA_ATTR: false,
            ALLOW_ARIA_ATTR: false,
          }),
        handleKeyDown: (_view, event) => keyboardRef.current(event),
        handleDrop: (_view, event) => {
          event.preventDefault();
          return true;
        },
        handlePaste: (_view, event) => {
          if (event.clipboardData?.files.length) {
            event.preventDefault();
            return true;
          }
          return false;
        },
        handleDOMEvents: {
          click: (_view, event) => {
            const anchor = (event.target as HTMLElement).closest("a");
            if (!anchor) return false;
            event.preventDefault();
            const href = safeLink(anchor.getAttribute("href"));
            if (href && (event.ctrlKey || event.metaKey))
              window.open(href, "_blank", "noopener,noreferrer");
            return true;
          },
        },
      },
      onUpdate: ({ editor }) => {
        publish(editor);
        selectionChanged(editor);
      },
      onSelectionUpdate: ({ editor }) => selectionChanged(editor),
    },
    [props.documentId],
  );

  const updateLayouts = useCallback(() => {
    if (!editor || !stageRef.current || editor.isDestroyed) return;
    const rect = stageRef.current.getBoundingClientRect();
    const next: BlockLayout[] = [];
    editor.state.doc.forEach((node, pos) => {
      const element = editor.view.nodeDOM(pos);
      const id = blockId(node.attrs.blockId);
      if (id && element instanceof HTMLElement) {
        const bounds = element.getBoundingClientRect();
        next.push({
          id,
          text: node.textContent,
          pos,
          size: node.nodeSize,
          top: bounds.top - rect.top,
          height: bounds.height,
        });
      }
    });
    setLayouts((previous) =>
      JSON.stringify(previous) === JSON.stringify(next) ? previous : next,
    );
  }, [editor]);

  const scheduleLayout = useCallback(() => {
    if (frames.current !== null) cancelAnimationFrame(frames.current);
    frames.current = requestAnimationFrame(() => {
      frames.current = null;
      updateLayouts();
    });
  }, [updateLayouts]);

  useLayoutEffect(() => {
    // Tiptap's focus command waits for an animation frame. Once React has
    // removed a block menu, focus synchronously so immediate typing is kept.
    if (focusVersion && editor && !editor.isDestroyed) editor.view.focus();
  }, [editor, focusVersion]);

  useLayoutEffect(() => {
    if (!editor || !stageRef.current) return;
    const observer = new ResizeObserver(scheduleLayout);
    observer.observe(stageRef.current);
    observer.observe(editor.view.dom);
    editor.on("transaction", scheduleLayout);
    window.addEventListener("resize", scheduleLayout);
    scheduleLayout();
    return () => {
      observer.disconnect();
      editor.off("transaction", scheduleLayout);
      window.removeEventListener("resize", scheduleLayout);
      if (frames.current !== null) cancelAnimationFrame(frames.current);
    };
  }, [editor, scheduleLayout]);
  useLayoutEffect(scheduleLayout, [
    props.ideasVisible,
    props.ideasPrimary,
    scheduleLayout,
  ]);
  useLayoutEffect(() => {
    for (const block of layouts) {
      const textarea = stageRef.current?.querySelector<HTMLTextAreaElement>(
        `textarea[data-idea-id="${block.id}"]`,
      );
      if (!textarea) continue;
      textarea.style.height = "0px";
      textarea.style.height = `${Math.min(Math.max(48, block.height), Math.max(48, textarea.scrollHeight))}px`;
    }
  }, [layouts, props.ideas, props.ideasVisible, ideasVersion]);
  useLayoutEffect(() => {
    if (menu?.mode === "insert")
      stageRef.current
        ?.querySelector<HTMLButtonElement>(
          `.rich-editor-menu button:nth-child(${menu.index + 1})`,
        )
        ?.focus();
  }, [menu?.blockId, menu?.mode, menu?.index]);

  useEffect(() => {
    ideasRef.current = { ...props.ideas };
    Object.assign(ideaUndoCache.current, props.ideas);
    for (const id of previousBlockIds.current)
      if (!Object.hasOwn(props.ideas, id)) ideaUndoCache.current[id] = "";
    rerender((value) => value + 1);
  }, [props.ideas]);

  useEffect(() => {
    if (!menu) return;
    const close = (event: PointerEvent) => {
      if (
        !(event.target instanceof Element) ||
        !event.target.closest(".rich-editor-menu, .rich-editor-block-controls")
      )
        setMenu(null);
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [Boolean(menu)]);

  const api = useMemo<RichEditorHandle | null>(
    () =>
      editor
        ? {
            getContent: () => editor.getJSON() as RichNode,
            getText: () => editor.getText({ blockSeparator: "\n\n" }),
            getBlocks: () => documentBlocks(editor.getJSON() as RichNode),
            replaceBlockText: (id, text) => {
              const block = findBlock(editor, id);
              if (!block) return;
              const value =
                block.node.type.name === "paragraph" ||
                block.node.type.name === "heading" ||
                block.node.type.name === "codeBlock"
                  ? {
                      type: block.node.type.name,
                      attrs: block.node.attrs,
                      ...(text ? { content: [{ type: "text", text }] } : {}),
                    }
                  : plainParagraph(text, id);
              standaloneChange(editor, () => {
                editor
                  .chain()
                  .insertContentAt(
                    { from: block.pos, to: block.pos + block.size },
                    value,
                  )
                  .run();
              });
            },
            insertAfterBlock: (id, text) => {
              const block = findBlock(editor, id);
              if (block)
                standaloneChange(editor, () => {
                  editor
                    .chain()
                    .insertContentAt(
                      block.pos + block.size,
                      plainParagraph(text),
                    )
                    .run();
                });
            },
            insertText: (text) => {
              standaloneChange(editor, () => {
                editor
                  .chain()
                  .focus()
                  .insertContent({ type: "text", text })
                  .run();
              });
              requestFocus((value) => value + 1);
            },
            undo: () => {
              editor.chain().focus().undo().run();
            },
          }
        : null,
    [editor],
  );
  const readyRef = useRef(props.onReady);
  readyRef.current = props.onReady;
  useEffect(() => {
    readyRef.current?.(api);
    return () => readyRef.current?.(null);
  }, [api]);

  useEffect(() => {
    if (!editor) return;
    // Persist generated IDs once; ordinary autosave prop echoes are never reapplied.
    if (
      JSON.stringify(propsRef.current.content) !==
      JSON.stringify(editor.getJSON())
    )
      publish(editor);
  }, [editor, publish]);

  useEffect(() => {
    if (!editor || props.content === lastIncomingContent.current) return;
    lastIncomingContent.current = props.content;
    // A local edit's roundtrip may arrive after the next keystroke. It must not
    // be mistaken for a remote restore and reset newer content or the cursor.
    if (
      emittedContent.current.has(props.content) ||
      JSON.stringify(props.content) === JSON.stringify(editor.getJSON())
    )
      return;
    const { from, to } = editor.state.selection;
    editor.commands.setContent(normalizeContent(props.content) as JSONContent, {
      emitUpdate: false,
    });
    previousBlockIds.current = new Set(
      documentBlocks(editor.getJSON() as RichNode).map((block) => block.id),
    );
    const maximum = editor.state.doc.content.size;
    editor.commands.setTextSelection({
      from: Math.min(from, maximum),
      to: Math.min(to, maximum),
    });
    selectionChanged(editor);
    scheduleLayout();
  }, [editor, props.content, scheduleLayout, selectionChanged]);

  function focusBlock(id: string) {
    if (!editor) return;
    const block = findBlock(editor, id);
    if (block) {
      editor
        .chain()
        .focus()
        .setTextSelection(block.pos + 1)
        .run();
      requestFocus((value) => value + 1);
    }
  }
  function applyCommand(index: number) {
    if (!editor || !menuRef.current) return;
    const current = menuRef.current;
    const command = COMMANDS[index];
    const block = findBlock(editor, current.blockId);
    setMenu(null);
    if (!block || !command) return;
    if (current.mode === "insert") {
      const id = crypto.randomUUID();
      let content: JSONContent = plainParagraph("", id);
      if (command.type === "heading")
        content = {
          type: "heading",
          attrs: { level: command.level, blockId: id },
        };
      else if (command.type === "codeBlock")
        content = { type: "codeBlock", attrs: { blockId: id } };
      else if (command.type === "blockquote")
        content = {
          type: "blockquote",
          attrs: { blockId: id },
          content: [{ type: "paragraph" }],
        };
      else if (command.type === "bulletList" || command.type === "orderedList")
        content = {
          type: command.type,
          attrs: { blockId: id },
          content: [{ type: "listItem", content: [{ type: "paragraph" }] }],
        };
      standaloneChange(editor, () => {
        editor
          .chain()
          .focus()
          .insertContentAt(block.pos + block.size, content)
          .run();
      });
      requestFocus((value) => value + 1);
      return;
    }
    let chain = editor.chain().focus();
    if (
      typeof current.slashPos === "number" &&
      editor.state.doc.textBetween(current.slashPos, current.slashPos + 1) ===
        "/"
    )
      chain = chain.deleteRange({
        from: current.slashPos,
        to: current.slashPos + 1,
      });
    else chain = chain.setTextSelection(block.pos + 1);
    chain = chain.clearNodes();
    standaloneChange(editor, () => {
      if (command.type === "paragraph") chain.setParagraph().run();
      else if (command.type === "heading")
        chain.setHeading({ level: command.level }).run();
      else if (command.type === "blockquote") chain.toggleBlockquote().run();
      else if (command.type === "codeBlock") chain.setCodeBlock().run();
      else if (command.type === "bulletList") chain.toggleBulletList().run();
      else chain.toggleOrderedList().run();
    });
    requestFocus((value) => value + 1);
  }
  keyboardRef.current = (event) => {
    if (!editor) return false;
    const current = menuRef.current;
    if (
      !current &&
      !event.shiftKey &&
      !event.metaKey &&
      !event.ctrlKey &&
      !event.altKey &&
      (event.key === "ArrowLeft" || event.key === "ArrowRight")
    ) {
      const transaction = collapseAllSelection(
        editor.state,
        event.key === "ArrowRight" ? "right" : "left",
      );
      if (transaction) {
        event.preventDefault();
        editor.view.dispatch(transaction);
        return true;
      }
    }
    if (current) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        setMenu({
          ...current,
          index:
            (current.index +
              (event.key === "ArrowDown" ? 1 : COMMANDS.length - 1)) %
            COMMANDS.length,
        });
        return true;
      }
      if (event.key === "Enter" || event.key === "Tab") {
        event.preventDefault();
        applyCommand(current.index);
        return true;
      }
      if (event.key === "Escape") {
        event.preventDefault();
        setMenu(null);
        focusBlock(current.blockId);
        return true;
      }
      setMenu(null);
    }
    if (
      event.key === "/" &&
      !event.metaKey &&
      !event.ctrlKey &&
      !event.altKey &&
      editor.state.selection.empty
    ) {
      const block = selectedBlock(editor);
      const layout = layouts.find((layout) => layout.id === block.id);
      if (block.id)
        setMenu({
          blockId: block.id,
          mode: "replace",
          index: 0,
          top: (layout?.top ?? 0) + 32,
          slashPos: editor.state.selection.from,
        });
    }
    return false;
  };

  function moveBlock(id: string, targetId: string, after: boolean) {
    if (!editor || id === targetId) return;
    const source = findBlock(editor, id),
      target = findBlock(editor, targetId);
    if (!source || !target) return;
    const targetPos = target.pos + (after ? target.size : 0);
    const transaction = editor.state.tr.delete(
      source.pos,
      source.pos + source.size,
    );
    transaction.insert(transaction.mapping.map(targetPos), source.node);
    standaloneChange(editor, () => editor.view.dispatch(transaction));
    requestFocus((value) => value + 1);
  }
  function deleteBlock(id: string) {
    if (!editor) return;
    const block = findBlock(editor, id);
    if (!block) return;
    standaloneChange(editor, () => {
      if (editor.state.doc.childCount === 1)
        editor.commands.insertContentAt(
          { from: block.pos, to: block.pos + block.size },
          plainParagraph(""),
        );
      else
        editor.commands.deleteRange({
          from: block.pos,
          to: block.pos + block.size,
        });
    });
  }
  function changeIdea(id: string, value: string) {
    if (!editor) return;
    ideasRef.current = { ...ideasRef.current, [id]: value };
    ideaUndoCache.current[id] = value;
    const content = editor.getJSON() as RichNode;
    emittedContent.current.add(content);
    propsRef.current.onChange(
      content,
      pruneIdeas(
        ideasRef.current,
        documentBlocks(content).map((block) => block.id),
      ),
    );
    propsRef.current.onSelection({
      text: value,
      blockId: id,
      kind: "idea",
      context: `Selected idea: ${value}\n\nCorresponding text: ${documentBlocks(content).find((block) => block.id === id)?.text ?? ""}`,
    });
    rerender((value) => value + 1);
  }
  function ideaKey(
    event: React.KeyboardEvent<HTMLTextAreaElement>,
    index: number,
  ) {
    if (event.key === "Enter" && !event.shiftKey && editor) {
      event.preventDefault();
      const next = layouts[index + 1];
      if (next)
        stageRef.current
          ?.querySelector<HTMLTextAreaElement>(
            `textarea[data-idea-id="${next.id}"]`,
          )
          ?.focus();
      else {
        const id = crypto.randomUUID();
        editor.commands.insertContentAt(
          editor.state.doc.content.size,
          plainParagraph("", id),
        );
        requestAnimationFrame(() =>
          stageRef.current
            ?.querySelector<HTMLTextAreaElement>(
              `textarea[data-idea-id="${id}"]`,
            )
            ?.focus(),
        );
      }
    }
    if (
      event.key === "Backspace" &&
      !event.currentTarget.value &&
      !layouts[index].text &&
      layouts.length > 1
    ) {
      event.preventDefault();
      deleteBlock(layouts[index].id);
      const previous = layouts[Math.max(0, index - 1)];
      requestAnimationFrame(() =>
        stageRef.current
          ?.querySelector<HTMLTextAreaElement>(
            `textarea[data-idea-id="${previous.id}"]`,
          )
          ?.focus(),
      );
    }
  }
  function indent(direction: 1 | -1) {
    if (!editor) return;
    if (editor.isActive("listItem")) {
      if (direction > 0) editor.chain().focus().sinkListItem("listItem").run();
      else editor.chain().focus().liftListItem("listItem").run();
      return;
    }
    const block = selectedBlock(editor);
    if (!block.id) return;
    const found = findBlock(editor, block.id);
    if (found)
      editor.view.dispatch(
        editor.state.tr.setNodeMarkup(found.pos, undefined, {
          ...found.node.attrs,
          indent: Math.max(
            0,
            Math.min(8, Number(found.node.attrs.indent ?? 0) + direction),
          ),
        }),
      );
    editor.commands.focus();
  }
  function setLink() {
    if (!editor) return;
    const value = window.prompt(
      "Link URL (https://, http:// or mailto:). Leave empty to remove.",
      String(editor.getAttributes("link").href ?? ""),
    );
    if (value === null) return;
    if (!value.trim()) {
      editor.chain().focus().extendMarkRange("link").unsetLink().run();
      return;
    }
    const href = safeLink(value);
    if (!href) {
      setNotice("Use a complete https://, http:// or mailto: link.");
      return;
    }
    setNotice("");
    editor.chain().focus().extendMarkRange("link").setLink({ href }).run();
  }
  function tool(
    label: string,
    text: string,
    run: () => void,
    active = false,
    disabled = false,
  ) {
    return (
      <button
        key={label}
        type="button"
        title={label}
        aria-label={label}
        aria-pressed={active}
        disabled={disabled}
        className={active ? "is-active" : ""}
        onMouseDown={(event) => event.preventDefault()}
        onClick={run}
      >
        {text}
      </button>
    );
  }

  if (!editor)
    return (
      <div className="rich-editor-loading" role="status">
        Preparing editor…
      </div>
    );
  const activeBlockType = editor.isActive("heading")
    ? `heading-${editor.getAttributes("heading").level}`
    : (["blockquote", "codeBlock", "bulletList", "orderedList"].find((type) =>
        editor.isActive(type),
      ) ?? "paragraph");
  return (
    <section
      className={`rich-editor ${props.ideasVisible ? "has-ideas" : ""} ${props.ideasPrimary ? "ideas-primary" : ""}`}
      aria-label="Rich text editor"
    >
      <div className="rich-toolbar" role="toolbar" aria-label="Text formatting">
        {tool(
          "Undo",
          "↶",
          () => editor.chain().focus().undo().run(),
          false,
          !editor.can().undo(),
        )}
        {tool(
          "Redo",
          "↷",
          () => editor.chain().focus().redo().run(),
          false,
          !editor.can().redo(),
        )}
        <select
          aria-label="Block type"
          value={activeBlockType}
          onChange={(event) => {
            const command = COMMANDS.findIndex((command) =>
              command.type === "heading"
                ? `heading-${command.level}` === event.target.value
                : command.type === event.target.value,
            );
            const block = selectedBlock(editor);
            if (block.id) {
              menuRef.current = {
                blockId: block.id,
                mode: "replace",
                top: 0,
                index: command,
              };
              applyCommand(command);
            }
          }}
        >
          {COMMANDS.map((command) => (
            <option
              key={command.label}
              value={
                command.type === "heading"
                  ? `heading-${command.level}`
                  : command.type
              }
            >
              {command.label}
            </option>
          ))}
        </select>
        <select
          aria-label="Font family"
          value={editor.getAttributes("textStyle").fontFamily ?? "Default"}
          onChange={(event) =>
            event.target.value === "Default"
              ? editor.chain().focus().unsetFontFamily().run()
              : editor.chain().focus().setFontFamily(event.target.value).run()
          }
        >
          {FONT_FAMILIES.map((font) => (
            <option key={font}>{font}</option>
          ))}
        </select>
        <select
          aria-label="Font size"
          value={String(
            editor.getAttributes("textStyle").fontSize ?? "16px",
          ).replace("px", "")}
          onChange={(event) =>
            editor.chain().focus().setFontSize(`${event.target.value}px`).run()
          }
        >
          {FONT_SIZES.map((size) => (
            <option key={size}>{size}</option>
          ))}
        </select>
        {tool(
          "Bold",
          "B",
          () => editor.chain().focus().toggleBold().run(),
          editor.isActive("bold"),
        )}
        {tool(
          "Italic",
          "I",
          () => editor.chain().focus().toggleItalic().run(),
          editor.isActive("italic"),
        )}
        {tool(
          "Underline",
          "U",
          () => editor.chain().focus().toggleUnderline().run(),
          editor.isActive("underline"),
        )}
        {tool(
          "Strikethrough",
          "S̶",
          () => editor.chain().focus().toggleStrike().run(),
          editor.isActive("strike"),
        )}
        {tool(
          "Inline code",
          "</>",
          () => editor.chain().focus().toggleCode().run(),
          editor.isActive("code"),
        )}
        {tool("Link", "↗", setLink, editor.isActive("link"))}
        {tool(
          "Subscript",
          "x₂",
          () => editor.chain().focus().toggleSubscript().run(),
          editor.isActive("subscript"),
        )}
        {tool(
          "Superscript",
          "x²",
          () => editor.chain().focus().toggleSuperscript().run(),
          editor.isActive("superscript"),
        )}
        <label title="Text color">
          A
          <input
            aria-label="Text color"
            type="color"
            value={editor.getAttributes("textStyle").color ?? "#1f2937"}
            onChange={(event) =>
              editor.chain().focus().setColor(event.target.value).run()
            }
          />
        </label>
        <label title="Highlight color">
          ▰
          <input
            aria-label="Highlight color"
            type="color"
            value={editor.getAttributes("highlight").color ?? "#fff0a8"}
            onChange={(event) =>
              editor
                .chain()
                .focus()
                .setHighlight({ color: event.target.value })
                .run()
            }
          />
        </label>
        {(["left", "center", "right", "justify"] as const).map(
          (alignment, index) =>
            tool(
              `Align ${alignment}`,
              ["≡←", "≡", "→≡", "☰"][index],
              () => editor.chain().focus().setTextAlign(alignment).run(),
              editor.isActive({ textAlign: alignment }),
            ),
        )}
        {tool("Outdent", "⇤", () => indent(-1))}
        {tool("Indent", "⇥", () => indent(1))}
        {tool("Clear formatting", "Tx", () =>
          editor.chain().focus().unsetAllMarks().clearNodes().run(),
        )}
      </div>
      {notice && (
        <div role="status" className="rich-editor-notice">
          {notice}
        </div>
      )}
      <div className="rich-editor-scroll" onScroll={scheduleLayout}>
        <div
          ref={stageRef}
          className="rich-editor-stage"
          style={{
            position: "relative",
            paddingLeft: 76,
            paddingRight: props.ideasVisible
              ? props.ideasPrimary
                ? "44%"
                : "33%"
              : 12,
            minHeight: "100%",
          }}
          onDragOver={(event) => {
            if (
              !event.dataTransfer.types.includes("application/x-oryn-block") ||
              !stageRef.current
            )
              return;
            event.preventDefault();
            event.dataTransfer.dropEffect = "move";
            const y =
              event.clientY - stageRef.current.getBoundingClientRect().top;
            const target =
              layouts.find((block) => y < block.top + block.height) ??
              layouts.at(-1);
            if (target) {
              const after = y > target.top + target.height / 2;
              setDrop({
                id: target.id,
                after,
                top: target.top + (after ? target.height : 0),
              });
            }
          }}
          onDragLeave={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node))
              setDrop(null);
          }}
          onDrop={(event) => {
            event.preventDefault();
            event.stopPropagation();
            const source = event.dataTransfer.getData(
              "application/x-oryn-block",
            );
            if (drop && blockId(source)) moveBlock(source, drop.id, drop.after);
            setDrop(null);
          }}
        >
          <EditorContent editor={editor} className="rich-editor-text" />
          {layouts.map((block, index) => (
            <div
              key={block.id}
              className={`rich-editor-block-controls ${selectedId === block.id ? "is-selected" : ""}`}
              style={{ position: "absolute", left: 0, top: block.top }}
            >
              <button
                type="button"
                title="Insert block"
                aria-label={`Insert block after ${index + 1}`}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() =>
                  setMenu({
                    blockId: block.id,
                    mode: "insert",
                    top: block.top + 28,
                    index: 0,
                  })
                }
              >
                +
              </button>
              <button
                type="button"
                draggable
                title="Drag block"
                aria-label={`Drag block ${index + 1}`}
                onDragStart={(event) => {
                  event.dataTransfer.setData(
                    "application/x-oryn-block",
                    block.id,
                  );
                  event.dataTransfer.effectAllowed = "move";
                  setMenu(null);
                }}
                onDragEnd={() => setDrop(null)}
                onClick={() => focusBlock(block.id)}
              >
                ⠿
              </button>
              <details>
                <summary aria-label={`More actions for block ${index + 1}`}>
                  ⋯
                </summary>
                <div>
                  <button
                    type="button"
                    disabled={index === 0}
                    onClick={() =>
                      moveBlock(block.id, layouts[index - 1].id, false)
                    }
                  >
                    Move up
                  </button>
                  <button
                    type="button"
                    disabled={index === layouts.length - 1}
                    onClick={() =>
                      moveBlock(block.id, layouts[index + 1].id, true)
                    }
                  >
                    Move down
                  </button>
                  <button type="button" onClick={() => deleteBlock(block.id)}>
                    Delete block
                  </button>
                </div>
              </details>
            </div>
          ))}
          {props.ideasVisible &&
            layouts.map((block, index) => (
              <div
                key={block.id}
                className={`rich-editor-idea ${selectedId === block.id ? "is-selected" : ""}`}
                style={{
                  position: "absolute",
                  right: 8,
                  width: props.ideasPrimary
                    ? "calc(44% - 24px)"
                    : "calc(33% - 24px)",
                  top: block.top,
                }}
              >
                <span aria-hidden="true" className="idea-connector" />
                <textarea
                  aria-label={`Idea for block ${index + 1}`}
                  placeholder="Idea…"
                  data-idea-id={block.id}
                  rows={1}
                  style={{
                    maxHeight: Math.max(48, block.height),
                    resize: "none",
                    overflowY: "auto",
                  }}
                  value={ideasRef.current[block.id] ?? ""}
                  onChange={(event) => changeIdea(block.id, event.target.value)}
                  onKeyDown={(event) => ideaKey(event, index)}
                  onFocus={() => {
                    setSelectedId(block.id);
                    const text = ideasRef.current[block.id] ?? "";
                    propsRef.current.onSelection({
                      text,
                      blockId: block.id,
                      kind: "idea",
                      context: `Selected idea: ${text}\n\nCorresponding text: ${block.text}`,
                    });
                  }}
                />
              </div>
            ))}
          {drop && (
            <div
              className="rich-editor-drop-line"
              style={{
                position: "absolute",
                top: drop.top,
                left: 76,
                right: 8,
                height: 3,
                pointerEvents: "none",
              }}
            />
          )}
          {menu && (
            <div
              className="rich-editor-menu"
              role="menu"
              aria-label="Block commands"
              style={{
                position: "absolute",
                top: menu.top,
                left: 76,
                zIndex: 10,
              }}
              onKeyDown={(event) => {
                if (
                  ["ArrowDown", "ArrowUp", "Escape", "Enter", "Tab"].includes(
                    event.key,
                  )
                )
                  keyboardRef.current(event.nativeEvent);
              }}
            >
              {COMMANDS.map((command, index) => (
                <button
                  key={command.label}
                  type="button"
                  role="menuitem"
                  tabIndex={menu.index === index ? 0 : -1}
                  className={menu.index === index ? "is-selected" : ""}
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={() => applyCommand(index)}
                >
                  {command.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      <span className="sr-only">
        Use slash for block commands. Control or Command click opens a link.
      </span>
    </section>
  );
}
