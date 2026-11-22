import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type FormEvent,
} from "react";
import type { EditorDocument, LocalUser, RichNode } from "./model";
import { EditorStorage } from "./storage";
import {
  RichEditor,
  type RichEditorHandle,
  type RichSelection,
} from "./RichEditor";
import {
  LocalAIService,
  buildQuestionPrompt,
  writingPrompt,
  textOf,
  completionMessage,
  isCompleteAnswer,
  thinkingNotice,
  IncompleteGenerationError,
  type GenerationResult,
} from "./ai";
import { findRelated, generateGraph, type RelatedMatch } from "./retrieval";
import {
  articleHTML,
  copyArticle,
  downloadFile,
  printArticle,
  standaloneHTML,
} from "./export";

type ModalKind =
  "users" | "export" | "graph" | "restore" | "password" | "help" | null;
type AIAction =
  "text" | "idea" | "comments" | "rewrite" | "fallacies" | "hemingway";
const emptySelection: RichSelection = { text: "", context: "", blockId: null };
const errorMessage = (error: unknown) =>
  error instanceof Error
    ? error.message
    : "The operation could not be completed.";
const words = (text: string) =>
  text.trim().split(/\s+/u).filter(Boolean).length;
const relativeDate = (time: number) => {
  const minutes = Math.max(0, Math.floor((Date.now() - time) / 60000));
  return minutes < 1
    ? "Just now"
    : minutes < 60
      ? `${minutes}m ago`
      : minutes < 1440
        ? `${Math.floor(minutes / 60)}h ago`
        : new Date(time).toLocaleDateString();
};

function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const paths: Record<string, ReactNode> = {
    plus: <path d="M12 5v14M5 12h14" />,
    search: (
      <>
        <circle cx="10.5" cy="10.5" r="6.5" />
        <path d="m16 16 4 4" />
      </>
    ),
    lock: (
      <>
        <rect x="5" y="10" width="14" height="11" rx="2" />
        <path d="M8 10V7a4 4 0 0 1 8 0v3M12 14v3" />
      </>
    ),
    open: (
      <>
        <rect x="5" y="10" width="14" height="11" rx="2" />
        <path d="M8 10V7a4 4 0 0 1 7-2" />
      </>
    ),
    user: (
      <>
        <circle cx="12" cy="8" r="4" />
        <path d="M4 21v-2a8 8 0 0 1 16 0v2" />
      </>
    ),
    arrow: <path d="m5 12 7-7 7 7M12 5v15" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    moon: <path d="M20 15.5A9 9 0 0 1 8.5 4 9 9 0 1 0 20 15.5Z" />,
    sun: (
      <>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1 1M18 18l1 1M5 19l1-1M18 6l1-1" />
      </>
    ),
    export: (
      <>
        <path d="M12 16V3m-4 4 4-4 4 4M5 13v7h14v-7" />
      </>
    ),
    document: (
      <>
        <path d="M6 3h8l4 4v14H6zM14 3v5h4M9 12h6M9 16h5" />
      </>
    ),
    dots: (
      <>
        <circle cx="5" cy="12" r="1" />
        <circle cx="12" cy="12" r="1" />
        <circle cx="19" cy="12" r="1" />
      </>
    ),
    spark: (
      <>
        <path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5Z" />
        <path d="m20 2 .5 1.5L22 4l-1.5.5L20 6l-.5-1.5L18 4l1.5-.5Z" />
      </>
    ),
    graph: (
      <>
        <circle cx="5" cy="7" r="3" />
        <circle cx="19" cy="7" r="3" />
        <circle cx="12" cy="19" r="3" />
        <path d="M8 7h8M7 10l3 6m7-6-3 6" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    trash: (
      <>
        <path d="M4 6h16M9 6V3h6v3M7 6l1 15h8l1-15M10 10v7M14 10v7" />
      </>
    ),
    copy: (
      <>
        <rect x="8" y="8" width="12" height="13" rx="2" />
        <path d="M16 8V3H3v13h5" />
      </>
    ),
    library: (
      <>
        <path d="M4 4h4v16H4zM10 4h4v16h-4zM16 5l4-1 3 15-4 1z" />
      </>
    ),
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] ?? paths.document}
    </svg>
  );
}

function Modal({
  title,
  children,
  onClose,
  wide = false,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  const panel = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const node = panel.current!;
    (
      node.querySelector(
        "input, select, textarea, button",
      ) as HTMLElement | null
    )?.focus();
    const handle = (event: KeyboardEvent) => {
      if (event.key === "Escape") close.current();
      if (event.key === "Tab") {
        const items = Array.from(
          node.querySelectorAll<HTMLElement>(
            "button:not(:disabled), input:not(:disabled), textarea, select, a[href]",
          ),
        ).filter((item) => item.offsetParent !== null);
        if (!items.length) return;
        if (event.shiftKey && document.activeElement === items[0]) {
          event.preventDefault();
          items.at(-1)!.focus();
        } else if (!event.shiftKey && document.activeElement === items.at(-1)) {
          event.preventDefault();
          items[0].focus();
        }
      }
    };
    document.addEventListener("keydown", handle);
    return () => {
      document.removeEventListener("keydown", handle);
      previous?.focus();
    };
  }, []);
  return (
    <div
      className="modal-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={`modal ${wide ? "modal-wide" : ""}`}
      >
        <div className="modal-heading">
          <h2>{title}</h2>
          <button
            className="icon-button"
            aria-label="Close dialog"
            onClick={onClose}
          >
            <Icon name="close" />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

function TopicGraphView({
  nodes,
  edges,
}: {
  nodes: { id: string; label: string }[];
  edges: { from: string; to: string; relation: string }[];
}) {
  const positions = new Map(
    nodes.map((node, index) => [
      node.id,
      { x: 120 + (index % 3) * 230, y: 65 + Math.floor(index / 3) * 130 },
    ]),
  );
  const height = Math.max(240, Math.ceil(nodes.length / 3) * 130);
  return (
    <div className="graph-canvas">
      <svg
        role="img"
        aria-label="Topic graph with classified idea relationships"
        viewBox={`0 0 720 ${height}`}
        style={{ minWidth: 560, width: "100%", minHeight: height }}
      >
        <defs>
          <marker
            id="idea-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
          </marker>
        </defs>
        {edges.map((edge, index) => {
          const from = positions.get(edge.from),
            to = positions.get(edge.to);
          if (!from || !to) return null;
          return (
            <g key={index} className="graph-edge">
              <line
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                markerEnd="url(#idea-arrow)"
              />
              <text
                x={(from.x + to.x) / 2}
                y={(from.y + to.y) / 2 - 8}
                textAnchor="middle"
              >
                {edge.relation}
              </text>
            </g>
          );
        })}
        {nodes.map((node) => {
          const point = positions.get(node.id)!;
          return (
            <g key={node.id}>
              <rect
                x={point.x - 90}
                y={point.y - 30}
                width="180"
                height="60"
                rx="8"
                className="graph-node-shape"
              />
              <foreignObject
                x={point.x - 83}
                y={point.y - 25}
                width="166"
                height="50"
              >
                <div className="graph-node-label">{node.label}</div>
              </foreignObject>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function App({ storage }: { storage: EditorStorage }) {
  const [documents, setDocuments] = useState<EditorDocument[]>([]);
  const [current, setCurrent] = useState<EditorDocument | null>(null);
  const [users, setUsers] = useState<LocalUser[]>([]);
  const [activeUser, setActiveUser] = useState<LocalUser | null>(
    storage.activeUser,
  );
  const [query, setQuery] = useState("");
  const [saveState, setSaveState] = useState("Saved locally");
  const [toast, setToast] = useState("");
  const [modal, setModal] = useState<ModalKind>(null);
  const [userMode, setUserMode] = useState<"unlock" | "create">("unlock");
  const [userId, setUserId] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [authBusy, setAuthBusy] = useState(false);
  const [restoreText, setRestoreText] = useState("");
  const [ideasVisible, setIdeasVisible] = useState(true);
  const [ideasPrimary, setIdeasPrimary] = useState(false);
  const [selection, setSelection] = useState<RichSelection>(emptySelection);
  const [dark, setDark] = useState(
    () => localStorage.getItem("editor-theme") === "dark",
  );
  const [mobilePanel, setMobilePanel] = useState("write");
  const [panelTab, setPanelTab] = useState<"ask" | "related">("ask");
  const [question, setQuestion] = useState("");
  const [scope, setScope] = useState<"general" | "context" | "article">(
    "context",
  );
  const [instruction, setInstruction] = useState("");
  const [ideaUndo, setIdeaUndo] = useState<{
    stamp: string;
    blockId: string;
    previous: string;
  } | null>(null);
  const [answer, setAnswer] = useState("");
  const [reasoning, setReasoning] = useState("");
  const [generationResult, setGenerationResult] = useState<GenerationResult | null>(null);
  const [thinkingMessage, setThinkingMessage] = useState("");
  const [thinkingMode, setThinkingMode] = useState<"default" | "on" | "off">("off");
  const [lastQuestion, setLastQuestion] = useState("");
  const [aiStatus, setAIStatus] = useState("");
  const [aiAvailable, setAIAvailable] = useState(false);
  const [aiReadyMessage, setAIReadyMessage] = useState("Checking local AI…");
  const [aiChecking, setAIChecking] = useState(false);
  const [aiBusy, setAIBusy] = useState(false);
  const [related, setRelated] = useState<RelatedMatch[]>([]);
  const [relatedLimit, setRelatedLimit] = useState(3);
  const [relatedEnabled, setRelatedEnabled] = useState(false);
  const [sourceScope, setSourceScope] = useState<
    "all" | "references" | "documents"
  >("all");
  const [sourceInclude, setSourceInclude] = useState("");
  const [sourceExclude, setSourceExclude] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [excludeUrl, setExcludeUrl] = useState("");
  const [advancedFilter, setAdvancedFilter] = useState(false);
  const [sourceDetail, setSourceDetail] = useState<RelatedMatch | null>(null);
  const [includeIdeas, setIncludeIdeas] = useState(false);
  const [exportMode, setExportMode] = useState("HTML");
  const [docMenu, setDocMenu] = useState(false);
  const [documentBusy, setDocumentBusy] = useState(false);
  const documentTransition = useRef(false);
  const [newMenu, setNewMenu] = useState(false);
  const currentRef = useRef(current);
  const dirty = useRef(false);
  const editVersion = useRef(0);
  const saveChain = useRef<Promise<void>>(Promise.resolve());
  const editor = useRef<RichEditorHandle | null>(null);
  const ai = useRef(useMemo(() => new LocalAIService(), []));
  const retryAIConnection = useRef<() => void>(() => {});
  const authTransition = useRef(false);
  const resultStamp = useRef<string | null>(null);
  const job = useRef<AbortController | null>(null);
  const epoch = useRef(storage.epoch);
  const refreshNumber = useRef(0);
  const pendingAuth = useRef<(() => Promise<void>) | null>(null);
  const importInput = useRef<HTMLInputElement>(null);
  const restoreInput = useRef<HTMLInputElement>(null);
  const inactivity = useRef(Date.now());
  const printCleanup = useRef<(() => void) | null>(null);
  const lastAutoBucket = useRef("");

  const cancelJob = useCallback((message = "") => {
    job.current?.abort();
    job.current = null;
    setAIBusy(false);
    setAIStatus(message);
    setReasoning("");
    setGenerationResult(null);
    setThinkingMessage("");
  }, []);
  const clearTransient = useCallback(() => {
    cancelJob();
    setAnswer("");
    setReasoning("");
    setGenerationResult(null);
    setLastQuestion("");
    setQuestion("");
    setInstruction("");
    setIdeaUndo(null);
    resultStamp.current = null;
    setQuery("");
    setSelection(emptySelection);
    setRelated([]);
    setSourceDetail(null);
    setSourceInclude("");
    setSourceExclude("");
    setSourceUrl("");
    setExcludeUrl("");
    lastAutoBucket.current = "";
    printCleanup.current?.();
    printCleanup.current = null;
    setModal((previous) =>
      authTransition.current &&
      ["users", "password", "restore"].includes(previous ?? "")
        ? previous
        : null,
    );
    setPassword("");
    setNewPassword("");
    setAuthError("");
    void ai.current.dispose();
  }, [cancelJob]);
  const replaceCurrent = useCallback((doc: EditorDocument | null) => {
    currentRef.current = doc;
    dirty.current = false;
    editVersion.current++;
    setCurrent(doc);
    setSaveState("Saved locally");
    setDocMenu(false);
  }, []);
  const reload = useCallback(async () => {
    const request = ++refreshNumber.current;
    const token = storage.epoch;
    const [visible, profiles] = await Promise.all([
      storage.listDocuments(),
      storage.users(),
    ]);
    if (request !== refreshNumber.current || token !== storage.epoch) return;
    setDocuments(visible);
    setUsers(profiles);
    setActiveUser(storage.activeUser);
    const selected = currentRef.current;
    if (selected) {
      const latest = visible.find((doc) => doc.id === selected.id);
      if (!latest) {
        clearTransient();
        replaceCurrent(null);
      } else if (!dirty.current && latest.revision !== selected.revision) {
        clearTransient();
        replaceCurrent(latest);
      }
    }
  }, [storage, clearTransient, replaceCurrent]);
  useEffect(() => {
    void reload().catch((error) => setToast(errorMessage(error)));
    const unsubscribe = storage.subscribe(() => {
      if (epoch.current !== storage.epoch) {
        epoch.current = storage.epoch;
        clearTransient();
        pendingAuth.current = null;
        setActiveUser(storage.activeUser);
        // Clear decrypted lists synchronously, before any IndexedDB read resolves.
        setDocuments((previous) =>
          previous.filter((doc) => doc.protection === "open"),
        );
        if (currentRef.current?.protection === "protected")
          replaceCurrent(null);
      }
      void reload().catch((error) => setToast(errorMessage(error)));
    });
    return unsubscribe;
  }, [storage, reload, clearTransient, replaceCurrent]);
  useEffect(() => {
    let live = true;
    let inFlight = false;
    let refreshQueued = false;
    let attempt = 0;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    let deadline: ReturnType<typeof setTimeout> | undefined;
    let stopCheck: (() => void) | undefined;
    // A finite startup retry burst avoids constant IPC when no runtime exists.
    // Returning to the page or pressing Retry starts a fresh burst.
    const retryDelays = [500, 1000, 2000, 4000, 8000];
    const check = async (restart = false) => {
      if (!live) return;
      if (restart) {
        attempt = 0;
        clearTimeout(retryTimer);
      }
      if (inFlight) {
        refreshQueued ||= restart;
        return;
      }
      inFlight = true;
      setAIChecking(true);
      let available = false;
      try {
        const result = await Promise.race([
          ai.current.capabilities(),
          new Promise<never>((_resolve, reject) => {
            stopCheck = () => reject(new Error("Connection check stopped."));
            deadline = setTimeout(
              () =>
                reject(
                  new Error(
                    "The local AI connection did not respond. Retry the connection.",
                  ),
                ),
              8000,
            );
          }),
        ]);
        if (!live) return;
        available = result.available;
        setAIAvailable(result.available);
        setAIReadyMessage(result.message);
      } catch (error) {
        if (!live) return;
        setAIAvailable(false);
        setAIReadyMessage(errorMessage(error));
      } finally {
        clearTimeout(deadline);
        stopCheck = undefined;
        inFlight = false;
        if (live) setAIChecking(false);
      }
      if (!live) return;
      if (refreshQueued) {
        refreshQueued = false;
        void check(true);
      } else if (
        !available &&
        attempt < retryDelays.length &&
        document.visibilityState === "visible"
      ) {
        retryTimer = setTimeout(() => void check(), retryDelays[attempt++]);
      }
    };
    const refresh = () => void check(true);
    const visible = () => {
      if (document.visibilityState === "visible") refresh();
      else clearTimeout(retryTimer);
    };
    retryAIConnection.current = refresh;
    window.addEventListener("focus", refresh);
    window.addEventListener("pageshow", refresh);
    document.addEventListener("visibilitychange", visible);
    refresh();
    return () => {
      live = false;
      clearTimeout(retryTimer);
      clearTimeout(deadline);
      stopCheck?.();
      retryAIConnection.current = () => {};
      window.removeEventListener("focus", refresh);
      window.removeEventListener("pageshow", refresh);
      document.removeEventListener("visibilitychange", visible);
    };
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    localStorage.setItem("editor-theme", dark ? "dark" : "light");
  }, [dark]);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 6500);
    return () => clearTimeout(timer);
  }, [toast]);

  const flush = useCallback(async () => {
    const operation = saveChain.current
      .catch(() => {})
      .then(async () => {
        if (!dirty.current || !currentRef.current) return;
        const snapshot = structuredClone(currentRef.current);
        const version = editVersion.current;
        const token = storage.epoch;
        setSaveState("Saving…");
        try {
          const saved = await storage.saveDocument(snapshot);
          if (token !== storage.epoch || currentRef.current?.id !== saved.id)
            return;
          if (version === editVersion.current) {
            dirty.current = false;
            currentRef.current = saved;
            setCurrent(saved);
            setSaveState("Saved locally");
          } else {
            currentRef.current = {
              ...currentRef.current,
              revision: saved.revision,
            };
            setCurrent(currentRef.current);
            setSaveState("Unsaved changes");
          }
          await reload();
        } catch (error) {
          setSaveState("Save failed");
          throw error;
        }
      });
    saveChain.current = operation;
    return operation;
  }, [storage, reload]);
  useEffect(() => {
    if (!current || !dirty.current) return;
    const timer = setTimeout(() => {
      void flush().catch((error) => setToast(errorMessage(error)));
    }, 600);
    return () => clearTimeout(timer);
  }, [current, flush]);
  const editDocument = useCallback(
    (changes: Partial<EditorDocument>) => {
      if (!currentRef.current) return;
      if (job.current)
        cancelJob("Cancellation requested because the document changed.");
      else setAIStatus("");
      setIdeaUndo(null);
      resultStamp.current = null;
      setAnswer("");
      setReasoning("");
      setGenerationResult(null);
      setThinkingMessage("");
      setLastQuestion("");
      setRelated([]);
      setSourceDetail(null);
      currentRef.current = {
        ...currentRef.current,
        ...changes,
        ...(changes.content || changes.ideas ? { graph: undefined } : {}),
        updatedAt: Date.now(),
      };
      dirty.current = true;
      editVersion.current++;
      setCurrent(currentRef.current);
      setSaveState("Unsaved changes");
    },
    [cancelJob],
  );
  const lock = useCallback(async () => {
    if (authTransition.current) {
      storage.lock();
      return;
    }
    try {
      await Promise.race([
        flush(),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("Saving timed out")), 2500),
        ),
      ]);
    } catch {
      setToast("Locked. The last changes could not be saved.");
    }
    storage.lock();
  }, [storage, flush, authBusy]);
  useEffect(() => {
    const activity = () => {
      inactivity.current = Date.now();
    };
    const timer = setInterval(() => {
      if (storage.activeUser && Date.now() - inactivity.current > 15 * 60000)
        void lock();
    }, 15000);
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (dirty.current) {
        event.preventDefault();
        event.returnValue = "";
      }
    };
    const pageHide = () => {
      job.current?.abort();
      storage.lock();
      void ai.current.dispose();
    };
    window.addEventListener("pointerdown", activity);
    window.addEventListener("keydown", activity);
    window.addEventListener("beforeunload", beforeUnload);
    window.addEventListener("pagehide", pageHide);
    return () => {
      clearInterval(timer);
      window.removeEventListener("pointerdown", activity);
      window.removeEventListener("keydown", activity);
      window.removeEventListener("beforeunload", beforeUnload);
      window.removeEventListener("pagehide", pageHide);
    };
  }, [storage, lock]);
  const transition = async (operation: () => Promise<void>) => {
    if (documentTransition.current) return;
    documentTransition.current = true;
    setDocumentBusy(true);
    try {
      await operation();
    } finally {
      documentTransition.current = false;
      setDocumentBusy(false);
    }
  };
  const safely = async (operation: () => Promise<void>) => {
    try {
      await operation();
    } catch (error) {
      setToast(errorMessage(error));
    }
  };
  const stamp = () =>
    JSON.stringify([
      storage.epoch,
      storage.activeUser?.id,
      currentRef.current?.id,
      currentRef.current?.protection,
      currentRef.current?.ownerId,
      editVersion.current,
    ]);
  const assertStamp = (expected: string) => {
    if (stamp() !== expected)
      throw new Error(
        "The document or local user changed. Try the action again.",
      );
  };
  const mountAuthorized = async (
    id: string,
    expectedEpoch: number,
    expectedOwner: string | undefined,
  ) => {
    const visible = await storage.listDocuments();
    if (
      expectedEpoch !== storage.epoch ||
      expectedOwner !== storage.activeUser?.id
    )
      throw new Error("The local user changed. Choose the document again.");
    const doc = visible.find((item) => item.id === id);
    if (
      doc?.protection === "protected" &&
      doc.ownerId !== storage.activeUser?.id
    )
      throw new Error("Unlock this document’s owner.");
    clearTransient();
    replaceCurrent(doc ?? null);
    await reload();
  };
  const selectDocument = async (doc: EditorDocument) =>
    transition(async () => {
      const expected = stamp();
      await flush();
      assertStamp(expected);
      clearTransient();
      await mountAuthorized(doc.id, storage.epoch, storage.activeUser?.id);
      setMobilePanel("write");
    });

  const showUsers = async (operation: (() => Promise<void>) | null = null) => {
    const token = storage.epoch;
    const profiles = await storage.users();
    if (token !== storage.epoch)
      throw new Error("The local user changed. Open the profile picker again.");
    setUsers(profiles);
    pendingAuth.current = operation;
    setUserMode(profiles.length ? "unlock" : "create");
    setUserId(profiles[0]?.id ?? "");
    setPassword("");
    setName("");
    setAuthError("");
    setModal("users");
  };
  const requireUser = (operation: () => Promise<void>) => {
    if (storage.activeUser) {
      void safely(operation);
      return;
    }
    void safely(() => showUsers(operation));
  };

  const createDocument = async (protection: "open" | "protected") =>
    transition(async () => {
      const expected = stamp();
      await flush();
      assertStamp(expected);
      clearTransient();
      const token = storage.epoch;
      const owner = storage.activeUser?.id;
      const doc = await storage.createDocument(protection);
      await mountAuthorized(doc.id, token, owner);
      setMobilePanel("write");
    });

  const authenticate = async (event: FormEvent) => {
    event.preventDefault();
    if (authTransition.current) return;
    const expected = stamp();
    authTransition.current = true;
    setAuthBusy(true);
    setAuthError("");
    const operation = pendingAuth.current;
    try {
      await flush();
      assertStamp(expected);
      if (userMode === "create")
        await storage.createUser(name.trim(), password);
      else await storage.unlockUser(userId, password);
      setPassword("");
      setModal(null);
      pendingAuth.current = null;
      await reload();
      if (operation) await operation();
    } catch (error) {
      setAuthError(errorMessage(error));
    } finally {
      authTransition.current = false;
      setAuthBusy(false);
    }
  };
  const changeProtection = async () =>
    transition(async () => {
      const expected = stamp();
      await flush();
      assertStamp(expected);
      const doc = currentRef.current;
      if (!doc) return;
      if (
        doc.protection === "protected" &&
        !confirm(
          "Make this document Open? Anyone using this browser will be able to read and edit its text and ideas.",
        )
      )
        return;
      if (
        doc.protection === "open" &&
        !confirm(
          "Protect this document for your local user? Earlier shared copies and exports cannot be recalled.",
        )
      )
        return;
      clearTransient();
      const token = storage.epoch;
      const owner = storage.activeUser?.id;
      const next = await storage.setProtection(
        doc.id,
        doc.protection === "open" ? "protected" : "open",
        doc.revision,
      );
      if (token !== storage.epoch || owner !== storage.activeUser?.id) {
        replaceCurrent(null);
        await reload();
        return;
      }
      await mountAuthorized(next.id, token, owner);
    });

  const clone = async () =>
    transition(async () => {
      const expected = stamp();
      await flush();
      assertStamp(expected);
      const doc = currentRef.current;
      if (!doc) return;
      clearTransient();
      const token = storage.epoch;
      const owner = storage.activeUser?.id;
      const cloned = await storage.cloneDocument(doc.id);
      await mountAuthorized(cloned.id, token, owner);
    });

  const remove = async () =>
    transition(async () => {
      const expected = stamp();
      const doc = currentRef.current;
      if (
        !doc ||
        !confirm(`Delete “${doc.title || "Untitled"}”? This cannot be undone.`)
      )
        return;
      await flush();
      assertStamp(expected);
      const fresh = currentRef.current!;
      await storage.deleteDocument(fresh.id, fresh.revision);
      if (stamp() === expected) {
        clearTransient();
        replaceCurrent(null);
      }
      await reload();
    });

  const startJob = () => {
    const doc = currentRef.current;
    if (documentTransition.current)
      throw new Error("Wait for the document to finish opening.");
    if (!doc) throw new Error("Choose a document first.");
    cancelJob();
    const controller = new AbortController();
    job.current = controller;
    const editStamp = editVersion.current;
    const token = storage.epoch;
    const id = doc.id;
    const result = { current: null as GenerationResult | null };
    const valid = () =>
      !controller.signal.aborted &&
      job.current === controller &&
      token === storage.epoch &&
      editStamp === editVersion.current &&
      id === currentRef.current?.id;
    resultStamp.current = stamp();
    setAnswer("");
    setLastQuestion("");
    setAIBusy(true);
    setAIStatus("Preparing local model…");
    return {
      doc: structuredClone(doc),
      controller,
      valid,
      result,
      options: {
        signal: controller.signal,
        ...(thinkingMode === "default" ? {} : { thinking: thinkingMode === "on" }),
        onReasoning: (text: string) => { if (valid()) setReasoning(text); },
        onResult: (response: GenerationResult) => {
          if (valid()) {
            result.current = response;
            setGenerationResult(response);
            setThinkingMessage(thinkingNotice(response));
          }
        },
        onStatus: (status: string) => {
          if (valid()) setAIStatus(status);
        },
      },
    };
  };
  const finishJob = (controller: AbortController) => {
    if (job.current === controller) {
      job.current = null;
      setAIBusy(false);
    }
  };
  const ask = async (
    event?: FormEvent,
    customQuestion?: string,
    customScope?: "general" | "context" | "article",
  ) => {
    event?.preventDefault();
    const prompt = customQuestion ?? question.trim();
    if (!prompt) return;
    const work = startJob();
    setAnswer("");
    setLastQuestion(prompt);
    setPanelTab("ask");
    setMobilePanel("assist");
    try {
      const result = await ai.current.generateResult(
        buildQuestionPrompt(
          prompt,
          customScope ?? scope,
          work.doc,
          selection.text,
        ),
        {
          ...work.options,
          onText: (text) => {
            if (work.valid()) setAnswer(text);
          },
        },
      );
      if (work.valid()) setAIStatus(completionMessage(result));
    } catch (error) {
      if (work.valid()) {
        setGenerationResult(null);
        setAIStatus(errorMessage(error));
      }
    } finally {
      finishJob(work.controller);
    }
  };
  const writeAction = async (action: AIAction) => {
    if (action === "fallacies" || action === "hemingway") {
      await ask(
        undefined,
        writingPrompt(action, textOf(currentRef.current!.content)),
        "general",
      );
      return;
    }
    const blocks = editor.current?.getBlocks() ?? [];
    const selected =
      blocks.find((block) => block.id === selection.blockId) ?? blocks[0];
    if (!selected) throw new Error("Add a paragraph first.");
    const doc = currentRef.current!;
    if (action === "text" && !doc.ideas[selected.id]?.trim())
      throw new Error(
        "Write an idea beside this paragraph before generating its text.",
      );
    if (
      action === "text" &&
      selected.text.trim() &&
      !confirm(
        "Replace this paragraph with generated text? You can undo the replacement.",
      )
    )
      return;
    if (
      action === "idea" &&
      doc.ideas[selected.id]?.trim() &&
      !confirm("Replace this paragraph’s idea?")
    )
      return;
    if (action === "rewrite" && !instruction.trim())
      throw new Error("Add a rewrite instruction first.");
    const work = startJob();
    setAnswer("");
    setLastQuestion(
      (
        {
          text: "Generate text",
          idea: "Generate idea",
          comments: "Implement [comments]",
          rewrite: "Rewrite article",
        } as const
      )[action],
    );
    setPanelTab("ask");
    try {
      if (action === "rewrite") {
        const originals = blocks.filter((block) => block.text.trim());
        const results: { id: string; text: string }[] = [];
        for (const [index, block] of originals.entries()) {
          if (!work.valid()) return;
          setGenerationResult(null);
          setReasoning("");
          const context = [
            originals[index - 1]?.text,
            work.doc.ideas[block.id],
            originals[index + 1]?.text,
          ]
            .filter(Boolean)
            .join("\n");
          const text = await ai.current.generate(
            writingPrompt("rewrite", block.text, instruction, context),
            {
              ...work.options,
              onText: (text) => {
                if (work.valid())
                  setAnswer(
                    results
                      .map((item) => item.text)
                      .concat(text)
                      .join("\n\n"),
                  );
              },
            },
          );
          results.push({ id: block.id, text });
        }
        if (work.valid()) {
          finishJob(work.controller);
          for (const result of results)
            editor.current?.insertAfterBlock(result.id, result.text);
          setThinkingMessage(work.result.current ? thinkingNotice(work.result.current) : "");
          setAIStatus("Rewritten paragraphs inserted after the originals.");
        }
      } else {
        const idx = blocks.findIndex((block) => block.id === selected.id);
        const context = [
          blocks[idx - 1]?.text,
          work.doc.ideas[selected.id],
          blocks[idx + 1]?.text,
        ]
          .filter(Boolean)
          .join("\n");
        const inputText =
          action === "text"
            ? work.doc.ideas[selected.id]
            : selection.kind === "text" && selection.text.trim()
              ? selection.text
              : selected.text;
        const text = await ai.current.generate(
          writingPrompt(action, inputText, instruction, context),
          {
            ...work.options,
            onText: (text) => {
              if (work.valid()) setAnswer(text);
            },
          },
        );
        if (work.valid()) {
          finishJob(work.controller);
          if (action === "idea") {
            const previous = currentRef.current!.ideas[selected.id] ?? "";
            editDocument({
              ideas: {
                ...currentRef.current!.ideas,
                [selected.id]: text.trim(),
              },
            });
            setIdeaUndo({ stamp: stamp(), blockId: selected.id, previous });
          } else if (action === "comments")
            editor.current?.insertAfterBlock(selected.id, text);
          else editor.current?.replaceBlockText(selected.id, text);
          setThinkingMessage(work.result.current ? thinkingNotice(work.result.current) : "");
          setAIStatus(
            action === "idea"
              ? "Idea generated locally."
              : "Completed locally. Original text edits can be undone.",
          );
        }
      }
    } catch (error) {
      if (work.valid()) {
        setGenerationResult(error instanceof IncompleteGenerationError ? error.result : null);
        setAIStatus(errorMessage(error));
      }
    } finally {
      finishJob(work.controller);
    }
  };
  const relatedSearch = async (useSelection = true) => {
    const work = startJob();
    setRelated([]);
    setPanelTab("related");
    try {
      const visible = await storage.listDocuments();
      if (!work.valid()) return;
      const results = await findRelated(
        ai.current,
        work.doc,
        visible,
        useSelection
          ? selection.text || textOf(work.doc.content)
          : textOf(work.doc.content),
        {
          ...work.options,
          scope: sourceScope,
          includeTitle: sourceInclude,
          excludeTitle: sourceExclude,
          includeUrl: sourceUrl,
          excludeUrl,
          advanced: advancedFilter,
          limit: 24,
        },
      );
      if (work.valid()) {
        setRelated(results);
        setRelatedLimit(3);
        setAIStatus(
          `${results.length} local match${results.length === 1 ? "" : "es"}`,
        );
      }
    } catch (error) {
      if (work.valid()) setAIStatus(errorMessage(error));
    } finally {
      finishJob(work.controller);
    }
  };
  useEffect(() => {
    if (
      !relatedEnabled ||
      !aiAvailable ||
      !current ||
      panelTab !== "related" ||
      aiBusy
    )
      return;
    const count = words(textOf(current.content));
    if (count < 10) return;
    const bucket = `${storage.epoch}:${current.id}:${current.protection}:${Math.floor(count / 10)}`;
    if (bucket === lastAutoBucket.current) return;
    const timer = setTimeout(() => {
      lastAutoBucket.current = bucket;
      void relatedSearch(false);
    }, 1200);
    return () => clearTimeout(timer);
  }, [current, relatedEnabled, aiAvailable, panelTab, aiBusy]);
  const graph = async () => {
    const work = startJob();
    try {
      const result = await generateGraph(ai.current, work.doc, work.options);
      if (work.valid()) {
        finishJob(work.controller);
        editDocument({ graph: result });
        setAIStatus("Graph generated locally");
      }
    } catch (error) {
      if (work.valid()) setAIStatus(errorMessage(error));
    } finally {
      finishJob(work.controller);
    }
  };

  const importReference = async (file: File) =>
    transition(async () => {
      if (file.size > 1024 * 1024 || !/\.(txt|md|markdown)$/i.test(file.name))
        throw new Error("Choose a text or Markdown file up to 1 MiB.");
      const expected = stamp();
      const token = storage.epoch;
      const owner = storage.activeUser?.id;
      const protection = currentRef.current?.protection ?? "open";
      const text = await file.text();
      if (token !== storage.epoch)
        throw new Error("The user changed. Choose the reference again.");
      assertStamp(expected);
      await flush();
      assertStamp(expected);
      const doc = await storage.createDocument(protection, "reference");
      assertStamp(expected);
      if (owner !== storage.activeUser?.id)
        throw new Error("The local user changed. Choose the reference again.");
      const content: RichNode = {
        type: "doc",
        content: text.split(/\n\s*\n/).map((paragraph) => ({
          type: "paragraph",
          attrs: { blockId: crypto.randomUUID() },
          content: paragraph ? [{ type: "text", text: paragraph }] : [],
        })),
      };
      const saved = await storage.saveDocument({
        ...doc,
        title: file.name,
        content,
      });
      assertStamp(expected);
      clearTransient();
      await mountAuthorized(saved.id, token, owner);
      setToast(
        `Reference imported as ${protection === "open" ? "shared / Open" : "Protected"}.`,
      );
    });

  const exportCurrent = async () => {
    const expected = stamp();
    await flush();
    assertStamp(expected);
    const doc = currentRef.current;
    if (!doc) return;
    if (exportMode === "HTML") {
      downloadFile(
        standaloneHTML(doc, includeIdeas),
        `${doc.title || "Untitled"}.html`,
        "text/html",
      );
      setToast("HTML downloaded. This exported copy is not encrypted.");
    } else if (exportMode === "Substack" || exportMode === "Medium") {
      await copyArticle(doc, includeIdeas);
      setToast("Article copied. Paste it into your publishing editor.");
    } else {
      printCleanup.current?.();
      printCleanup.current = printArticle(
        doc,
        exportMode === "Enhanced PDF" && includeIdeas,
      );
    }
  };
  const filtered = documents
    .filter(
      (doc) =>
        !query.trim() ||
        `${doc.title}\n${textOf(doc.content)}\n${Object.values(doc.ideas).join("\n")}`
          .toLocaleLowerCase()
          .includes(query.toLocaleLowerCase()),
    )
    .sort((a, b) => b.updatedAt - a.updatedAt);
  const openDocs = filtered.filter((doc) => doc.protection === "open");
  const protectedDocs = filtered.filter(
    (doc) => doc.protection === "protected",
  );
  const count = current ? words(textOf(current.content)) : 0;
  const ideaNodes = current
    ? Object.entries(current.ideas)
        .filter(([, idea]) => idea.trim())
        .map(([id, label]) => ({ id, label }))
    : [];
  const renderDoc = (doc: EditorDocument) => (
    <button
      key={doc.id}
      className={`document-card ${current?.id === doc.id ? "active" : ""}`}
      disabled={documentBusy}
      onClick={() => void safely(() => selectDocument(doc))}
    >
      <span className="document-card-title">
        <Icon
          name={doc.kind === "reference" ? "library" : "document"}
          size={15}
        />
        <strong>{doc.title || "Untitled"}</strong>
        {doc.protection === "protected" && <Icon name="lock" size={12} />}
      </span>
      <span className="document-preview">
        {textOf(doc.content).slice(0, 85) || "A fresh page, ready for you."}
      </span>
      <span className="document-date">{relativeDate(doc.updatedAt)}</span>
    </button>
  );

  return (
    <div className={`app ${mobilePanel}`}>
      <aside className="sidebar" aria-label="Documents">
        <a className="brand" href="./" aria-label="Editor home">
          <span className="brand-icon">e</span>
          <span>
            Editor<span className="brand-caption">A little room to think.</span>
          </span>
        </a>
        <div className="new-document-wrap">
          <button
            className="primary new-document"
            disabled={documentBusy}
            onClick={() => setNewMenu(!newMenu)}
          >
            <Icon name="plus" /> New document{" "}
            <span className="shortcut">＋</span>
          </button>
          {newMenu && (
            <div className="popover new-popover">
              <button
                onClick={() => {
                  setNewMenu(false);
                  void safely(() => createDocument("open"));
                }}
              >
                <Icon name="open" />
                <span>
                  Open document<small>Shared in this browser</small>
                </span>
              </button>
              <button
                onClick={() => {
                  setNewMenu(false);
                  requireUser(() => createDocument("protected"));
                }}
              >
                <Icon name="lock" />
                <span>
                  Protected document<small>Only your local user</small>
                </span>
              </button>
            </div>
          )}
        </div>
        <label className="search">
          <Icon name="search" size={16} />
          <input
            aria-label="Search documents"
            placeholder="Find a document…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <span>⌕</span>
        </label>
        <div className="document-lists">
          <div className="section-label">
            <span>SHARED / OPEN</span>
            <span>{openDocs.length}</span>
          </div>
          {openDocs.map(renderDoc)}
          {!openDocs.length && (
            <p className="small-empty">
              {query
                ? "No matching shared documents."
                : "Shared documents live here."}
            </p>
          )}
          <div className="section-label protected-label">
            <span>
              <Icon name="lock" size={12} /> MY PROTECTED
            </span>
            <span>{protectedDocs.length || ""}</span>
          </div>
          {activeUser ? (
            <>
              {protectedDocs.map(renderDoc)}
              {!protectedDocs.length && (
                <p className="small-empty">
                  Only you can unlock your protected work.
                </p>
              )}
            </>
          ) : (
            <button
              className="unlock-card"
              onClick={() =>
                requireUser(async () => {
                  await reload();
                })
              }
            >
              <Icon name="lock" />
              <span>
                Unlock your space
                <small>Your private documents stay hidden.</small>
              </span>
            </button>
          )}
        </div>
        <div className="sidebar-footer">
          <div className="local-note">
            <span className="status-dot" /> Stored on this device{" "}
            <button
              className="icon-button"
              aria-label="About local storage"
              onClick={() => setModal("help")}
            >
              ⓘ
            </button>
          </div>
          <div className="profile-row">
            <button
              className="profile-button"
              onClick={() => void safely(() => showUsers())}
            >
              <span className="avatar">
                {activeUser?.name.slice(0, 1).toUpperCase() || (
                  <Icon name="user" size={16} />
                )}
              </span>
              <span>
                {activeUser?.name || "Local users"}
                <small>
                  {activeUser
                    ? "Protected space unlocked"
                    : "No account needed for Open"}
                </small>
              </span>
            </button>
            {activeUser && (
              <button
                className="icon-button"
                aria-label="Lock protected documents"
                title="Lock protected documents"
                onClick={() => void lock()}
              >
                <Icon name="lock" size={16} />
              </button>
            )}
            <button
              className="icon-button"
              aria-label="Toggle dark theme"
              onClick={() => setDark(!dark)}
            >
              <Icon name={dark ? "sun" : "moon"} size={16} />
            </button>
          </div>
        </div>
      </aside>
      <nav className="mobile-navigation" aria-label="Workspace panels">
        {[
          ["documents", "Documents"],
          ["write", "Write"],
          ["assist", "Assistant"],
        ].map(([value, label]) => (
          <button
            key={value}
            className={mobilePanel === value ? "selected" : ""}
            onClick={() => setMobilePanel(value)}
          >
            {label}
          </button>
        ))}
      </nav>
      <main className="workspace" aria-busy={documentBusy}>
        {documentBusy && (
          <div className="document-loading" role="status">
            Opening your document…
          </div>
        )}
        <header className="workspace-topbar">
          <div className="breadcrumb">
            <Icon name="document" size={16} />
            <span>
              {current?.kind === "reference"
                ? "Reference library"
                : "Your workspace"}
            </span>
            <span className="breadcrumb-separator">/</span>
            <span>
              {current
                ? current.protection === "open"
                  ? "Shared document"
                  : "Protected document"
                : "Welcome"}
            </span>
          </div>
          <div className="top-actions">
            {current && (
              <>
                <span
                  className={`save-state ${saveState === "Save failed" ? "failed" : ""}`}
                  aria-live="polite"
                >
                  <Icon
                    name={saveState === "Saved locally" ? "check" : "dots"}
                    size={14}
                  />
                  {saveState}
                </span>
                {saveState === "Save failed" && (
                  <button
                    className="text-button"
                    onClick={() => void safely(flush)}
                  >
                    Retry
                  </button>
                )}
                <button
                  className="button subtle"
                  disabled={documentBusy}
                  onClick={() => {
                    setIncludeIdeas(false);
                    setModal("export");
                  }}
                >
                  <Icon name="export" size={15} /> Export
                </button>
              </>
            )}
          </div>
        </header>
        {current ? (
          <div className="writing-workspace" inert={documentBusy || undefined}>
            <div className="document-heading">
              <div className="document-eyebrow">
                <span className={`protection-badge ${current.protection}`}>
                  <Icon
                    name={current.protection === "open" ? "open" : "lock"}
                    size={12}
                  />
                  {current.protection === "open"
                    ? "SHARED / OPEN"
                    : `PROTECTED · ${activeUser?.name ?? ""}`}
                </span>
                <div className="document-tools">
                  <button
                    className={`text-button ${ideasVisible ? "selected" : ""}`}
                    onClick={() => setIdeasVisible(!ideasVisible)}
                  >
                    ◧ Ideas
                  </button>
                  {ideasVisible && (
                    <button
                      className="text-button"
                      onClick={() => setIdeasPrimary(!ideasPrimary)}
                    >
                      {ideasPrimary ? "Text first" : "Ideas first"}
                    </button>
                  )}
                  <button
                    className="icon-button"
                    aria-label="Topic graph"
                    disabled={ideaNodes.length < 2}
                    onClick={() => setModal("graph")}
                  >
                    <Icon name="graph" size={16} />
                  </button>
                  <div className="menu-anchor">
                    <button
                      className="icon-button"
                      aria-label="Document actions"
                      onClick={() => setDocMenu(!docMenu)}
                    >
                      <Icon name="dots" />
                    </button>
                    {docMenu && (
                      <div className="popover document-popover">
                        <button onClick={() => void safely(clone)}>
                          <Icon name="copy" /> Clone document
                        </button>
                        <button
                          onClick={() =>
                            current.protection === "open"
                              ? requireUser(changeProtection)
                              : void safely(changeProtection)
                          }
                        >
                          <Icon name="lock" />
                          {current.protection === "open"
                            ? "Protect document…"
                            : "Make shared / Open…"}
                        </button>
                        <button onClick={() => importInput.current?.click()}>
                          <Icon name="library" />
                          Import local reference…
                        </button>
                        <button
                          className="danger"
                          onClick={() => void safely(remove)}
                        >
                          <Icon name="trash" />
                          Delete document…
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              <input
                className="document-title"
                aria-label="Document title"
                placeholder="Untitled"
                value={current.title}
                onChange={(e) => editDocument({ title: e.target.value })}
              />
              {current.kind === "reference" && (
                <label className="reference-url">
                  Source URL (optional)
                  <input
                    type="url"
                    aria-label="Reference source URL"
                    maxLength={2048}
                    placeholder="https://…"
                    value={current.sourceUrl ?? ""}
                    onChange={(event) =>
                      editDocument({
                        sourceUrl: event.target.value || undefined,
                      })
                    }
                  />
                </label>
              )}
              <div className="document-meta">
                <span>{count.toLocaleString()} words</span>
                <span>·</span>
                <span>{Math.max(1, Math.ceil(count / 200))} min read</span>
                <span>·</span>
                <span>
                  {current.protection === "open"
                    ? "Anyone using this browser can edit"
                    : "Encrypted on this device"}
                </span>
              </div>
            </div>
            <RichEditor
              key={current.id}
              documentId={current.id}
              content={current.content}
              ideas={current.ideas}
              ideasVisible={ideasVisible}
              ideasPrimary={ideasPrimary}
              onChange={(content, ideas) =>
                editDocument({ content, ideas, graph: undefined })
              }
              onSelection={setSelection}
              onReady={(api) => {
                editor.current = api;
              }}
            />
            <div className="canvas-footer">
              <span>
                {current.protection === "protected" ? (
                  <>
                    <Icon name="lock" size={12} /> Protected for{" "}
                    {activeUser?.name}
                  </>
                ) : (
                  <>
                    <span className="status-dot" /> Your words stay on this
                    device
                  </>
                )}
              </span>
              <span>Use / for blocks · Select text to ask AI</span>
            </div>
          </div>
        ) : (
          <div className="welcome">
            <div className="welcome-mark">
              <Icon name="document" size={32} />
            </div>
            <span className="eyebrow">YOUR LOCAL WRITING SPACE</span>
            <h1>
              Make room
              <br />
              for a good idea.
            </h1>
            <p>
              A blank page, a thoughtful assistant,
              <br />
              and words that stay yours.
            </p>
            <button
              className="primary"
              onClick={() => void safely(() => createDocument("open"))}
            >
              <Icon name="plus" /> Start a document
            </button>
            <button
              className="text-button"
              onClick={() => requireUser(() => createDocument("protected"))}
            >
              <Icon name="lock" size={13} /> Or start with a protected page
            </button>
            <div className="welcome-features">
              <span>✎ Rich text & ideas</span>
              <span>◌ Local AI</span>
              <span>⌂ Stored here</span>
            </div>
          </div>
        )}
      </main>
      <aside
        className="assistant-panel"
        aria-label="Local AI assistant"
        inert={documentBusy || undefined}
      >
        <div className="assistant-heading">
          <span className="assistant-symbol">
            <Icon name="spark" size={19} />
          </span>
          <div>
            <strong>A little perspective</strong>
            <span>Your local writing assistant</span>
          </div>
          <span
            className={`runtime-dot ${aiAvailable ? "available" : ""}`}
            title={aiReadyMessage}
          />
        </div>
        <div className="panel-tabs">
          <button
            className={panelTab === "ask" ? "active" : ""}
            onClick={() => setPanelTab("ask")}
          >
            Ask AI
          </button>
          <button
            className={panelTab === "related" ? "active" : ""}
            onClick={() => setPanelTab("related")}
          >
            Related sources
          </button>
        </div>
        <div className="thinking-control">
          <label>Thinking
            <select aria-label="Thinking mode" value={thinkingMode} onChange={(event) => setThinkingMode(event.target.value as typeof thinkingMode)}>
              <option value="default">Model default</option>
              <option value="on">On</option>
              <option value="off">Off</option>
            </select>
          </label>
          <small>Applies to the next request. Support depends on the selected model.</small>
        </div>
        <div className="assistant-scroll">
          {thinkingMessage && <p className="thinking-notice" role="status">{thinkingMessage}</p>}
          {!aiAvailable && (
            <div className="runtime-message">
              <Icon name="spark" size={16} />
              <div>
                <p role="status">
                  {aiReadyMessage} Editing, protection and export work without
                  AI.
                </p>
                <button
                  className="text-button"
                  disabled={aiChecking}
                  onClick={() => retryAIConnection.current()}
                >
                  {aiChecking
                    ? "Checking connection…"
                    : "Retry local AI connection"}
                </button>
              </div>
            </div>
          )}
          {panelTab === "ask" ? (
            <>
              {aiStatus && !answer && !aiBusy && !lastQuestion && (
                <p className="job-status" role="status">
                  {aiStatus}
                </p>
              )}
              {answer || reasoning || aiBusy || lastQuestion ? (
                <section className="answer-card">
                  <div className="answer-heading">
                    <span>{lastQuestion || "Working locally"}</span>
                    <button
                      className="icon-button"
                      aria-label="Close AI answer"
                      onClick={() => {
                        cancelJob();
                        setAnswer("");
                        setLastQuestion("");
                      }}
                    >
                      <Icon name="close" size={14} />
                    </button>
                  </div>
                  {answer && <p className="answer-completion">{aiBusy ? "Final-answer preview · generating" : generationResult && isCompleteAnswer(generationResult) ? "Final answer · complete" : "Final-answer preview · incomplete or unconfirmed"}</p>}
                  <div className="ai-answer" aria-live="polite">
                    {answer || (aiBusy ? reasoning ? "Thinking locally…" : "Preparing your local model…" : reasoning ? "No final answer was produced." : "")}
                  </div>
                  {reasoning && <details className="model-reasoning"><summary>View model reasoning · separate from your document</summary><p>{reasoning}</p></details>}
                  {aiStatus && <p className="job-status">{aiStatus}</p>}
                  {answer && (
                    <button
                      className="text-button"
                      disabled={aiBusy || !current}
                      onClick={() => {
                        if (resultStamp.current !== stamp()) {
                          setToast(
                            "The document changed. Ask again before inserting this answer.",
                          );
                          return;
                        }
                        if ((!generationResult || !isCompleteAnswer(generationResult)) && !confirm("This final-answer preview is incomplete or its completion is unconfirmed. Insert it deliberately at the cursor? Your existing text will remain.")) return;
                        editor.current?.insertText(answer);
                        setToast("Answer inserted into the current document.");
                      }}
                    >
                      {generationResult && isCompleteAnswer(generationResult) ? "Insert at cursor ↗" : "Use incomplete preview…"}
                    </button>
                  )}
                </section>
              ) : (
                <div className="assistant-empty">
                  <span className="assistant-watermark">
                    <Icon name="spark" size={28} />
                  </span>
                  <h3>Think it through.</h3>
                  <p>
                    Find a clearer phrase, explore an idea,
                    <br />
                    or see your draft in a new light.
                  </p>
                  <div className="suggested-questions">
                    {[
                      "What could make this clearer?",
                      "Find logical fallacies",
                      "Hemingway analysis",
                    ].map((label, index) => (
                      <button
                        key={label}
                        disabled={!current || !aiAvailable || aiBusy}
                        onClick={() =>
                          void safely(() =>
                            index === 0
                              ? ask(undefined, label, "article")
                              : writeAction(
                                  index === 1 ? "fallacies" : "hemingway",
                                ),
                          )
                        }
                      >
                        {label}
                        <span>↗</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {current && (
                <div className="writing-actions">
                  <div className="section-label">WORK ON YOUR DRAFT</div>
                  {ideaUndo && (
                    <button
                      className="text-button"
                      onClick={() => {
                        if (ideaUndo.stamp !== stamp() || !currentRef.current) {
                          setIdeaUndo(null);
                          return;
                        }
                        editDocument({
                          ideas: {
                            ...currentRef.current.ideas,
                            [ideaUndo.blockId]: ideaUndo.previous,
                          },
                        });
                      }}
                    >
                      Undo generated idea
                    </button>
                  )}
                  <textarea
                    aria-label="Writing recommendation"
                    placeholder="A recommendation or rewrite instruction…"
                    value={instruction}
                    onChange={(e) => setInstruction(e.target.value)}
                    rows={2}
                  />
                  <div>
                    {[
                      ["idea", "Generate idea"],
                      ["text", "Generate text"],
                      ["comments", "Implement [comments]"],
                      ["rewrite", "Rewrite article"],
                    ].map(([action, label]) => (
                      <button
                        key={action}
                        className="button"
                        disabled={!aiAvailable || aiBusy}
                        onClick={() =>
                          void safely(() => writeAction(action as AIAction))
                        }
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="related-panel">
              <p className="muted">
                Find connections in your own documents and imported references.
                This is a local search.
              </p>
              <select
                aria-label="Source scope"
                value={sourceScope}
                onChange={(e) =>
                  setSourceScope(e.target.value as typeof sourceScope)
                }
              >
                <option value="all">
                  All sources available to this document
                </option>
                <option value="references">Reference library</option>
                <option value="documents">Other editor documents</option>
              </select>
              <details>
                <summary>Source filters</summary>
                <input
                  aria-label="Include source titles"
                  placeholder="Include title…"
                  value={sourceInclude}
                  onChange={(e) => setSourceInclude(e.target.value)}
                />
                <input
                  aria-label="Exclude source titles"
                  placeholder="Exclude title…"
                  value={sourceExclude}
                  onChange={(e) => setSourceExclude(e.target.value)}
                />
                <input
                  aria-label="Include source URLs"
                  placeholder="Include source URL…"
                  value={sourceUrl}
                  onChange={(e) => setSourceUrl(e.target.value)}
                />
                <input
                  aria-label="Exclude source URLs"
                  placeholder="Exclude source URL…"
                  value={excludeUrl}
                  onChange={(e) => setExcludeUrl(e.target.value)}
                />
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={advancedFilter}
                    onChange={(e) => setAdvancedFilter(e.target.checked)}
                  />
                  Filter matches with local AI
                </label>
              </details>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={relatedEnabled}
                  onChange={(e) => setRelatedEnabled(e.target.checked)}
                />
                Enable automatic related lookup
              </label>
              <button
                className="button"
                disabled={!current || !aiAvailable || aiBusy}
                onClick={() => void safely(() => relatedSearch(true))}
              >
                <Icon name="search" size={14} />
                Find sources for selection / draft
              </button>
              <button
                className="text-button"
                onClick={() => importInput.current?.click()}
              >
                <Icon name="plus" size={14} />
                Import{" "}
                {current?.protection === "protected"
                  ? "Protected"
                  : "Open"}{" "}
                text / Markdown reference
              </button>
              {aiStatus && <p className="job-status">{aiStatus}</p>}
              {related.slice(0, relatedLimit).map((match) => (
                <article className="source-card" key={match.documentId}>
                  <button
                    className="source-title"
                    onClick={() => setSourceDetail(match)}
                  >
                    {match.title || "Untitled"}
                  </button>
                  <small>
                    {match.kind === "reference" ? "Reference" : "Document"} ·{" "}
                    {(match.score * 100).toFixed(0)}% similarity
                  </small>
                  <p>{match.snippet}</p>
                  <button
                    className="text-button"
                    onClick={() => {
                      if (resultStamp.current !== stamp()) {
                        setRelated([]);
                        return;
                      }
                      editor.current?.insertText(
                        ` [${match.title || "Untitled"}${match.sourceUrl ? ` — ${match.sourceUrl}` : ""}] `,
                      );
                    }}
                  >
                    Insert reference ↗
                  </button>
                </article>
              ))}
              {related.length > relatedLimit && (
                <button
                  className="text-button"
                  onClick={() => setRelatedLimit(relatedLimit + 3)}
                >
                  Show three more
                </button>
              )}
            </div>
          )}
        </div>
        <div className="ask-composer">
          {selection.text && panelTab === "ask" && (
            <div className="context-preview">
              <span>Selected context</span>
              <p>{selection.text.slice(0, 220)}</p>
            </div>
          )}
          {panelTab === "ask" && (
            <form onSubmit={(event) => void safely(() => ask(event))}>
              <div className="scope-row">
                <label>
                  Use
                  <select
                    aria-label="AI context scope"
                    value={scope}
                    onChange={(e) => setScope(e.target.value as typeof scope)}
                  >
                    <option value="general">General · question only</option>
                    <option value="context">Context · selection / draft</option>
                    <option value="article">Article · full draft</option>
                  </select>
                </label>
                {scope === "context" && !selection.text && (
                  <small>Full draft fallback</small>
                )}
              </div>
              <textarea
                aria-label="Ask local AI"
                placeholder="Ask a question about your writing…"
                rows={3}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault();
                    void safely(() => ask());
                  }
                }}
              />
              <div className="composer-bottom">
                <span>
                  Local to your device <Icon name="lock" size={11} />
                </span>
                {aiBusy ? (
                  <button
                    className="stop-button"
                    type="button"
                    onClick={() =>
                      cancelJob(
                        "Cancellation requested. Partial output is shown above.",
                      )
                    }
                  >
                    ■ Stop
                  </button>
                ) : (
                  <button
                    className="send-button"
                    aria-label="Send question"
                    disabled={!current || !question.trim() || !aiAvailable}
                  >
                    <Icon name="arrow" size={17} />
                  </button>
                )}
              </div>
            </form>
          )}
          {aiBusy && panelTab === "related" && (
            <button
              className="stop-button"
              onClick={() => cancelJob("Cancellation requested.")}
            >
              ■ Stop local processing
            </button>
          )}
          <p className="assistant-footnote">
            AI can be mistaken. Keep your own perspective.
          </p>
        </div>
      </aside>
      <input
        ref={importInput}
        type="file"
        accept=".txt,.md,.markdown,text/plain,text/markdown"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) void safely(() => importReference(file));
        }}
      />
      <input
        ref={restoreInput}
        type="file"
        accept=".json,application/json"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (!file) return;
          void safely(async () => {
            if (file.size > 20 * 1024 * 1024)
              throw new Error("Backup must be smaller than 20 MiB.");
            setRestoreText(await file.text());
            setPassword("");
            setAuthError("");
            setModal("restore");
          });
        }}
      />
      {modal === "users" && (
        <Modal
          title={activeUser ? "Local users" : "Your protected space"}
          onClose={() => {
            if (!authBusy) {
              setModal(null);
              setPassword("");
              pendingAuth.current = null;
            }
          }}
        >
          <p className="muted">
            Open documents are shared. A local password unlocks only your
            protected documents. All users have the same access level.
          </p>
          {activeUser && (
            <div className="active-profile">
              <span className="avatar">{activeUser.name.slice(0, 1)}</span>
              <span>
                <strong>{activeUser.name}</strong>
                <small>Currently unlocked</small>
              </span>
              <button className="button" onClick={() => void lock()}>
                Lock
              </button>
            </div>
          )}
          <div className="segmented">
            <button
              className={userMode === "unlock" ? "selected" : ""}
              onClick={() => {
                setUserMode("unlock");
                setAuthError("");
                setPassword("");
              }}
            >
              Unlock user
            </button>
            <button
              className={userMode === "create" ? "selected" : ""}
              onClick={() => {
                setUserMode("create");
                setAuthError("");
                setPassword("");
              }}
            >
              Create local user
            </button>
          </div>
          <form onSubmit={(event) => void authenticate(event)}>
            {userMode === "create" ? (
              <label>
                Local name
                <input
                  required
                  maxLength={60}
                  autoComplete="off"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </label>
            ) : (
              <label>
                Local user
                <select
                  aria-label="Local user"
                  required
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                >
                  <option value="" disabled>
                    Choose a user
                  </option>
                  {users.map((user) => (
                    <option key={user.id} value={user.id}>
                      {user.name}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label>
              Password
              <input
                type="password"
                required
                minLength={userMode === "create" ? 12 : undefined}
                autoComplete={
                  userMode === "create" ? "new-password" : "current-password"
                }
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </label>
            {userMode === "create" && (
              <p className="form-hint">
                Use a strong private passphrase of at least 12 characters. There
                is no password reset or master key.
              </p>
            )}
            {authError && (
              <p role="alert" className="error-message">
                {authError}
              </p>
            )}
            <button
              className="primary"
              disabled={authBusy || (userMode === "unlock" && !users.length)}
            >
              {authBusy
                ? "Unlocking locally…"
                : userMode === "create"
                  ? "Create & unlock"
                  : "Unlock locally"}
            </button>
          </form>
          <div className="profile-utilities">
            {activeUser && (
              <>
                <button
                  className="text-button"
                  onClick={() => {
                    setPassword("");
                    setNewPassword("");
                    setAuthError("");
                    setModal("password");
                  }}
                >
                  Change my password
                </button>
                <button
                  className="text-button"
                  onClick={() =>
                    void safely(async () => {
                      const expected = stamp();
                      await flush();
                      assertStamp(expected);
                      const backup = await storage.exportBackup();
                      assertStamp(expected);
                      downloadFile(backup, "editor-encrypted-backup.json");
                      setToast(
                        "Your encrypted protected backup was downloaded. Keep your password with care.",
                      );
                    })
                  }
                >
                  Download my encrypted backup
                </button>
              </>
            )}
            <button
              className="text-button"
              onClick={() => restoreInput.current?.click()}
            >
              Restore an encrypted backup…
            </button>
          </div>
        </Modal>
      )}
      {modal === "password" && (
        <Modal
          title="Change your local password"
          onClose={() => {
            if (!authBusy) {
              setModal(null);
              setPassword("");
              setNewPassword("");
            }
          }}
        >
          <p className="muted">
            This changes how your existing keys are unlocked. It cannot revoke
            earlier backups or a previously stolen key.
          </p>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              if (authTransition.current) return;
              const expected = stamp();
              authTransition.current = true;
              setAuthBusy(true);
              setAuthError("");
              void flush()
                .then(() => {
                  assertStamp(expected);
                  return storage.changePassword(password, newPassword);
                })
                .then(() => {
                  setModal(null);
                  setPassword("");
                  setNewPassword("");
                  setToast(
                    "Password changed. New backups use the new password.",
                  );
                })
                .catch((error) => setAuthError(errorMessage(error)))
                .finally(() => {
                  authTransition.current = false;
                  setAuthBusy(false);
                });
            }}
          >
            <label>
              Current password
              <input
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </label>
            <label>
              New private passphrase
              <input
                type="password"
                autoComplete="new-password"
                required
                minLength={12}
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
              />
            </label>
            {authError && (
              <p className="error-message" role="alert">
                {authError}
              </p>
            )}
            <button className="primary" disabled={authBusy}>
              {authBusy ? "Updating…" : "Change password"}
            </button>
          </form>
        </Modal>
      )}
      {modal === "restore" && (
        <Modal
          title="Restore your protected documents"
          onClose={() => {
            if (!authBusy) {
              setModal(null);
              setPassword("");
              setRestoreText("");
            }
          }}
        >
          <p className="muted">
            Enter this backup’s password. Matching documents belonging to this
            user will be replaced after confirmation. Other users and Open
            documents are preserved.
          </p>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              if (
                authTransition.current ||
                !confirm(
                  "Restore this encrypted backup? Matching documents belonging to this user will be replaced with the backup versions. Other users and Open documents are preserved.",
                )
              )
                return;
              const expected = stamp();
              authTransition.current = true;
              setAuthBusy(true);
              setAuthError("");
              void flush()
                .then(() => {
                  assertStamp(expected);
                  return storage.restoreBackup(restoreText, password);
                })
                .then(async () => {
                  setModal(null);
                  setPassword("");
                  setRestoreText("");
                  await reload();
                  setToast(
                    "Encrypted backup restored. Unlock its user to read the documents.",
                  );
                })
                .catch((error) => setAuthError(errorMessage(error)))
                .finally(() => {
                  authTransition.current = false;
                  setAuthBusy(false);
                });
            }}
          >
            <label>
              Backup password
              <input
                type="password"
                autoComplete="off"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </label>
            {authError && (
              <p className="error-message" role="alert">
                {authError}
              </p>
            )}
            <button className="primary" disabled={authBusy}>
              {authBusy ? "Checking backup…" : "Validate & restore"}
            </button>
          </form>
        </Modal>
      )}
      {modal === "export" && current && (
        <Modal title="Export your writing" wide onClose={() => setModal(null)}>
          <div className="export-controls">
            <label>
              Format
              <select
                aria-label="Export format"
                value={exportMode}
                onChange={(e) => setExportMode(e.target.value)}
              >
                {[
                  "HTML",
                  "Substack",
                  "Medium",
                  "Plain PDF",
                  "Enhanced PDF",
                ].map((format) => (
                  <option key={format}>{format}</option>
                ))}
              </select>
            </label>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={includeIdeas}
                onChange={(e) => setIncludeIdeas(e.target.checked)}
                disabled={exportMode === "Plain PDF"}
              />
              Include ideas
            </label>
            <button
              className="primary"
              onClick={() => void safely(exportCurrent)}
            >
              {exportMode === "HTML"
                ? "Download HTML"
                : ["Substack", "Medium"].includes(exportMode)
                  ? "Copy rich article"
                  : "Print / save PDF"}
            </button>
          </div>
          <p className="form-hint">
            {current.protection === "protected"
              ? "This is a deliberate plaintext export of your protected document. "
              : ""}
            {["Substack", "Medium"].includes(exportMode)
              ? "Copies to your clipboard; paste into the service yourself."
              : exportMode.includes("PDF")
                ? "Uses your browser’s print / Save as PDF dialog."
                : "A self-contained HTML article."}
          </p>
          <article
            className="export-preview"
            dangerouslySetInnerHTML={{
              __html: articleHTML(
                current,
                includeIdeas && exportMode !== "Plain PDF",
              ),
            }}
          />
        </Modal>
      )}
      {modal === "graph" && current && (
        <Modal
          title="Connections between your ideas"
          wide
          onClose={() => {
            cancelJob();
            setModal(null);
          }}
        >
          <p className="muted">
            Local embeddings propose pairs; a local language model classifies
            each connection.
          </p>
          <TopicGraphView
            nodes={current.graph?.nodes ?? ideaNodes}
            edges={current.graph?.edges ?? []}
          />
          {current.graph?.edges.length ? (
            <ul className="graph-relations">
              {current.graph.edges.map((edge, index) => (
                <li key={index}>
                  <span>
                    {
                      current.graph!.nodes.find((node) => node.id === edge.from)
                        ?.label
                    }
                  </span>
                  <strong>{edge.relation}</strong>
                  <span>
                    {
                      current.graph!.nodes.find((node) => node.id === edge.to)
                        ?.label
                    }
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="form-hint">
              Generate to find relationships. No relationships have been
              inferred yet.
            </p>
          )}
          <div className="modal-actions">
            <button
              className="primary"
              disabled={!aiAvailable || aiBusy || ideaNodes.length < 2}
              onClick={() => void safely(graph)}
            >
              Generate connections locally
            </button>
            {aiBusy && (
              <button
                className="button"
                onClick={() => cancelJob("Cancellation requested.")}
              >
                Stop
              </button>
            )}
            <span className="job-status">{aiStatus}</span>
          </div>
        </Modal>
      )}
      {sourceDetail && (
        <Modal
          title={sourceDetail.title || "Local source"}
          wide
          onClose={() => setSourceDetail(null)}
        >
          <p className="muted">{sourceDetail.kind} · Local semantic match</p>
          <div className="source-full-text">
            {sourceDetail.fullText
              .split(sourceDetail.snippet)
              .map((part, index, all) => (
                <span key={index}>
                  {part}
                  {index < all.length - 1 && (
                    <mark>{sourceDetail.snippet}</mark>
                  )}
                </span>
              ))}
          </div>
          <button
            className="button"
            onClick={() => {
              if (resultStamp.current === stamp())
                editor.current?.insertText(` [${sourceDetail.title}] `);
              setSourceDetail(null);
            }}
          >
            Insert source reference
          </button>
        </Modal>
      )}
      {modal === "help" && (
        <Modal
          title="Your words, on this device"
          onClose={() => setModal(null)}
        >
          <div className="help-content">
            <p>
              <strong>Open documents are shared.</strong> Anyone using this
              browser profile can read, edit or delete them.
            </p>
            <p>
              <strong>Protected documents use your password.</strong> Each local
              user has independent keys. There is no administrator, master
              password or remote reset. Lock your space before handing over the
              computer; unlocked documents can be read by whoever is using it.
            </p>
            <p>
              Editor automatically locks protected work after 15 minutes without
              keyboard or pointer activity. Other app tabs cooperate when you
              lock or switch users.
            </p>
            <p>
              Documents, ideas and AI requests stay local. AI needs Oryn
              Browser; other browsers support writing, encryption and exports.
              Imported references use the current document’s protection, or Open
              if no document is selected.
            </p>
            <p>
              <strong>Keep backups.</strong> Clearing browser data, changing the
              site address or losing your password can make protected work
              unrecoverable. Download your encrypted backup from Local users.
            </p>
            <p>
              The website still delivers the code you run. A compromised host,
              extension or computer can capture a later unlock. Local users do
              not replace separate OS accounts. Previously Open copies and
              plaintext exports cannot be recalled.
            </p>
            <p>
              Offline editing works after the application has been cached.
              Updates take effect after all Editor tabs close.
            </p>
          </div>
        </Modal>
      )}
      {toast && (
        <div className="toast" role="status">
          <span>{toast}</span>
          <button
            className="icon-button"
            aria-label="Dismiss notification"
            onClick={() => setToast("")}
          >
            <Icon name="close" size={15} />
          </button>
        </div>
      )}
    </div>
  );
}
