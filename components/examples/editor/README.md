# Editor

A static, local writing application with rich text, paragraph ideas and optional
local AI through the public Oryn SDK. Licensed under [MIT](../LICENSE).

From the examples workspace:

```sh
pnpm install --frozen-lockfile
pnpm --filter @ocno/oryn-example-editor dev
```

Open `http://127.0.0.1:5192`. Build with
`pnpm --filter @ocno/oryn-example-editor build`; serve `editor/dist` on a dedicated,
stable HTTPS origin. There is no application backend or server password to set.
Changing the origin creates a different local workspace.

## Writing

Create an Open or Protected document. The editor supports headings, lists, quotes,
code, inline formatting, links, fonts, colors, alignment, indentation and undo/redo.
Use `/` or a block's `+` menu for structure. Drag the block handle to reorder,
or use its move buttons. Ideas stay attached to their blocks. Toggle Ideas or
Ideas first to change the writing layout.

Titles, body text and ideas autosave after about 600 ms. Saving failures stay
visible; Retry attempts the save again. If another tab changed the same document,
reload its current copy before retrying rather than overwriting it. The example
keeps deletion tombstones so stale tabs cannot recreate deleted or formerly Open
content.

Search, clone and delete documents in the sidebar. Imported `.txt` / `.md`
references can have an optional source URL for filtering/citation, are limited to
1 MiB, and use the current document's protection, or Open
when no document is selected. Markdown is imported as safe plain text.

## Shared and protected documents

- **Open** documents are shared with everyone using this app in the same browser
  profile. No account is needed; anyone can read, edit, export or delete them.
- **Protected** documents belong to a local user. Names are visible profile labels;
  passwords unlock independent random user keys and per-document keys. Titles,
  text, ideas and graphs are encrypted in IndexedDB using AES-256-GCM. Password
  keys use Argon2id in a bundled worker (64 MiB, 3 passes, 1 lane), with fresh salts
  and fresh encryption IVs. No master password or administrator exists.
- Lock before handing over the computer. Locking, switching users, closing a page
  or 15 minutes without keyboard/pointer activity clears protected UI and keys.
  Other open Editor tabs cooperate; they do not receive usable keys. Open work
  remains available. Local users are equal peers, not separate OS accounts.

Use a strong, private passphrase of at least 12 characters. Someone who knows or
can guess it can unlock that user's work. Forgotten passwords have no remote
reset. Browser storage can be cleared or evicted; keep encrypted backups. Storage
is bounded to 5 MiB per document. Reaching a limit reports a failure instead of
saving a partial record.

Protecting an Open document removes its current app-managed plaintext record;
it cannot recall previously read text, clipboard contents, exports or backups.
Making a Protected document Open requires its owner's unlock and confirmation.
Protected clones remain protected for the same owner.

## Local AI

Use a compatible Oryn Browser for AI. Other browsers retain editing, password
unlock and export. A deliberate AI action asks the browser to choose/approve a
compatible local model and shows download/loading progress. There is no cloud
inference fallback or simulated response. Model discovery/downloads are managed
by the browser and can need the network. Responses stream when the selected model
supports it; otherwise the completed response appears at once. The runtime-wide
streaming flag does not block model requests. Connection checks retry briefly and
can be refreshed with Retry local AI connection or by returning to the tab.

Ask AI supports General (question only), Context (selection, or a labeled full
article fallback) and Article. Generate text uses a paragraph's idea; Generate
idea summarizes its text. Implement comments interprets bracketed instructions
and inserts a new paragraph after the original. Rewrite article preserves each
original and inserts rewritten paragraphs after them. Replacement asks first;
stale responses cannot overwrite later edits. Stop requests cancellation and
session release.

Thinking offers **Model default / On / Off**, initially **Off**. The choice is
pinned for each request; changing it affects the next request. It asks Oryn to use
the selected model's supported explicit thinking control, rather than hiding
reasoning that was already generated. Unsupported or fixed-mode requests continue
normally and show the reported requested/effective mode. Viewing model reasoning
is a separate, collapsed display; it never becomes document text, an idea or an
export automatically.

The SDK explicitly discovers `chat-v2` before using separate answer/reasoning
channels, thinking metadata and completion reasons. The editor chooses a permitted
total output budget up to 4096 tokens, narrowed by the model/profile and available
context. Thinking and answer generation share that budget; a complete answer is
not guaranteed. Older peers keep their legacy budget of at most 512 tokens and
report thinking unsupported and completion unknown. Legacy raw text deltas are
withheld until result normalization, and tagged reasoning is not treated as prose.

Writing actions apply only a nonempty final answer with an explicit clean stop.
Limits, reasoning-only responses, unknown completion, cancellation, errors or
conflicting stream/final output preserve the original. A final-answer preview may
be used deliberately with confirmation when incomplete; it is never presented as
a successful automatic replacement. Sequential rewrite applies its generated
paragraphs only after every original block finishes successfully. Editing,
document changes, Stop, lock and user switches clear reasoning/control metadata and
invalidate late callbacks. A website update cannot add these native controls to
an older installed Oryn; native model/route qualification is separate from the
synthetic integration checks below.

Related sources uses embeddings over your documents and explicit reference
imports. Enable automatic lookup to refresh after each ten-word bucket; it is
off initially. Filters and an optional local model filter narrow results. Open
AI/retrieval uses **Open sources only**, even when a user is unlocked. Protected
work may use Open sources and its own owner's protected sources. No shared
plaintext vector cache exists. Sources are local, not internet search.

With at least two ideas, Topic graph uses local embeddings and a local model to
classify connections. Invalid output is rejected. AI requests that exceed the
model's reported input allowance fail visibly without silently cutting text.

## Exports and backups

HTML downloads a sanitized, self-contained article. Substack and Medium copy
rich HTML and plain text for you to paste yourself. Plain PDF / Enhanced PDF use
the browser's print/save dialog. Ideas are excluded by default; enhanced exports
can include them. Exports and clipboard contents are plaintext outside document
protection.

Local users → Download my encrypted backup includes only that owner's protected
records, with an authenticated completeness manifest. Open documents and other
users are excluded. Restore requires the backup password; replacing matching
owner documents requires confirmation. Other users/Open records cannot be
replaced. On an existing identity whose password changed, unlock that identity
first, then use the old backup's password. Restore preserves the current password.
Backups are bounded to 50 MiB and 1000 protected documents.

Changing a password rewraps the existing user key. It does not revoke previously
stolen keys or old backup passwords. Compromise recovery needs fresh identities
and keys; no claim of revoking old copies is made.

## Hosting and offline use

All scripts, styles, crypto and workers are bundled; there are no analytics,
external fonts or document upload endpoints. A service worker caches application
assets after a production visit so editing and local unlock work offline. It
caches no document data. Updates wait until existing app tabs close.

Use a dedicated origin, not just a path on a shared origin. The included
`_headers` file supports hosts that honor that format. The included Nginx
configuration supplies the same headers when running `editor/Dockerfile` through
the root Compose service. Other hosts must configure equivalent CSP, framing,
no-sniff and referrer headers themselves. HTTPS is supplied by your hosting proxy;
localhost is sufficient for development. The server still sees normal asset
requests and connection metadata.

The host delivers trusted application code. A compromised host/update, extension,
OS administrator or injected script could capture a later unlock. Source
availability and offline caching do not remove that trust. People controlling the
same browser can copy, delete or roll back its database; complete storage rollback
is not independently detectable. Encryption does not hide record existence/sizes,
protect an unlocked session or replace device access controls.

## Verification

```sh
pnpm --filter @ocno/oryn-example-editor typecheck
pnpm --filter @ocno/oryn-example-editor test:unit
pnpm --filter @ocno/oryn-example-editor build
pnpm --filter @ocno/oryn-example-editor exec playwright install chromium
pnpm --filter @ocno/oryn-example-editor test:browser
```

Unit tests exercise real AES-GCM, key envelopes, revision races, Alice/Bob/Open
isolation, backup tampering, editor transactions and controlled SDK lifecycles.
Browser tests use disposable profiles and synthetic content. Actual local model
inference additionally requires an Oryn Browser/runtime and suitable local models.
