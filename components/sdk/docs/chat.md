# Thinking and versioned chat

`chat-v2` revision 1 adds best-effort `thinking?: boolean`, a total generation budget
up to 8192 tokens (default 4096), separate final-answer and reasoning channels, and
truthful completion/usage metadata. Effective model profiles may impose lower limits.
`chat-v1` revision 1 remains unchanged, including its 512-token ceiling.

Use `selectChatContract(await client.getCapabilities(), { maxTokens })` before
requesting a model. Only an explicit `taskContracts` entry establishes v2 support;
the SDK version and `features.taskProfiles` alone do not. Request the returned
contract through `client.models.request`. A session's optional `taskProfile.thinking`
describes `support: "switchable" | "fixed" | "unsupported" | "unknown"` and its
known default (`boolean | null`). This is capability information, not a strict
boolean constraint. Do not put thinking in `requirements.parameterValues`.

The Browser performs native contract negotiation before exposing this discovery.
It opts into `chat-v2` using the existing private task-negotiation flow; an older
Engine can return only its supported subset. Before opt-in, a newer Engine keeps
the old capability/profile shape for older Browsers. Websites do not invoke this
private negotiation method or acquire another privilege through discovery.

After a deliberate user action, call
`invokeChat(client, session, { messages, thinking: false, maxTokens }, { signal, onEvent })`.
Both boolean values are accepted on v2 even when a model cannot honor them.
Omission keeps its default. The Engine applies a supported reviewed control before
generation; otherwise it keeps the default and acknowledges the request as unapplied.
This describes the configured generation mode, not hidden internal computation.

Text deltas use channel `text`; model-generated reasoning uses `reasoning`.
Only final answer text belongs in an article. The final result has `value.text`,
optional `value.reasoning`, `finishReason: "stop" | "length" | "unknown"`,
`usage`, and `metadata.thinking`. Usage contains nullable `inputTokens`,
`outputTokens`, `reasoningTokens`, and `answerTokens`; null means unavailable.
`maxTokens` counts the whole generated sequence, including reasoning. A length
termination or unknown finish is not proof of a complete answer.

`metadata.thinking` contains `requested` (boolean or null), `applied`, `effective`
(boolean or null), `reason`, and `source`. Engine reasons are `applied`,
`already_fixed`, `model_default`, `unsupported_model`, `fixed_mode`, or
`unknown_support`, with source `engine`. Applied requires a non-null request and
the same configured effective value. Output tags do not establish mode support.

`invokeChat` returns a `ChatResponse` wrapper with the original wire `result`,
its `contract`, normalized text/completion/usage, and thinking metadata.
`normalizeChatResult(result, contract, requestedThinking)` provides the same
validation/normalization for applications using their own guarded event loop.
Keep final streaming text consistent with the validated final result before editing
a document; cancellation and failure remain terminal events, not successful answers.

On an older peer, a request with a legacy-compatible budget continues without the
unsupported thinking field. The wrapper preserves the unchanged v1 result and adds
SDK-authored `requested`, `applied: false`, `effective: null`,
`reason: "unsupported_peer"`, `source: "sdk_compatibility"`. Completion and usage
remain unknown. Raw v1 results/events are never modified or relabeled v2. Recognized
or unfinished leading reasoning tags, or an orphaned reasoning closing tag, withhold
normalized answer text; raw content remains in the original result for deliberate
inspection. Ordinary literal tag examples are not deleted. Legacy output is not
certified final-answer-only, so do not automatically replace writing from it.

`selectChatContract(..., { requireV2: true })` can require the new result semantics.
Unsupported larger budgets fail without silent reduction. Fallback is decided
before invocation and never retries a potentially started job. Permission, model,
resource and malformed-input failures retain their normal error paths.

These additions require a compatible Browser/Engine and independently versioned
SDK release; installing JavaScript cannot upgrade an older native runtime. Development
artifacts are not a claim of published release qualification, and changed package
bytes must not replace an existing published npm version.
