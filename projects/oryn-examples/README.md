# Oryn Examples

Ordinary websites that consume the documented `@ocno/oryn-sdk` package through the same constrained boundary available to third parties.

```bash
pnpm install
pnpm test
```

The examples include transcription, chat and song generation with task-specific controls. The video-subtitles workload keeps exact Qwen ASR as its declared first choice, uses Qwen's separate forced aligner only for languages in that aligner's declared set, and uses an exact native-timestamp Whisper fallback for the quality profile when alignment is unsupported. The lower-overhead profile instead exposes bounded estimated timing for review. Each of these apps supports `pnpm dev` and builds as an ordinary static website without Electron, Node, Python, or browser-internal imports.

The [transcription demo](transcription/README.md) starts with a language picker, shows matching model choices, and requests the selected model when transcription starts. It accepts MP3 and WAV recordings up to 4 hours and 1 GiB, decodes and transcribes bounded sections locally, shows progress and partial text, and supports cancellation and text downloads.

Run `./scripts/bootstrap` from the monorepo root first. It produces
`vendor/ocno-oryn-sdk-0.1.0.tgz` from the included SDK source. The examples consume
that packaged SDK through the same package boundary available to third-party sites.

The video-subtitles demo also supports [local captioned video rendering](video-subtitles/README.md), original timed-span highlighting, and importing existing SRT files. It uses browser media APIs and the existing artifact save flow; no extra engine is installed.

Each example builds into an ordinary static website. Use its `pnpm dev` command
for a local development server, or serve its build output with your own static
host. Each included example is independent of Registry hosting. AI runs in the
visitor's Oryn Browser.
