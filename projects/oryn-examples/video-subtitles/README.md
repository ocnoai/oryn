# Local video captions

Choose a video and generate subtitles, or import an existing UTF-8 SRT / Oryn timing JSON after choosing the video.
Review the cue text/times. Select a caption style, press **Render captioned video**, preview the
result, and press **Save rendered video**. Output contains the entire source video (including
sections without captions), its default audio track, and burned-in captions. SRT export remains
available separately. CapCut can import the video; its captions are pixels, not editable native
CapCut caption objects.

## Multiple videos

Choose several videos in the source picker, or use **Add more videos to this project** after
opening one video. The queue only appears when there is more than one video; the existing
single-video editor, ending preview, render and save actions remain available.

1. **Create subtitles** processes videos sequentially, skipping videos that already have captions.
   A failure is shown on that video and the remaining queue continues. Stop the queue to cancel
   the current operation; completed transcripts are kept. Retry to process only missing transcripts.
2. Open each video to review and edit its captions in the existing editor, or import its own SRT /
   timing JSON. **Mark reviewed & next** advances to the next transcript needing review. Edits,
   timing repairs and resets revoke that video's approval and previous render. Switching videos
   preserves edits, timing hints and rendered results. Removing a video removes it from this queue.
3. Choose a shared caption design, animation and optional ending. These settings apply to every
   video; changing them clears all previous renders while keeping caption approvals.
4. **Render all videos** becomes available after every transcript is reviewed. Rendering checks
   caption validity and required word timings before starting, then runs sequentially. Cancellation
   or a failure keeps finished videos; **Render remaining videos** retries unfinished outputs.
5. **Download all (.zip)** saves one ZIP through Oryn's existing artifact/save dialog flow once all
   videos are rendered. Each video includes the shared ending, if selected. Duplicate filenames get
   numbered suffixes. Failed or cancelled downloads can be retried; individual saves remain available.

Keep the page open: source files, captions, reviews and rendered outputs live in page memory.
Existing per-video render limits still apply. The ZIP stores already-compressed videos without
recompressing them, computes checksums in bounded chunks and supports archives below 4 GiB.
Large batches consume more memory; use smaller batches or individual saves if needed.

## Optional ending

Before **Render captioned video**, use **Add an ending** to choose **Upload ending video** or
**Animated end card**. The end card has editable title, subtitle, accent color, and a 1–15 second
duration, with animated text and a moving glow. Uploaded clips play in full, fit inside the source
frame without stretching, and have an **Include ending video's audio** toggle.

Choose **Crossfade**, **Fade through black**, **Slide left**, or **Cut**. Transitions blend the held
final source frame into the start of the ending, so all source footage and speech play in full.
The transition lasts 0.1–2 seconds, capped at half the ending's duration for short clips. Ending
audio fades in with the transition. Animated cards are silent. Captions stay on the source segment.

**Preview ending & transition** uses the same painter as export and previews the ending's audio.
Changes invalidate previously rendered output. Select **No ending** to remove it. The combined
source and ending must fit the existing 30-minute/256 MiB export limits. Ending choices and files
stay in page memory and are not included in SRT or caption timing JSON.

## Caption studio

After generating or importing captions, use the live studio preview before rendering. Scrub the
video, press First caption, and drag anywhere on the preview to move the caption block. The arrow
keys move it by 1% (Shift: 5%); sliders and Top/Center/Bottom buttons provide another way to position
it. Caption boxes are clamped inside the picture, including portrait video.

Six editable presets: Classic, Clean outline, Bold social, Light card, Neon and Editorial. Adjust
sans/serif/monospace typeface, size, bold, uppercase, text/active-word/background colors, background
opacity, black outline, shadow, alignment and wrapping width. Size is a percentage of the video's
shorter dimension. The design applies globally to all cues. Local system fonts keep the editor offline.

Animation is separate from appearance: no animation, fade in/out, pop in, or spoken-word highlighting.
Only spoken-word highlighting requires word-level timing. Fades/pops operate at caption boundaries.
The preview and export use the same normalized design and canvas painter; export snapshots the
settings and pauses preview playback. Style changes invalidate stale rendered output. Styles are
remembered in browser local storage for this origin; Reset design restores defaults. No video or
transcript is saved to local storage. Timing JSON retains timing data, not design settings.

The transcript editor is expandable so long transcripts do not bury the design workspace. The
responsive layout stacks controls on narrow windows and provides keyboard alternatives to dragging.

## Timing and animation

The demo retains original timestamp spans when grouping model output into readable captions.
The existing Qwen forced-alignment request asks for `timestamps: "word"`. Whisper's existing
word timestamps are retained when returned. Word highlighting now requires usable single-word
spans for every cue: phrase-only output and standalone SRT cannot enable it. Plain captions remain
available. Each cue and the render panel display timing availability; missing timing blocks word
rendering rather than producing an all-yellow sentence.

Text corrections no longer unconditionally discard timing. A bounded sequence comparison keeps
unchanged words on their original spans, substitutes corrected spelling on existing word slots,
removes deleted words, and merges multiple recognized words into one corrected word using their
combined bounds. No proportional word times are invented. Inserted/split words or ambiguous
many-to-many replacements, as well as changed cue boundaries, require fresh alignment. Corrected
text still needs synchronization review: the retained bounds come from the original recognition.
Resetting restores the original generated captions/timings.

Use **Save captions with word timings (.json)** to retain alignment across sessions. The data-only
`oryn-caption-timing/1` format stores caption text, milliseconds and original/corrected spans;
imports validate finite ordered bounds, text consistency, size and entry counts. SRT remains the
portable text-only export. Importing a corrected SRT while generated captions for the same selected
video are still loaded reuses their timing when cue boundaries match. Standalone SRT import does
not pretend to recover missing word timings.

## Browser implementation and limits

`video-renderer.ts` validates the request and loads `video-export.ts`, which uses Mediabunny and
WebCodecs to decode and encode explicit media timestamps. The canvas painter is shared with preview.
Audio is decoded in bounded windows and normalized to 48 kHz stereo with OfflineAudioContext,
preserving source timing, silence, ending offsets and transition gain. No microphone, screen capture,
network upload, filesystem access, private Browser API, new runtime pack, or Engine change is used.
Output is saved through the existing SDK artifact creation/save-as/release flow and its normal
Browser permissions and quotas. Captions are drawn as text, not interpreted as markup.

- MP4/H.264/AAC is preferred when the browser advertises support; otherwise WebM/VP8/Opus.
  Recording can still fail if the device cannot allocate an encoder.
- Rendering speed depends on the device, maximum 30 minutes, a 1920×1080 bounding box with aspect ratio
  preserved and no upscaling, 30 fps, default 4 Mbps video and 128 kbps audio.
- Encoded output is buffered in memory and capped at 256 MiB. The size cap may be reached before
  the duration cap. Large/long sources require a future streaming exporter; four-hour transcription
  support does not imply four-hour video rendering.
- Switching tabs or hiding the window does not cancel export. Frames and audio use media timestamps,
  independent of `requestAnimationFrame`, playback clocks, and background timer frequency. Keep the
  page open; closing or reloading it ends the export. Cancellation discards partial output and releases
  decoders, encoders, audio samples, message ports and listeners.
- This is a re-encode of the default video/audio playback, not a lossless remux. HDR, additional
  audio tracks, original frame rate, metadata and native CapCut templates are not preserved.
- Source/edited-cue/style changes discard stale rendered output. Saving can be retried.
- Imported subtitle data is limited to 4 MiB / 10,000 non-overlapping cues (SRT numbering must be sequential). Timing JSON also has a 100,000-span cap. Invalid timings are
  rejected. Very long captions must be split to fit the image.

Implementation references: [Mediabunny media sinks](https://mediabunny.dev/guide/media-sinks),
[media sources](https://mediabunny.dev/guide/media-sources), and
[writing media files](https://mediabunny.dev/guide/writing-media-files).

## Validation

From the examples workspace, run `pnpm --filter @ocno/oryn-example-video-subtitles test`.
For real encoder integration checks, start this demo's Vite server on `127.0.0.1:5188`, then run
`scripts/render-smoke.cjs` with the workspace's existing Electron executable, with
`ELECTRON_RUN_AS_NODE` unset. Set `ORYN_RENDER_SMOKE_URL` to
`http://localhost:5188/tests/render-ui-smoke.html?built=1` for the UI/SDK-stub variant.
`ORYN_RENDER_SMOKE_REPORT` optionally selects an evidence output directory. Test-only Electron
imports belong to the harness, not the website bundle. `tests/fixture.mp4` is a synthetic two-second
blue frame with a 440 Hz tone, generated using FFmpeg; it contains no user media.

The encoder test checks playable MP4, decoded non-silent audio/duration, yellow caption pixels advancing between words,
cancellation and retry. The UI test imports SRT, renders, verifies chunked artifact transfer and
release via a stub bridge, and verifies edits invalidate saved output, punctuation corrections retain alignment, SRT without timing blocks word rendering, and timing JSON export/import preserves spans. It does not exercise the
native save dialog or assert new model/language conformance.

`tests/batch-ui-smoke.html` exercises multi-file selection, sequential transcription with a stub
model, failure/cancellation and resumption, isolated edits/imports, review gating, real batch
encoding with an uploaded ending, cancelled render resumption, ZIP save retry, duplicate names,
playable videos extracted from the ZIP, stale-output invalidation and returning to single-video
mode. Run the same Electron harness with `ORYN_RENDER_SMOKE_URL` set to
`http://127.0.0.1:5188/tests/batch-ui-smoke.html` after building. ZIP unit tests independently check
checksums, content, UTF-8 names, directory offsets and cancellation. No native save dialog or live
ASR model is exercised by these smoke tests.

`tests/ending-render-smoke.html` checks encoded animation and uploaded endings, total duration,
source audio, included/muted ending audio, portrait fitting, all transition painters, cancellation
during the ending, retry, and invalid-clip errors. `tests/ending-fixture.mp4` is a synthetic portrait
green clip with an 880 Hz tone. The UI smoke also exercises ending preview, both ending modes,
missing-file validation, stale-export invalidation, and removal.

Run `scripts/background-render-smoke.cjs` with the same Electron executable to exercise real tab
switching with normal background throttling. It detaches the rendering WebContentsView, matching
Oryn's tab switching, and checks exports finish while hidden with every expected video frame,
moving word highlights, correct duration, and decoded audio. Cases cover switching during the source,
an animated ending and an uploaded ending; starting hidden; cancelling and retrying; and WebM fallback.
Set `ORYN_BACKGROUND_SMOKE_REPORT` to write the JSON results to a file.

### Dynamic captions

Choose **Caption studio → Animation**:

- **Moving word box:** the accent-colored box follows the currently spoken word. Text contrast adjusts automatically.
- **Word-by-word reveal:** each word fades and rises into position as speech reaches it; earlier words remain visible until the caption ends.
- **Spoken-word bounce:** the active word briefly grows and returns to normal, using the accent color.

All three use the same styling and positioning controls as other effects and are burned into the exported video. Preview with Play or the scrubber. They require individual word timings from transcription or saved Oryn timing JSON. An SRT file alone only provides caption intervals; choose No animation, Fade or Pop in if word timings are unavailable. Ambiguous text edits or changed cue boundaries still require alignment. No new runtime or cloud service is involved.

### Repair an edited caption

Open **Edit caption text and timing**. Only captions missing complete word spans show **Repair word timings**. The panel has one repair action; model choice belongs to Oryn, using the existing speech-recognition/timing role. Choose a model that returns individual word timestamps (for example a compatible Whisper model). Model selection is handled by the browser.

1. Open the affected caption's repair panel. Existing measured word timings are prefilled, including partial timings kept from earlier attempts.
2. Click **Repair this caption locally**. The demo regenerates timestamps for the caption audio, then matches recognized words against your corrected text. It preserves measured slots for short corrections and merges boundaries when several recognized words become one corrected word. Text is never overwritten.
3. If some words remain unresolved, it automatically retries with up to 750 ms of audio context on each side. Only the selected caption changes. Each audio request is capped at 30 seconds.
4. Complete recovery immediately enables word animation for this cue. Partial recovery keeps every usable timing and opens the manual fields with only unresolved words blank. You may adjust existing values if a new word needs space. Listen with **Play this caption**, enter milliseconds from the start of the video and click **Apply word timings**.
5. Review synchronization and save Oryn timing JSON. Complete spans and optional partial `wordTimingHints` survive save/import; SRT contains only caption boundaries.

Recovery tolerates recognition differences rather than requiring exact text equality. It uses bounded sequence matching, anchored replacements and merges of existing measured spans. It does not fabricate boundaries by splitting a phrase evenly. Large rewrites, words never recognized, invalid timestamps and unsupported timestamp models may remain partially unresolved; known results are retained. Short inferred slot transfers should be reviewed in the preview.

Automatic repair requires Oryn's runtime, public SDK permissions and a compatible user-selected ASR model. Manual repair needs no inference. Maximum cue length is 30 seconds/2,000 characters/256 words; browser decoding still reads the source audio before cropping, retaining the existing memory cost. Public error codes remain visible and temporary artifacts/sessions are released on success, failure or cancellation.

Save your timing JSON before reloading the application, then reopen it with the same source video.
