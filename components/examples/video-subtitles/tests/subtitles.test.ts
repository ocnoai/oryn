// @ts-nocheck -- Executed by Node's type-stripper; this browser-only workspace does not ship Node types.
import assert from "node:assert/strict";
import test from "node:test";
import {
  cuesForWindow,
  planAudioWindows,
  serializeSubRip,
  type Cue,
} from "../src/subtitles.ts";
import { transcriptionProfile } from "../src/profiles.ts";

test("Qwen remains developer-selected and declares its exact timestamp aligner", () => {
  const quality = transcriptionProfile("quality");
  const faster = transcriptionProfile("faster");
  assert.equal(quality.preferredModel.repository, "Qwen/Qwen3-ASR-1.7B-hf");
  assert.equal(
    quality.preferredModel.revision,
    "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
  );
  assert.equal(
    quality.alignmentModel.repository,
    "Qwen/Qwen3-ForcedAligner-0.6B-hf",
  );
  assert.equal(
    quality.alignmentModel.revision,
    "c07281df297b9905d24a508279258cccf987a064",
  );
  assert.equal(
    quality.nativeTimingFallbackModel?.repository,
    "openai/whisper-large-v3",
  );
  assert.equal(quality.windowSeconds, 12);
  assert.equal(faster.preferredModel.repository, "Qwen/Qwen3-ASR-1.7B-hf");
  assert.equal(
    faster.preferredModel.revision,
    "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
  );
  assert.equal(
    faster.certifiedModels[0]?.repository,
    "openai/whisper-large-v3",
  );
  assert.equal(faster.nativeTimingFallbackModel, undefined);
  assert.equal(faster.windowSeconds, 15);
});

test("a 3:38 recording is covered by contiguous bounded ASR windows", () => {
  const windows = planAudioWindows(218 * 16_000, 16_000, 15);
  assert.equal(windows.length, 15);
  assert.deepEqual(windows[0], {
    index: 0,
    total: 15,
    startSample: 0,
    endSample: 240_000,
    startMs: 0,
    endMs: 15_000,
  });
  assert.equal(windows.at(-1)?.startMs, 210_000);
  assert.equal(windows.at(-1)?.endMs, 218_000);
  windows.slice(1).forEach((window, index) => {
    assert.equal(window.startSample, windows[index]?.endSample);
    assert.equal(window.startMs, windows[index]?.endMs);
  });
});

test("native timestamp segments are offset onto the original video timeline", () => {
  const window = planAudioWindows(30 * 16_000, 16_000, 15)[1]!;
  const result = cuesForWindow(
    {
      text: "Second window",
      language: "en",
      segments: [{ text: "Second window", start: 1.25, end: 3.5 }],
    },
    window,
  );
  assert.equal(result.timing, "native");
  assert.deepEqual(result.cues, [
    { text: "Second window", startMs: 16_250, endMs: 18_500, spans: [{ text: "Second window", startMs: 16_250, endMs: 18_500 }] },
  ]);
});

test("word timestamps become readable sentence cues without losing their real bounds", () => {
  const window = planAudioWindows(12 * 16_000, 16_000, 12)[0]!;
  const result = cuesForWindow(
    {
      text: "Prvá veta. Druhá veta.",
      language: "sk",
      segments: [
        { text: "Prvá", start: 0.2, end: 0.7 },
        { text: "veta.", start: 0.72, end: 1.2 },
        { text: "Druhá", start: 2, end: 2.5 },
        { text: "veta.", start: 2.52, end: 3 },
      ],
    },
    window,
  );
  assert.deepEqual(result.cues, [
    { text: "Prvá veta.", startMs: 200, endMs: 1_200, spans: [
      { text: "Prvá", startMs: 200, endMs: 700 },
      { text: "veta.", startMs: 720, endMs: 1200 },
    ] },
    { text: "Druhá veta.", startMs: 2_000, endMs: 3_000, spans: [
      { text: "Druhá", startMs: 2000, endMs: 2500 },
      { text: "veta.", startMs: 2520, endMs: 3000 },
    ] },
  ]);
});

test("text-only ASR results become disclosed window-timed cues rather than one video-long cue", () => {
  const windows = planAudioWindows(218 * 16_000, 16_000, 15);
  const cues: Cue[] = windows.flatMap(
    (window) =>
      cuesForWindow(
        {
          text: `Transcript from window ${window.index + 1}`,
          language: "en",
          segments: [],
        },
        window,
      ).cues,
  );
  assert.equal(cues.length, 15);
  assert.equal(cues[0]?.startMs, 0);
  assert.equal(cues[0]?.endMs, 15_000);
  assert.equal(cues.at(-1)?.startMs, 210_000);
  assert.equal(cues.at(-1)?.endMs, 218_000);
  const srt = serializeSubRip(cues);
  assert.equal((srt.match(/-->/gu) ?? []).length, 15);
  assert.match(srt, /00:03:30,000 --> 00:03:38,000/u);
});

test("long text within one window is split into independently timed readable cues", () => {
  const window = planAudioWindows(15 * 16_000, 16_000, 15)[0]!;
  const result = cuesForWindow(
    {
      text: Array.from({ length: 60 }, (_, index) => `word${index}`).join(" "),
      language: null,
    },
    window,
  );
  assert.equal(result.timing, "window-estimated");
  assert.ok(result.cues.length > 1);
  assert.ok(result.cues.every((cue) => cue.text.length <= 84));
  assert.equal(result.cues[0]?.startMs, 0);
  assert.equal(result.cues.at(-1)?.endMs, 15_000);
});

test('SRT import preserves Unicode and cue timing without invented word spans', async () => {
  const { parseSubRip } = await import('../src/subtitles.ts');
  const original = [{ startMs: 200, endMs: 1250, text: 'Prvá veta.' }];
  assert.deepEqual(parseSubRip('\uFEFF' + serializeSubRip(original)), original);
  assert.throws(() => parseSubRip('1\n00:00:01,000 --> 00:00:00,000\nBackwards'));
  assert.throws(() => parseSubRip('1\n00:99:00,000 --> 00:99:01,000\nInvalid'));
  assert.throws(() => parseSubRip(''));
});


test('timing companion round-trips word bounds and rejects malformed alignment', async () => {
  const {serializeCaptionTiming,parseCaptionTiming} = await import('../src/subtitles.ts');
  const cues = [{text:'Hello world',startMs:0,endMs:1000,spans:[
    {text:'Hello',startMs:0,endMs:400},{text:'world',startMs:600,endMs:1000},
  ]}];
  assert.deepEqual(parseCaptionTiming(serializeCaptionTiming(cues)),cues);
  assert.throws(()=>parseCaptionTiming(JSON.stringify({schema:'wrong',cues})));
  assert.throws(()=>serializeCaptionTiming([{...cues[0],text:'Unrelated text'}]));
  assert.throws(()=>serializeCaptionTiming([{...cues[0],spans:[{text:'Hello world',startMs:0,endMs:2000}]}]));
});
