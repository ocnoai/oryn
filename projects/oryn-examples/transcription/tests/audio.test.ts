// @ts-nocheck -- Browser workspace tests run directly in Node's type-stripper.
import assert from 'node:assert/strict';
import test from 'node:test';
import { MAX_DURATION_SECONDS, openAudio, pcmWave, planAudioWindows, SAMPLE_RATE } from '../src/audio.ts';

test('an 87 minute recording is covered through its final sample by bounded overlapping sections', () => {
  const windows = planAudioWindows(87 * 60);
  assert.equal(windows.length, 187);
  assert.deepEqual(windows[0], { index: 0, startSeconds: 0, endSeconds: 30 });
  assert.deepEqual(windows.at(-1), { index: 186, startSeconds: 5208, endSeconds: 5220 });
  windows.forEach((window, index) => {
    assert.equal(window.index, index);
    assert.ok(window.endSeconds - window.startSeconds <= 30);
    assert.ok(window.endSeconds > window.startSeconds);
    if (index) assert.equal(windows[index - 1].endSeconds - window.startSeconds, 2);
  });
});

test('short and exact-boundary recordings do not gain an empty trailing section', () => {
  assert.deepEqual(planAudioWindows(0.125), [{ index: 0, startSeconds: 0, endSeconds: 0.125 }]);
  assert.equal(planAudioWindows(30).length, 1);
  assert.equal(planAudioWindows(58).length, 2);
  assert.equal(planAudioWindows(58.001).length, 3);
  assert.equal(planAudioWindows(MAX_DURATION_SECONDS).at(-1).endSeconds, MAX_DURATION_SECONDS);
  for (const duration of [0, -1, NaN, Infinity, MAX_DURATION_SECONDS + 1]) {
    assert.throws(() => planAudioWindows(duration));
  }
});

test('a model profile can lower the section limit without losing any of an 87 minute recording', () => {
  const windows = planAudioWindows(87 * 60, 12);
  assert.equal(windows[0].endSeconds, 12);
  assert.equal(windows[1].startSeconds, 10.8);
  assert.equal(windows.at(-1).endSeconds, 5220);
  windows.forEach(window => assert.ok(window.endSeconds - window.startSeconds <= 12.000000001));
  for (const limit of [0, -1, NaN, 31]) assert.throws(() => planAudioWindows(5220, limit));
});

test('PCM encoding gives the runner a mono 16 kHz 16-bit WAV and clamps invalid sample values', async () => {
  const wave = pcmWave(new Float32Array([-2, -1, -0.5, 0, 0.5, 1, 2, NaN]));
  const bytes = await wave.arrayBuffer();
  const view = new DataView(bytes);
  const ascii = (start, end) => new TextDecoder().decode(bytes.slice(start, end));
  assert.equal(wave.type, 'audio/wav');
  assert.equal(ascii(0, 4), 'RIFF');
  assert.equal(ascii(8, 12), 'WAVE');
  assert.equal(view.getUint32(4, true), bytes.byteLength - 8);
  assert.equal(view.getUint16(20, true), 1);
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), SAMPLE_RATE);
  assert.equal(view.getUint32(28, true), SAMPLE_RATE * 2);
  assert.equal(view.getUint16(32, true), 2);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), 16);
  assert.deepEqual(Array.from({ length: 8 }, (_, index) => view.getInt16(44 + index * 2, true)),
    [-32768, -32768, -16384, 0, 16384, 32767, 32767, 0]);
  assert.equal(pcmWave(new Float32Array(30 * SAMPLE_RATE)).size, 960_044);
  assert.throws(() => pcmWave(new Float32Array(30 * SAMPLE_RATE + 1)));
});

test('actual WAV headers are accepted regardless of filename and MIME type', async () => {
  const file = new File([pcmWave(new Float32Array(SAMPLE_RATE))], 'recording.mp3', { type: 'audio/mpeg' });
  const opened = await openAudio(file, new AbortController().signal);
  assert.equal(opened.durationSeconds, 1);
  assert.equal(opened.windows.length, 1);
  await opened.dispose();
  await opened.dispose();
});

test('empty, disguised non-audio, and pre-cancelled inputs fail before decoding', async () => {
  await assert.rejects(openAudio(new File([], 'empty.mp3'), new AbortController().signal), /empty/);
  await assert.rejects(openAudio(new File(['not audio'], 'fake.mp3', { type: 'audio/mpeg' }),
    new AbortController().signal), /valid MP3 or WAV/);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(openAudio(new File(['not audio'], 'fake.mp3'), controller.signal), { name: 'AbortError' });
});
