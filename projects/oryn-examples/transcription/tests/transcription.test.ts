// @ts-nocheck -- Executed by Node's type-stripper; the browser workspace has no Node types.
import assert from 'node:assert/strict';
import test from 'node:test';
import { mergeTranscript, transcribeAudio } from '../src/transcription.ts';

const session = { sessionId: 'session', taskProfile: { limits: { audioDurationSeconds: 30 } } };

function source(durationSeconds = 60) {
  const windows = [];
  for (let startSeconds = 0; startSeconds < durationSeconds; startSeconds += 28) {
    const endSeconds = Math.min(durationSeconds, startSeconds + 30);
    windows.push({ index: windows.length, startSeconds, endSeconds });
    if (endSeconds === durationSeconds) break;
  }
  return { durationSeconds, windows, async readWindow() { return new Blob(['wave']); }, async dispose() {} };
}

function clientFixture(behavior = {}) {
  const listeners = new Set();
  const created = [], released = [], invoked = [], cancelled = [];
  let liveArtifacts = 0, peakArtifacts = 0;
  const emit = (index, type, data) => {
    const event = { event: type, sessionId: 'session', jobId: `job-${index}`, data, sequence: index, timestamp: 0 };
    for (const listener of listeners) listener(event);
  };
  const client = {
    artifacts: {
      async create(blob, options) {
        const index = created.length;
        created.push({ blob, options });
        peakArtifacts = Math.max(peakArtifacts, ++liveArtifacts);
        await behavior.create?.(index);
        return { id: `artifact-${index}` };
      },
      async release(id) { released.push(id); liveArtifacts--; },
    },
    subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    async invoke(request) {
      const index = invoked.length;
      invoked.push(request);
      if (behavior.invoke) return behavior.invoke(index, emit);
      // Deliberately complete before the invoke response, exercising terminal-event buffering.
      emit(index, 'job.completed', { result: { kind: 'structured', value: { text: `Section ${index}.`, language: 'en' } } });
      return { jobId: `job-${index}` };
    },
    async cancel(id) { cancelled.push(id); },
  };
  return { client, created, released, invoked, cancelled, listeners, get peakArtifacts() { return peakArtifacts; } };
}

test('an 87-minute recording runs every section with only one artifact live at a time', async () => {
  const audio = source(87 * 60);
  const fixture = clientFixture();
  const updates = [];
  const stages = [];
  const result = await transcribeAudio(fixture.client, session, audio, new AbortController().signal, {
    onTranscript: (text, seconds) => updates.push({ text, seconds }),
    onProgress: (window, index, total, stage) => stages.push({ window, index, total, stage }),
  });
  assert.equal(fixture.invoked.length, 187);
  assert.equal(fixture.created.length, fixture.released.length);
  assert.equal(fixture.peakArtifacts, 1);
  assert.equal(fixture.listeners.size, 0);
  assert.equal(result.completedSeconds, 5220);
  assert.equal(updates.at(-1).seconds, 5220);
  assert.match(result.text, /^Section 0\./);
  assert.match(result.text, /Section 186\.$/);
  assert.equal(result.language, 'en');
  assert.equal(stages.at(-1).total, 187);
  for (const request of fixture.invoked) {
    assert.equal(request.input.timestamps, false);
    assert.equal(request.input.language, 'auto');
  }
});

test('the selected language is passed unchanged through every section of an 87-minute recording', async () => {
  for (const language of ['sk', 'Slovak']) {
    const fixture = clientFixture();
    const result = await transcribeAudio(fixture.client, session, source(87 * 60), new AbortController().signal, {}, language);
    assert.equal(fixture.invoked.length, 187);
    assert.equal(result.completedSeconds, 5220);
    assert.equal(result.language, 'en'); // The result continues to report the runner's language.
    for (const request of fixture.invoked) assert.equal(request.input.language, language);
    assert.equal(fixture.released.length, 187);
    assert.equal(fixture.listeners.size, 0);
  }
});

test('an empty recording language fails before preparing or uploading audio', async () => {
  for (const language of ['', '   ']) {
    const fixture = clientFixture();
    await assert.rejects(transcribeAudio(fixture.client, session, source(), new AbortController().signal, {}, language), /Choose a recording language/);
    assert.equal(fixture.created.length, 0);
    assert.equal(fixture.invoked.length, 0);
  }
});

test('overlapping phrases merge across punctuation and case without dropping unmatched text', () => {
  assert.equal(mergeTranscript('We can begin. Welcome back!', 'welcome back to the recording.'), 'We can begin. Welcome back! to the recording.');
  assert.equal(mergeTranscript('First section ends.', 'Second section begins.'), 'First section ends. Second section begins.');
  assert.equal(mergeTranscript('Say yes', 'yes again'), 'Say yes yes again');
  assert.equal(mergeTranscript('Hello world', 'Hello world'), 'Hello world');
  assert.equal(mergeTranscript('  ', '  Voice  '), 'Voice');
  assert.equal(mergeTranscript('Voice', ' '), 'Voice');
  assert.equal(mergeTranscript('Prvý deň máme čas.', 'máme čas pokračovať.'), 'Prvý deň máme čas. pokračovať.');
});

test('a failed later job releases its audio and leaves the completed transcript visible', async () => {
  const fixture = clientFixture({
    async invoke(index, emit) {
      if (index === 1) emit(index, 'job.failed', { error: { message: 'Speech recognition failed.', code: 'ASR_INFERENCE_FAILED' } });
      else emit(index, 'job.completed', { result: { kind: 'structured', value: { text: 'Completed section.', language: null } } });
      return { jobId: `job-${index}` };
    },
  });
  const updates = [];
  await assert.rejects(transcribeAudio(fixture.client, session, source(), new AbortController().signal, {
    onTranscript: text => updates.push(text),
  }), /Speech recognition failed/);
  assert.deepEqual(updates, ['Completed section.']);
  assert.deepEqual(fixture.released, ['artifact-0', 'artifact-1']);
  assert.equal(fixture.invoked.length, 2);
  assert.equal(fixture.listeners.size, 0);
});

test('cancelling during artifact creation releases it and never invokes a job', async () => {
  const controller = new AbortController();
  const fixture = clientFixture({ create: () => controller.abort() });
  await assert.rejects(transcribeAudio(fixture.client, session, source(), controller.signal), { name: 'AbortError' });
  assert.deepEqual(fixture.released, ['artifact-0']);
  assert.equal(fixture.invoked.length, 0);
});

test('cancelling while invoke is pending cancels the returned job and removes its listener', async () => {
  const controller = new AbortController();
  let returnJob;
  let enteredInvoke;
  const entered = new Promise(resolve => { enteredInvoke = resolve; });
  const fixture = clientFixture({
    invoke() {
      enteredInvoke();
      return new Promise(resolve => { returnJob = resolve; });
    },
  });
  const operation = transcribeAudio(fixture.client, session, source(), controller.signal, {}, 'sk');
  await entered;
  controller.abort();
  returnJob({ jobId: 'job-0' });
  await assert.rejects(operation, { name: 'AbortError' });
  assert.deepEqual(fixture.cancelled, ['job-0']);
  assert.equal(fixture.invoked[0].input.language, 'sk');
  assert.deepEqual(fixture.released, ['artifact-0']);
  assert.equal(fixture.listeners.size, 0);
});

test('cancelling a running job stops later sections and retains earlier text', async () => {
  const controller = new AbortController();
  let enteredSecond;
  const entered = new Promise(resolve => { enteredSecond = resolve; });
  const fixture = clientFixture({
    async invoke(index, emit) {
      if (index === 0) emit(index, 'job.completed', { result: { kind: 'structured', value: { text: 'Saved first section.' } } });
      else enteredSecond();
      return { jobId: `job-${index}` };
    },
  });
  const updates = [];
  const operation = transcribeAudio(fixture.client, session, source(), controller.signal, { onTranscript: text => updates.push(text) });
  await entered;
  controller.abort();
  await assert.rejects(operation, { name: 'AbortError' });
  assert.deepEqual(updates, ['Saved first section.']);
  assert.deepEqual(fixture.cancelled, ['job-1']);
  assert.equal(fixture.released.length, 2);
});

test('malformed transcripts and invocation errors release their audio', async () => {
  for (const mode of ['bad-result', 'invoke-error']) {
    const fixture = clientFixture({
      async invoke(index, emit) {
        if (mode === 'invoke-error') throw new Error('Engine disconnected.');
        emit(index, 'job.completed', { result: { kind: 'structured', value: { text: 123 } } });
        return { jobId: `job-${index}` };
      },
    });
    await assert.rejects(transcribeAudio(fixture.client, session, source(), new AbortController().signal), /unexpected transcript|Engine disconnected/);
    assert.deepEqual(fixture.released, ['artifact-0']);
    assert.equal(fixture.listeners.size, 0);
  }
});

test('negotiated short model limits fail before audio preparation', async () => {
  const fixture = clientFixture();
  await assert.rejects(transcribeAudio(fixture.client, { ...session, taskProfile: { limits: { audioDurationSeconds: 15 } } }, source(), new AbortController().signal), /only 15 seconds/);
  assert.equal(fixture.created.length, 0);
});
