// @ts-nocheck -- Node type-stripper, no Node types in this browser workspace.
import assert from 'node:assert/strict';
import test from 'node:test';
import { drawCaptions, retainTiming, hasWordTiming, requireWordTiming } from '../src/video-renderer.ts';

const cue = { text: 'Hello world', startMs: 100, endMs: 1200, spans: [
  { text: 'Hello', startMs: 100, endMs: 400 },
  { text: 'world', startMs: 700, endMs: 1200 },
] };

test('corrections preserve known word slots but boundary changes require alignment', () => {
  assert.deepEqual(retainTiming({ text: cue.text, startMs: 100, endMs: 1200 }, cue).spans, cue.spans);
  assert.deepEqual(retainTiming({ text: 'Hello world!', startMs: 100, endMs: 1200 }, cue).spans, [cue.spans[0], {...cue.spans[1],text:'world!'}]);
  assert.equal(retainTiming({ text: 'Hello new world', startMs: 100, endMs: 1200 }, cue).spans, undefined);
  assert.equal(retainTiming({ text: cue.text, startMs: 200, endMs: 1200 }, cue).spans, undefined);
});

function draw(time, cues = [cue], style = 'highlight') {
  const calls = [];
  const context = {
    measureText: text => ({ width: text.length * 10 }),
    fillRect() {}, save() {}, restore() {}, beginPath() {}, roundRect() {}, fill() {}, translate() {}, scale() {}, strokeText() {},
    fillText(text) { calls.push([text, this.fillStyle]); },
  };
  drawCaptions(context, cues, time, 1280, 720, style);
  return calls;
}

test('highlight respects actual word bounds and pauses between words', () => {
  assert.deepEqual(draw(200), [['Hello', '#ffe34d'], ['world', '#ffffff']]);
  assert.deepEqual(draw(500), [['Hello', '#ffffff'], ['world', '#ffffff']]);
  assert.deepEqual(draw(800), [['Hello', '#ffffff'], ['world', '#ffe34d']]);
  assert.deepEqual(draw(1200), []);
  assert.deepEqual(draw(0), []);
});

test('phrase-only captions never turn the entire sentence yellow', () => {
  const phrase = [{ text: 'Hello world', startMs: 100, endMs: 1200 }];
  assert.deepEqual(draw(200, phrase), [['Hello', '#ffffff'], ['world', '#ffffff']]);
  assert.throws(()=>requireWordTiming(phrase), /no individual word timings/);
  assert.throws(()=>requireWordTiming([{...phrase[0],spans:phrase}]), /no individual word timings/);
  assert.deepEqual(draw(200, [cue], 'plain'), [['Hello', '#ffffff'], ['world', '#ffffff']]);
});


test('corrected names merge recognized word bounds without redistributing time', () => {
  const original = {text:'z kvót a dostať',startMs:100,endMs:1200,spans:[
    {text:'z',startMs:100,endMs:200},
    {text:'kvót',startMs:200,endMs:500},
    {text:'a',startMs:500,endMs:600},
    {text:'dostať',startMs:800,endMs:1200},
  ]};
  const corrected = retainTiming({...original,text:'z Clauda dostať',spans:undefined},original);
  assert.deepEqual(corrected.spans,[original.spans[0],{text:'Clauda',startMs:200,endMs:600},original.spans[3]]);
  assert.equal(hasWordTiming(corrected),true);
  assert.equal(retainTiming({...cue,text:'Hello many new words world',spans:undefined},cue).spans,undefined);
  assert.deepEqual(retainTiming({...cue,text:'world',spans:undefined},cue).spans,[cue.spans[1]]);
});
