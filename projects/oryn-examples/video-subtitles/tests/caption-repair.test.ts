// @ts-nocheck -- Node type-stripper, browser workspace.
import assert from 'node:assert/strict';
import test from 'node:test';
import {withWordBounds} from '../src/caption-repair.ts';
const cue={text:'Hello world!',startMs:1000,endMs:2500};
test('repair offsets cropped alignment and preserves corrected text and cue bounds',()=>{
  const repaired=withWordBounds(cue,recoveredSegments(cue,[{text:'hello',start:.1,end:.5},{text:'world',start:.7,end:1.4}],1000));
  assert.deepEqual(repaired,{...cue,spans:[{text:'Hello',startMs:1100,endMs:1500},{text:'world!',startMs:1700,endMs:2400}]});
  assert.equal(cue.spans,undefined);
});
test('recovery retains valid partials and rejects overlapping measured spans',()=>{
  assert.deepEqual(recoveredSegments(cue,[],1000),[undefined,undefined]);
  assert.deepEqual(recoveredSegments(cue,[{text:'Hello',start:0,end:.5},{text:'world',start:.7,end:2}],1000),[{startMs:1000,endMs:1500},undefined]);
  assert.deepEqual(recoveredSegments(cue,[{text:'Hello',start:0,end:.8},{text:'world',start:.7,end:1}],1000),[undefined,undefined]);
});
test('manual timings reject missing, fractional, overlapping and out-of-cue bounds',()=>{
  for(const bounds of [[],[{startMs:NaN,endMs:1500},{startMs:1700,endMs:2400}],[{startMs:1000.5,endMs:1500},{startMs:1700,endMs:2400}],[{startMs:1000,endMs:1800},{startMs:1700,endMs:2400}],[{startMs:999,endMs:1500},{startMs:1700,endMs:2400}]]) assert.throws(()=>withWordBounds(cue,bounds));
  assert.equal(withWordBounds(cue,[{startMs:1000,endMs:1500},{startMs:1700,endMs:2500}]).text,cue.text);
});

import {recoverWordTimings,recoveredSegments,mergeWordTimings} from '../src/caption-repair.ts';
test('corrections and merged recognized words retain measured boundaries',()=>{
  const cue={text:'We used Claude today',startMs:1000,endMs:4000};
  const spans=[{text:'We',startMs:1000,endMs:1200},{text:'used',startMs:1200,endMs:1500},{text:'cloud',startMs:1500,endMs:1700},{text:'a',startMs:1700,endMs:1800},{text:'today',startMs:1800,endMs:2200}];
  const bounds=recoverWordTimings(cue,spans);
  assert.equal(bounds.length,4);assert.deepEqual(bounds[2],{startMs:1500,endMs:1800});
  assert.equal(withWordBounds(cue,bounds).text,cue.text);
  assert.deepEqual(recoverWordTimings({...cue,text:'We used clouds today'},spans),bounds);
});
test('insertions preserve useful partial timings; unrelated words and phrases get no invented slots',()=>{
  const cue={text:'Hello new world',startMs:0,endMs:2000};
  const spans=[{text:'Hello',startMs:0,endMs:500},{text:'world',startMs:800,endMs:1500}];
  assert.deepEqual(recoverWordTimings(cue,spans),[{startMs:0,endMs:500},undefined,{startMs:800,endMs:1500}]);
  assert.deepEqual(recoverWordTimings({...cue,text:'entirely unrelated sentence'},spans),[undefined,undefined,undefined]);
  assert.deepEqual(recoverWordTimings(cue,[{text:'Hello new world',startMs:0,endMs:1500}]),[undefined,undefined,undefined]);
});
test('context retry offsets and keeps only measured words inside the edited cue',()=>{
  const cue={text:'Hello new world',startMs:1000,endMs:2000};
  const bounds=recoveredSegments(cue,[{text:'before',start:0,end:.4},{text:'Hello',start:.5,end:.7},{text:'new',start:.7,end:.9},{text:'world',start:.9,end:1.4},{text:'after',start:1.5,end:1.8}],500);
  assert.deepEqual(bounds,[{startMs:1000,endMs:1200},{startMs:1200,endMs:1400},{startMs:1400,endMs:1900}]);
});
test('merging partial recovery preserves known slots and rejects overlap',()=>{
  const cue={text:'Hello new world',startMs:0,endMs:2000};
  const known=[{startMs:0,endMs:500},undefined,{startMs:1000,endMs:1500}];
  assert.deepEqual(mergeWordTimings(cue,known,[undefined,{startMs:400,endMs:900},undefined]),known);
  assert.deepEqual(mergeWordTimings(cue,known,[undefined,{startMs:600,endMs:900},undefined])[1],{startMs:600,endMs:900});
  assert.equal(known[1],undefined);
});

import {serializeCaptionTiming,parseCaptionTiming} from '../src/subtitles.ts';
test('partial measured hints survive JSON save/import without enabling whole-cue animation',()=>{
  const partial={text:'Hello new world',startMs:0,endMs:2000,wordTimingHints:[{text:'Hello',startMs:0,endMs:500},{text:'world',startMs:1000,endMs:1500}]};
  const restored=parseCaptionTiming(serializeCaptionTiming([partial]))[0];
  assert.deepEqual(restored,partial);assert.equal(restored.spans,undefined);
  for(const hints of [[{text:'not-in-caption',startMs:0,endMs:500}],[{text:'Hello',startMs:0,endMs:1200},{text:'world',startMs:1000,endMs:1500}],[{text:'world',startMs:1000,endMs:2100}]]) assert.throws(()=>serializeCaptionTiming([{...partial,wordTimingHints:hints}]));
});
