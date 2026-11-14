// @ts-nocheck -- Node type-stripper, browser workspace.
import assert from 'node:assert/strict';
import test from 'node:test';
import {drawCaptions, requiresWordTiming, renderCaptionedVideo} from '../src/video-renderer.ts';
import {DEFAULT_DESIGN} from '../src/caption-design.ts';
const cue={text:'Hello world',startMs:0,endMs:1500,spans:[{text:'Hello',startMs:100,endMs:600},{text:'world',startMs:800,endMs:1400}]};
function paint(style,time,cues=[cue]) {
  const calls=[],stack=[];
  const ctx={globalAlpha:1,fillStyle:'',measureText:text=>({width:text.length*15}),
    save(){stack.push({globalAlpha:this.globalAlpha,fillStyle:this.fillStyle});},restore(){Object.assign(this,stack.pop());},
    beginPath(){},roundRect(x,y,w,h){calls.push({box:[x,y,w,h]});},fill(){},translate(){},
    scale(x){calls.push({scale:x});},strokeText(){},fillText(text,x,y){calls.push({text,x,y,color:this.fillStyle,alpha:this.globalAlpha});}};
  drawCaptions(ctx,cues,time,1280,720,style,{...DEFAULT_DESIGN,opacity:0});
  assert.equal(stack.length,0);
  return calls;
}
test('every word effect rejects missing timing before accessing browser APIs',async()=>{
  for(const style of ['highlight','word-box','word-reveal','word-pop']) {
    assert.ok(requiresWordTiming(style));
    await assert.rejects(renderCaptionedVideo({style,cues:[{...cue,spans:undefined}]}),/individual word timings/);
  }
  for(const style of ['plain','fade','pop']) assert.equal(requiresWordTiming(style),false);
});
test('reveal hides future words and retains stable positions across gaps',()=>{
  assert.equal(paint('word-reveal',50).length,0);
  const first=paint('word-reveal',300),gap=paint('word-reveal',700),both=paint('word-reveal',1000);
  assert.deepEqual(first.map(v=>v.text),['Hello']);
  assert.deepEqual(gap,first);
  assert.deepEqual(both.map(v=>v.text),['Hello','world']);
  assert.equal(first[0].x,both[0].x);
  assert.ok(paint('word-reveal',120)[0].alpha<1);
});
test('word box advances only with spoken words, with contrasting text',()=>{
  const first=paint('word-box',300),second=paint('word-box',1000);
  assert.ok(second.find(v=>v.box).box[0]>first.find(v=>v.box).box[0]);
  assert.equal(first.find(v=>v.text==='Hello').color,'#000000');
  assert.equal(paint('word-box',700).filter(v=>v.box).length,0);
});
test('bounce is bounded and settles; untimed preview never invents word effects',()=>{
  const scale=paint('word-pop',210).find(v=>v.scale).scale;
  assert.ok(scale>1&&scale<=1.14+Number.EPSILON);
  assert.equal(paint('word-pop',500).find(v=>v.scale).scale,1);
  assert.equal(paint('word-pop',700).filter(v=>v.scale).length,0);
  for(const style of ['word-box','word-pop','word-reveal']) {
    const calls=paint(style,300,[{...cue,spans:undefined}]);
    assert.deepEqual(calls.map(v=>v.text),['Hello','world']);
    assert.ok(calls.every(v=>v.color===DEFAULT_DESIGN.color));
  }
});
