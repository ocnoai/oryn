// @ts-nocheck -- Node type-stripper, browser workspace.
import assert from 'node:assert/strict';
import test from 'node:test';
import {normalizeDesign, captionBox, DEFAULT_DESIGN, CAPTION_PRESETS} from '../src/caption-design.ts';
import {drawCaptions} from '../src/video-renderer.ts';

test('all six presets have bounded supported design values', () => {
  assert.equal(Object.keys(CAPTION_PRESETS).length,6);
  for(const preset of Object.values(CAPTION_PRESETS)) assert.deepEqual(normalizeDesign(preset.design),preset.design);
  const sanitized=normalizeDesign({size:NaN,x:-100,y:Infinity,width:1000,color:'url(https://invalid)',opacity:4});
  assert.equal(sanitized.size,DEFAULT_DESIGN.size);
  assert.equal(sanitized.x,0); assert.equal(sanitized.y,82); assert.equal(sanitized.width,96);
  assert.equal(sanitized.color,'#ffffff'); assert.equal(sanitized.opacity,1);
});

test('caption placement clamps complete boxes within landscape and portrait frames', () => {
  for(const [width,height] of [[1920,1080],[608,1080]]) {
    for(const x of [0,50,100]) for(const y of [0,50,100]) {
      const box=captionBox(width,height,width*.8,height*.3,{...DEFAULT_DESIGN,x,y});
      assert.ok(box.x>=0&&box.y>=0);
      assert.ok(box.x+box.width<=width&&box.y+box.height<=height);
    }
  }
});

function paint(design,effect='plain',time=500) {
  const calls=[];
  const ctx={globalAlpha:1,measureText:text=>({width:text.length*15}),save(){},restore(){},beginPath(){},roundRect(){},fill(){},translate(){},scale(x){calls.push({scale:x});},strokeText(){},fillText(text,x,y){calls.push({text,x,y,color:this.fillStyle,alpha:this.globalAlpha,font:this.font});}};
  drawCaptions(ctx,[{text:'Hello world',startMs:0,endMs:2000}],time,1280,720,effect,{...DEFAULT_DESIGN,...design});
  return calls;
}

test('position, typography and colors reach the shared preview/export painter', () => {
  const top=paint({x:50,y:15,color:'#ff0000',font:'serif',uppercase:true,size:6});
  const bottom=paint({x:50,y:85});
  assert.ok(top[0].y<bottom[0].y);
  assert.equal(top[0].text,'HELLO'); assert.equal(top[0].color,'#ff0000'); assert.match(top[0].font,/Georgia/);
  const left=paint({align:'left'}),right=paint({align:'right'});
  assert.ok(left[0].x<right[0].x);
});

test('fade and pop animate entire captions without requiring word timing', () => {
  assert.ok(paint({},'fade',40)[0].alpha<1);
  assert.equal(paint({},'fade',500)[0].alpha,1);
  assert.ok(paint({},'pop',40)[0].scale<1);
  assert.equal(paint({},'pop',500)[0].scale,1);
});
