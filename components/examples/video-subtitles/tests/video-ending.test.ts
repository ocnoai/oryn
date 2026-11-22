// @ts-nocheck -- Node type-stripper, no Node types in this browser workspace.
import assert from 'node:assert/strict';
import test from 'node:test';
import {validateEnding, endingTransitionDuration, endingProgress, endingAudioLevel} from '../src/video-ending.ts';
import {renderCaptionedVideo} from '../src/video-renderer.ts';

const animation = {kind:'animation', title:'Thank you', subtitle:'', color:'#929bff', duration:3, transition:'crossfade', transitionSeconds:0.6};

test('ending validation rejects incomplete uploads and invalid animation inputs before allocating media', async () => {
  assert.doesNotThrow(() => validateEnding(animation));
  for (const patch of [{duration:NaN},{duration:0},{duration:16},{title:' '},{title:'x'.repeat(101)},{subtitle:'x'.repeat(161)},{color:'red'},{transition:'unknown'},{transitionSeconds:Infinity}]) {
    assert.throws(() => validateEnding({...animation,...patch}));
  }
  await assert.rejects(renderCaptionedVideo({ending:{kind:'video', transition:'crossfade', transitionSeconds:0.6}}), /Choose a playable ending video/);
});

test('short clips retain time after the transition, cuts start immediately, and progress is bounded', () => {
  assert.equal(endingTransitionDuration(animation, 0.4), 0.2);
  assert.equal(endingProgress(animation, -1, 3), 0);
  assert.equal(endingProgress(animation, 0.3, 3), 0.5);
  assert.equal(endingProgress(animation, 10, 3), 1);
  assert.equal(endingProgress({...animation,transition:'none'}, 0, 3), 1);
});

test('ending audio honors mute and fades in with the chosen transition', () => {
  const clip = {...animation,kind:'video',includeAudio:true};
  assert.equal(endingAudioLevel(animation, 2, 3), 0);
  assert.equal(endingAudioLevel({...clip,includeAudio:false}, 2, 3), 0);
  assert.equal(endingAudioLevel(clip, 0, 3), 0);
  assert.equal(endingAudioLevel(clip, 0.3, 3), 0.5);
  assert.equal(endingAudioLevel({...clip,transition:'fade-black'}, 0.3, 3), 0);
  assert.equal(endingAudioLevel(clip, 1, 3), 1);
});
