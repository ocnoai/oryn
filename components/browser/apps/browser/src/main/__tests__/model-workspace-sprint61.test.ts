import { describe, expect, it } from 'vitest';
import {
  forecastInput,
  numericControl,
  poseBoxes,
  tabularInput,
  numericMatrix,
  choiceValues,
} from '../../renderer/src/model-workspace-sprint61';
import { validateModelWorkspaceInput } from '../model-workspace/task-contract-registry';

describe('Sprint 61 workspace data and controls', () => {
  it('keeps tabular targets separate and preserves feature types and missing cells', () => {
    const value = tabularInput(
      'x\tnumeric\ncolor\tcategorical',
      '1\tred\n2\t',
      'cold\nwarm',
      '1.5\tblue',
      true,
    );
    expect(validateModelWorkspaceInput('tabular-classification-v1', value)).toEqual({
      columns: [
        { name: 'x', type: 'numeric' },
        { name: 'color', type: 'categorical' },
      ],
      trainingRows: [
        [1, 'red'],
        [2, null],
      ],
      targets: ['cold', 'warm'],
      queryRows: [[1.5, 'blue']],
    });
    for (const bad of ['1 2\tred', 'NaN\tred', '1\tred\textra'])
      expect(() =>
        tabularInput('x\tnumeric\ncolor\tcategorical', bad, 'cold\nwarm', '1\tred', true),
      ).toThrow();
  });
  it('bounds data grids and intersects declared operation choices', () => {
    expect(numericMatrix('1, 2\n3 4', true)).toEqual([
      [1, 2],
      [3, 4],
    ]);
    for (const bad of ['', '1, 2\n3', '1,,2', '.5 1'])
      expect(() => numericMatrix(bad, true)).toThrow();
    expect(
      choiceValues({ allOf: [{ enum: ['text', 'audio'] }, { const: 'text' }] }, [
        'text',
        'audio',
        'text-and-audio',
      ]),
    ).toEqual(['text']);
  });
  it('preserves ordered person boxes and rejects empty or inverted geometry', () => {
    expect(poseBoxes('0, 10, 50, 100\n50 0 100 90')).toEqual([
      [0, 0.1, 0.5, 1],
      [0.5, 0, 1, 0.9],
    ]);
    for (const bad of ['', '0 0 0 100', '10 0 5 100', '0 0 100', '0 0 101 100', '0,,100,100'])
      expect(() => poseBoxes(bad)).toThrow();
  });
  it('accepts only numerical history and preserves input order and native quantiles', () => {
    expect(forecastInput('1, 2\n3e1 -4', 2, '.1, .9')).toEqual({
      context: [1, 2, 30, -4],
      horizon: 2,
      quantiles: [0.1, 0.9],
    });
    for (const bad of ['', '1,,2', '1 undefined', 'NaN', 'Infinity', '1e999'])
      expect(() => forecastInput(bad, 2, '')).toThrow();
    expect(() =>
      validateModelWorkspaceInput('time-series-forecasting-v1', forecastInput('1 2', 2, '.9 .1')),
    ).toThrow();
  });
  it('keeps image sampling prompt-free and audio instructions bound to owned artifacts', () => {
    const id = 'artifact_aaaaaaaaaaaaaaaaaaaaaaaa';
    expect(
      validateModelWorkspaceInput('unconditional-image-generation-v1', {
        width: 32,
        height: 32,
        steps: 2,
        seed: 0,
      }),
    ).toBeTruthy();
    expect(() =>
      validateModelWorkspaceInput('unconditional-image-generation-v1', { prompt: '' }),
    ).toThrow();
    expect(
      validateModelWorkspaceInput('audio-question-answering-v1', {
        audioArtifactId: id,
        instruction: 'Describe the sound.',
      }),
    ).toBeTruthy();
    expect(() =>
      validateModelWorkspaceInput('audio-question-answering-v1', {
        audioPath: '/private/audio.wav',
        instruction: 'describe',
      }),
    ).toThrow();
  });
  it('reflects fixed geometry and narrowed native frame controls', () => {
    const base = { min: 1, max: 1024, step: 1 };
    expect(numericControl({ minimum: 8, maximum: 1024, const: 32 }, base)).toEqual({
      min: 32,
      max: 32,
      step: 1,
    });
    expect(
      numericControl(
        {
          allOf: [
            { minimum: 64, maximum: 1024, multipleOf: 16 },
            { maximum: 512, multipleOf: 32 },
          ],
        },
        base,
      ),
    ).toEqual({ min: 64, max: 512, step: 32 });
    expect(numericControl({ enum: [1, 9, 17] }, base)).toEqual({ min: 1, max: 17, step: 8 });
  });
});
