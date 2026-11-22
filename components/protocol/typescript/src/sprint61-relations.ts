/** Closed relational checks; these functions never grant runtime admission. */
export function validSprint61Input(contract: string, value: Record<string, unknown>): boolean {
  if (contract === "image-editing-v1") return !(value.referenceImageArtifactIds as string[] | undefined)?.includes(value.imageArtifactId as string);
  if (contract === "speech-synthesis-v1") {
    const reference = Object.hasOwn(value, "audioArtifactId");
    return ((value.voice ?? (reference ? "reference" : "af_heart")) === "reference") === reference &&
      (!Object.hasOwn(value, "referenceText") || reference);
  }
  if (contract === "pose-estimation-v1") return (value.boxes as number[][]).every(b => b[0]! < b[2]! && b[1]! < b[3]!);
  if (contract === "tabular-classification-v1" || contract === "tabular-regression-v1") {
    const columns = value.columns as {name: string; type: string}[];
    const targets = value.targets as (string | number)[];
    const training = value.trainingRows as (number | string | null)[][];
    const queries = value.queryRows as (number | string | null)[][];
    return new Set(columns.map(c => c.name.trim())).size === columns.length &&
      targets.length === training.length && [...training, ...queries].every(row =>
        row.length === columns.length && row.every((cell, i) => cell === null ||
          typeof cell === (columns[i]!.type === "numeric" ? "number" : "string"))) &&
      (contract !== "tabular-classification-v1" || (new Set(targets).size >= 2 && new Set(targets).size <= 64));
  }
  if (contract === "time-series-forecasting-v1") {
    const context = value.context as number[];
    const horizon = Number(value.horizon ?? 16);
    const times = value.timestamps as number[] | undefined;
    const frequency = value.frequencyMilliseconds as number | undefined;
    const quantiles = value.quantiles as number[] | undefined;
    const past = value.pastCovariates as number[][] | undefined;
    const future = value.futureCovariates as number[][] | undefined;
    return (times === undefined) === (frequency === undefined) &&
      (!times || (times.length === context.length && times.every((t, i) =>
        i === 0 || t - times[i - 1]! === frequency) && times.at(-1)! + horizon * frequency! <= 8640000000000000)) &&
      (!quantiles || quantiles.every((q, i) => i === 0 || q > quantiles[i - 1]!)) &&
      (past === undefined) === (future === undefined) &&
      (!past || (past.length === context.length && future!.length === horizon &&
        [...past, ...future!].every(row => row.length === past[0]!.length)));
  }
  if (contract === "audio-transformation-v1") {
    const sources = ["audioArtifactId", "referenceAudioArtifactId", "codes", "spectrogram"];
    const required: Record<string, string[]> = {enhance: ["audioArtifactId"], separate: ["audioArtifactId"],
      "voice-conversion": ["audioArtifactId", "referenceAudioArtifactId"], encode: ["audioArtifactId"],
      decode: ["codes"], vocode: ["spectrogram"]};
    const fields = required[String(value.operation)]!;
    return sources.every(s => Object.hasOwn(value, s) === fields.includes(s)) &&
      ["codes", "spectrogram"].every(field => !value[field] ||
        (value[field] as number[][]).every(row => row.length === (value[field] as number[][])[0]!.length));
  }
  if (contract === "multimodal-generation-v1") {
    const required = {image: ["imageArtifactId"], audio: ["audioArtifactId"], "image-and-audio": ["imageArtifactId", "audioArtifactId"]} as Record<string, string[]>;
    return ["imageArtifactId", "audioArtifactId"].every(field => Object.hasOwn(value, field) === required[String(value.inputMode)]!.includes(field));
  }
  if (contract === "video-editing-v1") return Number(value.startSeconds ?? 0) + Number(value.durationSeconds ?? 1) <= 30 &&
    (value.operation === "restore" ? !Object.hasOwn(value, "prompt") && !Object.hasOwn(value, "maskVideoArtifactId") : Object.hasOwn(value, "prompt"));
  return true;
}

export function validSprint61Output(contract: string, value: Record<string, unknown>): boolean {
  if (contract === "pose-estimation-v1") return (value.poses as {keypoints: {id: number; x: number; y: number; visible: boolean}[]}[]).every(pose =>
    new Set(pose.keypoints.map(p => p.id)).size === pose.keypoints.length && pose.keypoints.every(p =>
      p.visible === (p.x >= 0 && p.x < Number(value.width) && p.y >= 0 && p.y < Number(value.height))));
  const expected = ({"unconditional-image-generation-v1": "image/png", "image-text-to-video-v1": "video/mp4",
    "video-editing-v1": "video/mp4", "audio-transformation-v1": "audio/wav", "multimodal-generation-v1": "audio/wav"} as Record<string, string>)[contract];
  if (expected && value.kind === "artifact" && (value.artifact as {mimeType: string}).mimeType !== expected) return false;
  if (expected && value.kind === "multi" && (value.outputs as {kind: string; artifact?: {mimeType: string}}[]).some(item => item.kind === "artifact" && item.artifact!.mimeType !== expected)) return false;
  if (contract === "tabular-classification-v1") {
    const classes = value.classes as string[];
    return new Set(classes).size === classes.length &&
      (value.predictions as {label: string; probabilities: number[]}[]).every(p =>
        classes.includes(p.label) && p.probabilities.length === classes.length &&
        Math.abs(p.probabilities.reduce((a, b) => a + b, 0) - 1) <= 0.000001 &&
        p.probabilities[classes.indexOf(p.label)] === Math.max(...p.probabilities));
  }
  if (contract === "time-series-forecasting-v1") {
    const count = (value.predictions as number[]).length;
    const quantiles = value.quantiles as {level: number; values: number[]}[];
    const times = value.timestamps as number[] | undefined;
    return quantiles.every((q, i) => q.values.length === count && (i === 0 ||
      q.level > quantiles[i - 1]!.level && q.values.every((v, j) => v >= quantiles[i - 1]!.values[j]!))) &&
      (!times || times.length === count && times.every((t, i) => i === 0 || t > times[i - 1]!));
  }
  return true;
}

export function validSprint61ResultForInput(contract: string, result: Record<string, unknown>, input: Record<string, unknown>): boolean {
  const value = result.value as Record<string, unknown>;
  if (contract === "pose-estimation-v1") return (value.poses as {box: number[]}[]).length === (input.boxes as number[][]).length &&
    (value.poses as {box: number[]}[]).every((pose, i) => {
      const box = (input.boxes as number[][])[i]!;
      return pose.box.every((coordinate, j) => coordinate === box[j]! * Number(j % 2 === 0 ? value.width : value.height));
    });
  if (contract === "tabular-classification-v1" || contract === "tabular-regression-v1") {
    return (value.predictions as unknown[]).length === (input.queryRows as unknown[]).length &&
      (contract !== "tabular-classification-v1" ||
        (value.classes as string[]).length === new Set(input.targets as string[]).size &&
        (value.classes as string[]).every(c => (input.targets as string[]).includes(c)));
  }
  if (contract === "time-series-forecasting-v1") {
    const horizon = Number(input.horizon ?? 16);
    const times = input.timestamps as number[] | undefined;
    const expected = times ? Array.from({length: horizon}, (_, i) => times.at(-1)! + (i + 1) * Number(input.frequencyMilliseconds)) : undefined;
    return (value.predictions as number[]).length === horizon &&
      JSON.stringify((value.quantiles as {level: number}[]).map(q => q.level)) === JSON.stringify(input.quantiles ?? []) &&
      JSON.stringify(value.timestamps) === JSON.stringify(expected);
  }
  if (contract === "audio-transformation-v1") return result.kind === (input.operation === "encode" ? "structured" : input.operation === "separate" ? "multi" : "artifact");
  if (contract === "multimodal-generation-v1") return result.kind === ({text: "structured", audio: "artifact", "text-and-audio": "multi"} as Record<string, string>)[String(input.outputMode ?? "text")];
  return true;
}
