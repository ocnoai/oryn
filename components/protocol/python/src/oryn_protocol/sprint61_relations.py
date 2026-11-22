"""Cross-field data constraints for additive contracts; no execution authority."""


def valid_input(contract, value):
    if contract == "image-editing-v1":
        return value["imageArtifactId"] not in value.get("referenceImageArtifactIds", [])
    if contract == "speech-synthesis-v1":
        reference = "audioArtifactId" in value
        return ((value.get("voice", "reference" if reference else "af_heart") == "reference") == reference
                and ("referenceText" not in value or reference))
    if contract == "embeddings-v1":
        return not (value.get("inputType") == "query" and value.get("contextual") is True)
    if contract == "pose-estimation-v1":
        return all(b[0] < b[2] and b[1] < b[3] for b in value["boxes"])
    if contract in {"tabular-classification-v1", "tabular-regression-v1"}:
        columns = value["columns"]
        if len({c["name"].strip() for c in columns}) != len(columns):
            return False
        if len(value["targets"]) != len(value["trainingRows"]):
            return False
        for row in [*value["trainingRows"], *value["queryRows"]]:
            if len(row) != len(columns):
                return False
            for c, cell in zip(columns, row, strict=True):
                if cell is not None and (
                    (c["type"] == "numeric" and (isinstance(cell, bool) or not isinstance(cell, (int, float))))
                    or (c["type"] == "categorical" and not isinstance(cell, str))
                ):
                    return False
        if contract == "tabular-classification-v1":
            return 2 <= len(set(value["targets"])) <= 64
    elif contract == "time-series-forecasting-v1":
        horizon = value.get("horizon", 16)
        times, frequency = value.get("timestamps"), value.get("frequencyMilliseconds")
        if (times is None) != (frequency is None):
            return False
        if times is not None and (
            len(times) != len(value["context"])
            or any(b - a != frequency for a, b in zip(times[:-1], times[1:], strict=True))
            or times[-1] + horizon * frequency > 8640000000000000
        ):
            return False
        q = value.get("quantiles", [])
        if q != sorted(set(q)):
            return False
        past, future = value.get("pastCovariates"), value.get("futureCovariates")
        if (past is None) != (future is None):
            return False
        if past is not None and (
            len(past) != len(value["context"]) or len(future) != horizon
            or any(len(row) != len(past[0]) for row in [*past, *future])
        ):
            return False
    elif contract == "audio-transformation-v1":
        operation = value["operation"]
        required = {"enhance": {"audioArtifactId"}, "separate": {"audioArtifactId"},
                    "voice-conversion": {"audioArtifactId", "referenceAudioArtifactId"},
                    "encode": {"audioArtifactId"}, "decode": {"codes"}, "vocode": {"spectrogram"}}[operation]
        sources = {"audioArtifactId", "referenceAudioArtifactId", "codes", "spectrogram"}
        if set(value) & sources != required:
            return False
        for field in ("codes", "spectrogram"):
            matrix = value.get(field)
            if matrix and any(len(row) != len(matrix[0]) for row in matrix):
                return False
    elif contract == "multimodal-generation-v1":
        required = {"image": {"imageArtifactId"}, "audio": {"audioArtifactId"},
                    "image-and-audio": {"imageArtifactId", "audioArtifactId"}}[value["inputMode"]]
        return set(value) & {"imageArtifactId", "audioArtifactId"} == required
    elif contract == "video-editing-v1":
        if value.get("startSeconds", 0) + value.get("durationSeconds", 1) > 30:
            return False
        if value.get("operation", "edit") == "restore":
            return not (set(value) & {"prompt", "maskVideoArtifactId"})
        return "prompt" in value
    return True


def valid_output(contract, value):
    if contract == "pose-estimation-v1":
        return all(len({p["id"] for p in pose["keypoints"]}) == len(pose["keypoints"])
                   and all(p["visible"] == (0 <= p["x"] < value["width"] and 0 <= p["y"] < value["height"]) for p in pose["keypoints"])
                   for pose in value["poses"])
    if contract == "tabular-classification-v1":
        classes = value["classes"]
        return len(set(classes)) == len(classes) and all(
            p["label"] in classes and len(p["probabilities"]) == len(classes)
            and abs(sum(p["probabilities"]) - 1) <= 0.000001
            and p["probabilities"][classes.index(p["label"])] == max(p["probabilities"])
            for p in value["predictions"]
        )
    if contract == "time-series-forecasting-v1":
        horizon = len(value["predictions"])
        quantiles = value["quantiles"]
        levels = [q["level"] for q in quantiles]
        times = value.get("timestamps")
        return (
            levels == sorted(set(levels))
            and all(len(q["values"]) == horizon for q in quantiles)
            and all(all(a <= b for a, b in zip(lo["values"], hi["values"], strict=True))
                    for lo, hi in zip(quantiles[:-1], quantiles[1:], strict=True))
            and (times is None or (len(times) == horizon and all(
                a < b for a, b in zip(times[:-1], times[1:], strict=True))))
        )
    return True


def valid_result_for_input(contract, result, source):
    """Check input-derived cardinality after schema validation, before publication."""
    if contract == "pose-estimation-v1":
        value = result["value"]
        return len(value["poses"]) == len(source["boxes"]) and all(
            pose["box"] == [b[0]*value["width"], b[1]*value["height"], b[2]*value["width"], b[3]*value["height"]]
            for pose, b in zip(value["poses"], source["boxes"], strict=True))
    if contract in {"tabular-classification-v1", "tabular-regression-v1"}:
        value = result["value"]
        return len(value["predictions"]) == len(source["queryRows"]) and (
            contract != "tabular-classification-v1" or set(value["classes"]) == set(source["targets"]))
    if contract == "time-series-forecasting-v1":
        value = result["value"]
        times = source.get("timestamps")
        expected = None if times is None else [times[-1] + (i + 1) * source["frequencyMilliseconds"]
                                               for i in range(source.get("horizon", 16))]
        return (len(value["predictions"]) == source.get("horizon", 16)
                and [q["level"] for q in value["quantiles"]] == source.get("quantiles", [])
                and value.get("timestamps") == expected)
    if contract == "audio-transformation-v1":
        operation = source["operation"]
        if operation == "encode":
            return result["kind"] == "structured"
        return result["kind"] == ("multi" if operation == "separate" else "artifact")
    if contract == "multimodal-generation-v1":
        return result["kind"] == {"text": "structured", "audio": "artifact",
                                  "text-and-audio": "multi"}[source.get("outputMode", "text")]
    return True


def valid_result_media(contract, result):
    expected = {"unconditional-image-generation-v1": "image/png",
                "image-text-to-video-v1": "video/mp4", "video-editing-v1": "video/mp4",
                "audio-transformation-v1": "audio/wav", "multimodal-generation-v1": "audio/wav"}.get(contract)
    if expected is None:
        return True
    if result["kind"] == "artifact":
        return result["artifact"]["mimeType"] == expected
    if result["kind"] == "multi":
        return all(item["kind"] != "artifact" or item["artifact"]["mimeType"] == expected
                   for item in result["outputs"])
    return True
