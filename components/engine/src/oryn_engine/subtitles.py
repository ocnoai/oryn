from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

MAX_MEDIA_DURATION_MS = 4 * 60 * 60 * 1000
MAX_ALIGNMENT_WINDOW_MS = 5 * 60 * 1000
MAX_SEGMENTS = 20_000
MAX_SEGMENT_TEXT = 2_000
MAX_TOTAL_TEXT = 2_000_000


class SubtitleContractError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class TimedSegment:
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class AlignmentWindow:
    start_ms: int
    end_ms: int


def alignment_windows(duration_ms: int) -> tuple[AlignmentWindow, ...]:
    """Return non-overlapping, timebase-preserving windows accepted by Qwen's aligner."""

    if not isinstance(duration_ms, int) or isinstance(duration_ms, bool):
        raise SubtitleContractError("ALIGNMENT_LIMIT_INVALID", "Media duration is invalid.")
    if not 1 <= duration_ms <= MAX_MEDIA_DURATION_MS:
        raise SubtitleContractError("ALIGNMENT_LIMIT_INVALID", "Media duration is unsupported.")
    return tuple(
        AlignmentWindow(offset, min(duration_ms, offset + MAX_ALIGNMENT_WINDOW_MS))
        for offset in range(0, duration_ms, MAX_ALIGNMENT_WINDOW_MS)
    )


def normalize_segments(values: Iterable[object]) -> tuple[TimedSegment, ...]:
    raw = list(values)
    if not 1 <= len(raw) <= MAX_SEGMENTS:
        raise SubtitleContractError("SUBTITLE_SERIALIZATION_INVALID", "Cue count is invalid.")
    result: list[TimedSegment] = []
    previous_end = 0
    total_text = 0
    for value in raw:
        if isinstance(value, TimedSegment):
            start_ms, end_ms, text = value.start_ms, value.end_ms, value.text
        elif isinstance(value, dict) and set(value) == {"startMs", "endMs", "text"}:
            start_ms, end_ms, text = value["startMs"], value["endMs"], value["text"]
        else:
            raise SubtitleContractError(
                "SUBTITLE_SERIALIZATION_INVALID", "A subtitle cue is invalid."
            )
        if (
            not isinstance(start_ms, int)
            or isinstance(start_ms, bool)
            or not isinstance(end_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(text, str)
            or not 0 <= start_ms < end_ms <= MAX_MEDIA_DURATION_MS
            or start_ms < previous_end
            or not 1 <= len(text) <= MAX_SEGMENT_TEXT
        ):
            raise SubtitleContractError(
                "SUBTITLE_SERIALIZATION_INVALID", "Subtitle cue bounds are invalid."
            )
        normalized = _normalize_text(text)
        total_text += len(normalized)
        if total_text > MAX_TOTAL_TEXT:
            raise SubtitleContractError(
                "SUBTITLE_SERIALIZATION_INVALID", "Subtitle text is too large."
            )
        result.append(TimedSegment(start_ms, end_ms, normalized))
        previous_end = end_ms
    return tuple(result)


def serialize_subrip(values: Iterable[object]) -> bytes:
    """Serialize inert, deterministic UTF-8 SubRip with normalized CRLF line endings."""

    segments = normalize_segments(values)
    blocks = [
        f"{index}\r\n{_timestamp(item.start_ms)} --> {_timestamp(item.end_ms)}\r\n{item.text}"
        for index, item in enumerate(segments, start=1)
    ]
    return ("\r\n\r\n".join(blocks) + "\r\n").encode("utf-8")


def reconcile_alignment_windows(
    windowed_segments: Iterable[tuple[AlignmentWindow, Iterable[object]]],
) -> tuple[TimedSegment, ...]:
    """Convert window-relative cues to the original zero-based media timeline."""

    combined: list[TimedSegment] = []
    expected_window_start = 0
    for window, values in windowed_segments:
        if (
            window.start_ms != expected_window_start
            or not window.start_ms < window.end_ms
            or window.end_ms - window.start_ms > MAX_ALIGNMENT_WINDOW_MS
        ):
            raise SubtitleContractError(
                "ALIGNMENT_OUTPUT_INVALID", "Alignment windows are not contiguous."
            )
        relative = normalize_segments(values)
        for item in relative:
            if item.end_ms > window.end_ms - window.start_ms:
                raise SubtitleContractError(
                    "ALIGNMENT_OUTPUT_INVALID", "An aligned cue exceeds its source window."
                )
            combined.append(
                TimedSegment(
                    window.start_ms + item.start_ms,
                    window.start_ms + item.end_ms,
                    item.text,
                )
            )
        expected_window_start = window.end_ms
    return normalize_segments(combined)


def _normalize_text(value: str) -> str:
    # Newlines are allowed inside SubRip cues, but blank lines and numeric/timestamp-looking
    # structural lines are flattened so user edits cannot manufacture an extra cue block.
    lines = [re.sub(r"[\t ]+", " ", line).strip() for line in value.replace("\r", "\n").split("\n")]
    text = " ".join(line for line in lines if line)
    if not text or "-->" in text:
        raise SubtitleContractError(
            "SUBTITLE_SERIALIZATION_INVALID", "Subtitle cue text is invalid."
        )
    return text


def _timestamp(value: int) -> str:
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
