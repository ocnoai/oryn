from __future__ import annotations

import pytest

from oryn_engine.subtitles import (
    MAX_ALIGNMENT_WINDOW_MS,
    AlignmentWindow,
    SubtitleContractError,
    TimedSegment,
    alignment_windows,
    reconcile_alignment_windows,
    serialize_subrip,
)


def test_subrip_is_deterministic_utf8_and_round_trip_shaped() -> None:
    output = serialize_subrip(
        [
            {"startMs": 0, "endMs": 1_250, "text": "Ahoj, svet 👋"},
            {"startMs": 1_250, "endMs": 3_001, "text": "Second\nline"},
        ]
    )
    assert (
        output
        == (
            "1\r\n00:00:00,000 --> 00:00:01,250\r\nAhoj, svet 👋\r\n\r\n"
            "2\r\n00:00:01,250 --> 00:00:03,001\r\nSecond line\r\n"
        ).encode()
    )


@pytest.mark.parametrize(
    "segments",
    [
        [{"startMs": 2, "endMs": 2, "text": "zero"}],
        [
            {"startMs": 0, "endMs": 10, "text": "one"},
            {"startMs": 9, "endMs": 20, "text": "overlap"},
        ],
        [{"startMs": 0, "endMs": 10, "text": "1 --> 2"}],
    ],
)
def test_subrip_rejects_structurally_invalid_cues(segments: list[dict[str, object]]) -> None:
    with pytest.raises(SubtitleContractError) as error:
        serialize_subrip(segments)
    assert error.value.code == "SUBTITLE_SERIALIZATION_INVALID"


def test_alignment_windows_never_exceed_five_minutes_and_preserve_timebase() -> None:
    windows = alignment_windows(MAX_ALIGNMENT_WINDOW_MS * 2 + 123)
    assert windows == (
        AlignmentWindow(0, MAX_ALIGNMENT_WINDOW_MS),
        AlignmentWindow(MAX_ALIGNMENT_WINDOW_MS, MAX_ALIGNMENT_WINDOW_MS * 2),
        AlignmentWindow(MAX_ALIGNMENT_WINDOW_MS * 2, MAX_ALIGNMENT_WINDOW_MS * 2 + 123),
    )
    result = reconcile_alignment_windows(
        [
            (windows[0], [TimedSegment(10, 20, "one")]),
            (windows[1], [TimedSegment(0, 20, "two")]),
            (windows[2], [TimedSegment(0, 123, "three")]),
        ]
    )
    assert result[1] == TimedSegment(MAX_ALIGNMENT_WINDOW_MS, MAX_ALIGNMENT_WINDOW_MS + 20, "two")
    assert result[-1].end_ms == MAX_ALIGNMENT_WINDOW_MS * 2 + 123


def test_alignment_rejects_gap_or_window_relative_overflow() -> None:
    with pytest.raises(SubtitleContractError):
        reconcile_alignment_windows([(AlignmentWindow(10, 100), [TimedSegment(0, 20, "gap")])])
    with pytest.raises(SubtitleContractError):
        reconcile_alignment_windows([(AlignmentWindow(0, 100), [TimedSegment(0, 101, "overflow")])])
