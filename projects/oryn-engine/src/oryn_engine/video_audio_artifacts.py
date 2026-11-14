"""Bounded native MP4 artifacts containing both video and stereo audio."""

from __future__ import annotations

from fractions import Fraction

MAX_PIXELS = 96 * 1024 * 1024
MAX_BYTES = 256 * 1024 * 1024


def write_video_audio(path, frames, audio, *, fps, sample_rate, event):
    import av
    import numpy as np

    frames, audio = np.asarray(frames), np.asarray(audio)
    if (
        frames.ndim != 4
        or frames.shape[-1] != 3
        or not 1 <= len(frames) <= 360
        or frames.shape[0] * frames.shape[1] * frames.shape[2] > MAX_PIXELS
        or frames.dtype != np.uint8
        and not np.issubdtype(frames.dtype, np.floating)
        or audio.ndim != 2
        or audio.shape[0] != 2
        or not np.issubdtype(audio.dtype, np.floating)
        or not np.isfinite(audio).all()
        or np.max(np.abs(audio), initial=0) > 1.00001
        or fps != 24
        or sample_rate != 32000
    ):
        raise ValueError("VIDEO_AUDIO_OUTPUT")
    if np.issubdtype(frames.dtype, np.floating):
        if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise ValueError("VIDEO_AUDIO_PIXELS")
        frames = np.rint(frames * 255).astype(np.uint8)
    samples = (len(frames) * sample_rate + fps - 1) // fps
    # Native 40 Hz audio latents round the duration by at most half a hop.
    # Trim only the terminal grid surplus; never stretch or resample a track.
    if abs(samples - audio.shape[1]) > 400:
        raise ValueError("VIDEO_AUDIO_DURATION")
    samples = min(samples, audio.shape[1])
    audio = np.ascontiguousarray(audio[:, :samples], dtype=np.float32)
    # A common multiple of the video and audio clocks preserves the exact
    # sample duration in MP4 edit lists instead of rounding it to milliseconds.
    with av.open(str(path), "w", format="mp4", options={"movie_timescale": "96000"}) as container:
        video = container.add_stream("libx264", rate=fps)
        video.width, video.height = frames.shape[2], frames.shape[1]
        video.pix_fmt = "yuv420p"
        video.options = {"crf": "18", "preset": "fast"}
        video.codec_context.thread_count = 1
        sound = container.add_stream("aac", rate=sample_rate)
        sound.layout = "stereo"
        sound.bit_rate = 128000
        sound.codec_context.thread_count = 1
        position = 0
        for i, pixels in enumerate(frames):
            event("encoding-video-audio", 0.88 + 0.08 * i / len(frames))
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in video.encode(frame):
                container.mux(packet)
            end = min(samples, ((i + 1) * sample_rate + fps - 1) // fps)
            while position < end:
                stop = min(position + 1024, end)
                chunk = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(audio[:, position:stop]), format="fltp", layout="stereo"
                )
                chunk.sample_rate = sample_rate
                chunk.pts = position
                chunk.time_base = Fraction(1, sample_rate)
                for packet in sound.encode(chunk):
                    container.mux(packet)
                position = stop
        for packet in video.encode():
            container.mux(packet)
        for packet in sound.encode():
            container.mux(packet)
    return samples


def validate_video_audio(path, result, value):
    import av
    import numpy as np

    if (
        path.is_symlink()
        or not path.is_file()
        or not 1 <= path.stat().st_size <= MAX_BYTES
        or result.get("relativeOutput") != path.name
        or result.get("mediaType") != "video/mp4"
        or result.get("sampleRate") != 32000
        or result.get("channels") != 2
        or any(result.get(k) != value[k] for k in ("width", "height", "frames", "fps"))
    ):
        raise ValueError("VIDEO_AUDIO_ARTIFACT")
    target = (value["frames"] * 32000 + 23) // 24
    expected = result.get("samples")
    if type(expected) is not int or not target - 400 <= expected <= target:
        raise ValueError("VIDEO_AUDIO_ARTIFACT")
    try:
        with av.open(str(path), "r") as container:
            if (
                len(container.streams) != 2
                or len(container.streams.video) != 1
                or len(container.streams.audio) != 1
            ):
                raise ValueError("VIDEO_AUDIO_STREAMS")
            video, sound = container.streams.video[0], container.streams.audio[0]
            if (
                (video.width, video.height, video.average_rate)
                != (value["width"], value["height"], Fraction(24))
                or sound.sample_rate != 32000
                or sound.channels != 2
                or sound.start_time != 0
                or sound.duration is None
                or sound.duration * sound.time_base * 32000 != expected
            ):
                raise ValueError("VIDEO_AUDIO_GEOMETRY")
            count = 0
            for frame in container.decode(video):
                count += 1
                if count > value["frames"] or (frame.width, frame.height) != (
                    value["width"],
                    value["height"],
                ):
                    raise ValueError("VIDEO_AUDIO_FRAMES")
            if count != value["frames"]:
                raise ValueError("VIDEO_AUDIO_FRAMES")
        with av.open(str(path), "r") as container:
            count = 0
            for frame in container.decode(audio=0):
                count += frame.samples
                if (
                    count > expected + 1024
                    or frame.sample_rate != 32000
                    or len(frame.layout.channels) != 2
                    or not np.isfinite(frame.to_ndarray()).all()
                ):
                    raise ValueError("VIDEO_AUDIO_SAMPLES")
            if not expected <= count <= expected + 1024:
                raise ValueError("VIDEO_AUDIO_SAMPLES")
    except (av.error.FFmpegError, EOFError) as exc:
        raise ValueError("VIDEO_AUDIO_ARTIFACT") from exc
