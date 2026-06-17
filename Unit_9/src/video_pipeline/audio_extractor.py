"""Audio extraction from video files via an ``ffmpeg`` subprocess wrapper."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import get_logger

log = get_logger("audio_extractor")

SUPPORTED_SUFFIXES = {".mp4", ".mkv"}


class FFmpegError(RuntimeError):
    """Raised when an ``ffmpeg``/``ffprobe`` invocation fails."""


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FFmpegError(
            f"`{name}` was not found on PATH. Install FFmpeg and ensure it is available."
        )
    return path


def validate_video(video_path: Path) -> Path:
    """Validate that *video_path* exists and has a supported container suffix."""
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if video_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported video suffix '{video_path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )
    return video_path


def probe_duration(video_path: Path) -> float | None:
    """Return the media duration in seconds via ``ffprobe``, or ``None`` if unknown."""
    ffprobe = _require_binary("ffprobe")
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(out.stdout)
        duration = data.get("format", {}).get("duration")
        return float(duration) if duration is not None else None
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as exc:
        log.warning("Could not probe duration for %s: %s", video_path, exc)
        return None


def extract_audio(
    video_path: Path,
    output_path: Path | None = None,
    sample_rate: int = 16_000,
) -> Path:
    """Extract a mono WAV track suitable for Whisper.

    Args:
        video_path: Source ``.mp4`` or ``.mkv`` file.
        output_path: Destination ``.wav`` path. When ``None`` a temporary file is
            created (the caller is responsible for cleanup).
        sample_rate: Target sample rate in Hz. Whisper expects 16 kHz.

    Returns:
        Path to the extracted WAV file.
    """
    ffmpeg = _require_binary("ffmpeg")
    video_path = validate_video(video_path)

    if output_path is None:
        fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="vp_audio_")
        Path(tmp).unlink(missing_ok=True)  # ffmpeg writes the file itself
        import os

        os.close(fd)
        output_path = Path(tmp)
    else:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-vn",                      # drop video
        "-ac", "1",                 # mono
        "-ar", str(sample_rate),    # resample
        "-c:a", "pcm_s16le",        # 16-bit PCM WAV
        str(output_path),
    ]
    log.info("Extracting audio: %s -> %s", video_path.name, output_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(
            f"ffmpeg audio extraction failed (exit {result.returncode}):\n{result.stderr}"
        )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise FFmpegError(f"ffmpeg produced no audio output at {output_path}")
    return output_path
