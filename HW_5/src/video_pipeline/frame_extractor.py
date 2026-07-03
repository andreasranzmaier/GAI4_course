"""Frame extraction from video files via an ``ffmpeg`` subprocess wrapper."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import get_logger
from .audio_extractor import FFmpegError, _require_binary, validate_video

log = get_logger("frame_extractor")


@dataclass(slots=True)
class ExtractedFrame:
    """A single frame sampled from a video."""

    frame_number: int  # 1-based index among extracted frames
    timestamp: float   # approximate time in the source video, seconds
    path: Path         # PNG file on disk


def extract_frames(
    video_path: Path,
    fps: float = 1.0,
    output_dir: Path | None = None,
) -> list[ExtractedFrame]:
    """Sample frames from *video_path* at *fps* frames per second.

    Args:
        video_path: Source ``.mp4`` or ``.mkv`` file.
        fps: Frames to sample per second of video (e.g. ``1.0`` = one per second,
            ``0.5`` = one every two seconds).
        output_dir: Directory to write PNG frames into. When ``None`` a temporary
            directory is created (the caller is responsible for cleanup).

    Returns:
        Ordered list of :class:`ExtractedFrame`.
    """
    ffmpeg = _require_binary("ffmpeg")
    video_path = validate_video(video_path)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="vp_frames_"))
    else:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(output_dir / "frame_%06d.png")
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-vsync", "vfr",
        pattern,
    ]
    log.info("Extracting frames at %.3g fps: %s -> %s", fps, video_path.name, output_dir)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(
            f"ffmpeg frame extraction failed (exit {result.returncode}):\n{result.stderr}"
        )

    frame_paths = sorted(output_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FFmpegError(f"ffmpeg produced no frames in {output_dir}")

    frames = [
        ExtractedFrame(
            frame_number=i,
            timestamp=round((i - 1) / fps, 3),
            path=path,
        )
        for i, path in enumerate(frame_paths, start=1)
    ]
    log.info("Extracted %d frames", len(frames))
    return frames
