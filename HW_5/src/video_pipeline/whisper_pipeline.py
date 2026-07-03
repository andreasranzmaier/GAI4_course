"""Pipeline A: Speech-to-Text using ``openai/whisper-large-v3`` via transformers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from . import free_memory, get_device_and_dtype, get_logger
from .audio_extractor import extract_audio, probe_duration, validate_video
from .models import WHISPER_MODEL_ID, TranscriptSegment, WhisperResult

log = get_logger("whisper")


class WhisperPipeline:
    """Loads Whisper Large-V3 once and transcribes one or more videos.

    Use as a context manager so the model is unloaded from GPU memory on exit::

        with WhisperPipeline() as wp:
            result = wp.transcribe("clip.mp4")
    """

    def __init__(
        self,
        model_id: str = WHISPER_MODEL_ID,
        batch_size: int = 8,
        chunk_length_s: int = 30,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.chunk_length_s = chunk_length_s
        self._pipe = None  # transformers ASR pipeline, lazily loaded

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        """Load the ASR pipeline onto the best available device."""
        if self._pipe is not None:
            return
        from transformers import pipeline

        device, dtype = get_device_and_dtype()
        # transformers `pipeline` accepts a device string ("cuda:0", "mps", "cpu").
        log.info("Loading %s on %s (%s)", self.model_id, device, dtype)
        self._pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.model_id,
            dtype=dtype,
            device=device,
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
        )

    def unload(self) -> None:
        """Release the model and free GPU memory."""
        if self._pipe is not None:
            log.info("Unloading Whisper model")
            del self._pipe
            self._pipe = None
            free_memory()

    def __enter__(self) -> "WhisperPipeline":
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    # -- inference ---------------------------------------------------------
    def transcribe(self, video_path: Path | str, language: str | None = None) -> WhisperResult:
        """Transcribe a single video and return a :class:`WhisperResult`."""
        if self._pipe is None:
            self.load()
        assert self._pipe is not None

        video_path = validate_video(Path(video_path))
        duration = probe_duration(video_path)

        audio_path: Path | None = None
        try:
            audio_path = extract_audio(video_path)
            log.info("Transcribing %s", video_path.name)
            call_kwargs: dict = {"return_timestamps": True}
            if language:
                call_kwargs["generate_kwargs"] = {"language": language}
            raw = self._pipe(str(audio_path), **call_kwargs)
        finally:
            if audio_path is not None:
                audio_path.unlink(missing_ok=True)

        segments: list[TranscriptSegment] = []
        for idx, chunk in enumerate(raw.get("chunks", []) or []):
            ts = chunk.get("timestamp") or (None, None)
            segments.append(
                TranscriptSegment(
                    id=idx,
                    start=ts[0],
                    end=ts[1],
                    text=(chunk.get("text") or "").strip(),
                )
            )

        result = WhisperResult(
            video_path=str(video_path),
            model=self.model_id,
            language=language,
            duration=duration,
            text=(raw.get("text") or "").strip(),
            segments=segments,
        )
        log.info("Transcribed %s: %d segments", video_path.name, len(segments))
        return result

    def transcribe_batch(
        self, video_paths: list[Path | str], language: str | None = None
    ) -> list[WhisperResult]:
        """Transcribe multiple videos, reusing the loaded model across all of them."""
        if self._pipe is None:
            self.load()
        results: list[WhisperResult] = []
        for i, path in enumerate(video_paths, start=1):
            log.info("Batch item %d/%d", i, len(video_paths))
            results.append(self.transcribe(path, language=language))
        return results


def transcribe_video(
    video_path: Path | str,
    language: str | None = None,
    batch_size: int = 8,
    chunk_length_s: int = 30,
) -> WhisperResult:
    """Convenience wrapper: load, transcribe one video, and unload."""
    with WhisperPipeline(batch_size=batch_size, chunk_length_s=chunk_length_s) as wp:
        return wp.transcribe(video_path, language=language)
