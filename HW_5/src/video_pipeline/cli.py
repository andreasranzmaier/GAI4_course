"""Typer CLI entry point for the video intelligence pipeline.

Subcommands:

* ``whisper``   — Pipeline A: speech-to-text (Whisper Large-V3).
* ``ocr``       — Pipeline B: on-screen text OCR (Qwen2.5-VL-7B).
* ``translate`` — translate a transcript with Qwen2.5-VL (text-only).
* ``summarize`` — summarise a transcript/translation with Qwen2.5-VL (text-only).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from . import __version__, configure_logging
from .models import OCRResult, SummaryResult, TranslationResult, WhisperResult

app = typer.Typer(
    name="video-pipeline",
    help="Speech-to-text, OCR, translation and summarisation for video.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"video-pipeline {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable debug logging.")
    ] = False,
    _version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True,
                     help="Show version and exit."),
    ] = False,
) -> None:
    """Configure logging for all subcommands."""
    configure_logging(logging.DEBUG if verbose else logging.INFO)


def _write_json(
    model: WhisperResult | OCRResult | TranslationResult | SummaryResult, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[green]✓[/green] Wrote {path}")


def _read_text_field(path: Path, field: str) -> str:
    """Pull a text field (e.g. ``text`` or ``summary``) out of a result JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get(field)
    if not value:
        raise typer.BadParameter(f"No '{field}' found in {path}")
    return value


@app.command()
def whisper(
    videos: Annotated[
        list[Path], typer.Argument(help="One or more .mp4/.mkv video files.")
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output JSON file (single video) or directory."),
    ] = None,
    language: Annotated[
        Optional[str], typer.Option("--language", "-l", help="Force a language code (e.g. de).")
    ] = None,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Whisper chunk batch size.")
    ] = 8,
    chunk_length: Annotated[
        int, typer.Option("--chunk-length", help="Chunk length in seconds.")
    ] = 30,
) -> None:
    """Pipeline A: transcribe speech from one or more videos with Whisper Large-V3."""
    from .whisper_pipeline import WhisperPipeline

    multi = len(videos) > 1
    with WhisperPipeline(batch_size=batch_size, chunk_length_s=chunk_length) as wp:
        for video in videos:
            result = wp.transcribe(video, language=language)
            if output is None:
                dest = video.with_suffix(".transcript.json")
            elif multi:
                dest = output / f"{video.stem}.transcript.json"
            else:
                dest = output
            _write_json(result, dest)
            console.print(
                f"[cyan]{video.name}[/cyan]: {len(result.segments)} segments, "
                f"{len(result.text)} chars"
            )


@app.command()
def ocr(
    video: Annotated[Path, typer.Argument(help="A .mp4/.mkv video file.")],
    fps: Annotated[
        float, typer.Option("--fps", help="Frames sampled per second of video.")
    ] = 1.0,
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Output JSON file.")
    ] = None,
    max_new_tokens: Annotated[
        int, typer.Option("--max-new-tokens", help="Max tokens generated per frame.")
    ] = 512,
) -> None:
    """Pipeline B: OCR on-screen text from video frames with Qwen2.5-VL-7B."""
    from .ocr_pipeline import OCRPipeline

    with OCRPipeline(max_new_tokens=max_new_tokens) as op:
        result = op.run(video, fps=fps)
    dest = output or video.with_suffix(".ocr.json")
    _write_json(result, dest)
    console.print(
        f"[cyan]{video.name}[/cyan]: {result.frame_count} frames, "
        f"{len(result.unique_text_blocks)} unique text blocks"
    )


@app.command()
def translate(
    transcript: Annotated[
        Path, typer.Argument(help="A transcript JSON file (Whisper output).")
    ],
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Output JSON file.")
    ] = None,
    source: Annotated[
        str, typer.Option("--source", "-s", help="Source language name.")
    ] = "German",
    target: Annotated[
        str, typer.Option("--target", "-t", help="Target language name.")
    ] = "English",
    field: Annotated[
        str, typer.Option("--field", help="JSON field holding the text to translate.")
    ] = "text",
    max_new_tokens: Annotated[
        int, typer.Option("--max-new-tokens", help="Max tokens generated.")
    ] = 1024,
) -> None:
    """Translate a transcript with Qwen2.5-VL (text-only)."""
    from .qwen_text import translate_text

    text = _read_text_field(transcript, field)
    result = translate_text(
        text, source_path=str(transcript), source=source, target=target,
        max_new_tokens=max_new_tokens,
    )
    dest = output or transcript.with_suffix(".translation.json")
    _write_json(result, dest)
    console.print(
        f"[cyan]{transcript.name}[/cyan]: {source} -> {target}, "
        f"{len(result.text)} chars"
    )


@app.command()
def summarize(
    transcript: Annotated[
        Path, typer.Argument(help="A transcript or translation JSON file.")
    ],
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Output JSON file.")
    ] = None,
    language: Annotated[
        str, typer.Option("--language", "-l", help="Language of the text / summary.")
    ] = "English",
    field: Annotated[
        str, typer.Option("--field", help="JSON field holding the text to summarise.")
    ] = "text",
    max_new_tokens: Annotated[
        int, typer.Option("--max-new-tokens", help="Max tokens generated.")
    ] = 1024,
) -> None:
    """Summarise a transcript/translation with Qwen2.5-VL (text-only)."""
    from .qwen_text import summarize_text

    text = _read_text_field(transcript, field)
    result = summarize_text(
        text, source_path=str(transcript), language=language,
        max_new_tokens=max_new_tokens,
    )
    dest = output or transcript.with_suffix(".summary.json")
    _write_json(result, dest)
    console.print(
        f"[cyan]{transcript.name}[/cyan]: summarised to {len(result.summary)} chars"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
