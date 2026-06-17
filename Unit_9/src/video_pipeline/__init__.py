"""Video Intelligence Pipeline.

Speech-to-text (Whisper Large-V3) and video OCR (Qwen2.5-VL) pipelines.
"""

from __future__ import annotations

import gc
import logging

__version__ = "0.1.0"

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, idempotently."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt="%H:%M:%S")
    root.setLevel(level)
    # Quiet down noisy third-party loggers.
    for noisy in ("urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(f"video_pipeline.{name}")


def get_device_and_dtype() -> tuple[str, "object"]:
    """Pick the best available compute device and matching dtype.

    Returns a ``(device, dtype)`` tuple where ``device`` is a concrete string such
    as ``"cuda:0"``, ``"mps"`` or ``"cpu"``, and ``dtype`` is a ``torch.dtype``.

    Only CUDA GPUs with a compute capability supported by the installed torch
    build are considered, so older incompatible GPUs are skipped automatically.
    """
    import torch

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                major, _ = torch.cuda.get_device_capability(i)
            except Exception:  # pragma: no cover - unsupported GPU probe
                continue
            if major >= 7:  # Volta+ : float16 is well supported
                return f"cuda:{i}", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def free_memory() -> None:
    """Release cached GPU memory and run a garbage collection pass.

    Call this after unloading a model so a subsequent pipeline starts clean.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover - torch optional at import time
        pass


__all__ = [
    "__version__",
    "configure_logging",
    "get_logger",
    "get_device_and_dtype",
    "free_memory",
]
