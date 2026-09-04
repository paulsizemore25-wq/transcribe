"""Speech recognition backed by faster-whisper (CTranslate2).

Everything here runs locally: the only network access is the one-time model
download from Hugging Face, after which the weights live in a local cache and
the program works fully offline.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .audio import Audio
from .errors import EngineError
from .transcript import Segment, Transcript, Word

# Rough VRAM/RAM ordering, smallest first.  `large-v3-turbo` is the sweet spot:
# near-large-v3 accuracy at a fraction of the compute, which matters a lot when
# the whole point is that this runs on your own machine.
MODELS = (
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3",
    "large-v3-turbo", "turbo",
    "distil-small.en", "distil-medium.en", "distil-large-v2", "distil-large-v3", "distil-large-v3.5",
)
DEFAULT_MODEL = "large-v3-turbo"

ProgressCallback = Callable[[Segment, float | None], None]


@dataclass
class EngineOptions:
    """How the model itself is loaded."""

    model: str = DEFAULT_MODEL
    device: str = "auto"          # auto | cpu | cuda
    compute_type: str = "auto"    # auto | int8 | int8_float16 | float16 | float32 | ...
    cpu_threads: int = 0          # 0 lets CTranslate2 decide
    download_root: str | None = None
    local_files_only: bool = False


@dataclass
class DecodeOptions:
    """How a given file is decoded.  Defaults favour accuracy over speed."""

    language: str | None = None            # None = autodetect
    task: str = "transcribe"               # transcribe | translate (to English)
    beam_size: int = 5
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    vad_filter: bool = True
    vad_min_silence_ms: int = 500
    word_timestamps: bool = False
    initial_prompt: str | None = None
    hotwords: str | None = None
    condition_on_previous_text: bool = True
    # Guards against Whisper's classic failure mode: looping text over silence.
    hallucination_silence_threshold: float | None = 2.0
    batch_size: int = 0                    # >1 enables the batched pipeline
    extra: dict = field(default_factory=dict)


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_compute_type(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    # int8 on CPU is ~4x faster than float32 with no meaningful accuracy loss;
    # float16 is the natural choice on any CUDA card that supports it.
    return "float16" if device == "cuda" else "int8"


class WhisperEngine:
    """Thin, testable wrapper around faster-whisper's WhisperModel."""

    def __init__(self, options: EngineOptions | None = None) -> None:
        self.options = options or EngineOptions()
        self.device = resolve_device(self.options.device)
        self.compute_type = resolve_compute_type(self.options.compute_type, self.device)
        self._model = None
        self._batched = None

    # -- model loading -----------------------------------------------------

    def load(self):
        """Load (downloading on first use) the Whisper weights."""
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise EngineError(
                "faster-whisper is not installed. Run: pip install -r requirements.txt"
            ) from exc

        try:
            self._model = WhisperModel(
                self.options.model,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.options.cpu_threads,
                download_root=self.options.download_root,
                local_files_only=self.options.local_files_only,
            )
        except Exception as exc:
            raise EngineError(self._explain_load_failure(exc)) from exc
        return self._model

    def _explain_load_failure(self, exc: Exception) -> str:
        detail = f"{type(exc).__name__}: {exc}"
        message = str(exc).lower()
        network_hints = (
            "connect", "network", "offline", "resolve", "timed out", "timeout",
            "outgoing traffic", "cached snapshot", "proxy", "ssl", "403", "404",
        )
        if any(hint in message for hint in network_hints):
            return (
                f"could not fetch the '{self.options.model}' model ({detail}).\n"
                "The first run needs internet access to download the weights from "
                "Hugging Face; after that everything is offline. If you are behind a "
                "proxy, set HTTPS_PROXY, or pre-download the model on another machine "
                "and point --model at the local directory."
            )
        if "cuda" in message or "cublas" in message or "cudnn" in message:
            return (
                f"CUDA initialisation failed ({detail}).\n"
                "Re-run with --device cpu, or install the cuBLAS/cuDNN runtime that "
                "CTranslate2 expects."
            )
        return f"could not load the '{self.options.model}' model ({detail})."

    # -- transcription -----------------------------------------------------

    def transcribe(
        self,
        audio: Audio,
        source: str,
        options: DecodeOptions | None = None,
        progress: ProgressCallback | None = None,
    ) -> Transcript:
        options = options or DecodeOptions()
        model = self.load()
        started = time.monotonic()

        kwargs = dict(
            language=options.language,
            task=options.task,
            beam_size=options.beam_size,
            temperature=list(options.temperature),
            word_timestamps=options.word_timestamps,
            initial_prompt=options.initial_prompt,
            condition_on_previous_text=options.condition_on_previous_text,
            vad_filter=options.vad_filter,
            vad_parameters={"min_silence_duration_ms": options.vad_min_silence_ms},
        )
        if options.hotwords:
            kwargs["hotwords"] = options.hotwords
        if options.hallucination_silence_threshold and options.word_timestamps:
            # faster-whisper only applies this when word timestamps are on.
            kwargs["hallucination_silence_threshold"] = options.hallucination_silence_threshold
        kwargs.update(options.extra)

        runner = model
        if options.batch_size and options.batch_size > 1:
            runner = self._batched_pipeline(model)
            kwargs["batch_size"] = options.batch_size
            kwargs.pop("condition_on_previous_text", None)
            kwargs.pop("hallucination_silence_threshold", None)

        try:
            raw_segments, info = runner.transcribe(audio.samples, **kwargs)
            segments = list(self._collect(raw_segments, audio.duration, progress))
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"transcription failed for {source}: {type(exc).__name__}: {exc}") from exc

        return Transcript(
            source=source,
            segments=segments,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            duration=getattr(info, "duration", None) or audio.duration,
            model=self.options.model,
            task=options.task,
            elapsed=time.monotonic() - started,
        )

    def _batched_pipeline(self, model):
        if self._batched is None:
            try:
                from faster_whisper import BatchedInferencePipeline
            except ImportError as exc:
                raise EngineError(
                    "--batch-size needs a newer faster-whisper (>=1.1). Upgrade or drop the flag."
                ) from exc
            self._batched = BatchedInferencePipeline(model=model)
        return self._batched

    @staticmethod
    def _collect(
        raw_segments: Iterable,
        duration: float | None,
        progress: ProgressCallback | None,
    ):
        for index, raw in enumerate(raw_segments):
            segment = Segment(
                index=index,
                start=float(raw.start),
                end=float(raw.end),
                text=(raw.text or "").strip(),
                words=[
                    Word(
                        start=float(word.start),
                        end=float(word.end),
                        word=word.word,
                        probability=getattr(word, "probability", None),
                    )
                    for word in (getattr(raw, "words", None) or [])
                ],
                avg_logprob=getattr(raw, "avg_logprob", None),
                no_speech_prob=getattr(raw, "no_speech_prob", None),
                compression_ratio=getattr(raw, "compression_ratio", None),
            )
            if progress is not None:
                fraction = None
                if duration and duration > 0:
                    fraction = min(1.0, segment.end / duration)
                progress(segment, fraction)
            yield segment


def default_cpu_threads() -> int:
    """A sane thread count when the user has not asked for one."""
    return max(1, (os.cpu_count() or 4))
