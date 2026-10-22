"""Speech recognition backed by faster-whisper (CTranslate2).

Everything here runs locally: the only network access is the one-time model
download from Hugging Face, after which the weights live in a local cache and
the program works fully offline.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .audio import Audio
from .errors import EngineError, ModelLoadError
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

if sys.platform == "win32":
    # CTranslate2's Windows wheel already bundles cuDNN; only cuBLAS is missing.
    _CUDA_PIP_PACKAGES = "nvidia-cublas-cu12"
else:
    _CUDA_PIP_PACKAGES = 'nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"'

CUDA_SETUP_HINT = (
    "To use the GPU, install the CUDA 12 math libraries into this environment:\n"
    f"        python -m pip install {_CUDA_PIP_PACKAGES}\n"
    "        (or: python -m pip install -e \".[cuda]\" from the project folder)\n"
    "        Pass --device cpu to skip the GPU entirely."
)

_CUDA_FAILURE_HINTS = (
    "cublas", "cudnn", "cuda", "cudart", "nvrtc", "libcu", "gpu",
    "no kernel image", "out of memory",
)


def _is_cuda_failure(exc: Exception) -> bool:
    return any(hint in str(exc).lower() for hint in _CUDA_FAILURE_HINTS)


def _is_out_of_memory(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


OUT_OF_MEMORY_HINT = (
    "        The model does not fit in GPU memory. Try --compute-type int8_float16\n"
    "        (about half the memory, near-identical accuracy), close other programs\n"
    "        using the GPU, or use --model large-v3-turbo."
)


# cuBLAS needs cuBLASLt, so that one is loaded first.
_WINDOWS_CUDA_DLLS = ("cublasLt64_12.dll", "cublas64_12.dll")
_registered_cuda_dirs: list = []  # keeps add_dll_directory handles alive


def _nvidia_bin_dirs() -> list[Path]:
    """The bin/ folders of pip-installed NVIDIA packages (nvidia-cublas-cu12...).

    `nvidia` is a namespace package spread over every site-packages directory
    that has one, hence `submodule_search_locations` rather than a single path.
    """
    spec = importlib.util.find_spec("nvidia")
    roots = getattr(spec, "submodule_search_locations", None) or []
    return [bin_dir for root in roots for bin_dir in sorted(Path(root).glob("*/bin"))]


def _register_cuda_libraries() -> None:
    """Make pip-installed CUDA libraries visible to CTranslate2 on Windows.

    pip puts the DLLs in site-packages/nvidia/*/bin, which Windows never
    searches.  CTranslate2 loads cuBLAS lazily *by bare name* on the first GPU
    computation, and a by-name load only looks at the standard search order
    (PATH among it) — not at os.add_dll_directory folders.  So: preload each
    DLL by full path, as ctranslate2/__init__.py does with its own DLLs (once
    a module is loaded, Windows hands it back to any later by-name load), and
    put the folders on PATH for anything that loads them indirectly.
    """
    if sys.platform != "win32" or _registered_cuda_dirs:
        return
    import ctypes

    bin_dirs = _nvidia_bin_dirs()
    if not bin_dirs:
        return

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    missing = [str(bin_dir) for bin_dir in bin_dirs if str(bin_dir) not in path_entries]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing + path_entries)

    for bin_dir in bin_dirs:
        try:
            _registered_cuda_dirs.append(os.add_dll_directory(str(bin_dir)))
        except (AttributeError, OSError):
            _registered_cuda_dirs.append(str(bin_dir))

    for name in _WINDOWS_CUDA_DLLS:
        candidate = next((d / name for d in bin_dirs if (d / name).is_file()), None)
        if candidate is None:
            continue
        try:
            ctypes.CDLL(str(candidate))
        except OSError:
            pass  # reported properly by cuda_library_status / the real load


# What faster-whisper needs from a model repo; a snapshot without them is an
# interrupted download.
_MODEL_FILES = ("config.json", "model.bin")


def cached_model_path(model: str, cache_dir: str | None = None) -> str | None:
    """The local copy of a model, found without touching the network.

    Given a bare name, faster-whisper asks Hugging Face for the latest commit
    on every run, for weights that never change: a round trip at best, a
    timeout on a flaky connection, and a hard failure behind a broken proxy.
    """
    if os.path.isdir(model):
        return model
    try:
        from faster_whisper.utils import download_model

        path = download_model(model, local_files_only=True, cache_dir=cache_dir)
    except Exception:
        return None
    if all(os.path.isfile(os.path.join(path, name)) for name in _MODEL_FILES):
        return path
    return None


@dataclass
class EngineOptions:
    """How the model itself is loaded."""

    model: str = DEFAULT_MODEL
    device: str = "auto"          # auto | cpu | cuda
    compute_type: str = "auto"    # auto | int8 | int8_float16 | float16 | float32 | ...
    cpu_threads: int = 0          # 0 = default_cpu_threads()
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
    # Batched pipeline only: False emits one segment per ~30 s chunk, which
    # is slightly faster and fine for plain text but useless for subtitles.
    segment_timestamps: bool = True
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

    def __init__(
        self,
        options: EngineOptions | None = None,
        warn: Callable[[str], None] | None = None,
    ) -> None:
        self.options = options or EngineOptions()
        self.device = resolve_device(self.options.device)
        self.compute_type = resolve_compute_type(self.options.compute_type, self.device)
        # Only a device *we* picked may be silently swapped for another one; an
        # explicit --device cuda must fail loudly instead.
        self._device_was_auto = self.options.device == "auto"
        self._warn = warn or (lambda message: None)
        self._model = None
        self._batched = None
        self._warned_unbatched = False
        self._batch_limit: int | None = None

    # -- model loading -----------------------------------------------------

    def load(self):
        """Load (downloading on first use) the Whisper weights."""
        if self._model is not None:
            return self._model
        try:
            self._model = self._create_model()
        except Exception as exc:
            if not self._can_fall_back(exc):
                raise ModelLoadError(self._explain_load_failure(exc)) from exc
            self._fall_back_to_cpu(exc)
            try:
                self._model = self._create_model()
            except Exception as retry_exc:
                raise ModelLoadError(self._explain_load_failure(retry_exc)) from retry_exc
        return self._model

    def _create_model(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise EngineError(
                "faster-whisper is not installed. Run: pip install -r requirements.txt"
            ) from exc

        if self.device == "cuda":
            _register_cuda_libraries()

        model_path = cached_model_path(self.options.model, self.options.download_root)
        return WhisperModel(
            model_path or self.options.model,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.options.cpu_threads or default_cpu_threads(),
            download_root=self.options.download_root,
            local_files_only=self.options.local_files_only,
        )

    # -- GPU fallback ------------------------------------------------------

    def _can_fall_back(self, exc: Exception) -> bool:
        """CUDA problems are recoverable when we, not the user, chose the GPU."""
        if isinstance(exc, EngineError) or not self._device_was_auto or self.device != "cuda":
            return False
        return _is_cuda_failure(exc)

    def _fall_back_to_cpu(self, exc: Exception) -> None:
        self._warn(
            f"GPU unavailable ({type(exc).__name__}: {exc}) — falling back to CPU.\n"
            f"        {CUDA_SETUP_HINT}"
        )
        self.device = "cpu"
        self.compute_type = resolve_compute_type(self.options.compute_type, "cpu")
        self._model = None
        self._batched = None

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
        if _is_cuda_failure(exc):
            return f"the GPU could not be used ({detail}).\n{CUDA_SETUP_HINT}"
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
        self.load()
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

        batched = bool(options.batch_size and options.batch_size > 1)
        if batched and not options.vad_filter:
            # The batched pipeline cuts long audio at the pauses VAD finds and
            # refuses anything over 30 s without them.
            if not self._warned_unbatched:
                self._warn("--no-vad turns batching off: the batched pipeline needs VAD to split the audio.")
                self._warned_unbatched = True
            batched = False
        if batched:
            # A size that already ran out of memory on an earlier file is not retried.
            kwargs["batch_size"] = min(options.batch_size, self._batch_limit or options.batch_size)
            kwargs["without_timestamps"] = not options.segment_timestamps
            kwargs.pop("condition_on_previous_text", None)
            kwargs.pop("hallucination_silence_threshold", None)

        try:
            segments, info = self._run_within_memory(audio, kwargs, batched, progress)
        except EngineError:
            raise
        except Exception as exc:
            if not self._can_fall_back(exc):
                raise self._transcription_failed(source, exc) from exc
            # CTranslate2 only loads the CUDA libraries when it first decodes,
            # so a broken GPU install surfaces here rather than at load time.
            self._fall_back_to_cpu(exc)
            try:
                segments, info = self._run(audio, kwargs, batched, progress)
            except EngineError:
                raise
            except Exception as retry_exc:
                raise self._transcription_failed(source, retry_exc) from retry_exc

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

    def _run_within_memory(self, audio: Audio, kwargs: dict, batched: bool, progress):
        """_run, halving the batch size each time the GPU runs out of memory.

        How big a batch fits depends on the card, the model and whatever else
        is using video memory, so a fixed --batch-size cannot be right everywhere.
        """
        while True:
            try:
                return self._run(audio, kwargs, batched, progress)
            except EngineError:
                raise
            except Exception as exc:
                size = kwargs.get("batch_size", 1)
                if not (batched and size > 1 and _is_out_of_memory(exc)):
                    raise
            # Outside the except block, so the traceback no longer pins the model.
            smaller = size // 2
            self._warn(f"the GPU ran out of memory at batch size {size}; reloading the model to retry with {smaller}.")
            kwargs["batch_size"] = self._batch_limit = smaller
            # Reusing the model after a CUDA out-of-memory error failed on the
            # user's card ("parallel_for failed: cudaErrorInvalidDevice"), so
            # start from a fresh one; _run loads it again.
            self._model = None
            self._batched = None
            gc.collect()

    def _transcription_failed(self, source: str, exc: Exception) -> EngineError:
        message = f"transcription failed for {source}: {type(exc).__name__}: {exc}"
        if self._batch_limit is not None and self.device == "cuda":
            # The GPU ran out of memory earlier and did not recover in this run;
            # a fresh run at the smaller size never hits that state at all.
            message += (
                f"\n        The GPU ran out of memory earlier in this run. Start again with"
                f" --batch-size {self._batch_limit} (or lower), and consider"
                " --compute-type int8_float16."
            )
        elif _is_out_of_memory(exc):
            message += f"\n{OUT_OF_MEMORY_HINT}"
        return EngineError(message)

    def _run(self, audio: Audio, kwargs: dict, batched: bool, progress):
        model = self.load()
        runner = model
        if batched:
            runner = self._batched_pipeline(model)
        raw_segments, info = runner.transcribe(audio.samples, **kwargs)
        # The generator is consumed here, inside the caller's error handling,
        # because that is where CTranslate2 actually touches the GPU.
        return list(self._collect(raw_segments, audio.duration, progress)), info

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


def cuda_library_status() -> tuple[bool, str]:
    """Actually try to load cuBLAS, the library CTranslate2 needs on a GPU.

    Reporting "your GPU is detected" is useless if the math libraries are
    missing — that is exactly the case that fails halfway through a long file.
    """
    import ctypes

    _register_cuda_libraries()
    if sys.platform == "win32":
        # winmode=0 is the classic LoadLibrary search — the one CTranslate2's
        # by-name load uses.  Python's default would also search
        # add_dll_directory folders and could report OK while CTranslate2 fails.
        name, kwargs = "cublas64_12.dll", {"winmode": 0}
    else:
        name, kwargs = "libcublas.so.12", {}
    try:
        ctypes.CDLL(name, **kwargs)
    except OSError as exc:
        return False, f"{name} not loadable ({exc})"
    return True, f"{name} loaded"


def default_cpu_threads() -> int:
    """About one thread per physical core, when the user has not asked.

    CTranslate2 on its own uses 4 threads however big the machine is.  Logical
    CPUs are usually two per core, and OpenMP slows down when oversubscribed,
    hence the halving; never fewer than CTranslate2's 4, and gains flatten out
    long before 16.
    """
    return max(4, min(16, (os.cpu_count() or 4) // 2))
