"""Audio decoding.

Whisper wants 16 kHz mono float32 samples.  Getting there from an .m4a (AAC in
an MP4 container) normally means shelling out to ffmpeg, which is one more
thing to install.  PyAV ships the ffmpeg libraries inside its wheel, so the
primary path here has no system dependencies at all; the ffmpeg CLI is only
used as a fallback when PyAV is missing or chokes on a file.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import AudioDecodeError

SAMPLE_RATE = 16000

# Containers Whisper is routinely pointed at.  Anything ffmpeg can open will
# actually work; this list only decides what gets picked up when the user
# points the CLI at a directory.
AUDIO_EXTENSIONS = frozenset(
    {
        ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".oga", ".opus", ".aac",
        ".wma", ".aiff", ".aif", ".alac", ".amr", ".caf", ".mp4", ".m4b",
        ".m4v", ".mov", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".ts",
    }
)


@dataclass(frozen=True)
class Audio:
    """Decoded PCM plus the bits of metadata the rest of the program needs."""

    samples: np.ndarray  # float32, mono, in [-1, 1]
    sample_rate: int = SAMPLE_RATE

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate)


def load_audio(path: Path | str, sample_rate: int = SAMPLE_RATE) -> Audio:
    """Decode `path` to mono float32 samples at `sample_rate`."""
    path = Path(path)
    if not path.is_file():
        raise AudioDecodeError(f"no such file: {path}")

    try:
        samples = _decode_with_pyav(path, sample_rate)
    except _PyAVUnavailable:
        samples = _decode_with_ffmpeg(path, sample_rate)
    except AudioDecodeError:
        if not shutil.which("ffmpeg"):
            raise
        samples = _decode_with_ffmpeg(path, sample_rate)

    if samples.size == 0:
        raise AudioDecodeError(f"decoded no audio from {path} (empty or corrupt file?)")
    return Audio(samples=samples, sample_rate=sample_rate)


def probe_duration(path: Path | str) -> float | None:
    """Return the container's reported duration in seconds, or None.

    Used only to render a progress percentage, so a miss is harmless.
    """
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(path)) as container:
            if container.duration is not None:
                return float(container.duration) / av.time_base
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is not None and stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
    except Exception:
        return None
    return None


class _PyAVUnavailable(Exception):
    """PyAV is not installed; the caller should fall back to the ffmpeg CLI."""


def _decode_with_pyav(path: Path, sample_rate: int) -> np.ndarray:
    try:
        import av
        from av.audio.resampler import AudioResampler
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise _PyAVUnavailable(str(exc)) from exc

    resampler = AudioResampler(format="s16", layout="mono", rate=sample_rate)
    chunks: list[np.ndarray] = []
    try:
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise AudioDecodeError(f"{path} contains no audio stream")
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))
            for resampled in resampler.resample(None):  # flush
                chunks.append(resampled.to_ndarray().reshape(-1))
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError(f"could not decode {path}: {exc}") from exc

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return _to_float32(np.concatenate(chunks))


def _decode_with_ffmpeg(path: Path, sample_rate: int) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioDecodeError(
            "PyAV is not installed and ffmpeg was not found on PATH. "
            "Install the project dependencies (pip install -r requirements.txt) "
            "or install ffmpeg."
        )
    command = [
        ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(path),
        "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-",
    ]
    process = subprocess.run(command, capture_output=True)
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", "replace").strip()
        raise AudioDecodeError(f"ffmpeg failed on {path}: {detail}")
    return _to_float32(np.frombuffer(process.stdout, dtype=np.int16))


def _to_float32(pcm: np.ndarray) -> np.ndarray:
    return (pcm.astype(np.float32) / 32768.0).copy()
