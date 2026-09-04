import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _write_m4a(path: Path, seconds: float = 2.0, sample_rate: int = 44100) -> Path:
    """Encode a real AAC/m4a file so the decoder is exercised for real."""
    av = pytest.importorskip("av")

    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    wave = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    pcm = (wave * 32767).astype(np.int16).reshape(1, -1)

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("aac", rate=sample_rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.rate = sample_rate
        frame.time_base = Fraction(1, sample_rate)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


@pytest.fixture
def m4a_file(tmp_path: Path) -> Path:
    return _write_m4a(tmp_path / "sample.m4a")


@pytest.fixture
def make_m4a(tmp_path: Path):
    def factory(name: str, seconds: float = 2.0) -> Path:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        return _write_m4a(target, seconds=seconds)

    return factory
