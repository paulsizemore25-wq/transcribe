import numpy as np
import pytest

from transcribe.audio import SAMPLE_RATE, load_audio, probe_duration
from transcribe.errors import AudioDecodeError


def test_decodes_m4a_to_mono_float32_at_16k(m4a_file):
    audio = load_audio(m4a_file)

    assert audio.sample_rate == SAMPLE_RATE
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    assert audio.duration == pytest.approx(2.0, abs=0.2)
    assert float(np.abs(audio.samples).max()) <= 1.0
    assert float(np.abs(audio.samples).max()) > 0.05  # actually carries signal


def test_ffmpeg_fallback_matches_pyav(m4a_file):
    import shutil

    from transcribe.audio import _decode_with_ffmpeg

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg CLI not installed")
    pyav = load_audio(m4a_file).samples
    ffmpeg = _decode_with_ffmpeg(m4a_file, SAMPLE_RATE)

    assert ffmpeg.dtype == np.float32 and ffmpeg.flags.writeable
    assert len(ffmpeg) == pytest.approx(len(pyav), abs=SAMPLE_RATE // 10)
    assert float(np.abs(ffmpeg).max()) == pytest.approx(float(np.abs(pyav).max()), abs=0.05)


def test_probe_duration_matches_decoded_length(m4a_file):
    assert probe_duration(m4a_file) == pytest.approx(load_audio(m4a_file).duration, abs=0.2)


def test_missing_file_is_a_clean_error(tmp_path):
    with pytest.raises(AudioDecodeError, match="no such file"):
        load_audio(tmp_path / "nope.m4a")


def test_garbage_file_is_a_clean_error(tmp_path):
    broken = tmp_path / "broken.m4a"
    broken.write_bytes(b"this is not an mp4 container")
    with pytest.raises(AudioDecodeError):
        load_audio(broken)
