import numpy as np
import pytest

from transcribe.audio import Audio
from transcribe.engine import (
    DecodeOptions,
    EngineOptions,
    WhisperEngine,
    resolve_compute_type,
    resolve_device,
)
from transcribe.errors import EngineError


class FakeWord:
    def __init__(self, start, end, word, probability=0.9):
        self.start, self.end, self.word, self.probability = start, end, word, probability


class FakeSegment:
    def __init__(self, start, end, text, words=None):
        self.start, self.end, self.text, self.words = start, end, text, words or []
        self.avg_logprob, self.no_speech_prob, self.compression_ratio = -0.2, 0.01, 1.3


class FakeInfo:
    language = "en"
    language_probability = 0.97
    duration = 4.0


class FakeModel:
    """Stands in for faster_whisper.WhisperModel; records how it was called."""

    def __init__(self, segments=None):
        self.segments = segments or [
            FakeSegment(0.0, 2.0, " Hello there. ", [FakeWord(0.0, 0.5, "Hello")]),
            FakeSegment(2.0, 4.0, " Second line."),
        ]
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter(self.segments), FakeInfo()


@pytest.fixture
def audio():
    return Audio(samples=np.zeros(16000 * 4, dtype=np.float32))


def engine_with(fake_model, **options):
    engine = WhisperEngine(EngineOptions(**options))
    engine._model = fake_model
    return engine


def test_compute_type_defaults_per_device():
    assert resolve_compute_type("auto", "cpu") == "int8"
    assert resolve_compute_type("auto", "cuda") == "float16"
    assert resolve_compute_type("float32", "cuda") == "float32"


def test_resolve_device_honours_explicit_choice():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}


def test_transcribe_maps_segments_and_metadata(audio):
    model = FakeModel()
    transcript = engine_with(model, model="large-v3-turbo").transcribe(audio, source="a.m4a")

    assert transcript.text == "Hello there. Second line."
    assert transcript.language == "en"
    assert transcript.language_probability == 0.97
    assert transcript.model == "large-v3-turbo"
    assert transcript.elapsed is not None
    assert [segment.index for segment in transcript.segments] == [0, 1]
    assert transcript.segments[0].text == "Hello there."  # whitespace stripped
    assert transcript.segments[0].words[0].word == "Hello"
    assert transcript.segments[0].avg_logprob == -0.2


def test_decode_options_reach_the_backend(audio):
    model = FakeModel()
    options = DecodeOptions(
        language="es", task="translate", beam_size=3, word_timestamps=True,
        initial_prompt="Kubernetes", hotwords="Anthropic", vad_filter=False,
    )
    engine_with(model).transcribe(audio, source="a.m4a", options=options)

    _, kwargs = model.calls[0]
    assert kwargs["language"] == "es"
    assert kwargs["task"] == "translate"
    assert kwargs["beam_size"] == 3
    assert kwargs["word_timestamps"] is True
    assert kwargs["initial_prompt"] == "Kubernetes"
    assert kwargs["hotwords"] == "Anthropic"
    assert kwargs["vad_filter"] is False
    # temperature fallback keeps decoding from getting stuck
    assert kwargs["temperature"][0] == 0.0 and len(kwargs["temperature"]) > 1


def test_vad_parameters_passed_when_enabled(audio):
    model = FakeModel()
    engine_with(model).transcribe(audio, source="a.m4a", options=DecodeOptions(vad_min_silence_ms=250))
    _, kwargs = model.calls[0]
    assert kwargs["vad_filter"] is True
    assert kwargs["vad_parameters"] == {"min_silence_duration_ms": 250}


def test_hallucination_guard_only_with_word_timestamps(audio):
    model = FakeModel()
    engine = engine_with(model)
    engine.transcribe(audio, source="a.m4a", options=DecodeOptions(word_timestamps=False))
    assert "hallucination_silence_threshold" not in model.calls[0][1]

    engine.transcribe(audio, source="a.m4a", options=DecodeOptions(word_timestamps=True))
    assert model.calls[1][1]["hallucination_silence_threshold"] == 2.0


def test_progress_callback_reports_fractions(audio):
    seen = []
    engine_with(FakeModel()).transcribe(
        audio, source="a.m4a", progress=lambda segment, fraction: seen.append((segment.index, fraction))
    )
    assert [index for index, _ in seen] == [0, 1]
    assert seen[0][1] == pytest.approx(0.5)
    assert seen[1][1] == pytest.approx(1.0)


def test_backend_exception_becomes_engine_error(audio):
    class Boom(FakeModel):
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("cuda out of memory")

    with pytest.raises(EngineError, match="transcription failed for a.m4a"):
        engine_with(Boom()).transcribe(audio, source="a.m4a")


def test_load_failure_message_mentions_the_download(monkeypatch):
    engine = WhisperEngine(EngineOptions(model="tiny", local_files_only=True))
    with pytest.raises(EngineError) as excinfo:
        engine.load()
    assert "tiny" in str(excinfo.value)


class CudaBrokenModel(FakeModel):
    """Reproduces the Windows failure: CTranslate2 finds the GPU, but the CUDA
    math libraries are missing, and it only says so once decoding starts."""

    def transcribe(self, audio, **kwargs):
        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")


def test_auto_device_falls_back_to_cpu_when_cuda_is_broken(audio, monkeypatch):
    warnings = []
    engine = WhisperEngine(EngineOptions(device="auto"), warn=warnings.append)
    engine.device = "cuda"  # pretend a GPU was detected
    engine.compute_type = "float16"

    models = [CudaBrokenModel(), FakeModel()]
    monkeypatch.setattr(engine, "_create_model", lambda: models.pop(0))
    engine._model = models.pop(0)  # the broken one is already "loaded"

    transcript = engine.transcribe(audio, source="a.m4a")

    assert transcript.text == "Hello there. Second line."
    assert engine.device == "cpu"
    assert engine.compute_type == "int8"
    assert warnings and "falling back to CPU" in warnings[0]
    assert "nvidia-cublas-cu12" in warnings[0]  # tells the user how to fix it


def test_explicit_cuda_request_is_not_silently_downgraded(audio):
    engine = WhisperEngine(EngineOptions(device="cuda"))
    engine._model = CudaBrokenModel()

    with pytest.raises(EngineError) as excinfo:
        engine.transcribe(audio, source="a.m4a")
    assert "cublas64_12.dll" in str(excinfo.value)
    assert engine.device == "cuda"  # unchanged; the user asked for it explicitly


def test_non_cuda_errors_do_not_trigger_a_retry(audio, monkeypatch):
    engine = WhisperEngine(EngineOptions(device="auto"))
    engine.device = "cuda"

    class Broken(FakeModel):
        def transcribe(self, audio, **kwargs):
            raise ValueError("invalid audio")

    engine._model = Broken()
    monkeypatch.setattr(engine, "_create_model", lambda: pytest.fail("must not reload"))

    with pytest.raises(EngineError, match="invalid audio"):
        engine.transcribe(audio, source="a.m4a")


def test_load_failure_falls_back_too(audio, monkeypatch):
    engine = WhisperEngine(EngineOptions(device="auto"), warn=lambda message: None)
    engine.device = "cuda"
    attempts = []

    def create():
        attempts.append(engine.device)
        if engine.device == "cuda":
            raise RuntimeError("CUDA driver version is insufficient")
        return FakeModel()

    monkeypatch.setattr(engine, "_create_model", create)
    assert engine.load() is not None
    assert attempts == ["cuda", "cpu"]


def test_cuda_failure_detection():
    from transcribe.engine import _is_cuda_failure

    assert _is_cuda_failure(RuntimeError("Library cublas64_12.dll is not found"))
    assert _is_cuda_failure(RuntimeError("libcudnn_ops.so.9: cannot open shared object file"))
    assert _is_cuda_failure(RuntimeError("CUDA failed with error out of memory"))
    assert not _is_cuda_failure(RuntimeError("model file is corrupt"))
