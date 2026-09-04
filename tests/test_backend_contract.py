"""Contract tests against the real faster-whisper API.

The model weights cannot be downloaded in CI, but the argument names we send
can still be checked against the real signatures — that is where a silent
breakage after a dependency upgrade would show up first.
"""

import inspect

import numpy as np
import pytest

from transcribe.audio import Audio
from transcribe.engine import DecodeOptions, MODELS, WhisperEngine

faster_whisper = pytest.importorskip("faster_whisper")


class RecordingModel:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        return iter(()), type("Info", (), {"language": "en", "language_probability": 1.0, "duration": 1.0})()


def _kwargs_for(options: DecodeOptions) -> dict:
    engine = WhisperEngine()
    model = RecordingModel()
    engine._model = model
    if options.batch_size and options.batch_size > 1:
        engine._batched = model  # skip constructing the real pipeline
    engine.transcribe(Audio(np.zeros(16000, dtype=np.float32)), source="x.m4a", options=options)
    return model.kwargs


@pytest.mark.parametrize(
    "options",
    [
        DecodeOptions(),
        DecodeOptions(word_timestamps=True, hotwords="Anthropic", language="en", task="translate"),
        DecodeOptions(vad_filter=False, condition_on_previous_text=False, beam_size=1),
    ],
)
def test_kwargs_bind_to_the_real_transcribe_signature(options):
    signature = inspect.signature(faster_whisper.WhisperModel.transcribe)
    signature.bind(None, None, **_kwargs_for(options))  # self, audio, **ours


def test_batched_kwargs_bind_to_the_real_pipeline_signature():
    kwargs = _kwargs_for(DecodeOptions(batch_size=8))
    signature = inspect.signature(faster_whisper.BatchedInferencePipeline.transcribe)
    signature.bind(None, None, **kwargs)
    assert kwargs["batch_size"] == 8
    # the batched pipeline rejects these, so the engine must strip them
    assert "condition_on_previous_text" not in kwargs
    assert "hallucination_silence_threshold" not in kwargs


def test_advertised_models_exist_in_the_backend():
    from faster_whisper.utils import _MODELS

    assert set(MODELS) <= set(_MODELS)
