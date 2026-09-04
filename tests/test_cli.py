"""End-to-end CLI tests.

Audio decoding is real (a genuine .m4a is encoded by the fixture); only the
neural network is stubbed out, so these run in milliseconds without downloading
gigabytes of weights.
"""

import json

import pytest

from transcribe import cli
from transcribe.errors import AudioDecodeError
from transcribe.transcript import Segment, Transcript


class StubEngine:
    """Replaces WhisperEngine; records what it was asked to do."""

    instances = []

    def __init__(self, options=None):
        self.options = options
        self.calls = []
        StubEngine.instances.append(self)

    def transcribe(self, audio, source, options=None, progress=None):
        self.calls.append({"source": source, "duration": audio.duration, "options": options})
        segments = [
            Segment(0, 0.0, 1.5, "This is a local transcript."),
            Segment(1, 1.5, 2.0, "Second sentence."),
        ]
        if progress is not None:
            for segment in segments:
                progress(segment, segment.end / max(audio.duration, 1e-6))
        return Transcript(
            source=source,
            segments=segments,
            language="en",
            language_probability=0.99,
            duration=audio.duration,
            model=getattr(self.options, "model", None),
            elapsed=0.01,
        )


@pytest.fixture(autouse=True)
def stub_engine(monkeypatch):
    StubEngine.instances = []
    monkeypatch.setattr(cli, "WhisperEngine", StubEngine)
    return StubEngine


def test_transcribes_an_m4a_next_to_the_input(m4a_file, capsys):
    assert cli.main([str(m4a_file)]) == 0

    output = m4a_file.with_suffix(".txt")
    assert output.exists()
    assert output.read_text() == "This is a local transcript. Second sentence.\n"


def test_multiple_formats_and_output_dir(m4a_file, tmp_path):
    out = tmp_path / "results"
    assert cli.main([str(m4a_file), "--format", "txt,srt,json", "-o", str(out)]) == 0

    assert (out / "sample.txt").exists()
    assert (out / "sample.srt").read_text().startswith("1\n00:00:00,000 --> 00:00:01,500")
    payload = json.loads((out / "sample.json").read_text())
    assert payload["language"] == "en"
    assert len(payload["segments"]) == 2


def test_stdout_flag_prints_transcript(m4a_file, capsys):
    assert cli.main([str(m4a_file), "--stdout", "-q"]) == 0
    assert capsys.readouterr().out == "This is a local transcript. Second sentence.\n"


def test_directory_input_picks_up_audio_only(make_m4a, tmp_path, monkeypatch):
    first = make_m4a("batch/one.m4a")
    second = make_m4a("batch/two.m4a")
    (tmp_path / "batch" / "notes.txt").write_text("ignore me")

    assert cli.main([str(tmp_path / "batch")]) == 0
    assert first.with_suffix(".txt").exists()
    assert second.with_suffix(".txt").exists()
    assert {call["source"] for call in StubEngine.instances[0].calls} == {str(first), str(second)}


def test_recursive_walks_subdirectories(make_m4a, tmp_path):
    nested = make_m4a("library/2024/talk.m4a")

    assert cli.main([str(tmp_path / "library")]) == 2  # nothing at the top level
    assert cli.main([str(tmp_path / "library"), "--recursive"]) == 0
    assert nested.with_suffix(".txt").exists()


def test_skip_existing_leaves_completed_work_alone(m4a_file):
    output = m4a_file.with_suffix(".txt")
    output.write_text("previous run\n")

    assert cli.main([str(m4a_file), "--skip-existing"]) == 0
    assert output.read_text() == "previous run\n"
    assert StubEngine.instances[0].calls == []


def test_reruns_overwrite_by_default(m4a_file):
    output = m4a_file.with_suffix(".txt")
    output.write_text("previous run\n")

    assert cli.main([str(m4a_file)]) == 0
    assert output.read_text().startswith("This is a local transcript.")


def test_flags_are_forwarded_to_the_decoder(m4a_file):
    cli.main([
        str(m4a_file), "--language", "de", "--translate", "--beam-size", "1",
        "--word-timestamps", "--no-vad", "--no-context", "--prompt", "Acme Corp",
        "--hotwords", "Kubernetes", "--model", "large-v3", "--device", "cpu",
    ])
    engine = StubEngine.instances[0]
    options = engine.calls[0]["options"]

    assert engine.options.model == "large-v3"
    assert engine.options.device == "cpu"
    assert options.language == "de"
    assert options.task == "translate"
    assert options.beam_size == 1
    assert options.word_timestamps is True
    assert options.vad_filter is False
    assert options.condition_on_previous_text is False
    assert options.initial_prompt == "Acme Corp"
    assert options.hotwords == "Kubernetes"


def test_unknown_format_is_rejected(m4a_file, capsys):
    assert cli.main([str(m4a_file), "--format", "docx"]) == 2
    assert "unknown format" in capsys.readouterr().err


def test_missing_input_is_reported(tmp_path, capsys):
    assert cli.main([str(tmp_path / "ghost.m4a")]) == 2
    assert "no such file" in capsys.readouterr().err


def test_one_bad_file_does_not_abort_the_batch(make_m4a, tmp_path, capsys):
    good = make_m4a("mixed/good.m4a")
    bad = tmp_path / "mixed" / "bad.m4a"
    bad.write_bytes(b"not audio")

    assert cli.main([str(tmp_path / "mixed")]) == 1
    assert good.with_suffix(".txt").exists()
    assert "bad.m4a" in capsys.readouterr().err


def test_no_arguments_prints_help(capsys):
    assert cli.main([]) == 2
    assert "usage: transcribe" in capsys.readouterr().err


def test_list_models(capsys):
    assert cli.main(["--list-models"]) == 0
    assert "large-v3-turbo" in capsys.readouterr().out


def test_quiet_suppresses_progress(m4a_file, capsys):
    assert cli.main([str(m4a_file), "-q"]) == 0
    assert capsys.readouterr().err == ""


def test_verbose_prints_segments(m4a_file, capsys):
    assert cli.main([str(m4a_file), "-v"]) == 0
    assert "This is a local transcript." in capsys.readouterr().err


def test_check_reports_the_local_setup(capsys):
    assert cli.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "faster_whisper" in out
    assert "device" in out
