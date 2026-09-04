import json

import pytest

from transcribe.formats import FORMATS, render
from transcribe.transcript import Segment, Transcript, Word


@pytest.fixture
def transcript():
    return Transcript(
        source="talk.m4a",
        model="large-v3-turbo",
        language="en",
        language_probability=0.98,
        duration=3671.5,
        elapsed=42.0,
        segments=[
            Segment(0, 0.0, 3.25, "Hello there.", words=[Word(0.0, 0.4, "Hello", 0.9)]),
            Segment(1, 3.25, 3661.75, "A much later line."),
            Segment(2, 3661.75, 3665.0, "   "),  # blank segments are dropped
        ],
    )


def test_every_format_renders_non_empty(transcript):
    for fmt in FORMATS:
        assert render(transcript, fmt).strip(), fmt


def test_txt_joins_segments_and_drops_blanks(transcript):
    assert render(transcript, "txt") == "Hello there. A much later line.\n"


def test_srt_numbering_and_timestamps(transcript):
    srt = render(transcript, "srt")
    assert srt.startswith("1\n00:00:00,000 --> 00:00:03,250\nHello there.\n")
    assert "2\n00:00:03,250 --> 01:01:01,750\n" in srt
    assert "3\n" not in srt  # the blank segment is not numbered


def test_vtt_header_and_dot_separator(transcript):
    vtt = render(transcript, "vtt")
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:03.250" in vtt


def test_tsv_uses_millisecond_integers(transcript):
    rows = render(transcript, "tsv").splitlines()
    assert rows[0] == "start\tend\ttext"
    assert rows[1] == "0\t3250\tHello there."


def test_json_carries_metadata_and_words(transcript):
    payload = json.loads(render(transcript, "json"))
    assert payload["language"] == "en"
    assert payload["model"] == "large-v3-turbo"
    assert payload["text"] == "Hello there. A much later line."
    assert payload["segments"][0]["words"][0]["word"] == "Hello"


def test_markdown_has_heading_and_timecodes(transcript):
    md = render(transcript, "md")
    assert md.startswith("# Transcript — talk.m4a")
    assert "**[0:00]** Hello there." in md
    assert "**Duration:** 1:01:12" in md


def test_max_line_width_wraps_subtitles(transcript):
    srt = render(transcript, "srt", max_line_width=10)
    assert "A much\nlater\nline." in srt


def test_unknown_format_rejected(transcript):
    with pytest.raises(ValueError, match="unknown output format"):
        render(transcript, "docx")
