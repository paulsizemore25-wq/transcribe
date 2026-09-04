"""Rendering a Transcript into the file formats people actually want."""

from __future__ import annotations

import json
import textwrap

from .transcript import Transcript

FORMATS = ("txt", "srt", "vtt", "json", "tsv", "md")

# Written next to (or instead of) the wrapped text, so subtitles stay readable.
DEFAULT_MAX_LINE_WIDTH = 0  # 0 disables wrapping


def render(transcript: Transcript, fmt: str, max_line_width: int = DEFAULT_MAX_LINE_WIDTH) -> str:
    try:
        renderer = _RENDERERS[fmt]
    except KeyError:
        raise ValueError(f"unknown output format: {fmt!r} (choose from {', '.join(FORMATS)})") from None
    return renderer(transcript, max_line_width)


def _render_txt(transcript: Transcript, max_line_width: int) -> str:
    paragraphs = [segment.text.strip() for segment in transcript.segments if segment.text.strip()]
    if not paragraphs:
        return ""
    body = " ".join(paragraphs)
    if max_line_width > 0:
        body = "\n".join(textwrap.wrap(body, width=max_line_width))
    return body + "\n"


def _render_md(transcript: Transcript, max_line_width: int) -> str:
    lines = [f"# Transcript — {transcript.source}", ""]
    meta = []
    if transcript.model:
        meta.append(f"**Model:** {transcript.model}")
    if transcript.language:
        confidence = (
            f" ({transcript.language_probability:.0%} confidence)"
            if transcript.language_probability is not None
            else ""
        )
        meta.append(f"**Language:** {transcript.language}{confidence}")
    if transcript.duration:
        meta.append(f"**Duration:** {_clock(transcript.duration)}")
    if meta:
        lines.extend(["  \n".join(meta), ""])
    for segment in transcript.segments:
        text = segment.text.strip()
        if not text:
            continue
        if max_line_width > 0:
            text = "\n".join(textwrap.wrap(text, width=max_line_width))
        lines.append(f"**[{_clock(segment.start)}]** {text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_srt(transcript: Transcript, max_line_width: int) -> str:
    blocks = []
    counter = 0
    for segment in transcript.segments:
        text = _wrap(segment.text.strip(), max_line_width)
        if not text:
            continue
        counter += 1
        blocks.append(
            f"{counter}\n"
            f"{_timestamp(segment.start, ',')} --> {_timestamp(segment.end, ',')}\n"
            f"{text}\n"
        )
    return "\n".join(blocks)


def _render_vtt(transcript: Transcript, max_line_width: int) -> str:
    blocks = ["WEBVTT\n"]
    for segment in transcript.segments:
        text = _wrap(segment.text.strip(), max_line_width)
        if not text:
            continue
        blocks.append(
            f"{_timestamp(segment.start, '.')} --> {_timestamp(segment.end, '.')}\n{text}\n"
        )
    return "\n".join(blocks)


def _render_tsv(transcript: Transcript, max_line_width: int) -> str:
    rows = ["start\tend\ttext"]
    for segment in transcript.segments:
        text = segment.text.strip().replace("\t", " ")
        if not text:
            continue
        rows.append(f"{int(round(segment.start * 1000))}\t{int(round(segment.end * 1000))}\t{text}")
    return "\n".join(rows) + "\n"


def _render_json(transcript: Transcript, max_line_width: int) -> str:
    return json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2) + "\n"


def _wrap(text: str, max_line_width: int) -> str:
    if not text or max_line_width <= 0:
        return text
    return "\n".join(textwrap.wrap(text, width=max_line_width))


def _timestamp(seconds: float, decimal_marker: str) -> str:
    seconds = max(0.0, float(seconds))
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{milliseconds:03d}"


def _clock(seconds: float) -> str:
    total = int(round(max(0.0, float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


_RENDERERS = {
    "txt": _render_txt,
    "srt": _render_srt,
    "vtt": _render_vtt,
    "json": _render_json,
    "tsv": _render_tsv,
    "md": _render_md,
}
