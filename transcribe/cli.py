"""Command-line interface: `transcribe meeting.m4a`."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .audio import AUDIO_EXTENSIONS, load_audio
from .engine import (
    DEFAULT_MODEL,
    MODELS,
    DecodeOptions,
    EngineOptions,
    WhisperEngine,
    resolve_compute_type,
    resolve_device,
)
from .errors import TranscribeError
from .formats import FORMATS, render
from .transcript import Segment, Transcript

EPILOG = """\
examples:
  transcribe interview.m4a                       # writes interview.txt next to it
  transcribe interview.m4a --format txt,srt,json # several formats at once
  transcribe interview.m4a --model large-v3      # most accurate, slowest
  transcribe interview.m4a --language en         # skip autodetect
  transcribe ./recordings --recursive -o ./out   # whole folder
  transcribe lecture.m4a --word-timestamps --format json
  transcribe notes.m4a --stdout > notes.txt      # pipe it somewhere
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Accurate, fully local speech-to-text for .m4a and other audio/video files.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="audio/video files, or directories of them")
    parser.add_argument("--version", action="version", version=f"transcribe {__version__}")

    out = parser.add_argument_group("output")
    out.add_argument(
        "-f", "--format", default="txt", metavar="LIST",
        help=f"comma-separated output formats: {', '.join(FORMATS)} (default: txt)",
    )
    out.add_argument("-o", "--output-dir", type=Path, help="write results here (default: beside each input)")
    out.add_argument("--stdout", action="store_true", help="also print the transcript to stdout")
    out.add_argument("--skip-existing", action="store_true", help="leave files that already have output alone")
    out.add_argument(
        "--max-line-width", type=int, default=0, metavar="N",
        help="wrap text/subtitles at N characters (default: no wrapping)",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "-m", "--model", default=DEFAULT_MODEL, metavar="NAME",
        help=f"model name or path to a converted model directory (default: {DEFAULT_MODEL})",
    )
    model.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="default: auto")
    model.add_argument(
        "--compute-type", default="auto", metavar="TYPE",
        help="int8, int8_float16, float16, float32, ... (default: auto — int8 on CPU, float16 on GPU)",
    )
    model.add_argument("--threads", type=int, default=0, metavar="N", help="CPU threads (default: auto)")
    model.add_argument("--model-dir", type=Path, metavar="DIR", help="where to cache downloaded models")
    model.add_argument("--offline", action="store_true", help="never touch the network; use cached models only")
    model.add_argument("--list-models", action="store_true", help="print the built-in model names and exit")

    decode = parser.add_argument_group("transcription")
    decode.add_argument("-l", "--language", metavar="CODE", help="e.g. en, es, de (default: autodetect)")
    decode.add_argument("--translate", action="store_true", help="translate the speech into English")
    decode.add_argument("--beam-size", type=int, default=5, metavar="N", help="default: 5 (1 = greedy, faster)")
    decode.add_argument("--word-timestamps", action="store_true", help="per-word timings (json/srt detail)")
    decode.add_argument("--no-vad", action="store_true", help="disable voice-activity filtering of silence")
    decode.add_argument(
        "--prompt", metavar="TEXT",
        help="context to prime the model, e.g. names and jargon it should spell correctly",
    )
    decode.add_argument("--hotwords", metavar="TEXT", help="words to bias the decoder towards")
    decode.add_argument(
        "--no-context", action="store_true",
        help="do not condition each window on the previous text (helps if it starts looping)",
    )
    decode.add_argument(
        "--batch-size", type=int, default=0, metavar="N",
        help="batch N chunks together — much faster on a GPU, needs more memory",
    )

    misc = parser.add_argument_group("other")
    misc.add_argument("-r", "--recursive", action="store_true", help="descend into subdirectories")
    misc.add_argument("-v", "--verbose", action="store_true", help="print segments as they are decoded")
    misc.add_argument("-q", "--quiet", action="store_true", help="only report errors")
    misc.add_argument("--check", action="store_true", help="report the local setup (versions, device, cache) and exit")
    return parser


def _use_utf8_streams() -> None:
    """Windows consoles default to cp1252, which cannot print most non-English
    transcripts (or the arrows below) and raises UnicodeEncodeError when the
    output is piped to a file.  UTF-8 everywhere keeps stdout matching the
    files we write."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # not a reconfigurable text stream (e.g. captured in tests)


def main(argv: list[str] | None = None) -> int:
    _use_utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        print("\n".join(MODELS))
        return 0
    if args.check:
        return _run_check(args)
    if not args.inputs:
        parser.print_help(sys.stderr)
        return 2

    try:
        formats = _parse_formats(args.format)
        files = _collect_inputs(args.inputs, recursive=args.recursive)
    except TranscribeError as exc:
        print(f"transcribe: {exc}", file=sys.stderr)
        return 2

    if not files:
        print("transcribe: no audio or video files found in the given paths", file=sys.stderr)
        return 2

    def warn(message: str) -> None:
        # Always shown, even with --quiet: it changes how long the run takes.
        print(f"transcribe: {message}", file=sys.stderr, flush=True)

    engine = WhisperEngine(
        warn=warn,
        options=EngineOptions(
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.threads,
            download_root=str(args.model_dir) if args.model_dir else None,
            local_files_only=args.offline,
        ),
    )
    decode_options = DecodeOptions(
        language=args.language,
        task="translate" if args.translate else "transcribe",
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        word_timestamps=args.word_timestamps,
        initial_prompt=args.prompt,
        hotwords=args.hotwords,
        condition_on_previous_text=not args.no_context,
        batch_size=args.batch_size,
    )

    if not args.quiet:
        device = resolve_device(args.device)
        compute = resolve_compute_type(args.compute_type, device)
        _log(f"model {args.model} on {device} ({compute}) — {len(files)} file(s)", args)

    failures = 0
    written_any = False
    for path in files:
        try:
            wrote = _process_file(path, engine, decode_options, formats, args)
            written_any = written_any or wrote
        except TranscribeError as exc:
            failures += 1
            print(f"transcribe: {path}: {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\ntranscribe: interrupted", file=sys.stderr)
            return 130

    if failures:
        print(f"transcribe: {failures} of {len(files)} file(s) failed", file=sys.stderr)
        return 1
    if not written_any and not args.stdout:
        _log("nothing to do (everything already had output; drop --skip-existing to redo it)", args)
    return 0


def _run_check(args: argparse.Namespace) -> int:
    """Print what the program can see locally — the first thing to look at
    when something is slow, missing, or picking the wrong device."""
    import os
    import platform
    import shutil

    device = resolve_device(args.device)
    lines = [
        f"transcribe {__version__}",
        f"python       {platform.python_version()} ({platform.system()} {platform.machine()})",
        f"cpu threads  {args.threads or os.cpu_count() or '?'}",
        f"device       {device} ({resolve_compute_type(args.compute_type, device)})",
        f"default model {DEFAULT_MODEL}",
    ]

    for package in ("faster_whisper", "ctranslate2", "av", "numpy"):
        try:
            module = __import__(package)
            lines.append(f"{package:<12} {getattr(module, '__version__', 'installed')}")
        except ImportError:
            lines.append(f"{package:<12} MISSING — run: pip install -r requirements.txt")

    if device == "cuda":
        from .engine import CUDA_SETUP_HINT, cuda_library_status

        ok, detail = cuda_library_status()
        lines.append(f"cuda libs    {'OK — ' if ok else 'MISSING — '}{detail}")
        if not ok:
            lines.append(f"             {CUDA_SETUP_HINT.splitlines()[0]}")
            lines.append("             " + CUDA_SETUP_HINT.splitlines()[1].strip())
            lines.append("             (transcription still works — it falls back to the CPU)")

    ffmpeg = shutil.which("ffmpeg")
    lines.append(f"ffmpeg cli   {ffmpeg or 'not found (not required — PyAV decodes audio)'}")

    cache = args.model_dir or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    lines.append(f"model cache  {cache}{'' if Path(cache).exists() else ' (empty — first run will download)'}")

    print("\n".join(lines))
    return 0


def _process_file(
    path: Path,
    engine: WhisperEngine,
    decode_options: DecodeOptions,
    formats: list[str],
    args: argparse.Namespace,
) -> bool:
    targets = {fmt: _output_path(path, fmt, args.output_dir) for fmt in formats}
    if args.skip_existing and all(target.exists() for target in targets.values()):
        _log(f"skip {path.name} (output already exists)", args)
        return False

    _log(f"→ {path.name}", args)
    started = time.monotonic()
    audio = load_audio(path)
    progress = _make_progress(path, audio.duration, args)
    transcript = engine.transcribe(audio, source=str(path), options=decode_options, progress=progress)
    _clear_progress(args)

    for fmt, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(transcript, fmt, args.max_line_width), encoding="utf-8")
        _log(f"  wrote {target}", args)

    if args.stdout:
        sys.stdout.write(render(transcript, formats[0], args.max_line_width))
        sys.stdout.flush()

    elapsed = time.monotonic() - started
    _log(f"  {_summary(transcript, audio.duration, elapsed)}", args)
    return True


def _summary(transcript: Transcript, duration: float, elapsed: float) -> str:
    speed = f"{duration / elapsed:.1f}x realtime" if elapsed > 0 else "—"
    language = transcript.language or "?"
    if transcript.language_probability is not None:
        language += f" {transcript.language_probability:.0%}"
    words = len(transcript.text.split())
    return f"{_clock(duration)} audio, {words} words, language {language}, {elapsed:.1f}s ({speed})"


def _make_progress(path: Path, duration: float, args: argparse.Namespace):
    if args.quiet:
        return None

    def report(segment: Segment, fraction: float | None) -> None:
        if args.verbose:
            print(f"  [{_clock(segment.start)} → {_clock(segment.end)}] {segment.text}", file=sys.stderr)
        elif sys.stderr.isatty():
            percent = f"{fraction * 100:5.1f}%" if fraction is not None else "  ...."
            line = f"  {percent}  {_clock(segment.end)} / {_clock(duration)}"
            print(f"\r{line}", end="", file=sys.stderr, flush=True)

    return report


def _clear_progress(args: argparse.Namespace) -> None:
    if not args.quiet and not args.verbose and sys.stderr.isatty():
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr, flush=True)


def _parse_formats(raw: str) -> list[str]:
    formats: list[str] = []
    for piece in raw.split(","):
        fmt = piece.strip().lower().lstrip(".")
        if not fmt:
            continue
        if fmt not in FORMATS:
            raise TranscribeError(f"unknown format {fmt!r}; choose from {', '.join(FORMATS)}")
        if fmt not in formats:
            formats.append(fmt)
    if not formats:
        raise TranscribeError("no output format requested")
    return formats


def _collect_inputs(inputs: list[Path], recursive: bool) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for entry in inputs:
        if entry.is_dir():
            pattern = "**/*" if recursive else "*"
            candidates = sorted(p for p in entry.glob(pattern) if p.is_file())
            found = [p for p in candidates if p.suffix.lower() in AUDIO_EXTENSIONS]
            if not found:
                print(f"transcribe: no audio files in {entry}", file=sys.stderr)
            files.extend(found)
        elif entry.is_file():
            files.append(entry)
        else:
            raise TranscribeError(f"no such file or directory: {entry}")

    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _output_path(source: Path, fmt: str, output_dir: Path | None) -> Path:
    directory = output_dir if output_dir is not None else source.parent
    return directory / f"{source.stem}.{fmt}"


def _log(message: str, args: argparse.Namespace) -> None:
    if not args.quiet:
        print(message, file=sys.stderr, flush=True)


def _clock(seconds: float) -> str:
    total = int(round(max(0.0, float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
