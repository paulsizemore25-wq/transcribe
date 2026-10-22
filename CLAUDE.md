# CLAUDE.md

Local, offline speech-to-text CLI: `transcribe recording.m4a` → `recording.txt`.
Built on faster-whisper (CTranslate2). Audio and transcripts must never leave
the user's machine.

## Layout

- `transcribe/audio.py` — decode any audio/video to 16 kHz mono float32 via
  PyAV (bundles ffmpeg libs, so no system ffmpeg); ffmpeg CLI is a fallback.
- `transcribe/engine.py` — model loading, device/compute-type selection,
  decode options, GPU→CPU fallback, Windows CUDA DLL handling.
- `transcribe/transcript.py` — backend-independent result type.
- `transcribe/formats.py` — txt, srt, vtt, json, tsv, md renderers.
- `transcribe/cli.py` — argparse, batching, progress, `--check`, exit codes.
- `install.sh` (macOS/Linux), `install.ps1` + `install.bat` (Windows).

## Develop and test

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

Tests encode real .m4a/.ogg files on the fly, so decoding is exercised for
real; only the Whisper model is stubbed. `tests/test_backend_contract.py`
binds the kwargs we send against faster-whisper's real signatures — keep it
passing after any dependency upgrade.

## Limits of the cloud sandbox

- huggingface.co is blocked, so real model weights cannot be downloaded here.
  Nothing involving actual transcription quality can be verified in a cloud
  session; the user's machine is the only real test.
- No Windows and no GPU. Windows/CUDA code paths are unit-tested with
  monkeypatching only. Say so plainly when reporting such changes.
- The container is recycled between sessions; rebuild `.venv` if missing.

## The user's machine

- Windows, PowerShell, Python 3.13, NVIDIA GPU, 20 CPU threads.
- Project checked out at `D:\transcribe`, venv activated, editable install
  (`git pull` updates the tool, no reinstall needed).
- Uses `--model large-v3 --device cuda --batch-size 16`.

## Hard-won facts (don't re-learn these)

- **Windows GPU:** CTranslate2's Windows wheel bundles `cudnn64_9.dll` but not
  cuBLAS. `pip install nvidia-cublas-cu12` puts `cublas64_12.dll` and
  `cublasLt64_12.dll` in `site-packages/nvidia/cublas/bin/` (`nvidia` is a
  namespace package — no `__init__.py`).
- CTranslate2 loads cuBLAS lazily **by bare name** at the first GPU
  computation, so the failure appears mid-transcription, not at model load.
  A by-name load ignores `os.add_dll_directory` folders. The fix is to preload
  the DLLs by full path (as `ctranslate2/__init__.py` does for its own) and
  prepend the folders to PATH — see `_register_cuda_libraries`.
- `--check` must probe with `ctypes.CDLL(name, winmode=0)` (classic search
  order); Python's default search would report OK while CTranslate2 fails.
- `--device auto` falls back to CPU on CUDA errors with a warning; an explicit
  `--device cuda` fails loudly. Keep that distinction.
- Windows consoles default to cp1252: `cli._use_utf8_streams()` reconfigures
  stdout/stderr to UTF-8. Output files are always written as UTF-8.
- The batched pipeline rejects `condition_on_previous_text` and
  `hallucination_silence_threshold`; the engine strips them.
- PyAV's wheel has no libvorbis encoder: test .ogg files are made with the
  native `vorbis` encoder (`strict=experimental`). Decoding is unaffected.
- faster-whisper given a model *name* asks the Hub for the latest commit on
  every run, and a `ProxyError` there is fatal even with the model cached.
  `engine.cached_model_path` resolves the cache first; only a miss downloads.
- CTranslate2's default is 4 CPU threads regardless of the machine;
  `default_cpu_threads()` picks ~one per physical core instead.
- The batched pipeline (faster-whisper 1.2) raises "No clip timestamps found"
  without VAD on audio over 30 s, so `--no-vad` disables batching. It uses
  only the first temperature, and defaults to `without_timestamps=True`
  (one segment per ~30 s chunk); the CLI asks for timestamps unless the only
  output format is txt.
- Decoding is ~5 s per 30 min of audio here; the CLI decodes the next file
  on a daemon thread while the current one transcribes.

## Privacy

Our package imports no networking code. The only network access is
huggingface_hub's one-time model download; once a model is cached, runs make
no network requests at all. Never add telemetry, uploads, or cloud APIs.

## Status

- The Windows cuBLAS fix (commit 7e0c210) is not yet confirmed on the user's
  machine. If GPU problems come up, ask for `transcribe --check --device cuda`
  output first.
- Not implemented: speaker diarization (would need pyannote.audio).
