# transcribe

Accurate speech-to-text for `.m4a` (and most other audio/video) files that runs
entirely on your own machine. It wraps OpenAI's Whisper models via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2), so
after the one-time model download nothing you record ever leaves your computer.

```
transcribe interview.m4a
```

→ `interview.txt` next to the original. That's the whole workflow.

## Why this setup

- **Accurate.** Whisper `large-v3-turbo` by default, with beam search,
  temperature fallback, voice-activity filtering and a hallucination guard —
  the settings that matter for real recordings, on by default.
- **Local and private.** No API keys, no uploads, no per-minute billing. Works
  on a plane once the model is cached.
- **No system dependencies.** PyAV ships the ffmpeg libraries inside its wheel,
  so `.m4a`/AAC decoding works without installing ffmpeg yourself. (If you
  already have the `ffmpeg` binary, it is used as a fallback for exotic files.)

## Install

Requires Python 3.9+.

```bash
git clone https://github.com/paulsizemore25-wq/transcribe.git
cd transcribe
./install.sh
```

That creates `.venv`, installs everything, and prints a diagnostic summary.
Then either activate the environment (`source .venv/bin/activate`) and run
`transcribe`, or call `./.venv/bin/transcribe` directly.

Manual equivalent:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
transcribe --check
```

The first transcription downloads the model (~1.5 GB for `large-v3-turbo`) into
`~/.cache/huggingface`. Every run after that is offline.

## Usage

```bash
transcribe interview.m4a                        # → interview.txt
transcribe interview.m4a -f txt,srt,json        # several formats at once
transcribe interview.m4a --model large-v3       # most accurate, slower
transcribe interview.m4a --language en          # skip language autodetect
transcribe interview.m4a --stdout | pbcopy      # straight to the clipboard
transcribe ./recordings -r -o ./transcripts     # a whole folder, recursively
transcribe lecture.m4a --word-timestamps -f json
transcribe podcast.m4a --prompt "Guests: Ana Ruiz, Kwame Osei. Topic: CRISPR."
```

### Output formats

| Format | What you get |
| --- | --- |
| `txt` | Plain running text (default) |
| `srt` | Subtitles with timecodes |
| `vtt` | WebVTT subtitles |
| `json` | Segments, timings, language, per-word timings, confidence scores |
| `tsv` | `start`/`end` (ms) and text — easy to load into a spreadsheet |
| `md` | Markdown with a header and `[mm:ss]` timecodes, nice for notes |

### Models

| Model | Size | Speed on CPU | Use it when |
| --- | --- | --- | --- |
| `tiny` / `base` | 75 MB / 145 MB | very fast | quick drafts, keyword spotting |
| `small` | 480 MB | fast | decent quality on a laptop |
| `medium` | 1.5 GB | slow | good quality without a GPU |
| **`large-v3-turbo`** (default) | 1.6 GB | moderate | best accuracy-per-second; the right default |
| `large-v3` | 3.1 GB | slowest | maximum accuracy, hard audio, many languages |
| `distil-large-v3.5` | 1.5 GB | fast | English-only, near-large quality |

`transcribe --list-models` prints them all. `--model` also accepts a path to a
local CTranslate2 model directory, for fully air-gapped machines.

### Getting the most accurate result

1. **Name the language** (`--language en`) if you know it — autodetect
   occasionally guesses wrong on the first few seconds and taints the whole run.
2. **Prime the model** with `--prompt` for proper nouns, jargon and acronyms:
   `--prompt "Speakers: Dr. Nakamura, Priya Venkatesan. Terms: GLP-1, HbA1c."`
   Use `--hotwords` to bias towards specific words more strongly.
3. **Step up to `--model large-v3`** for accents, crosstalk, or noisy rooms.
4. **If the text starts looping or repeating**, add `--no-context` — Whisper
   sometimes gets stuck echoing its own previous output.
5. **Leave VAD on.** It removes silence before decoding, which is the single
   biggest cause of invented sentences. `--no-vad` disables it if you need
   every last utterance from very quiet audio.

### Speed

- On an NVIDIA GPU: `--device cuda --batch-size 16` is dramatically faster.
- On CPU: `--beam-size 1` roughly halves the time at a small accuracy cost, and
  a smaller model helps more than any flag.
- `--threads N` pins the CPU thread count; the default uses what CTranslate2
  thinks is best.
- Apple Silicon runs on the CPU backend with int8 quantisation (CTranslate2 has
  no Metal backend); `large-v3-turbo` still runs comfortably faster than
  realtime on an M-series chip.

### Batch jobs

```bash
transcribe ./podcast-archive -r -o ./transcripts -f txt,srt --skip-existing
```

`--skip-existing` makes the run resumable: files that already have output are
left alone, so an interrupted batch can simply be re-run. One unreadable file
never aborts the batch — it is reported at the end and the exit code is `1`.

### Full option list

```
transcribe --help
transcribe --check       # versions, device, compute type, model cache location
```

## How it works

```
.m4a ─► PyAV decode ─► 16 kHz mono float32 ─► VAD ─► Whisper (CTranslate2) ─► segments ─► txt/srt/vtt/json/tsv/md
```

- `transcribe/audio.py` — decoding, PyAV first with an ffmpeg CLI fallback.
- `transcribe/engine.py` — model loading, device/compute-type selection,
  decode options, streaming of segments.
- `transcribe/transcript.py` — the backend-independent result type.
- `transcribe/formats.py` — renderers for each output format.
- `transcribe/cli.py` — argument parsing, batching, progress, exit codes.

## Development

```bash
.venv/bin/python -m pytest
```

The suite encodes a real `.m4a` on the fly, so audio decoding is exercised for
real; only the neural network is stubbed, which keeps the tests fast and
offline. `tests/test_backend_contract.py` additionally checks the arguments we
pass against faster-whisper's real signatures, so a dependency upgrade that
renames an option fails loudly instead of at 3 a.m. on a long recording.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `could not fetch the ... model` | The first run needs internet access to Hugging Face. Behind a proxy, set `HTTPS_PROXY`; or copy a model directory over and pass it to `--model`. |
| `CUDA initialisation failed` | Run with `--device cpu`, or install the cuBLAS/cuDNN runtime CTranslate2 expects. |
| Repeated/looping sentences | `--no-context`, and keep VAD enabled. |
| Wrong language detected | Pass `--language`. |
| Very slow on CPU | Smaller `--model`, `--beam-size 1`, or a GPU with `--batch-size`. |
| `decoded no audio` | The file is empty or corrupt; check it plays. |

Not included: speaker diarization ("who said what"). Whisper does not do it;
pipe the JSON output into pyannote.audio if you need speaker labels.

## License

MIT
