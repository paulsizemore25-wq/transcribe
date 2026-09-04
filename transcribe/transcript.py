"""The in-memory result of a transcription, independent of any backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float | None = None


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    # Whisper's own quality signals, kept so `--format json` can expose them.
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


@dataclass
class Transcript:
    source: str
    segments: list[Segment]
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None
    model: str | None = None
    task: str = "transcribe"
    elapsed: float | None = None

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model": self.model,
            "task": self.task,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "elapsed": self.elapsed,
            "text": self.text,
            "segments": [
                {
                    "id": segment.index,
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                    "compression_ratio": segment.compression_ratio,
                    "words": [
                        {
                            "start": round(word.start, 3),
                            "end": round(word.end, 3),
                            "word": word.word,
                            "probability": word.probability,
                        }
                        for word in segment.words
                    ],
                }
                for segment in self.segments
            ],
        }
