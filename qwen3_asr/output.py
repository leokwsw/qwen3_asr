"""Structured transcription result and JSON serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .subtitle import Cue, format_vtt, is_cjk_width_char


@dataclass
class WordTimestamp:
    word: str
    start_ms: int
    end_ms: int


@dataclass
class SegmentResult:
    start_ms: int
    end_ms: int
    text: str
    words: list[WordTimestamp] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    language: str
    duration_ms: int
    text: str
    segments: list[SegmentResult] = field(default_factory=list)
    vtt: str = ""

    def to_json(self) -> str:
        payload = {
            "transcription_info": {
                "language": self.language,
                "duration": round(self.duration_ms / 1000.0, 3),
            },
            "text": self.text,
            "word_count": count_words(self.text),
            "segments": [
                {
                    "start": round(seg.start_ms / 1000.0, 3),
                    "end": round(seg.end_ms / 1000.0, 3),
                    "text": seg.text,
                    "words": [
                        {
                            "word": w.word,
                            "start": round(w.start_ms / 1000.0, 3),
                            "end": round(w.end_ms / 1000.0, 3),
                        }
                        for w in seg.words
                    ],
                    "word_count": count_words(seg.text),
                }
                for seg in self.segments
            ],
            "vtt": self.vtt,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def count_words(text: str) -> int:
    count = 0
    in_non_cjk_word = False
    for ch in text:
        if is_cjk_width_char(ch):
            if in_non_cjk_word:
                count += 1
                in_non_cjk_word = False
            count += 1
        elif ch.isspace():
            if in_non_cjk_word:
                count += 1
                in_non_cjk_word = False
        else:
            in_non_cjk_word = True
    if in_non_cjk_word:
        count += 1
    return count


def result_from_segments(
    language: str,
    duration_ms: int,
    segments: list[SegmentResult],
    cues: list[Cue] | None = None,
) -> TranscriptionResult:
    text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
    vtt = format_vtt(cues if cues is not None else [Cue(s.start_ms, s.end_ms, s.text) for s in segments])
    return TranscriptionResult(
        language=language or "",
        duration_ms=duration_ms,
        text=text,
        segments=segments,
        vtt=vtt,
    )
