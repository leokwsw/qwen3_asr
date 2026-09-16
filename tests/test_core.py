from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen3_asr.audio import parse_wav_buffer, resample, skip_silence, split_segments, load_audio_from_bytes
from qwen3_asr.download import resolve_model_id
from qwen3_asr.engine import _delta_text, _max_new_tokens
from qwen3_asr.languages import language_to_iso639, normalize_language
from qwen3_asr.output import TranscriptionResult, WordTimestamp, count_words
from qwen3_asr.subtitle import Cue, format_srt, format_vtt, group_words_to_cues


def _pcm16_wav(samples: np.ndarray, sr: int = 16000) -> bytes:
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    data = pcm.tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    dat = b"data" + struct.pack("<I", len(data)) + data
    return header + fmt + dat


def test_language_aliases() -> None:
    assert normalize_language("en") == "English"
    assert normalize_language("zh") == "Chinese"
    assert normalize_language("chinese") == "Chinese"
    assert language_to_iso639("English") == "en"
    assert language_to_iso639("Cantonese") == "yue"
    assert normalize_language("not-a-lang") is None


def test_count_words_cjk_and_latin() -> None:
    assert count_words("Hello world") == 2
    assert count_words("你好世界") == 4
    assert count_words("Hello 世界") == 3


def test_subtitle_roundtrip() -> None:
    words = [
        {"text": "Hello", "start_ms": 0, "end_ms": 400},
        {"text": "world.", "start_ms": 420, "end_ms": 900},
    ]
    cues = group_words_to_cues(words, 2000)
    assert cues
    srt = format_srt(cues)
    vtt = format_vtt(cues)
    assert "Hello world." in srt
    assert srt.splitlines()[1].count(",") == 2
    assert vtt.startswith("WEBVTT")
    assert "." in vtt.splitlines()[3]


def test_json_output_shape() -> None:
    result = TranscriptionResult(
        language="en",
        duration_ms=1234,
        text="Hello",
        segments=[],
        vtt="WEBVTT\n\n",
    )
    payload = result.to_json()
    assert '"language": "en"' in payload
    assert '"duration": 1.234' in payload


def test_parse_wav_and_resample() -> None:
    tone = np.sin(np.linspace(0, 8 * np.pi, 8000)).astype(np.float32) * 0.2
    wav = _pcm16_wav(tone, sr=8000)
    samples = parse_wav_buffer(wav)
    assert samples is not None
    assert abs(len(samples) - 16000) < 2
    up = resample(tone, 8000, 16000)
    assert len(up) == 16000
    from_bytes = load_audio_from_bytes(wav, "tone.wav")
    assert from_bytes is not None
    assert abs(len(from_bytes) - 16000) < 2


def test_skip_silence_and_segments() -> None:
    sr = 16000
    speech = np.ones(sr, dtype=np.float32) * 0.2
    silence = np.zeros(sr, dtype=np.float32)
    mixed = np.concatenate([speech, silence, speech])
    compacted = skip_silence(mixed, sample_rate=sr, min_silence_s=0.4)
    assert compacted.size < mixed.size
    spans = split_segments(mixed, target_sec=1.0, search_sec=0.4, sample_rate=sr)
    assert len(spans) >= 2
    assert spans[0][0] == 0
    assert spans[-1][1] == mixed.size


def test_model_alias_and_delta_text() -> None:
    assert resolve_model_id("qwen3-asr-0.6b") == "Qwen/Qwen3-ASR-0.6B-hf"
    assert _delta_text("", "hello") == "hello "
    assert _delta_text("hello", "hello world").strip() == "world"
    assert _max_new_tokens(1.0) >= 64
    _ = WordTimestamp("hi", 0, 100)
    _ = Cue(0, 100, "hi")
