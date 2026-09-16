from __future__ import annotations

import struct
from typing import Any

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from qwen3_asr.engine import Timing
from qwen3_asr.output import SegmentResult, TranscriptionResult, WordTimestamp, result_from_segments
from qwen3_asr.server import create_app


def _pcm16_wav(samples: np.ndarray, sr: int = 16000) -> bytes:
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    data = pcm.tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    dat = b"data" + struct.pack("<I", len(data)) + data
    return header + fmt + dat


class FakeEngine:
    def __init__(self, *, aligner: bool = False) -> None:
        self.model_id = "Qwen/Qwen3-ASR-0.6B-hf"
        self.aligner_id = "Qwen/Qwen3-ForcedAligner-0.6B-hf" if aligner else None
        self.aligner_model = object() if aligner else None
        self.max_new_tokens = None
        self.timing = Timing(total_ms=12.5)
        self.last_kwargs: dict[str, Any] | None = None

    def transcribe_samples(self, samples: np.ndarray, **kwargs: Any) -> str | TranscriptionResult:
        self.last_kwargs = kwargs
        language = kwargs.get("language") or "English"
        duration_ms = max(1, int(len(samples) / 16))
        words = (
            [WordTimestamp("hello", 0, 400), WordTimestamp("world", 420, 900)]
            if self.aligner_model
            else []
        )
        seg = SegmentResult(0, duration_ms, "hello world", words=words)
        if kwargs.get("return_result"):
            return result_from_segments(str(language), duration_ms, [seg])
        return "hello world"

    def align_samples(self, samples: np.ndarray, transcript: str, language: str) -> list[WordTimestamp]:
        return [WordTimestamp("hello", 0, 400), WordTimestamp("world", 420, 900)]


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app(FakeEngine())) as test_client:
        yield test_client


@pytest.fixture
def align_client() -> TestClient:
    with TestClient(create_app(FakeEngine(aligner=True))) as test_client:
        yield test_client


def test_docs_and_openapi(client: TestClient) -> None:
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert b"swagger" in docs.text.lower().encode() or "Swagger UI" in docs.text or "swagger-ui" in docs.text
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    paths = spec.json()["paths"]
    assert "/v1/transcribe" in paths
    assert "/v1/align" in paths
    assert "/v1/audio/transcriptions" in paths
    assert "/v1/audio/translations" in paths
    root = client.get("/", follow_redirects=False)
    assert root.status_code in {307, 302}
    assert root.headers["location"].endswith("/docs")


def test_health_and_metadata(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["device"] == "cpu"
    assert body["model_id"] == "Qwen/Qwen3-ASR-0.6B-hf"

    languages = client.get("/v1/languages")
    assert languages.status_code == 200
    codes = {item["code"] for item in languages.json()}
    assert {"en", "zh", "ja"} <= codes

    models = client.get("/v1/models")
    body = models.json()
    assert body["object"] == "list"
    ids = {item["id"] for item in body["data"]}
    assert "whisper-1" in ids
    assert "qwen3-asr-0.6b" in ids
    retrieved = client.get("/v1/models/whisper-1")
    assert retrieved.status_code == 200
    assert retrieved.json()["id"] == "whisper-1"
    assert retrieved.json()["object"] == "model"
    alias = client.get("/models")
    assert alias.status_code == 200
    assert alias.json()["object"] == "list"


def test_transcribe_upload(client: TestClient) -> None:
    tone = np.sin(np.linspace(0, 8 * np.pi, 16000)).astype(np.float32) * 0.2
    wav = _pcm16_wav(tone)
    response = client.post(
        "/v1/transcribe",
        files={"file": ("tone.wav", wav, "audio/wav")},
        data={"language": "en", "include_srt": "true", "skip_silence": "true"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"] == "hello world"
    assert body["transcription_info"]["language"] == "English"
    assert body["word_count"] == 2
    assert body["srt"]
    assert "hello world" in body["srt"]
    assert body["inference_ms"] == 12.5


def test_transcribe_rejects_bad_language(client: TestClient) -> None:
    wav = _pcm16_wav(np.zeros(1600, dtype=np.float32))
    response = client.post(
        "/v1/transcribe",
        files={"file": ("tone.wav", wav, "audio/wav")},
        data={"language": "klingon"},
    )
    assert response.status_code == 400
    assert "Unsupported language" in response.json()["detail"]


def test_transcribe_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/v1/transcribe",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400


def test_align_requires_aligner(client: TestClient) -> None:
    wav = _pcm16_wav(np.zeros(1600, dtype=np.float32))
    response = client.post(
        "/v1/align",
        files={"file": ("tone.wav", wav, "audio/wav")},
        data={"transcript": "hello world", "language": "en"},
    )
    assert response.status_code == 400
    assert "Aligner is not loaded" in response.json()["detail"]


def test_align_with_aligner(align_client: TestClient) -> None:
    wav = _pcm16_wav(np.zeros(1600, dtype=np.float32))
    response = align_client.post(
        "/v1/align",
        files={"file": ("tone.wav", wav, "audio/wav")},
        data={"transcript": "hello world", "language": "en"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["language"] == "English"
    assert body["words"][0]["text"] == "hello"
    assert body["words"][0]["start"] == 0.0
    assert body["words"][0]["end"] == 0.4


def _wav() -> bytes:
    tone = np.sin(np.linspace(0, 8 * np.pi, 16000)).astype(np.float32) * 0.2
    return _pcm16_wav(tone)


def test_openai_transcriptions_json(client: TestClient) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("tone.wav", _wav(), "audio/wav")},
        data={"model": "whisper-1", "language": "en"},
        headers={"Authorization": "Bearer sk-not-needed"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {"text": "hello world"}


def test_openai_transcriptions_text_srt_vtt(client: TestClient) -> None:
    wav = _wav()
    text = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("tone.wav", wav, "audio/wav")},
        data={"model": "whisper-1", "response_format": "text"},
    )
    assert text.status_code == 200
    assert text.text == "hello world"

    srt = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("tone.wav", wav, "audio/wav")},
        data={"model": "whisper-1", "response_format": "srt"},
    )
    assert srt.status_code == 200
    assert "hello world" in srt.text

    vtt = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("tone.wav", wav, "audio/wav")},
        data={"model": "whisper-1", "response_format": "vtt"},
    )
    assert vtt.status_code == 200
    assert vtt.text.startswith("WEBVTT")


def test_openai_transcriptions_verbose_json_and_words(align_client: TestClient) -> None:
    response = align_client.post(
        "/v1/audio/transcriptions",
        files={"file": ("tone.wav", _wav(), "audio/wav")},
        data={
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task"] == "transcribe"
    assert body["language"] == "english"
    assert body["text"] == "hello world"
    assert body["segments"]
    assert body["words"][0]["word"] == "hello"
    assert body["words"][0]["start"] == 0.0


def test_openai_transcriptions_alias_path_and_errors(client: TestClient) -> None:
    ok = client.post(
        "/audio/transcriptions",
        files={"file": ("tone.wav", _wav(), "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert ok.status_code == 200
    assert ok.json()["text"] == "hello world"

    bad_format = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("tone.wav", _wav(), "audio/wav")},
        data={"model": "whisper-1", "response_format": "pdf"},
    )
    assert bad_format.status_code == 400
    assert bad_format.json()["error"]["type"] == "invalid_request_error"

    streamed = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("tone.wav", _wav(), "audio/wav")},
        data={"model": "whisper-1", "stream": "true"},
    )
    assert streamed.status_code == 400
    assert "stream" in streamed.json()["error"]["message"]


def test_openai_translations_into_english() -> None:
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        response = client.post(
            "/v1/audio/translations",
            files={"file": ("tone.wav", _wav(), "audio/wav")},
            data={"model": "whisper-1", "response_format": "verbose_json"},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task"] == "translate"
    assert body["language"] == "english"
    assert body["text"] == "hello world"
    assert engine.last_kwargs is not None
    assert engine.last_kwargs["language"] == "English"
