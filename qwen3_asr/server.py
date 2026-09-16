"""FastAPI HTTP service for Qwen3-ASR (Swagger UI at /docs)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from . import audio
from .download import KNOWN_MODELS
from .engine import AsrEngine, configure_cpu_threads
from .languages import LANGUAGE_TABLE, normalize_language, supported_language_list
from .output import TranscriptionResult, WordTimestamp, count_words
from .subtitle import format_srt, group_words_to_cues, segment_to_cue


def _package_version() -> str:
    try:
        return pkg_version("qwen3-asr-cpu")
    except PackageNotFoundError:
        return "0.1.2"


class WordOut(BaseModel):
    word: str
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str
    words: list[WordOut] = Field(default_factory=list)
    word_count: int = 0


class TranscriptionInfo(BaseModel):
    language: str
    duration: float = Field(description="Audio duration in seconds")


class TranscribeResponse(BaseModel):
    transcription_info: TranscriptionInfo
    text: str
    word_count: int
    segments: list[SegmentOut]
    vtt: str = ""
    srt: str | None = None
    inference_ms: float | None = Field(default=None, description="Inference time in milliseconds")


class AlignWordOut(BaseModel):
    text: str
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")


class AlignResponse(BaseModel):
    language: str
    transcript: str
    words: list[AlignWordOut]


class LanguageOut(BaseModel):
    name: str
    code: str


class ModelOut(BaseModel):
    name: str
    repo: str
    description: str


class HealthResponse(BaseModel):
    status: str
    ready: bool
    device: str = "cpu"
    model_id: str | None = None
    aligner_id: str | None = None
    version: str


class InfoResponse(HealthResponse):
    languages: list[LanguageOut]
    models: list[ModelOut]
    threads: int | None = None


def _result_to_response(result: TranscriptionResult, *, include_srt: bool, inference_ms: float | None) -> TranscribeResponse:
    if any(seg.words for seg in result.segments):
        cues = group_words_to_cues(
            [
                {"text": w.word, "start_ms": w.start_ms, "end_ms": w.end_ms}
                for seg in result.segments
                for w in seg.words
            ],
            result.duration_ms,
        )
    else:
        cues = [segment_to_cue(s.start_ms, s.end_ms, s.text) for s in result.segments]
    return TranscribeResponse(
        transcription_info=TranscriptionInfo(
            language=result.language,
            duration=round(result.duration_ms / 1000.0, 3),
        ),
        text=result.text,
        word_count=count_words(result.text),
        segments=[
            SegmentOut(
                start=round(seg.start_ms / 1000.0, 3),
                end=round(seg.end_ms / 1000.0, 3),
                text=seg.text,
                words=[
                    WordOut(
                        word=w.word,
                        start=round(w.start_ms / 1000.0, 3),
                        end=round(w.end_ms / 1000.0, 3),
                    )
                    for w in seg.words
                ],
                word_count=count_words(seg.text),
            )
            for seg in result.segments
        ],
        vtt=result.vtt,
        srt=format_srt(cues) if include_srt else None,
        inference_ms=round(inference_ms, 1) if inference_ms is not None else None,
    )


class EngineState:
    def __init__(self, engine: AsrEngine) -> None:
        self.engine = engine
        self.lock = threading.Lock()
        self.max_upload_bytes = int(os.environ.get("QWEN3_ASR_MAX_UPLOAD_MB", "100")) * 1024 * 1024


def _require_engine(app: FastAPI) -> EngineState:
    state: EngineState | None = getattr(app.state, "asr", None)
    if state is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ASR engine is not loaded")
    return state


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit",
        )
    return data


def _decode_audio(data: bytes, filename: str | None) -> Any:
    samples = audio.load_audio_from_bytes(data, filename or "audio.wav")
    if samples is None or getattr(samples, "size", 0) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode audio. Upload WAV/MP3/FLAC/OGG or a common video container.",
        )
    return samples


def _parse_language(language: str | None) -> str | None:
    if not language:
        return None
    lang = normalize_language(language)
    if lang is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language: {language}. Supported: {supported_language_list()}",
        )
    return lang


def create_app(engine: AsrEngine) -> FastAPI:
    """Build a FastAPI app around a preloaded CPU ASR engine."""

    engine_state = EngineState(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.asr = engine_state
        yield

    app = FastAPI(
        title="qwen3-asr",
        summary="CPU-only Qwen3-ASR speech-to-text HTTP API",
        description=(
            "Upload audio and get transcriptions from Qwen3-ASR running on CPU.\n\n"
            "- Interactive docs: `/docs` (Swagger UI) and `/redoc`\n"
            "- Model is loaded once at process start and reused for every request\n"
            "- Inference is serialized (one transcription at a time) to avoid CPU oversubscription"
        ),
        version=_package_version(),
        lifespan=lifespan,
        contact={"name": "qwen3-asr"},
        license_info={"name": "MIT"},
        openapi_tags=[
            {"name": "service", "description": "Health and model metadata"},
            {"name": "asr", "description": "Speech-to-text and forced alignment"},
        ],
    )
    app.state.asr = engine_state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("QWEN3_ASR_CORS_ORIGINS", "*").split(","),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        state = _require_engine(app)
        return HealthResponse(
            status="ok",
            ready=True,
            device="cpu",
            model_id=state.engine.model_id,
            aligner_id=state.engine.aligner_id,
            version=_package_version(),
        )

    @app.get("/v1/info", response_model=InfoResponse, tags=["service"])
    def info() -> InfoResponse:
        state = _require_engine(app)
        try:
            import torch

            threads = int(torch.get_num_threads())
        except Exception:
            threads = None
        return InfoResponse(
            status="ok",
            ready=True,
            device="cpu",
            model_id=state.engine.model_id,
            aligner_id=state.engine.aligner_id,
            version=_package_version(),
            threads=threads,
            languages=[LanguageOut(name=name, code=code) for name, code in LANGUAGE_TABLE],
            models=[
                ModelOut(name=m.name, repo=m.repo, description=m.description) for m in KNOWN_MODELS
            ],
        )

    @app.get("/v1/languages", response_model=list[LanguageOut], tags=["service"])
    def languages() -> list[LanguageOut]:
        return [LanguageOut(name=name, code=code) for name, code in LANGUAGE_TABLE]

    @app.get("/v1/models", response_model=list[ModelOut], tags=["service"])
    def models() -> list[ModelOut]:
        return [ModelOut(name=m.name, repo=m.repo, description=m.description) for m in KNOWN_MODELS]

    @app.post(
        "/v1/transcribe",
        response_model=TranscribeResponse,
        response_model_exclude_none=True,
        tags=["asr"],
        summary="Transcribe an uploaded audio or video file",
    )
    async def transcribe(
        file: Annotated[UploadFile, File(description="Audio or video file (WAV, MP3, FLAC, MP4, ...)")],
        language: Annotated[
            str | None,
            Form(description="Force language (en, zh, Japanese, ...). Omit to auto-detect."),
        ] = None,
        prompt: Annotated[str | None, Form(description="Optional system prompt for biasing")] = None,
        segment_sec: Annotated[float, Form(description="Segment target seconds (0 = full file)")] = 0.0,
        search_sec: Annotated[float, Form(description="Silence search window ± seconds")] = 3.0,
        stream: Annotated[bool, Form(description="Process audio in streaming chunks")] = False,
        stream_chunk_sec: Annotated[float, Form(description="Chunk size when stream=true")] = 2.0,
        skip_silence: Annotated[bool, Form(description="Drop long silent spans before inference")] = False,
        include_srt: Annotated[bool, Form(description="Include SRT subtitle text in the response")] = False,
        max_new_tokens: Annotated[int, Form(description="Max generated tokens (0 = auto)")] = 0,
    ) -> TranscribeResponse:
        state = _require_engine(app)
        lang = _parse_language(language)
        data = await _read_upload(file, state.max_upload_bytes)
        samples = await asyncio.to_thread(_decode_audio, data, file.filename)

        def _run() -> TranscriptionResult:
            with state.lock:
                previous = state.engine.max_new_tokens
                if max_new_tokens > 0:
                    state.engine.max_new_tokens = max_new_tokens
                try:
                    result = state.engine.transcribe_samples(
                        samples,
                        language=lang,
                        prompt=prompt,
                        stream=stream,
                        stream_chunk_sec=stream_chunk_sec,
                        segment_sec=segment_sec,
                        search_sec=search_sec,
                        skip_silence=skip_silence,
                        return_result=True,
                    )
                finally:
                    state.engine.max_new_tokens = previous
            assert isinstance(result, TranscriptionResult)
            return result

        try:
            result = await asyncio.to_thread(_run)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        inference_ms = getattr(getattr(state.engine, "timing", None), "total_ms", None)
        return _result_to_response(result, include_srt=include_srt, inference_ms=inference_ms)

    @app.post(
        "/v1/align",
        response_model=AlignResponse,
        tags=["asr"],
        summary="Force-align a transcript to audio (word timestamps)",
    )
    async def align(
        file: Annotated[UploadFile, File(description="Audio or video file")],
        transcript: Annotated[str, Form(description="Transcript to align")],
        language: Annotated[str, Form(description="Language used for word splitting")] = "English",
    ) -> AlignResponse:
        state = _require_engine(app)
        if getattr(state.engine, "aligner_model", None) is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Aligner is not loaded. Start the server with --aligner-dir qwen3-aligner-0.6b",
            )
        text = transcript.strip()
        if not text:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="transcript is required")
        lang = _parse_language(language) or language
        data = await _read_upload(file, state.max_upload_bytes)
        samples = await asyncio.to_thread(_decode_audio, data, file.filename)

        def _run() -> list[WordTimestamp]:
            with state.lock:
                return state.engine.align_samples(samples, text, lang)

        try:
            words = await asyncio.to_thread(_run)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        return AlignResponse(
            language=lang,
            transcript=text,
            words=[
                AlignWordOut(text=w.word, start=round(w.start_ms / 1000.0, 3), end=round(w.end_ms / 1000.0, 3))
                for w in words
            ],
        )

    return app


def handle_serve_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="qwen3-asr serve",
        description="Start the FastAPI HTTP service (Swagger UI at /docs)",
    )
    parser.add_argument(
        "-d",
        "--model-dir",
        dest="model_dir",
        default=os.environ.get("QWEN3_ASR_MODEL"),
        help="Model directory or alias (or QWEN3_ASR_MODEL)",
    )
    parser.add_argument(
        "--aligner-dir",
        default=os.environ.get("QWEN3_ASR_ALIGNER"),
        help="Optional ForcedAligner model for /v1/align and word timestamps",
    )
    parser.add_argument("--host", default=os.environ.get("QWEN3_ASR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("QWEN3_ASR_PORT", "8000")))
    parser.add_argument("-t", type=int, default=0, dest="threads", help="CPU thread count (default: all cores)")
    args = parser.parse_args(argv)

    if not args.model_dir:
        parser.print_help()
        print("\nError: -d / --model-dir (or QWEN3_ASR_MODEL) is required", file=sys.stderr)
        return 1

    n_threads = configure_cpu_threads(args.threads)
    print(f"CPU threads: {n_threads}", file=sys.stderr)
    try:
        engine = AsrEngine.load(
            args.model_dir,
            aligner_dir=args.aligner_dir,
            n_threads=n_threads,
            verbose=1,
        )
    except Exception as exc:
        print(f"Failed to load model from {args.model_dir}: {exc}", file=sys.stderr)
        return 1

    try:
        import uvicorn
    except ImportError:
        print('Missing dependency: uvicorn. Install with: pip install -e ".[serve]"', file=sys.stderr)
        return 1

    app = create_app(engine)
    print(f"Swagger UI: http://{args.host}:{args.port}/docs", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    return handle_serve_command(list(sys.argv[1:] if argv is None else argv))


__all__ = ["create_app", "handle_serve_command", "main"]
