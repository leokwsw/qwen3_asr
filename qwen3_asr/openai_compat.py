"""OpenAI Audio API compatible models and response formatting."""

from __future__ import annotations

from typing import Iterable, Literal

from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from .download import KNOWN_MODELS
from .languages import normalize_language
from .output import TranscriptionResult
from .subtitle import format_srt, group_words_to_cues, segment_to_cue

OPENAI_RESPONSE_FORMATS = frozenset({"json", "text", "srt", "verbose_json", "vtt"})
OPENAI_DROP_IN_MODELS = ("whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe")
ResponseFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]


class OpenAIErrorBody(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorBody


class OpenAIModel(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "qwen3-asr"


class OpenAIModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[OpenAIModel]


class OpenAITranscription(BaseModel):
    text: str


class OpenAITranscriptionWord(BaseModel):
    word: str
    start: float
    end: float


class OpenAITranscriptionSegment(BaseModel):
    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: list[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.0


class OpenAIVerboseTranscription(BaseModel):
    task: Literal["transcribe", "translate"]
    language: str
    duration: float
    text: str
    segments: list[OpenAITranscriptionSegment] = Field(default_factory=list)
    words: list[OpenAITranscriptionWord] | None = None


def openai_error(status_code: int, message: str, *, err_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=OpenAIErrorResponse(
            error=OpenAIErrorBody(message=message, type=err_type)
        ).model_dump(),
    )


def openai_language_name(language: str | None) -> str:
    canonical = normalize_language(language) if language else None
    if canonical:
        return canonical.lower()
    return (language or "english").strip().lower() or "english"


def cues_from_result(result: TranscriptionResult):
    if any(seg.words for seg in result.segments):
        return group_words_to_cues(
            [
                {"text": w.word, "start_ms": w.start_ms, "end_ms": w.end_ms}
                for seg in result.segments
                for w in seg.words
            ],
            result.duration_ms,
        )
    return [segment_to_cue(s.start_ms, s.end_ms, s.text) for s in result.segments]


def parse_timestamp_granularities(values: Iterable[str] | str | None) -> set[str]:
    if values is None:
        return {"segment"}
    if isinstance(values, str):
        parts = [values]
    else:
        parts = list(values)
    out = {part.strip().lower() for part in parts if part and str(part).strip()}
    return out or {"segment"}


def format_openai_audio_response(
    result: TranscriptionResult,
    *,
    response_format: str,
    task: Literal["transcribe", "translate"],
    timestamp_granularities: Iterable[str] | str | None = None,
    temperature: float = 0.0,
) -> Response:
    fmt = (response_format or "json").strip().lower()
    if fmt not in OPENAI_RESPONSE_FORMATS:
        return openai_error(
            400,
            f"Invalid response_format '{response_format}'. "
            "Supported: json, text, srt, verbose_json, vtt.",
        )
    cues = cues_from_result(result)
    if fmt == "text":
        return PlainTextResponse(result.text)
    if fmt == "srt":
        return PlainTextResponse(format_srt(cues), media_type="text/plain; charset=utf-8")
    if fmt == "vtt":
        return PlainTextResponse(result.vtt or "", media_type="text/plain; charset=utf-8")
    if fmt == "json":
        return JSONResponse(OpenAITranscription(text=result.text).model_dump())

    grains = parse_timestamp_granularities(timestamp_granularities)
    duration = round(result.duration_ms / 1000.0, 3)
    segments = [
        OpenAITranscriptionSegment(
            id=idx,
            seek=int(seg.start_ms),
            start=round(seg.start_ms / 1000.0, 3),
            end=round(seg.end_ms / 1000.0, 3),
            text=(" " + seg.text) if seg.text and not seg.text.startswith(" ") else seg.text,
            temperature=temperature,
        )
        for idx, seg in enumerate(result.segments)
    ]
    payload = OpenAIVerboseTranscription(
        task=task,
        language=openai_language_name(result.language),
        duration=duration,
        text=result.text,
        segments=segments,
        words=[
            OpenAITranscriptionWord(
                word=w.word,
                start=round(w.start_ms / 1000.0, 3),
                end=round(w.end_ms / 1000.0, 3),
            )
            for seg in result.segments
            for w in seg.words
        ]
        if "word" in grains
        else None,
    )
    return JSONResponse(payload.model_dump(exclude_none=True))


def openai_model_catalog(loaded_model_id: str | None = None) -> list[OpenAIModel]:
    seen: set[str] = set()
    models: list[OpenAIModel] = []
    for model_id in (*OPENAI_DROP_IN_MODELS, loaded_model_id, *(m.name for m in KNOWN_MODELS), *(m.repo for m in KNOWN_MODELS)):
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        models.append(OpenAIModel(id=model_id))
    return models
