"""CPU-only Qwen3-ASR engine (Hugging Face Transformers + PyTorch)."""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import audio
from .download import is_hf_model_dir, resolve_model_id
from .languages import normalize_language, supported_language_list
from .output import SegmentResult, TranscriptionResult, WordTimestamp, result_from_segments
from .subtitle import group_words_to_cues, segment_to_cue


def configure_cpu_threads(n_threads: int | None = None) -> int:
    threads = n_threads if n_threads and n_threads > 0 else (os.cpu_count() or 4)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, min(threads, 4)))
    except Exception:
        pass
    return threads


def default_dtype():
    import torch

    if sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return torch.bfloat16
    return torch.float32


def _load_transformers():
    try:
        from transformers import AutoProcessor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: transformers. Install with: pip install -e python/"
        ) from exc

    asr_cls = None
    aligner_cls = None
    try:
        from transformers import AutoModelForMultimodalLM

        asr_cls = AutoModelForMultimodalLM
    except ImportError:
        try:
            from transformers import Qwen3ASRForConditionalGeneration

            asr_cls = Qwen3ASRForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "This Python CPU build needs transformers with Qwen3-ASR support "
                "(AutoModelForMultimodalLM or Qwen3ASRForConditionalGeneration). "
                "Upgrade with: pip install -U 'transformers>=5.13.0'"
            ) from exc
    try:
        from transformers import AutoModelForTokenClassification

        aligner_cls = AutoModelForTokenClassification
    except ImportError:
        aligner_cls = None
    return AutoProcessor, asr_cls, aligner_cls


def _max_new_tokens(duration_s: float, override: int | None = None) -> int:
    if override is not None and override > 0:
        return override
    return int(min(2048, max(64, duration_s * 12 + 32)))


def _from_pretrained(cls: Any, model_id: str, torch_dtype: Any) -> Any:
    kwargs = {"low_cpu_mem_usage": True}
    try:
        return cls.from_pretrained(model_id, dtype=torch_dtype, **kwargs)
    except TypeError:
        try:
            return cls.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
        except TypeError:
            return cls.from_pretrained(model_id, torch_dtype=torch_dtype)


def _move_inputs(inputs: Any, device: Any, dtype: Any) -> Any:
    if hasattr(inputs, "to"):
        try:
            return inputs.to(device, dtype)
        except TypeError:
            return inputs.to(device)
    moved: dict[str, Any] = {}
    for key, value in dict(inputs).items():
        if hasattr(value, "to"):
            is_float = bool(getattr(value, "is_floating_point", lambda: False)())
            moved[key] = value.to(device=device, dtype=dtype) if is_float else value.to(device)
        else:
            moved[key] = value
    return moved


@dataclass
class Timing:
    load_ms: float = 0.0
    audio_ms: float = 0.0
    encode_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    audio_s: float = 0.0

    def report(self, file=sys.stderr) -> None:
        print(f"Inference: {self.total_ms:.0f} ms (load {self.load_ms:.0f} ms)", file=file)
        if self.audio_s > 0 and self.total_ms > 0:
            infer_s = self.total_ms / 1000.0
            print(
                f"Audio: {self.audio_s:.1f} s processed in {infer_s:.1f} s "
                f"({self.audio_s / infer_s:.2f}x realtime)",
                file=file,
            )


@dataclass
class AsrEngine:
    model_id: str
    processor: Any
    model: Any
    device: Any
    dtype: Any
    aligner_id: str | None = None
    aligner_processor: Any = None
    aligner_model: Any = None
    language: str | None = None
    prompt: str | None = None
    max_new_tokens: int | None = None
    timing: Timing = field(default_factory=Timing)

    @classmethod
    def load(
        cls,
        model_dir: str,
        *,
        aligner_dir: str | None = None,
        n_threads: int | None = None,
        dtype: Any | None = None,
        verbose: int = 1,
    ) -> "AsrEngine":
        configure_cpu_threads(n_threads)
        import torch

        AutoProcessor, asr_cls, aligner_cls = _load_transformers()
        model_id = resolve_model_id(model_dir)
        if Path(model_id).exists() and not is_hf_model_dir(model_id):
            raise FileNotFoundError(
                f"'{model_id}' is missing config.json. "
                "Download Hugging Face format weights with:\n"
                "  qwen3-asr download qwen3-asr-0.6b"
            )

        torch_dtype = dtype or default_dtype()
        if verbose >= 1:
            print(
                f"Optimizations: PyTorch CPU, {torch_dtype} | "
                f"{torch.get_num_threads()} threads | {platform.machine()} | device=cpu",
                file=sys.stderr,
            )

        t0 = time.perf_counter()
        processor = AutoProcessor.from_pretrained(model_id)
        model = _from_pretrained(asr_cls, model_id, torch_dtype)
        model.to("cpu")
        model.eval()
        load_ms = (time.perf_counter() - t0) * 1000.0

        engine = cls(
            model_id=str(model_id),
            processor=processor,
            model=model,
            device=torch.device("cpu"),
            dtype=torch_dtype,
        )
        engine.timing.load_ms = load_ms

        if aligner_dir:
            engine._load_aligner(aligner_dir, aligner_cls, torch_dtype, verbose)
        return engine

    def _load_aligner(self, aligner_dir: str, aligner_cls: Any, torch_dtype: Any, verbose: int) -> None:
        if aligner_cls is None:
            raise RuntimeError("transformers is missing AutoModelForTokenClassification")
        AutoProcessor, _, _ = _load_transformers()
        aligner_id = resolve_model_id(aligner_dir)
        if verbose >= 1:
            print(f"Loading aligner: {aligner_id}", file=sys.stderr)
        self.aligner_id = str(aligner_id)
        self.aligner_processor = AutoProcessor.from_pretrained(aligner_id)
        self.aligner_model = _from_pretrained(aligner_cls, aligner_id, torch_dtype)
        self.aligner_model.to("cpu")
        self.aligner_model.eval()

    def _prepare_inputs(self, wav_path: str, language: str | None, prompt: str | None):
        kwargs: dict[str, Any] = {"audio": wav_path}
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt
        inputs = self.processor.apply_transcription_request(**kwargs)
        return _move_inputs(inputs, self.device, self.dtype)

    def _generate(self, inputs: Any, max_new_tokens: int) -> Any:
        import torch

        with torch.inference_mode():
            return self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    def _decode(self, output_ids: Any, prompt_len: int) -> tuple[str, str]:
        generated = output_ids[:, prompt_len:]
        try:
            parsed = self.processor.decode(generated, return_format="parsed")[0]
            if isinstance(parsed, dict):
                return str(parsed.get("language") or ""), str(parsed.get("transcription") or "")
        except Exception:
            pass
        try:
            text = self.processor.decode(generated, return_format="transcription_only")[0]
            return "", str(text)
        except Exception:
            raw = self.processor.decode(generated)[0]
            return "", str(raw)

    def transcribe_samples(
        self,
        samples: np.ndarray,
        *,
        language: str | None = None,
        prompt: str | None = None,
        stream: bool = False,
        stream_chunk_sec: float = 2.0,
        segment_sec: float = 0.0,
        search_sec: float = 3.0,
        skip_silence: bool = False,
        token_cb: Callable[[str], None] | None = None,
        return_result: bool = False,
    ) -> str | TranscriptionResult:
        samples = np.asarray(samples, dtype=np.float32)
        if skip_silence:
            samples = audio.skip_silence(samples)
        duration_s = audio.duration_seconds(samples)
        self.timing.audio_s = duration_s
        lang = normalize_language(language or self.language)
        if language and lang is None:
            raise ValueError(
                f"Unsupported language for --language: {language}\n"
                f"Supported languages: {supported_language_list()}"
            )
        prompt_text = prompt if prompt is not None else self.prompt
        max_new = _max_new_tokens(duration_s, self.max_new_tokens)

        if stream:
            text = self._transcribe_stream(
                samples,
                language=lang,
                prompt=prompt_text,
                chunk_sec=stream_chunk_sec,
                token_cb=token_cb,
            )
            if return_result:
                duration_ms = int(duration_s * 1000)
                seg = SegmentResult(0, duration_ms, text)
                return result_from_segments(lang or "", duration_ms, [seg])
            return text

        if segment_sec and segment_sec > 0:
            return self._transcribe_segmented(
                samples,
                language=lang,
                prompt=prompt_text,
                segment_sec=segment_sec,
                search_sec=search_sec,
                token_cb=token_cb,
                return_result=return_result,
            )

        wav_path = audio.write_temp_wav(samples)
        try:
            t0 = time.perf_counter()
            inputs = self._prepare_inputs(wav_path, lang, prompt_text)
            output_ids = self._generate(inputs, max_new)
            self.timing.total_ms = (time.perf_counter() - t0) * 1000.0
            language_out, text = self._decode(output_ids, inputs["input_ids"].shape[1])
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        if token_cb:
            token_cb(text)
        if not return_result:
            return text

        duration_ms = int(duration_s * 1000)
        words: list[WordTimestamp] = []
        if self.aligner_model is not None and text.strip():
            words = self.align_samples(samples, text, language_out or lang or "English")
        segments = [
            SegmentResult(
                start_ms=words[0].start_ms if words else 0,
                end_ms=words[-1].end_ms if words else duration_ms,
                text=text,
                words=words,
            )
        ]
        if words:
            cues = group_words_to_cues(
                [{"text": w.word, "start_ms": w.start_ms, "end_ms": w.end_ms} for w in words],
                duration_ms,
            )
        else:
            cues = [segment_to_cue(0, duration_ms, text)]
        return result_from_segments(language_out or lang or "", duration_ms, segments, cues)

    def _transcribe_segmented(
        self,
        samples: np.ndarray,
        *,
        language: str | None,
        prompt: str | None,
        segment_sec: float,
        search_sec: float,
        token_cb: Callable[[str], None] | None,
        return_result: bool,
    ) -> str | TranscriptionResult:
        spans = audio.split_segments(samples, segment_sec, search_sec)
        texts: list[str] = []
        segments: list[SegmentResult] = []
        t0 = time.perf_counter()
        past = prompt or ""
        for start, end in spans:
            chunk = samples[start:end]
            if chunk.size == 0:
                continue
            wav_path = audio.write_temp_wav(chunk)
            try:
                chunk_prompt = past if past else prompt
                inputs = self._prepare_inputs(wav_path, language, chunk_prompt)
                output_ids = self._generate(
                    inputs, _max_new_tokens(audio.duration_seconds(chunk), self.max_new_tokens)
                )
                lang_out, text = self._decode(output_ids, inputs["input_ids"].shape[1])
            finally:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass
            text = text.strip()
            if not text:
                continue
            if token_cb:
                token_cb(text + " ")
            texts.append(text)
            past = text
            segments.append(
                SegmentResult(
                    start_ms=int(start / audio.SAMPLE_RATE * 1000),
                    end_ms=int(end / audio.SAMPLE_RATE * 1000),
                    text=text,
                )
            )
        self.timing.total_ms = (time.perf_counter() - t0) * 1000.0
        joined = " ".join(texts).strip()
        if return_result:
            duration_ms = int(audio.duration_seconds(samples) * 1000)
            cues = [segment_to_cue(s.start_ms, s.end_ms, s.text) for s in segments]
            detected = language or ""
            return result_from_segments(detected, duration_ms, segments, cues)
        return joined

    def _transcribe_stream(
        self,
        samples: np.ndarray,
        *,
        language: str | None,
        prompt: str | None,
        chunk_sec: float,
        token_cb: Callable[[str], None] | None,
    ) -> str:
        chunk = max(1.0, chunk_sec)
        step = int(chunk * audio.SAMPLE_RATE)
        committed = ""
        t0 = time.perf_counter()
        start = 0
        n = samples.size
        while start < n:
            end = min(n, start + step)
            # Grow a little overlap so word boundaries are less likely to split.
            window_start = max(0, start - int(0.2 * audio.SAMPLE_RATE))
            piece = samples[window_start:end]
            wav_path = audio.write_temp_wav(piece)
            try:
                inputs = self._prepare_inputs(wav_path, language, prompt or committed[-200:] or None)
                output_ids = self._generate(
                    inputs, _max_new_tokens(audio.duration_seconds(piece), self.max_new_tokens)
                )
                _lang, text = self._decode(output_ids, inputs["input_ids"].shape[1])
            finally:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass
            text = text.strip()
            delta = _delta_text(committed, text)
            if delta:
                committed = (committed + delta).strip()
                if token_cb:
                    token_cb(delta)
            start = end
        self.timing.total_ms = (time.perf_counter() - t0) * 1000.0
        return committed.strip()

    def transcribe_file(self, path: str, **kwargs: Any) -> str | TranscriptionResult:
        t0 = time.perf_counter()
        samples = audio.load_audio(path)
        self.timing.audio_ms = (time.perf_counter() - t0) * 1000.0
        if samples is None:
            raise RuntimeError(f"Failed to load audio: {path}")
        return self.transcribe_samples(samples, **kwargs)

    def align_samples(
        self,
        samples: np.ndarray,
        transcript: str,
        language: str,
    ) -> list[WordTimestamp]:
        import torch

        if self.aligner_model is None or self.aligner_processor is None:
            raise RuntimeError("Aligner model is not loaded. Pass --aligner-dir / aligner_dir.")
        lang = normalize_language(language) or language
        wav_path = audio.write_temp_wav(samples)
        try:
            prepare = self.aligner_processor.prepare_forced_aligner_inputs
            prepared = prepare(audio=wav_path, transcript=transcript, language=lang)
            if isinstance(prepared, tuple):
                aligner_inputs, word_lists = prepared
            else:
                aligner_inputs, word_lists = prepared, None
            aligner_inputs = _move_inputs(aligner_inputs, self.device, self.dtype)
            with torch.inference_mode():
                outputs = self.aligner_model(**aligner_inputs)
            ts_id = getattr(self.aligner_model.config, "timestamp_token_id", None)
            decoded = self.aligner_processor.decode_forced_alignment(
                logits=outputs.logits,
                input_ids=aligner_inputs["input_ids"],
                word_lists=word_lists,
                timestamp_token_id=ts_id,
            )[0]
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        words: list[WordTimestamp] = []
        for item in decoded:
            text = str(item.get("text") or item.get("word") or "")
            start = item.get("start_time", item.get("start", 0.0))
            end = item.get("end_time", item.get("end", start))
            words.append(
                WordTimestamp(
                    word=text,
                    start_ms=int(float(start) * 1000),
                    end_ms=int(float(end) * 1000),
                )
            )
        return words


def _delta_text(committed: str, latest: str) -> str:
    """Return the new suffix of `latest` that is not already in `committed`."""
    latest = latest.strip()
    committed = committed.strip()
    if not latest:
        return ""
    if not committed:
        return latest + " "
    if latest.startswith(committed):
        return latest[len(committed) :].lstrip()
    # Longest prefix of latest that matches a suffix of committed.
    max_k = min(len(committed), len(latest))
    for k in range(max_k, 0, -1):
        if committed.endswith(latest[:k]):
            return latest[k:].lstrip()
    if latest in committed:
        return ""
    return " " + latest
