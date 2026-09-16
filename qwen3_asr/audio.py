"""WAV / stdin / video loading, resampling, silence handling."""

from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".flv",
    ".ts",
    ".mpg",
    ".mpeg",
    ".wmv",
}


def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def resample(samples: np.ndarray, src_sr: int, dst_sr: int = SAMPLE_RATE) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if src_sr == dst_sr or samples.size == 0:
        return samples
    n_out = int(round(samples.size * dst_sr / src_sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def _pcm16_to_f32(raw: bytes) -> np.ndarray:
    if len(raw) < 2:
        return np.zeros(0, dtype=np.float32)
    if len(raw) % 2:
        raw = raw[:-1]
    pcm = np.frombuffer(raw, dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def parse_wav_buffer(data: bytes) -> np.ndarray | None:
    """Parse 16-bit PCM WAV; resample to 16 kHz mono. Returns None on failure."""
    if len(data) < 44 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        print("parse_wav_buffer: not a valid WAV file", file=sys.stderr)
        return None

    channels = 0
    sample_rate = 0
    bits_per_sample = 0
    audio_format = 0
    pcm_data: bytes | None = None

    p = 12
    while p + 8 <= len(data):
        chunk_id = data[p : p + 4]
        chunk_size = struct.unpack_from("<I", data, p + 4)[0]
        if p + 8 + chunk_size > len(data):
            break
        body = data[p + 8 : p + 8 + chunk_size]
        if chunk_id == b"fmt " and chunk_size >= 16:
            audio_format, channels, sample_rate = struct.unpack_from("<HHI", body, 0)
            bits_per_sample = struct.unpack_from("<H", body, 14)[0]
        elif chunk_id == b"data":
            pcm_data = body
        p += 8 + chunk_size
        if chunk_size & 1:
            p += 1

    if audio_format != 1 or bits_per_sample != 16 or channels < 1 or pcm_data is None:
        print(
            f"parse_wav_buffer: unsupported format (need 16-bit PCM, got fmt={audio_format} bits={bits_per_sample})",
            file=sys.stderr,
        )
        return None

    frame_bytes = channels * 2
    n_frames = len(pcm_data) // frame_bytes
    pcm = np.frombuffer(pcm_data[: n_frames * frame_bytes], dtype="<i2").reshape(n_frames, channels)
    samples = pcm.astype(np.float32).mean(axis=1) / 32768.0
    if sample_rate != SAMPLE_RATE:
        samples = resample(samples, sample_rate, SAMPLE_RATE)
    return samples.astype(np.float32)


def load_wav(path: str) -> np.ndarray | None:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        print(f"Error: cannot read {path}: {exc}", file=sys.stderr)
        return None
    return parse_wav_buffer(data)


def load_with_soundfile(path: str) -> np.ndarray | None:
    try:
        import soundfile as sf
    except ImportError:
        return None
    try:
        samples, sr = sf.read(path, always_2d=True, dtype="float32")
    except Exception:
        return None
    mono = samples.mean(axis=1)
    return resample(mono, int(sr), SAMPLE_RATE)


def extract_audio_from_video(path: str) -> np.ndarray | None:
    try:
        output = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                path,
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-f",
                "s16le",
                "pipe:1",
            ],
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        print("Error: ffmpeg not found — install it to process video files", file=sys.stderr)
        print("  macOS:  brew install ffmpeg", file=sys.stderr)
        print("  Linux:  sudo apt install ffmpeg", file=sys.stderr)
        return None
    if output.returncode != 0:
        err = output.stderr.decode("utf-8", errors="replace")
        print(f"Error: ffmpeg failed:\n{err}", file=sys.stderr)
        return None
    return _pcm16_to_f32(output.stdout)


def load_audio(path: str) -> np.ndarray | None:
    if is_video_file(path):
        return extract_audio_from_video(path)
    samples = load_with_soundfile(path)
    if samples is not None:
        return samples
    return load_wav(path)


def read_pcm_stdin() -> np.ndarray | None:
    buf = sys.stdin.buffer.read()
    if len(buf) < 4:
        print("read_pcm_stdin: no data on stdin", file=sys.stderr)
        return None
    if buf[0:4] == b"RIFF":
        return parse_wav_buffer(buf)
    return _pcm16_to_f32(buf)


def write_temp_wav(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    import soundfile as sf

    samples = np.asarray(samples, dtype=np.float32)
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="qwen3-asr-")
    os.close(fd)
    sf.write(path, samples, sample_rate, subtype="PCM_16")
    return path


def duration_seconds(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    if sample_rate <= 0:
        return 0.0
    return float(len(samples)) / float(sample_rate)


def skip_silence(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: int = 30,
    thresh: float = 0.012,
    min_silence_s: float = 0.6,
) -> np.ndarray:
    """Drop long silent spans; keep a little padding around speech."""
    samples = np.asarray(samples, dtype=np.float32)
    frame = max(1, int(sample_rate * frame_ms / 1000))
    if samples.size < frame * 2:
        return samples
    n_frames = samples.size // frame
    rms = np.sqrt(np.mean(samples[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))
    speech = rms >= thresh
    min_silent_frames = max(1, int(min_silence_s * 1000 / frame_ms))
    keep = np.ones(n_frames, dtype=bool)
    run = 0
    start = 0
    for i, is_speech in enumerate(speech):
        if is_speech:
            if run >= min_silent_frames:
                keep[start : i] = False
                pad = min(2, i - start)
                keep[start : start + pad] = True
                keep[i - pad : i] = True
            run = 0
            start = i + 1
        else:
            if run == 0:
                start = i
            run += 1
    if run >= min_silent_frames:
        keep[start:] = False
        pad = min(2, n_frames - start)
        keep[start : start + pad] = True
    kept = np.repeat(keep, frame)
    tail = samples[n_frames * frame :]
    out = np.concatenate([samples[: kept.size][kept], tail]) if kept.size else tail
    return out.astype(np.float32)


def split_segments(
    samples: np.ndarray,
    target_sec: float,
    search_sec: float = 3.0,
    sample_rate: int = SAMPLE_RATE,
) -> list[tuple[int, int]]:
    """Return (start_sample, end_sample) spans near silence around target_sec."""
    samples = np.asarray(samples, dtype=np.float32)
    n = samples.size
    if target_sec <= 0 or n == 0:
        return [(0, n)]
    target = int(target_sec * sample_rate)
    search = int(max(0.0, search_sec) * sample_rate)
    frame = max(1, sample_rate // 50)  # 20 ms
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        remaining = n - start
        if remaining <= target + search:
            spans.append((start, n))
            break
        center = start + target
        lo = max(start + frame, center - search)
        hi = min(n - 1, center + search)
        best = center
        best_rms = float("inf")
        pos = lo
        while pos + frame <= hi:
            window = samples[pos : pos + frame]
            rms = float(np.sqrt(np.mean(window * window)))
            if rms < best_rms:
                best_rms = rms
                best = pos + frame // 2
            pos += frame
        end = max(start + frame, min(n, best))
        spans.append((start, end))
        start = end
    return spans


def wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(samples, dtype=np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
