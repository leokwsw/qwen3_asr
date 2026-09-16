"""Optional live microphone capture (sounddevice)."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator

import numpy as np

from .audio import SAMPLE_RATE, resample


def _require_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Live capture needs sounddevice. Install with: pip install 'qwen3-asr-cpu[live]'"
        ) from exc
    return sd


def list_devices() -> None:
    sd = _require_sounddevice()
    devices = sd.query_devices()
    print("Audio input devices:", file=sys.stderr)
    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) <= 0:
            continue
        default = " (default)" if idx == sd.default.device[0] else ""
        print(
            f"  [{idx}] {dev['name']}  {dev['max_input_channels']}ch  {int(dev['default_samplerate'])} Hz{default}",
            file=sys.stderr,
        )


def find_device(name: str | None) -> int | None:
    sd = _require_sounddevice()
    if name is None:
        default = sd.default.device[0]
        return None if default is None or default < 0 else int(default)
    needle = name.lower()
    for idx, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels", 0)) <= 0:
            continue
        if needle in str(dev["name"]).lower():
            return idx
    return None


def capture_chunks(
    device: int | None,
    *,
    block_sec: float = 0.1,
) -> tuple[Iterator[np.ndarray], float]:
    sd = _require_sounddevice()
    info = sd.query_devices(device)
    device_rate = float(info["default_samplerate"])
    channels = 1 if int(info["max_input_channels"]) >= 1 else int(info["max_input_channels"])
    blocksize = max(1, int(device_rate * block_sec))

    def gen() -> Iterator[np.ndarray]:
        with sd.InputStream(
            device=device,
            channels=channels,
            samplerate=device_rate,
            dtype="float32",
            blocksize=blocksize,
        ) as stream:
            while True:
                data, overflowed = stream.read(blocksize)
                if overflowed:
                    print("warning: input overflow", file=sys.stderr)
                mono = np.asarray(data, dtype=np.float32)
                if mono.ndim > 1:
                    mono = mono.mean(axis=1)
                yield resample(mono, int(device_rate), SAMPLE_RATE)

    return gen(), SAMPLE_RATE


def energy(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))


def vad_collect(
    chunks: Iterator[np.ndarray],
    *,
    threshold: float = 0.02,
    min_silence_s: float = 0.6,
    min_speech_s: float = 0.4,
    max_speech_s: float = 20.0,
    running,
) -> Iterator[np.ndarray]:
    """Yield speech segments using a simple energy VAD."""
    buf: list[np.ndarray] = []
    voiced = False
    silence_samples = 0
    speech_samples = 0
    min_silence = int(min_silence_s * SAMPLE_RATE)
    min_speech = int(min_speech_s * SAMPLE_RATE)
    max_speech = int(max_speech_s * SAMPLE_RATE)

    for chunk in chunks:
        if running is not None and not running():
            break
        rms = energy(chunk)
        if rms >= threshold:
            voiced = True
            silence_samples = 0
            buf.append(chunk)
            speech_samples += chunk.size
            if speech_samples >= max_speech:
                yield np.concatenate(buf)
                buf = []
                voiced = False
                speech_samples = 0
        elif voiced:
            buf.append(chunk)
            silence_samples += chunk.size
            if silence_samples >= min_silence:
                audio = np.concatenate(buf)
                buf = []
                voiced = False
                speech_samples = 0
                silence_samples = 0
                if audio.size >= min_speech:
                    yield audio
        # drop leading silence
    if buf:
        audio = np.concatenate(buf)
        if audio.size >= min_speech:
            yield audio


def sleep_while_running(seconds: float, running) -> None:
    end = time.time() + seconds
    while time.time() < end:
        if running is not None and not running():
            return
        time.sleep(0.05)
