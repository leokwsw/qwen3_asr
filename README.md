# qwen3-asr

CPU-only Python CLI and library for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B).

This is a Transformers + PyTorch **CPU** runtime. Inference always runs on `device=cpu`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# optional live microphone support
pip install -e ".[live]"
```

PyTorch is forced onto `device=cpu`. Apple Silicon uses `bfloat16` on CPU; other platforms default to `float32`.

The Hugging Face Qwen3-ASR models need a recent `transformers` (5.13+ recommended):

```bash
pip install -U "transformers>=5.13.0"
```

## Download model

Weights must be in Hugging Face format (`config.json` + processor files).

```bash
qwen3-asr download qwen3-asr-0.6b
# or: python -m qwen3_asr download qwen3-asr-0.6b
```

Known aliases: `qwen3-asr-0.6b`, `qwen3-asr-1.7b`, `qwen3-aligner-0.6b`.

## Transcribe

```bash
qwen3-asr -d qwen3-asr-0.6b -i audio.wav
qwen3-asr -d qwen3-asr-0.6b -i audio.wav --silent
qwen3-asr -d qwen3-asr-0.6b -i audio.wav --language zh
qwen3-asr -d qwen3-asr-0.6b -i audio.wav --stream
qwen3-asr -d qwen3-asr-0.6b -i long.wav -S 30
qwen3-asr -d qwen3-asr-0.6b -i audio.wav --srt --vtt --json out.json
qwen3-asr -d qwen3-asr-0.6b -i audio.wav --aligner-dir qwen3-aligner-0.6b --srt
cat audio.wav | qwen3-asr -d qwen3-asr-0.6b --stdin
```

Live capture (needs `sounddevice`):

```bash
qwen3-asr --list-devices
qwen3-asr -d qwen3-asr-0.6b --live --vad
```

Forced alignment:

```bash
qwen3-asr -d qwen3-aligner-0.6b -i audio.wav --align "Hello world" --align-language English
```

## Library

```python
from qwen3_asr import AsrEngine

engine = AsrEngine.load("qwen3-asr-0.6b")  # CPU
text = engine.transcribe_file("audio.wav")
print(text)

result = engine.transcribe_file("audio.wav", return_result=True)
print(result.language, result.text)
```

Always on CPU: `model.to("cpu")`. No CUDA or MPS dispatch.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -q
```
