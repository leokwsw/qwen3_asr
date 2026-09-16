# qwen3-asr

CPU-only Python CLI, HTTP service, and library for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B).

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

## HTTP service (FastAPI + Swagger UI)

Start a local API after the model is available. The process loads the checkpoint once and reuses it for every request.

```bash
qwen3-asr serve -d qwen3-asr-0.6b
# optional word timestamps / /v1/align
qwen3-asr serve -d qwen3-asr-0.6b --aligner-dir qwen3-aligner-0.6b --host 0.0.0.0 --port 8000
```

Then open:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Service and loaded model |
| `GET` | `/v1/info` | Model, languages, known checkpoints |
| `GET` | `/v1/languages` | Supported languages |
| `GET` | `/v1/models` | Known model aliases |
| `POST` | `/v1/transcribe` | Upload audio/video and transcribe |
| `POST` | `/v1/align` | Forced alignment (needs `--aligner-dir`) |

```bash
curl -sS -F "file=@audio.wav" -F "language=zh" http://127.0.0.1:8000/v1/transcribe
curl -sS http://127.0.0.1:8000/health
```

Environment overrides: `QWEN3_ASR_MODEL`, `QWEN3_ASR_ALIGNER`, `QWEN3_ASR_HOST`, `QWEN3_ASR_PORT`, `QWEN3_ASR_MAX_UPLOAD_MB` (default 100). Keep a single Uvicorn worker; extra workers reload the model into RAM again.

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

The HTTP app is also usable as a library:

```python
from qwen3_asr.engine import AsrEngine
from qwen3_asr.server import create_app

engine = AsrEngine.load("qwen3-asr-0.6b")
app = create_app(engine)
```

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -q
```
