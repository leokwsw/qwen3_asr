FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf-cache \
    QWEN3_ASR_HOST=0.0.0.0 \
    QWEN3_ASR_PORT=8000 \
    QWEN3_ASR_MODEL=/models/qwen3-asr-0.6b

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY qwen3_asr ./qwen3_asr
COPY deploy/docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir --no-deps . \
    && pip install --no-cache-dir \
        "transformers>=5.13.0" \
        "numpy>=1.24" \
        "soundfile>=0.12" \
        "huggingface_hub>=0.23" \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.32" \
        "python-multipart>=0.0.9"

EXPOSE 8000
VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

ENTRYPOINT ["/entrypoint.sh"]
