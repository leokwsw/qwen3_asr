#!/bin/sh
set -eu

MODEL="${QWEN3_ASR_MODEL:-/models/qwen3-asr-0.6b}"
ALIAS="${QWEN3_ASR_MODEL_ALIAS:-qwen3-asr-0.6b}"
PORT="${QWEN3_ASR_PORT:-8000}"
THREADS="${QWEN3_ASR_THREADS:-0}"

if [ ! -f "${MODEL}/config.json" ]; then
  echo "Model missing at ${MODEL}; downloading ${ALIAS}" >&2
  mkdir -p "${MODEL}"
  qwen3-asr download "${ALIAS}" --output "${MODEL}"
fi

aligner_args=""
if [ -n "${QWEN3_ASR_ALIGNER:-}" ]; then
  if [ ! -f "${QWEN3_ASR_ALIGNER}/config.json" ]; then
    echo "Aligner missing at ${QWEN3_ASR_ALIGNER}; downloading qwen3-aligner-0.6b" >&2
    mkdir -p "${QWEN3_ASR_ALIGNER}"
    qwen3-asr download qwen3-aligner-0.6b --output "${QWEN3_ASR_ALIGNER}"
  fi
  aligner_args="--aligner-dir ${QWEN3_ASR_ALIGNER}"
fi

# shellcheck disable=SC2086
exec qwen3-asr serve \
  -d "${MODEL}" \
  ${aligner_args} \
  --host 0.0.0.0 \
  --port "${PORT}" \
  -t "${THREADS}"
