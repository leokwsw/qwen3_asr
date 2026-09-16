"""Download Hugging Face Qwen3-ASR checkpoints used by the Python CPU runtime."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelInfo:
    name: str
    repo: str
    description: str


KNOWN_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        name="qwen3-asr-0.6b",
        repo="Qwen/Qwen3-ASR-0.6B-hf",
        description="Qwen3-ASR 0.6B (Hugging Face format) — fast CPU default",
    ),
    ModelInfo(
        name="qwen3-asr-1.7b",
        repo="Qwen/Qwen3-ASR-1.7B-hf",
        description="Qwen3-ASR 1.7B (Hugging Face format) — higher accuracy",
    ),
    ModelInfo(
        name="qwen3-aligner-0.6b",
        repo="Qwen/Qwen3-ForcedAligner-0.6B-hf",
        description="Qwen3 ForcedAligner 0.6B — word-level timestamps",
    ),
)

ALIAS_TO_REPO = {m.name: m.repo for m in KNOWN_MODELS}
# Also accept the original (non-hf) repo IDs and map them to HF-format weights.
ALIAS_TO_REPO.update(
    {
        "Qwen/Qwen3-ASR-0.6B": "Qwen/Qwen3-ASR-0.6B-hf",
        "Qwen/Qwen3-ASR-1.7B": "Qwen/Qwen3-ASR-1.7B-hf",
        "Qwen/Qwen3-ForcedAligner-0.6B": "Qwen/Qwen3-ForcedAligner-0.6B-hf",
    }
)


def find_model(name: str) -> ModelInfo | None:
    key = name.lower()
    return next((m for m in KNOWN_MODELS if m.name == key), None)


def list_models() -> None:
    print("Available models:\n", file=sys.stderr)
    for model in KNOWN_MODELS:
        print(f"  {model.name:<24} {model.description}", file=sys.stderr)
    print(file=sys.stderr)
    print("Usage: qwen3-asr download <model-name> [--output <dir>]", file=sys.stderr)


def resolve_model_id(name_or_path: str) -> str:
    """Map CLI aliases to a local directory or Hugging Face repo id."""
    path = Path(name_or_path)
    if path.exists():
        return str(path)
    if name_or_path in ALIAS_TO_REPO:
        return ALIAS_TO_REPO[name_or_path]
    found = find_model(name_or_path)
    if found:
        return found.repo
    return name_or_path


def is_hf_model_dir(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "config.json").is_file()


def download_model(name: str, output: str | None = None) -> Path:
    info = find_model(name)
    repo = info.repo if info else name
    dest = Path(output) if output else Path(info.name if info else Path(repo).name)
    dest.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    print(f"Downloading {repo} -> {dest.resolve()}", file=sys.stderr)
    snapshot_download(repo_id=repo, local_dir=str(dest))
    print(f"Done. Use: qwen3-asr -d {dest} -i audio.wav", file=sys.stderr)
    return dest


def handle_download_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qwen3-asr download", add_help=True)
    parser.add_argument("model", nargs="?", help="Model alias or Hugging Face repo id")
    parser.add_argument("--list", action="store_true", help="List known models")
    parser.add_argument("--output", "-o", help="Destination directory")
    args = parser.parse_args(argv)
    if args.list or not args.model:
        list_models()
        return 0 if args.list or not args.model else 1
    try:
        download_model(args.model, args.output)
    except Exception as exc:
        print(f"Error: download failed: {exc}", file=sys.stderr)
        return 1
    return 0
