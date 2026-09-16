"""qwen3-asr — Qwen3-ASR speech-to-text (Python, CPU-only)."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from . import audio
from .download import handle_download_command, list_models
from .engine import AsrEngine, configure_cpu_threads
from .languages import normalize_language, supported_language_list
from .output import WordTimestamp
from .subtitle import format_srt, format_vtt, group_words_to_cues, segment_to_cue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen3-asr",
        description="qwen3-asr — Qwen3-ASR speech-to-text (Python, CPU-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog="Subcommands:\n  download   Download a Hugging Face checkpoint\n  serve      Start the FastAPI HTTP service (Swagger UI at /docs)",
    )
    parser.add_argument("-h", "--help", action="help", help="Show this help")
    parser.add_argument(
        "-d",
        "--model-dir",
        dest="model_dir",
        help="Model directory or alias (qwen3-asr-0.6b / Hugging Face repo)",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="inputs",
        action="append",
        default=[],
        help="Input WAV/audio/video file. Repeat for multiple files.",
    )
    parser.add_argument("--stdin", action="store_true", help="Read audio from stdin (WAV or raw s16le 16 kHz)")
    parser.add_argument("--live", action="store_true", help="Live capture from audio input device")
    parser.add_argument("--device", help="Input device name for live capture")
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices and exit")
    parser.add_argument("--vad", action="store_true", help="Live VAD mode: transcribe each speech segment")
    parser.add_argument("-t", type=int, default=0, dest="threads", help="CPU thread count (default: all cores)")
    parser.add_argument("-S", type=float, default=0.0, dest="segment_sec", help="Segment target seconds (0 = full)")
    parser.add_argument("-W", type=float, default=3.0, dest="search_sec", help="Silence search window ± seconds")
    parser.add_argument("--stream", action="store_true", help="Streaming mode: process in chunks")
    parser.add_argument("--stream-chunk-sec", type=float, default=2.0, help="Chunk size for streaming (default 2.0)")
    parser.add_argument("--skip-silence", action="store_true", help="Drop long silent spans before inference")
    parser.add_argument("--prompt", help="System prompt for biasing")
    parser.add_argument("--language", help="Force output language (en, zh, ja, English, ...)")
    parser.add_argument("--align", dest="align_text", help="Align transcript to audio (word-level timestamps)")
    parser.add_argument("--align-language", default="English", help="Language for word splitting (default English)")
    parser.add_argument("--srt", nargs="?", const="", default=None, help="Write SRT (default: <input>.srt)")
    parser.add_argument("--vtt", nargs="?", const="", default=None, help="Write WebVTT (default: <input>.vtt)")
    parser.add_argument("--json", nargs="?", const="", default=None, help="Write JSON (stdout when path omitted)")
    parser.add_argument("--aligner-dir", help="ForcedAligner model for word timestamps / sentence subtitles")
    parser.add_argument("--profile", action="store_true", help="Print timing breakdown")
    parser.add_argument("--silent", action="store_true", help="Transcript only, no status output")
    parser.add_argument("--max-new-tokens", type=int, default=0, help="Max generated tokens (0 = auto)")
    return parser


def default_output_path(input_path: str, extension: str) -> str:
    stem = Path(input_path).stem or input_path
    parent = str(Path(input_path).parent) if Path(input_path).parent != Path("") else "."
    return f"{parent}/{stem}.{extension}"


def _expand_inputs(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(part for part in value.split(",") if part)
    return out


def _stream_token(piece: str) -> None:
    sys.stdout.write(piece)
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "download":
        return handle_download_command(argv[1:])
    if argv and argv[0] == "serve":
        try:
            from .server import handle_serve_command
        except ImportError:
            print('Missing HTTP extras. Install with: pip install -e ".[serve]"', file=sys.stderr)
            return 1
        return handle_serve_command(argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)

    verbosity = 0 if args.silent else 1
    input_files = _expand_inputs(args.inputs)

    if args.list_devices:
        from . import live

        try:
            live.list_devices()
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0

    if not args.model_dir:
        parser.print_help()
        print("\nModel management:", file=sys.stderr)
        print("  qwen3-asr download [--list] [<model>] [--output <dir>]", file=sys.stderr)
        print("HTTP service:", file=sys.stderr)
        print("  qwen3-asr serve -d qwen3-asr-0.6b [--host 127.0.0.1] [--port 8000]", file=sys.stderr)
        return 1

    input_count = sum(bool(x) for x in (bool(input_files), args.stdin, args.live))
    if input_count == 0:
        parser.print_help()
        return 1
    if input_count > 1:
        print("Error: -i, --stdin, and --live are mutually exclusive", file=sys.stderr)
        return 1

    srt_requested = args.srt is not None
    vtt_requested = args.vtt is not None
    json_requested = args.json is not None
    output_requested = srt_requested or vtt_requested or json_requested

    if len(input_files) > 1 and (output_requested or args.align_text or args.stream):
        print(
            "Error: multiple -i inputs currently support plain-text output only "
            "(no --srt/--vtt/--json/--align/--stream)",
            file=sys.stderr,
        )
        return 1
    if output_requested and not input_files:
        print("Error: --srt/--vtt/--json requires -i <file>", file=sys.stderr)
        return 1

    input_wav = input_files[0] if input_files else None
    srt_path = args.srt
    vtt_path = args.vtt
    json_path = args.json
    if srt_requested and srt_path == "":
        srt_path = default_output_path(input_wav, "srt")
    if vtt_requested and vtt_path == "":
        vtt_path = default_output_path(input_wav, "vtt")

    n_threads = configure_cpu_threads(args.threads)
    if verbosity >= 1:
        print(f"CPU threads: {n_threads}", file=sys.stderr)

    try:
        engine = AsrEngine.load(
            args.model_dir,
            aligner_dir=args.aligner_dir,
            n_threads=n_threads,
            verbose=verbosity,
        )
    except Exception as exc:
        print(f"Failed to load model from {args.model_dir}: {exc}", file=sys.stderr)
        if not Path(args.model_dir).exists() and "/" not in args.model_dir:
            print(file=sys.stderr)
            list_models()
        return 1

    engine.language = normalize_language(args.language) if args.language else None
    if args.language and engine.language is None:
        print(f"Unsupported language for --language: {args.language}", file=sys.stderr)
        print(f"Supported languages: {supported_language_list()}", file=sys.stderr)
        return 1
    engine.prompt = args.prompt
    engine.max_new_tokens = args.max_new_tokens or None

    if args.live:
        return _run_live(engine, args, verbosity)

    if args.align_text:
        return _run_align(engine, args, input_wav)

    emit_tokens = verbosity > 0 and not (json_requested and json_path == "")
    want_structured = output_requested or bool(args.aligner_dir)

    def run_one(samples, stream: bool) -> str:
        token_cb = _stream_token if (stream or emit_tokens) and not want_structured else None
        text = engine.transcribe_samples(
            samples,
            language=engine.language,
            prompt=engine.prompt,
            stream=stream,
            stream_chunk_sec=args.stream_chunk_sec,
            segment_sec=args.segment_sec,
            search_sec=args.search_sec,
            skip_silence=args.skip_silence,
            token_cb=token_cb,
            return_result=False,
        )
        return str(text)

    if len(input_files) > 1:
        for path in input_files:
            samples = audio.load_audio(path)
            if samples is None:
                print(f"Failed to load audio: {path}", file=sys.stderr)
                return 1
            print(run_one(samples, False))
        return 0

    if args.stdin:
        samples = audio.read_pcm_stdin()
    else:
        samples = audio.load_audio(input_wav)  # type: ignore[arg-type]
    if samples is None:
        print("Failed to load audio", file=sys.stderr)
        return 1

    if want_structured:
        result = engine.transcribe_samples(
            samples,
            language=engine.language,
            prompt=engine.prompt,
            stream=args.stream,
            stream_chunk_sec=args.stream_chunk_sec,
            segment_sec=args.segment_sec,
            search_sec=args.search_sec,
            skip_silence=args.skip_silence,
            return_result=True,
        )
        assert not isinstance(result, str)
        cues = (
            group_words_to_cues(
                [{"text": w.word, "start_ms": w.start_ms, "end_ms": w.end_ms} for seg in result.segments for w in seg.words],
                result.duration_ms,
            )
            if any(seg.words for seg in result.segments)
            else [segment_to_cue(s.start_ms, s.end_ms, s.text) for s in result.segments]
        )
        if srt_requested and srt_path:
            Path(srt_path).write_text(format_srt(cues), encoding="utf-8")
            if verbosity:
                print(f"Wrote {srt_path}", file=sys.stderr)
        if vtt_requested and vtt_path:
            Path(vtt_path).write_text(format_vtt(cues), encoding="utf-8")
            if verbosity:
                print(f"Wrote {vtt_path}", file=sys.stderr)
        if json_requested:
            payload = result.to_json()
            if json_path:
                Path(json_path).write_text(payload, encoding="utf-8")
                if verbosity:
                    print(f"Wrote {json_path}", file=sys.stderr)
            else:
                sys.stdout.write(payload)
        elif emit_tokens:
            print(result.text)
        if verbosity >= 1 or args.profile:
            engine.timing.report()
        return 0

    text = run_one(samples, args.stream)
    if emit_tokens and not args.stream:
        print()
        print(text)
    elif not emit_tokens:
        print(text)
    elif args.stream:
        print()
    if verbosity >= 1 or args.profile:
        engine.timing.report()
    return 0


def _run_align(engine: AsrEngine, args: argparse.Namespace, input_wav: str | None) -> int:
    if args.stdin:
        samples = audio.read_pcm_stdin()
    else:
        samples = audio.load_audio(input_wav) if input_wav else None
    if samples is None:
        print("Failed to load audio", file=sys.stderr)
        return 1
    lang = normalize_language(args.align_language) or args.align_language
    if engine.aligner_model is None:
        # Alignment can also be done with the same checkpoint if it is an aligner.
        try:
            engine._load_aligner(args.model_dir, type(engine.model), engine.dtype, verbose=1)
        except Exception:
            print("Error: --align requires a ForcedAligner model (-d qwen3-aligner-0.6b)", file=sys.stderr)
            return 1
    try:
        words: list[WordTimestamp] = engine.align_samples(samples, args.align_text, lang)
    except Exception as exc:
        print(f"Alignment failed: {exc}", file=sys.stderr)
        return 1
    print("[")
    for i, word in enumerate(words):
        comma = "," if i + 1 < len(words) else ""
        escaped = word.word.replace("\\", "\\\\").replace('"', '\\"')
        print(f'  {{"text": "{escaped}", "start": {word.start_ms:.0f}, "end": {word.end_ms:.0f}}}{comma}')
    print("]")
    return 0


def _run_live(engine: AsrEngine, args: argparse.Namespace, verbosity: int) -> int:
    from . import live

    try:
        device_id = live.find_device(args.device)
        if args.device and device_id is None:
            print(f"Error: No input device matching '{args.device}'", file=sys.stderr)
            live.list_devices()
            return 1
        chunks, _sr = live.capture_chunks(device_id)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    stop = {"flag": False}

    def _handle(_sig, _frame) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    running = lambda: not stop["flag"]

    if verbosity:
        mode = "VAD" if args.vad else ("streaming" if args.stream else "segmented")
        print(f"Listening ({mode})... press Ctrl+C to stop\n", file=sys.stderr)

    if args.vad:
        for speech in live.vad_collect(chunks, running=running):
            text = engine.transcribe_samples(
                speech,
                language=engine.language,
                prompt=engine.prompt,
                skip_silence=False,
            )
            print(str(text).strip())
            sys.stdout.flush()
        return 0

    chunk_sec = args.stream_chunk_sec if args.stream else (args.segment_sec or 5.0)
    buf: list = []
    samples_needed = int(chunk_sec * audio.SAMPLE_RATE)
    committed = ""
    try:
        for piece in chunks:
            if not running():
                break
            buf.append(piece)
            n = sum(p.size for p in buf)
            if n < samples_needed:
                continue
            samples = np_concat(buf)
            buf = []
            if args.stream:
                engine.prompt = committed[-200:] or args.prompt
            text = str(
                engine.transcribe_samples(
                    samples,
                    language=engine.language,
                    prompt=engine.prompt,
                    skip_silence=args.skip_silence,
                )
            ).strip()
            if not text:
                continue
            if args.stream:
                delta = text if not committed else text
                print(delta)
                committed = (committed + " " + text).strip()
            else:
                print(text)
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    return 0


def np_concat(parts):
    import numpy as np

    return np.concatenate(parts)


if __name__ == "__main__":
    sys.exit(main())
