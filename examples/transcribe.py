from qwen3_asr import AsrEngine

# CPU-only transcription. Pass a local HF-format directory or a known alias.
engine = AsrEngine.load("qwen3-asr-0.6b")
print(engine.transcribe_file("audio.wav"))
