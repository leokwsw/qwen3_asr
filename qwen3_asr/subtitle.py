"""SRT / WebVTT cue grouping."""

from __future__ import annotations

from dataclasses import dataclass

MAX_WIDTH = 84
MAX_DUR_MS = 10000.0
MIN_DUR_MS = 1000
GAP_BREAK_MS = 2800.0
TAIL_MS = 300
TS_GRID_MS = 80.0


@dataclass
class Cue:
    start_ms: int
    end_ms: int
    text: str


@dataclass
class Word:
    text: str
    start_ms: float
    end_ms: float


def is_cjk_width_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFF60
    )


def text_width(text: str) -> int:
    return sum(2 if is_cjk_width_char(ch) else 1 for ch in text)


def cjk_join(left: str, right: str) -> bool:
    return bool(left) and bool(right) and is_cjk_width_char(left[-1]) and is_cjk_width_char(right[0])


def join_words(words: list[Word]) -> str:
    out: list[str] = []
    for idx, word in enumerate(words):
        if idx > 0 and not cjk_join(words[idx - 1].text, word.text):
            out.append(" ")
        out.append(word.text)
    return "".join(out)


def _sentence_tail(text: str) -> str | None:
    stripped = text.rstrip("\"'」』）)]》")
    return stripped[-1] if stripped else None


def is_sentence_end(text: str) -> bool:
    return _sentence_tail(text) in {".", "!", "?", "…", "。", "！", "？"}


def is_clause_end(text: str) -> bool:
    return _sentence_tail(text) in {",", ";", ":", "，", "；", "：", "、"}


def _cue_width_with(cur: list[Word], next_word: Word) -> int:
    width = 0
    seq = cur + [next_word]
    for idx, word in enumerate(seq):
        if idx > 0 and not cjk_join(seq[idx - 1].text, word.text):
            width += 1
        width += text_width(word.text)
    return width


def _would_overflow(cur: list[Word], next_word: Word) -> bool:
    return _cue_width_with(cur, next_word) > MAX_WIDTH or (next_word.end_ms - cur[0].start_ms) > MAX_DUR_MS


def group_words_to_cues(words: list[dict] | list[Word], audio_end_ms: int) -> list[Cue]:
    normalized: list[Word] = []
    for word in words:
        if isinstance(word, Word):
            item = word
        else:
            start = float(word.get("start_ms", word.get("start_time", 0) * 1000))
            end = float(word.get("end_ms", word.get("end_time", start / 1000) * 1000))
            item = Word(text=str(word.get("text", word.get("word", ""))), start_ms=start, end_ms=end)
        end_ms = max(item.end_ms, item.start_ms)
        if end_ms == item.start_ms:
            end_ms = item.start_ms + TS_GRID_MS
        normalized.append(Word(text=item.text, start_ms=item.start_ms, end_ms=end_ms))
    if not normalized:
        return []

    groups: list[list[Word]] = []
    cur: list[Word] = []
    soft_break: int | None = None
    for idx, word in enumerate(normalized):
        if cur and _would_overflow(cur, word):
            if soft_break is not None:
                groups.append(cur[:soft_break])
                cur = cur[soft_break:]
            else:
                groups.append(cur)
                cur = []
            soft_break = None
        cur.append(word)
        nxt = normalized[idx + 1] if idx + 1 < len(normalized) else None
        gap_break = nxt is not None and (nxt.start_ms - word.end_ms) > GAP_BREAK_MS
        if is_sentence_end(word.text) or idx + 1 == len(normalized) or gap_break:
            groups.append(cur)
            cur = []
            soft_break = None
        elif is_clause_end(word.text):
            soft_break = len(cur)
    if cur:
        groups.append(cur)
    return _groups_to_cues(groups, audio_end_ms)


def _groups_to_cues(groups: list[list[Word]], audio_end_ms: int) -> list[Cue]:
    cues: list[Cue] = []
    for group in groups:
        if not group:
            continue
        start_ms = int(group[0].start_ms)
        end_ms = int(group[-1].end_ms) + TAIL_MS
        if end_ms - start_ms < MIN_DUR_MS:
            end_ms = start_ms + MIN_DUR_MS
        cues.append(Cue(start_ms=start_ms, end_ms=end_ms, text=join_words(group)))
    for idx, cue in enumerate(cues):
        clamp_to = cues[idx + 1].start_ms if idx + 1 < len(cues) else audio_end_ms
        cue.end_ms = min(cue.end_ms, clamp_to)
        if cue.end_ms <= cue.start_ms:
            cue.end_ms = cue.start_ms + int(TS_GRID_MS)
    return cues


def segment_to_cue(start_ms: int, end_ms: int, text: str) -> Cue:
    return Cue(start_ms=start_ms, end_ms=end_ms, text=text.strip())


def format_time(ms: int, vtt: bool) -> str:
    h = ms // 3_600_000
    m = (ms % 3_600_000) // 60_000
    s = (ms % 60_000) // 1_000
    millis = ms % 1_000
    sep = "." if vtt else ","
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{millis:03d}"


def format_cues(cues: list[Cue], vtt: bool = False, start_index: int = 1) -> str:
    out: list[str] = []
    idx = start_index
    for cue in cues:
        if not cue.text.strip():
            continue
        out.append(str(idx))
        out.append(f"{format_time(cue.start_ms, vtt)} --> {format_time(cue.end_ms, vtt)}")
        out.append(cue.text.strip())
        out.append("")
        idx += 1
    return "\n".join(out) + ("\n" if out else "")


def format_srt(cues: list[Cue]) -> str:
    return format_cues(cues, vtt=False, start_index=1)


def format_vtt(cues: list[Cue]) -> str:
    body = format_cues(cues, vtt=True, start_index=1)
    return "WEBVTT\n\n" + body
