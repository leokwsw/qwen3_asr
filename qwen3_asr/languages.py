"""Supported Qwen3-ASR languages: English name + ISO 639 code."""

from __future__ import annotations

LANGUAGE_TABLE: tuple[tuple[str, str], ...] = (
    ("Chinese", "zh"),
    ("English", "en"),
    ("Cantonese", "yue"),
    ("Arabic", "ar"),
    ("German", "de"),
    ("French", "fr"),
    ("Spanish", "es"),
    ("Portuguese", "pt"),
    ("Indonesian", "id"),
    ("Italian", "it"),
    ("Korean", "ko"),
    ("Russian", "ru"),
    ("Thai", "th"),
    ("Vietnamese", "vi"),
    ("Japanese", "ja"),
    ("Turkish", "tr"),
    ("Hindi", "hi"),
    ("Malay", "ms"),
    ("Dutch", "nl"),
    ("Swedish", "sv"),
    ("Danish", "da"),
    ("Finnish", "fi"),
    ("Polish", "pl"),
    ("Czech", "cs"),
    ("Filipino", "fil"),
    ("Persian", "fa"),
    ("Greek", "el"),
    ("Romanian", "ro"),
    ("Hungarian", "hu"),
    ("Macedonian", "mk"),
)

SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(name for name, _ in LANGUAGE_TABLE)
_NAME_TO_CODE = {name.lower(): code for name, code in LANGUAGE_TABLE}
_CODE_TO_NAME = {code.lower(): name for name, code in LANGUAGE_TABLE}
for name, _ in LANGUAGE_TABLE:
    _NAME_TO_CODE[name.lower()] = next(code for n, code in LANGUAGE_TABLE if n == name)


def normalize_language(language: str | None) -> str | None:
    """Accept English names (`en`/`zh`/`Chinese`) and return canonical English names."""
    if language is None:
        return None
    trimmed = language.strip()
    if not trimmed:
        return None
    key = trimmed.lower()
    if key in _CODE_TO_NAME:
        return _CODE_TO_NAME[key]
    if key in _NAME_TO_CODE:
        # Title-case the canonical name from the table.
        for name, _code in LANGUAGE_TABLE:
            if name.lower() == key:
                return name
    return None


def language_to_iso639(name: str | None) -> str | None:
    canonical = normalize_language(name)
    if canonical is None:
        return None
    return _NAME_TO_CODE[canonical.lower()]


def supported_language_list() -> str:
    return ",".join(SUPPORTED_LANGUAGES)
