from __future__ import annotations

import re


_NUM_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(r"[^\w\s']")
_SP_RE = re.compile(r"\s+")
_TEXT_WS_RE = re.compile(r"\s+")


def normalize_vqa_answer(text: str) -> str:
    value = str(text or "").strip().lower()
    value = value.replace("\n", " ").replace("\t", " ")
    value = _PUNCT_RE.sub(" ", value)
    words = [_NUM_WORDS.get(word, word) for word in _SP_RE.split(value) if word]
    words = [word for word in words if word not in _ARTICLES]
    return " ".join(words).strip()


def normalize_text_answer(text: str) -> str:
    value = str(text or "").replace("\n", " ").replace("\t", " ").strip().lower()
    value = (
        value.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return _TEXT_WS_RE.sub(" ", value).strip()
