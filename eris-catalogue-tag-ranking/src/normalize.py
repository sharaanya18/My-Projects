"""Title normalization shared by feature extraction and grouping."""
from __future__ import annotations

import re
import unicodedata

_PUNCT_DIGITS = re.compile(r"[0-9]+|[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_title(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _PUNCT_DIGITS.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text
