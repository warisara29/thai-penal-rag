"""Tokenizer for lexical retrieval (BM25).

Thai has no word spaces, so proper tokenisation wants pythainlp. To keep the
scaffold dependency-free and runnable now, the default is a character n-gram
tokeniser (works reasonably for Thai lexical matching). Swap in pythainlp for
the production baseline — it's a one-line change behind this interface.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_THAI = re.compile(r"[฀-๿]+")
_ALNUM = re.compile(r"[A-Za-z0-9๐-๙]+")


def char_ngram_tokens(text: str, n: int = 3) -> list[str]:
    toks: list[str] = []
    for chunk in _WS.split(text):
        for m in _THAI.finditer(chunk):
            s = m.group()
            if len(s) <= n:
                toks.append(s)
            else:
                toks.extend(s[i:i + n] for i in range(len(s) - n + 1))
        toks.extend(m.group().lower() for m in _ALNUM.finditer(chunk))
    return toks


def pythainlp_tokens(text: str) -> list[str]:
    """Production tokeniser (needs `pip install pythainlp`)."""
    from pythainlp import word_tokenize  # lazy import
    return [t for t in word_tokenize(text, engine="newmm") if t.strip()]


def get_tokenizer(name: str = "char_ngram"):
    if name == "char_ngram":
        return char_ngram_tokens
    if name == "pythainlp":
        return pythainlp_tokens
    raise ValueError(f"unknown tokenizer {name!r} (use 'char_ngram' or 'pythainlp')")
