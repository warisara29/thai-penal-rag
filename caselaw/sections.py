"""Penal-Code citation normalization + corpus section index (tasks A2 / A4).

The HF dataset ships ``deka_citations.jsonl`` with ``sections_cited`` already parsed to
Arabic-numeral strings such as ``"ป.อ. มาตรา 269/5"``. This module turns those (and raw
judgment text) into the SAME section-id scheme the penal corpus uses, so citations can be
joined onto the 444-section Book-Title-Chapter tree.

Two id styles appear in the corpus, both handled here:
    slash   มาตรา ๒๖๙/๕   -> section_id "269/5"
    word    มาตรา ๒๗๗ ทวิ -> section_id "277/2"
The word -> slash convention (ทวิ->/2, ตรี->/3, ...) is NOT hardcoded for resolution: it is
derived from the corpus headings themselves by :class:`CorpusSectionIndex`.

Scope filter: only the current Penal Code (``ป.อ.``). Criminal procedure (``ป.วิ.อ.``), the
civil/commercial codes, and the pre-1957 ``กฎหมายลักษณะอาญา`` are dropped.
"""

from __future__ import annotations

import re

# ---- Thai digit handling -------------------------------------------------------------
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def to_arabic(s: str) -> str:
    return s.translate(_THAI_DIGITS)


# Inserted-section ordinal words, longest-first so "เอกาทศ" matches before "ทศ".
# Values follow Thai legislative drafting: ทวิ = the 2nd in the run (original = implied 1st),
# so มาตรา 277 ทวิ is the corpus's "277/2". Used only for the raw-text extractor's canonical
# form; authoritative resolution still goes through the corpus index.
_ORDINAL_WORDS = [
    ("เอกาทศ", 11), ("ทวาทศ", 12), ("จัตวา", 4), ("เบญจ", 5), ("สัตต", 7),
    ("อัฏฐ", 8), ("ทวิ", 2), ("ตรี", 3), ("ฉ", 6), ("นว", 9), ("ทศ", 10),
]
_WORD_RE = "|".join(w for w, _ in _ORDINAL_WORDS)

# A single parsed Penal cite tail: base number, optional /n, optional ordinal word.
_TAIL_RE = re.compile(rf"(\d+)\s*(?:/\s*(\d+))?\s*({_WORD_RE})?")


def parse_penal_cite(raw: str) -> str | None:
    """Normalize ONE ``sections_cited`` string to a canonical cite key, or ``None`` if it is
    not a Penal-Code citation.

    Canonical keys mirror the corpus id scheme where possible:
        "288"        plain
        "269/5"      slash form
        "277 ทวิ"    word form (resolved to a slash id by the corpus index)
    วรรค (paragraph) and (1)-style item qualifiers are stripped to section level.
    """
    s = raw.strip()
    # ป.อ. but NOT ป.วิ.อ. / ป.อ.ท. etc.  (ป.วิ.อ. starts "ป.ว", so prefix test is safe.)
    if not s.startswith("ป.อ."):
        return None
    tail = s.split("มาตรา", 1)[1] if "มาตรา" in s else s[len("ป.อ."):]
    tail = to_arabic(tail)
    tail = re.split(r"วรรค|\(", tail, maxsplit=1)[0].strip()  # drop paragraph / item
    m = _TAIL_RE.match(tail)
    if not m:
        return None
    base, slash, word = m.group(1), m.group(2), m.group(3)
    if slash:
        return f"{base}/{slash}"
    if word:
        return f"{base} {word}"
    return base


# ---- Raw-text extractor (task A2 core; independent cross-check for A3) ----------------
# Law-context markers. When one is seen while scanning left-to-right, subsequent bare
# "มาตรา N" runs are attributed to that code until the next marker. This handles judgments
# that name the code once and then list sections (incl. "ประกอบ"/"และ"/comma runs).
_PENAL_MARKERS = ("ประมวลกฎหมายอาญา", "ป.อ.")
_OTHER_MARKERS = (
    "ประมวลกฎหมายวิธีพิจารณาความอาญา", "ป.วิ.อ.",
    "ประมวลกฎหมายวิธีพิจารณาความแพ่ง", "ป.วิ.พ.",
    "ประมวลกฎหมายแพ่งและพาณิชย์", "ป.พ.พ.",
    "กฎหมายลักษณะอาญา", "ประมวลรัษฎากร", "ป.รัษฎากร",
    "พระราชบัญญัติ", "พ.ร.บ.", "พระราชกำหนด", "พ.ร.ก.", "ประมวลกฎหมายที่ดิน",
)
# A marker followed (soon) by a section run. Tokenizer splits text into (marker | section).
_MARKER_RE = re.compile("|".join(re.escape(m) for m in _PENAL_MARKERS + _OTHER_MARKERS))
# "มาตรา" or "ม." introduces a run; numbers/separators/ordinals continue it.
_SEC_INTRO_RE = re.compile(r"มาตรา|ม\.")
_NUM_RUN_RE = re.compile(
    rf"(\d+(?:/\d+)?)\s*({_WORD_RE})?"            # one section token (+optional ordinal word)
)
# separators that keep a run going: comma, และ, ประกอบ, spaces (no ^ anchor: .match(t,pos)
# already anchors at pos, and ^ would only fire at offset 0). Deliberately conservative: the
# dash form "288-60" is left unsplit, since en-dash also marks ranges/dates and over-splitting
# hurts precision — this extractor only corroborates the curated dataset field, not vice versa.
_RUN_SEP_RE = re.compile(r"[\s,.]*(?:และ|ประกอบ|กับ|ที่)?\s*")


def extract_from_text(text: str) -> list[str]:
    """Extract Penal-Code section citations from a raw judgment ``text``.

    Returns canonical cite keys (see :func:`parse_penal_cite`) in order of first appearance,
    deduplicated. Context-aware: a bare "มาตรา 288, 80" run is only kept when the nearest
    preceding law marker is the Penal Code. วรรค / (1) qualifiers are dropped to section level.
    """
    t = to_arabic(text)
    # Build a position-ordered list of markers so we know the law context at any offset.
    markers = [(m.start(), _law_of(m.group())) for m in _MARKER_RE.finditer(t)]
    out: list[str] = []
    seen: set[str] = set()
    for intro in _SEC_INTRO_RE.finditer(t):
        law = _context_at(markers, intro.start())
        if law != "penal":
            continue
        # walk the comma/และ/ประกอบ-separated number run after this "มาตรา"
        pos = intro.end()
        while True:
            sep = _RUN_SEP_RE.match(t, pos)
            pos = sep.end() if sep else pos
            nm = _NUM_RUN_RE.match(t, pos)
            if not nm:
                break
            base, word = nm.group(1), nm.group(2)
            key = f"{base} {word}" if word and "/" not in base else base
            if key not in seen:
                seen.add(key)
                out.append(key)
            pos = nm.end()
            # stop the run if the next non-space char is not a continuation separator
            nxt = t[pos:pos + 12]
            if not re.match(r"[\s,.]*(?:\d|และ|ประกอบ|กับ)", nxt):
                break
    return out


def _law_of(marker: str) -> str:
    return "penal" if marker in _PENAL_MARKERS else "other"


def _context_at(markers: list[tuple[int, str]], pos: int) -> str | None:
    """Law context of the nearest marker at or before ``pos`` (None if none precedes)."""
    law = None
    for start, l in markers:
        if start <= pos:
            law = l
        else:
            break
    return law


# ---- Corpus section index (task A4) ---------------------------------------------------
_HEADING_RE = re.compile(rf"มาตรา\s+([๐-๙\d]+)\s*(?:/\s*([๐-๙\d]+))?\s*({_WORD_RE})?")


class CorpusSectionIndex:
    """Resolve a canonical cite key (from :func:`parse_penal_cite` / :func:`extract_from_text`)
    to a corpus ``section_id``. Keys are built from the corpus itself, so the ทวิ/ตรี -> /n
    convention is learned from headings rather than assumed."""

    def __init__(self, section_ids: list[str], headings: dict[str, str]):
        self._ids = set(section_ids)
        self._key: dict[str, str] = {}
        for sid in section_ids:
            self._key[sid] = sid  # plain "288" and slash "269/5" resolve directly
        for sid, heading in headings.items():
            m = _HEADING_RE.search(heading or "")
            if not m:
                continue
            base = to_arabic(m.group(1))
            slash, word = m.group(2), m.group(3)
            if word:  # "มาตรา ๒๗๗ ทวิ" -> accept "277 ทวิ" and "277ทวิ"
                self._key[f"{base} {word}"] = sid
                self._key[f"{base}{word}"] = sid
            elif slash:  # "มาตรา ๒๙/๑" -> accept "29/1"
                self._key[f"{base}/{to_arabic(slash)}"] = sid

    @classmethod
    def from_sections_jsonl(cls, path: str) -> "CorpusSectionIndex":
        import json
        ids, heads = [], {}
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = str(d["section_id"])
            ids.append(sid)
            heads[sid] = d.get("heading", "")
        return cls(ids, heads)

    def resolve(self, cite_key: str | None) -> str | None:
        if not cite_key:
            return None
        return self._key.get(cite_key)

    def __contains__(self, cite_key: str) -> bool:
        return cite_key in self._key
