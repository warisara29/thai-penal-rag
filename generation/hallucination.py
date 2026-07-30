"""Hallucinated-section rate (§6b) — deterministic, no LLM.

Extract every มาตรา the answer cites, then flag:
  hard : cited section NOT in the corpus  (invented authority)
  soft : cited section not in what was retrieved for that item (unsupported cite)
Section id form matches the corpus: arabic base, optional /n or ทวิ/ตรี ordinal.
"""

from __future__ import annotations

import re

THAI2AR = {ord(t): str(i) for i, t in enumerate("๐๑๒๓๔๕๖๗๘๙")}
ORDINALS = {"ทวิ": 2, "ตรี": 3, "จัตวา": 4, "เบญจ": 5, "ฉ": 6,
            "สัตต": 7, "อัฏฐ": 8, "นว": 9, "ทศ": 10, "เอกาทศ": 11, "ทวาทศ": 12}
_ORD = "|".join(ORDINALS)
CITE = re.compile(r"มาตรา\s*([๐-๙\d]+(?:/[๐-๙\d]+)?)\s*(" + _ORD + r")?")


def norm(num: str) -> str:
    return num.translate(THAI2AR)


def cited_sections(answer: str) -> list[str]:
    out = []
    for m in CITE.finditer(answer):
        base = norm(m.group(1))
        ordw = m.group(2)
        if ordw and "/" not in base:
            base = f"{base}/{ORDINALS[ordw]}"
        out.append(base)
    return list(dict.fromkeys(out))  # unique, ordered


def flag(answer: str, corpus_ids: set[str], retrieved_ids: set[str] | None = None) -> dict:
    cites = cited_sections(answer)
    hard = [c for c in cites if c not in corpus_ids]
    soft = [c for c in cites if retrieved_ids is not None and c in corpus_ids
            and c not in retrieved_ids]
    return {"cited": cites, "hard": hard, "soft": soft,
            "hard_rate": (len(hard) / len(cites)) if cites else 0.0,
            "soft_rate": (len(soft) / len(cites)) if cites else 0.0}


if __name__ == "__main__":
    corpus = {"334", "335", "80"}
    r = flag("ความผิดตามมาตรา ๓๓๔ และมาตรา 999 ประกอบมาตรา ๓๓๕ ทวิ", corpus, {"334"})
    assert r["cited"] == ["334", "999", "335/2"], r["cited"]
    assert r["hard"] == ["999", "335/2"], r["hard"]
    assert r["soft"] == [], r["soft"]  # 335/2 is hard (not in corpus), not soft
    print("hallucination self-test ok:", r)
