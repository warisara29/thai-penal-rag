"""C — fact categories for fact-guided traversal.

A question's facts are mapped (one LLM call) to the general-part chapters of the Penal Code
(ภาค 1 ลักษณะ 1, หมวด 4–8 = มาตรา 59–94). Those chapters are the *general provisions* that a
hybrid/dense retriever misses because they share little wording with the offence-specific query
— exactly the gap the court-weighted graph fills. F(q) then restricts the KG's reserved-slot
candidates to the matching chapter(s), so the graph contributes a legally relevant general
provision rather than any co-cited section.

The chapter→section map is derived from the corpus `chapter` metadata (not hardcoded); the
fact categories below just attach human-readable cues + the authoritative มาตรา range for the
prompt. The prompt string is kept verbatim for Appendix A.
"""

from __future__ import annotations

import json
import re

# หมวด (chapter number within ภาค 1 ลักษณะ 1) -> (heading, มาตรา range, example facts)
FACT_CATEGORIES = {
    4: ("ความรับผิดในทางอาญา", "59–79",
        "อายุผู้กระทำ, ป้องกันโดยชอบ, ความจำเป็น, บันดาลโทสะ, เจตนา/ประมาท, สำคัญผิด, รับสารภาพ/บรรเทาโทษ"),
    5: ("การพยายามกระทำความผิด", "80–82",
        "ลงมือกระทำแล้วไม่บรรลุผล, ยับยั้งเอง, กระทำไปตลอดแล้วแต่ไม่สำเร็จ"),
    6: ("ตัวการและผู้สนับสนุน", "83–89",
        "ร่วมกันกระทำผิด, ใช้/จ้างวาน/โฆษณาให้กระทำ, ผู้สนับสนุนช่วยเหลือก่อน/ขณะกระทำ"),
    7: ("การกระทำความผิดหลายบทหรือหลายกระทง", "90–91",
        "การกระทำเดียวผิดกฎหมายหลายบท, กระทำหลายกรรมต่างกัน"),
    8: ("การกระทำความผิดอีก", "92–94",
        "เคยต้องคำพิพากษาถึงที่สุดให้จำคุกแล้วกระทำผิดซ้ำ (เพิ่มโทษ)"),
}

# ---- Prompt (verbatim → Appendix A) --------------------------------------------------
FACT_SYSTEM = (
    "คุณเป็นผู้ช่วยวิเคราะห์ข้อเท็จจริงในคำถามกฎหมายอาญาไทย หน้าที่ของคุณคือระบุว่า "
    "ข้อเท็จจริงในคำถามเข้าข่ายหลักทั่วไปในภาค 1 ลักษณะ 1 ของประมวลกฎหมายอาญาหมวดใดบ้าง "
    "ตอบเป็น JSON เท่านั้น ห้ามอธิบายนอกเหนือจาก JSON"
)


def build_fact_user_prompt(question: str) -> str:
    cats = "\n".join(
        f"  {n}. {head} (มาตรา {rng}) — เช่น {ex}"
        for n, (head, rng, ex) in FACT_CATEGORIES.items()
    )
    return (
        f"คำถาม:\n{question}\n\n"
        f"หมวดหลักทั่วไป (ภาค 1 ลักษณะ 1):\n{cats}\n\n"
        "พิจารณาว่าข้อเท็จจริงในคำถามเกี่ยวข้องกับหมวดใดบ้าง (เลือกได้หลายหมวด หรือไม่เลือกเลย"
        " ถ้าคำถามเป็นเรื่องฐานความผิดล้วน ๆ ไม่มีประเด็นหลักทั่วไป) "
        'ตอบ JSON รูปแบบ: {"categories": [หมายเลขหมวด], "reason": "เหตุผลสั้น ๆ"}'
    )


def parse_fact_categories(data: dict) -> list[int]:
    """Validate the LLM JSON → list of valid chapter numbers (4–8)."""
    out = []
    for c in (data or {}).get("categories", []) or []:
        try:
            n = int(c)
        except (TypeError, ValueError):
            continue
        if n in FACT_CATEGORIES and n not in out:
            out.append(n)
    return out


# ---- Chapter → section map (from corpus) ---------------------------------------------
def load_chapter_sections(sections_path: str) -> dict[int, set[str]]:
    """{chapter_number: {section_id,…}} for ภาค 1 ลักษณะ 1 only (book 1, title 1)."""
    out: dict[int, set[str]] = {}
    for line in open(sections_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if (d.get("book") or {}).get("number") != "1" or (d.get("title") or {}).get("number") != "1":
            continue
        ch = (d.get("chapter") or {}).get("number")
        if ch is None:
            continue
        m = re.match(r"\d+", str(ch))
        if not m:
            continue
        out.setdefault(int(m.group()), set()).add(str(d["section_id"]))
    return out


def allowed_sections(categories: list[int], chapter_sections: dict[int, set[str]]) -> set[str]:
    """Union of section ids for the matched fact chapters (the F(q) general-provision set)."""
    s: set[str] = set()
    for c in categories:
        s |= chapter_sections.get(c, set())
    return s
