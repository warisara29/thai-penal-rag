"""Shared generation prompt — identical across A1-A4, C0, C1 (fair-comparison §4c).

The generator answers ONLY from the provided มาตรา context; with no context
(C0 closed-book) it must answer from memory or decline; for out-of-scope
questions it must decline rather than invent a section.
"""

from __future__ import annotations

from retrieval.base import Corpus

GEN_SYSTEM = (
    "คุณเป็นผู้ช่วยตอบคำถามกฎหมายอาญาไทย ตอบโดยอ้างอิงเฉพาะ 'ตัวบทที่ให้มา' เท่านั้น "
    "ระบุเลขมาตราที่ใช้ตอบเสมอ ถ้าตัวบทที่ให้มาไม่พอจะตอบ หรือคำถามไม่อยู่ในขอบเขตประมวลกฎหมายอาญา "
    "ให้ตอบว่า 'คำถามนี้ไม่อยู่ในขอบเขตของประมวลกฎหมายอาญา' อย่าแต่งเลขมาตราหรือโทษที่ไม่ปรากฏในตัวบท"
)


def build_context(context_ids: list[str], corpus: Corpus) -> str:
    if not context_ids:
        return "(ไม่มีตัวบทประกอบ)"
    blocks = []
    for sid in context_ids:
        s = corpus.by_id.get(sid)
        if s:
            blocks.append(f"{s.heading}\n{s.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, context_ids: list[str], corpus: Corpus) -> str:
    return (f"[ตัวบทที่ให้มา]\n{build_context(context_ids, corpus)}\n\n"
            f"[คำถาม]\n{question}\n\n"
            f"ตอบเป็นภาษาไทย พร้อมระบุเลขมาตราที่ใช้")
