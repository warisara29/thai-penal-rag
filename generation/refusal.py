"""Refusal accuracy on `unanswerable` items (§6b) — deterministic, no LLM.

A correct system DECLINES out-of-scope questions instead of inventing a มาตรา.
We detect refusal by cue phrases; precision/recall/F1 are computed treating
"refused" as the positive class over the unanswerable subset vs. the rest.

The cue list is a starting point — tune it against your generator's actual
refusal wording, and cross-check a sample against the LLM judge.
"""

from __future__ import annotations

REFUSAL_CUES = [
    "ไม่อยู่ในขอบเขต", "ไม่สามารถตอบ", "ไม่ครอบคลุม", "อยู่นอกขอบเขต",
    "ไม่เกี่ยวกับประมวลกฎหมายอาญา", "ไม่ปรากฏในประมวลกฎหมายอาญา",
    "ตอบจากประมวลกฎหมายอาญาไม่ได้", "ไม่มีข้อมูล",
]


def is_refusal(answer: str) -> bool:
    return any(cue in answer for cue in REFUSAL_CUES)


def score(items: list[dict]) -> dict:
    """items: [{question_type, answer}]. Positive class = 'should refuse' (unanswerable)."""
    tp = fp = fn = tn = 0
    for it in items:
        should = it["question_type"] == "unanswerable"
        did = is_refusal(it["answer"])
        if should and did:
            tp += 1
        elif should and not did:
            fn += 1
        elif not should and did:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec and (prec + rec) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "refusal_precision": prec, "refusal_recall": rec, "refusal_f1": f1}


if __name__ == "__main__":
    rows = [
        {"question_type": "unanswerable", "answer": "คำถามนี้ไม่อยู่ในขอบเขตของประมวลกฎหมายอาญา"},
        {"question_type": "unanswerable", "answer": "ต้องใช้เอกสาร ก ข ค"},           # missed refusal
        {"question_type": "penalty", "answer": "ผิดมาตรา ๓๓๔ โทษจำคุก"},
        {"question_type": "lookup", "answer": "ไม่สามารถตอบได้"},                       # false refusal
    ]
    s = score(rows)
    assert s["tp"] == 1 and s["fn"] == 1 and s["fp"] == 1 and s["tn"] == 1, s
    print("refusal self-test ok:", s)
