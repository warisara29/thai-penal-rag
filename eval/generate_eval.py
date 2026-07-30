"""
Generate a 250+ item evaluation set for the Thai Penal Code RAG benchmark,
LLM-drafted from the corpus (the "Eval set generation" box in the flow).

Design — gold is fixed by CONSTRUCTION, not by the model:
  We select a corpus section (or offence+general-provision pair) ourselves and
  hand its text to the drafter. The drafter only writes the natural-language
  question, the reference answer, and atomic answer_claims grounded in that
  text. gold_sections / supporting_sections are set by us, so retrieval labels
  can never be a model hallucination. The drafter (Claude Opus 4.8) is NOT the
  generator-under-test and NOT the judge — no eval-loop confound.

Six question types (schema.json enum): lookup, penalty, multi_hop, definition,
exception, unanswerable. Per-type targets live in generation_config.json.

Pools are built from data/sections.jsonl by structural + keyword filters and
sampled stratified across ลักษณะ (titles) so items don't clump in one area.

Usage:
  # 1. See the plan + one sample prompt per type. No API key needed.
  python eval/generate_eval.py --dry-run

  # 2. Generate for real (needs ANTHROPIC_API_KEY or `ant auth login`).
  python eval/generate_eval.py --out eval/eval_set.generated.jsonl --workers 4

  # 3. Then gate the output:
  python eval/validate_eval.py eval/eval_set.generated.jsonl \
         --schema eval/schema.json --sections data/sections.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SEED = 20260729  # deterministic sampling -> reproducible eval set

# --- structured-output schema the drafter must return (per item) ------------
DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "reference_answer": {"type": "string"},
        "answer_claims": {"type": "array", "items": {"type": "string"}},
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
    },
    "required": ["question", "reference_answer", "answer_claims", "difficulty"],
    "additionalProperties": False,
}

SYSTEM = (
    "คุณเป็นผู้เชี่ยวชาญกฎหมายอาญาไทยที่กำลังสร้างชุดข้อสอบทองคำ (gold eval set) "
    "เพื่อวัดระบบค้นคืนกฎหมาย (RAG) บนประมวลกฎหมายอาญา "
    "เขียนคำถามให้เป็นธรรมชาติแบบที่ประชาชนหรือนักกฎหมายถามจริง "
    "คำตอบอ้างอิงและ answer_claims ต้องมีที่มาจากตัวบทที่ให้ไว้เท่านั้น ห้ามแต่งเติมมาตราหรือโทษที่ไม่ปรากฏ "
    "answer_claims คือข้อเท็จจริงย่อย ๆ ที่ตรวจสอบได้ทีละข้อ"
)

# per-type instruction; {ctx} is the section text block we inject
TYPE_PROMPTS = {
    "lookup": (
        "ประเภท: lookup (ถามหามาตราเดียวที่ตอบตรง ๆ)\n"
        "จากตัวบทต่อไปนี้ ให้ตั้งคำถามหนึ่งข้อที่มีคำตอบตรงอยู่ในมาตรานี้ "
        "โดยคำถามต้องไม่เอ่ยเลขมาตรา เขียน reference_answer ที่อ้างถึงมาตรานี้และสาระสำคัญ\n\n{ctx}"
    ),
    "penalty": (
        "ประเภท: penalty (ถามฐานความผิดและ/หรืออัตราโทษ)\n"
        "จากตัวบทต่อไปนี้ซึ่งมีการกำหนดโทษ ให้ตั้งคำถามที่ผู้ถามต้องการทราบว่าการกระทำนี้ผิดฐานใดและมีโทษอย่างไร "
        "reference_answer ต้องระบุฐานความผิดและอัตราโทษตามที่ปรากฏในตัวบท ห้ามคำนวณโทษเกินกว่าที่เขียนไว้\n\n{ctx}"
    ),
    "definition": (
        "ประเภท: definition (ถามความหมายของถ้อยคำตามบทนิยาม)\n"
        "จากบทนิยามต่อไปนี้ ให้เลือกถ้อยคำที่นิยามไว้ 'หนึ่งคำ' แล้วตั้งคำถามถามความหมายของคำนั้น "
        "(นี่คือคำถามข้อที่ {k} จากมาตรานี้ กรุณาเลือกถ้อยคำคนละคำกับข้ออื่นเท่าที่ทำได้) "
        "reference_answer ต้องยกความหมายตามตัวบท\n\n{ctx}"
    ),
    "exception": (
        "ประเภท: exception (เหตุยกเว้นความผิด/ยกเว้นโทษ/ลดหย่อน)\n"
        "จากตัวบทต่อไปนี้ซึ่งเป็นเหตุยกเว้นหรือลดหย่อน ให้ตั้งคำถามในลักษณะสถานการณ์ว่าผู้กระทำยังต้องรับผิดหรือรับโทษหรือไม่ "
        "reference_answer ต้องอธิบายผลตามตัวบท (เช่น ไม่เป็นความผิด / ไม่ต้องรับโทษ / ลดโทษ)\n\n{ctx}"
    ),
    "multi_hop": (
        "ประเภท: multi_hop (ต้องใช้ทั้งฐานความผิดในภาค 2 และบทบัญญัติทั่วไปในภาค 1)\n"
        "ให้ตั้งคำถามสถานการณ์ที่ต้องอ้างทั้งมาตราฐานความผิด (gold) และบทบัญญัติทั่วไป (supporting) ด้านล่าง "
        "reference_answer ต้องอ้างทั้งสองมาตราและอธิบายผลทางกฎหมายที่เชื่อมกัน\n\n"
        "[มาตราฐานความผิด — gold]\n{gold_ctx}\n\n[บทบัญญัติทั่วไป — supporting: {gp_label}]\n{sup_ctx}"
    ),
    "unanswerable": (
        "ประเภท: unanswerable (คำถามนอกขอบเขตประมวลกฎหมายอาญา เพื่อทดสอบการปฏิเสธ)\n"
        "ให้แต่งคำถามกฎหมายที่ผู้ใช้จริงอาจถาม แต่ 'ไม่อยู่' ในประมวลกฎหมายอาญา ในหัวข้อ: {topic}\n"
        "reference_answer ต้องเป็นข้อความปฏิเสธว่าคำถามไม่อยู่ในขอบเขตของประมวลกฎหมายอาญา ตอบจากประมวลกฎหมายอาญาไม่ได้ "
        "และให้ answer_claims เป็นลิสต์ว่าง []"
    ),
}


def load_sections(path: Path):
    secs = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    return secs, {s["section_id"]: s for s in secs}


def title_key(s: dict) -> str:
    b = (s.get("book") or {}).get("number", "?")
    t = (s.get("title") or {}).get("number", "-")
    return f"B{b}.T{t}"


def stratified(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Pick n sections spread across ลักษณะ (round-robin over title buckets)."""
    buckets = defaultdict(list)
    for s in pool:
        buckets[title_key(s)].append(s)
    for b in buckets.values():
        rng.shuffle(b)
    order = list(buckets)
    rng.shuffle(order)
    picked, i = [], 0
    while len(picked) < min(n, len(pool)):
        progressed = False
        for k in order:
            if buckets[k]:
                picked.append(buckets[k].pop())
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
        i += 1
    return picked


def build_pools(secs: list[dict], gp_ids: set[str]):
    """Candidate sections per type, by structural + keyword filters."""
    def is_penalty(s):
        return "ระวางโทษ" in s["text"]

    def is_definition(s):
        return s["section_id"] == "1" or "หมายความว่า" in s["text"] or "หมายความรวมถึง" in s["text"]

    def book(s):
        return (s.get("book") or {}).get("number")

    def is_exception(s):
        # explicit exculpatory / mitigating language, any book
        # (Book-1 justification+excuse AND Book-2 offence-specific exceptions, e.g. defamation 329-331)
        return any(k in s["text"] for k in (
            "ไม่ต้องรับโทษ", "ไม่เป็นความผิด", "ไม่มีความผิด",
            "ผู้นั้นไม่ต้องรับโทษ", "ได้รับยกเว้นโทษ", "ยกเว้นโทษ", "ลดโทษ", "ศาลจะไม่ลงโทษ"))

    offences = [s for s in secs if book(s) == "2" and is_penalty(s)]
    return {
        "lookup": [s for s in secs if book(s) in ("2", "3")],
        "penalty": [s for s in secs if is_penalty(s) and book(s) in ("2", "3")],
        "definition": [s for s in secs if is_definition(s)],
        "exception": [s for s in secs if is_exception(s)],
        "multi_hop_offences": offences,
    }


def ctx_block(s: dict) -> str:
    return f"{s['heading']}\n{s['text']}"


def build_tasks(cfg, secs, by_id, rng):
    """Return a list of task dicts: {qtype, gold, supporting, prompt}."""
    gp_ids = {g["section_id"] for g in cfg["general_provisions"]}
    pools = build_pools(secs, gp_ids)
    tasks, report = [], []

    def add(qtype, prompt, gold, supporting=None):
        tasks.append({"qtype": qtype, "prompt": prompt, "gold": gold,
                      "supporting": supporting or []})

    for qtype in ("lookup", "penalty", "exception"):
        target = cfg["targets"][qtype]
        picked = stratified(pools[qtype], target, rng)
        report.append((qtype, len(picked), target))
        for s in picked:
            add(qtype, TYPE_PROMPTS[qtype].format(ctx=ctx_block(s)), [s["section_id"]])

    # definition: pool is tiny (มาตรา 1 holds many terms) -> sample WITH replacement,
    # asking the drafter to pick a different defined term each time.
    dtarget = cfg["targets"]["definition"]
    dpool = pools["definition"] or []
    report.append(("definition", dtarget if dpool else 0, dtarget))
    for k in range(dtarget if dpool else 0):
        s = dpool[k % len(dpool)]
        add("definition", TYPE_PROMPTS["definition"].format(k=k + 1, ctx=ctx_block(s)),
            [s["section_id"]])

    # multi_hop: offence x general provision
    target = cfg["targets"]["multi_hop"]
    offences = stratified(pools["multi_hop_offences"], target, rng)
    gps = cfg["general_provisions"]
    report.append(("multi_hop", len(offences), target))
    for i, off in enumerate(offences):
        gp = gps[i % len(gps)]
        sup = by_id.get(gp["section_id"])
        if not sup:
            continue
        prompt = TYPE_PROMPTS["multi_hop"].format(
            gold_ctx=ctx_block(off), sup_ctx=ctx_block(sup), gp_label=gp["label"])
        add("multi_hop", prompt, [off["section_id"]], [gp["section_id"]])

    # unanswerable: out-of-scope topics, cycled
    target = cfg["targets"]["unanswerable"]
    topics = cfg["unanswerable_topics"]
    report.append(("unanswerable", target, target))
    for i in range(target):
        prompt = TYPE_PROMPTS["unanswerable"].format(topic=topics[i % len(topics)])
        add("unanswerable", prompt, [])

    # top up lookup so the set clears 250 even when definition/exception pools fall short
    floor = cfg.get("total_floor", 255)
    if len(tasks) < floor:
        used = {t["gold"][0] for t in tasks if t["qtype"] == "lookup" and t["gold"]}
        extra_pool = [s for s in pools["lookup"] if s["section_id"] not in used]
        extra = stratified(extra_pool, floor - len(tasks), rng)
        report.append(("lookup(topup)", len(extra), floor - len(tasks)))
        for s in extra:
            add("lookup", TYPE_PROMPTS["lookup"].format(ctx=ctx_block(s)), [s["section_id"]])

    return tasks, report


def draft_one(client, model, task):
    """Call the drafter; return a fully-assembled eval item dict (no id yet)."""
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium",
                       "format": {"type": "json_schema", "schema": DRAFT_SCHEMA}},
        system=SYSTEM,
        messages=[{"role": "user", "content": task["prompt"]}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    d = json.loads(text)
    item = {
        "question": d["question"].strip(),
        "question_type": task["qtype"],
        "gold_sections": task["gold"],
        "supporting_sections": task["supporting"],
        "reference_answer": d["reference_answer"].strip(),
        "answer_claims": [] if task["qtype"] == "unanswerable" else d.get("answer_claims", []),
        "difficulty": d.get("difficulty", "medium"),
        "annotator": "llm-drafted:" + model,
    }
    return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("eval/generation_config.json"))
    ap.add_argument("--sections", type=Path, default=Path("data/sections.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("eval/eval_set.generated.jsonl"))
    ap.add_argument("--dry-run", action="store_true",
                    help="build the plan + print one sample prompt per type; no API calls")
    ap.add_argument("--limit", type=int, default=0, help="cap total items (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text("utf-8"))
    secs, by_id = load_sections(args.sections)
    rng = random.Random(SEED)
    tasks, report = build_tasks(cfg, secs, by_id, rng)
    if args.limit:
        tasks = tasks[: args.limit]

    print("=== generation plan (pool_picked / target) ===")
    for qtype, got, target in report:
        flag = "" if got >= target else "  <-- POOL SHORTFALL"
        print(f"  {qtype:13s} {got:3d} / {target}{flag}")
    print(f"  {'TOTAL':13s} {len(tasks)} tasks")

    if args.dry_run:
        print("\n=== one sample prompt per type ===")
        seen = set()
        for t in tasks:
            if t["qtype"] in seen:
                continue
            seen.add(t["qtype"])
            print(f"\n--- {t['qtype']} | gold={t['gold']} supporting={t['supporting']} ---")
            print(t["prompt"][:600])
        print("\n(dry run — no items written. Re-run without --dry-run to draft via the API.)")
        return

    import anthropic  # imported lazily so --dry-run needs no SDK/key
    client = anthropic.Anthropic()
    model = cfg["drafter_model"]

    items, failures = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(draft_one, client, model, t): i for i, t in enumerate(tasks)}
        for done, fut in enumerate(as_completed(futs), 1):
            try:
                items.append((futs[fut], fut.result()))
            except Exception as e:
                failures += 1
                print(f"  ! task {futs[fut]} failed: {e}", file=sys.stderr)
            if done % 20 == 0:
                print(f"  drafted {done}/{len(tasks)} (failures={failures})")

    items.sort(key=lambda x: x[0])  # stable order -> stable ids
    start = cfg["id_start"]
    lines = []
    for offset, (_, it) in enumerate(items):
        it = {"id": f"{cfg['id_prefix']}-{start + offset:04d}", **it}
        lines.append(json.dumps(it, ensure_ascii=False))
    args.out.write_text("\n".join(lines), "utf-8")
    print(f"\n✓ wrote {len(lines)} items -> {args.out}  (failures={failures})")
    print(f"  next: python eval/validate_eval.py {args.out} "
          f"--schema eval/schema.json --sections {args.sections}")


if __name__ == "__main__":
    main()
