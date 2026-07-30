# Annotation guide — Thai Penal Code RAG eval set

Goal: a gold set that lets us score **retrieval** and **generation separately**,
and that stresses the cases where PageIndex / KG beat a plain vector baseline.

## Workflow
1. Write the question in Thai as a real user would ask it.
2. Open the corpus (`sections.jsonl`) and find the section(s) that answer it.
3. Fill one JSON object per line following `schema.json`. Validate with
   `python validate_eval.py eval_set.jsonl --sections ../data/sections.jsonl`.

## The fields that matter most
- **`gold_sections`** — the section(s) that *directly* answer the question.
  This is the retrieval label (Recall@k / MRR are scored against it). Keep it
  tight: only sections a correct answer must cite.
- **`supporting_sections`** — general provisions (usually Book 1 / ภาค 1) needed
  to *reason* but not the headline answer — e.g. มาตรา 80 (พยายาม), 83 (ตัวการ),
  59 (เจตนา). Filling these well is what makes the set measure multi-hop.
- **`answer_claims`** — break the reference answer into atomic facts. Each claim
  is later checked for grounding against gold+supporting sections. This catches
  "right section, wrong penalty" and invented numbers.

## Target distribution (aim, per ~250 items)
| question_type | share | why |
|---|---|---|
| `lookup`       | 25% | baseline; single section |
| `multi_hop`    | 30% | **the discriminating cases** — offence + general provision |
| `penalty`      | 20% | penalty stated/derived correctly |
| `definition`   | 10% | บทนิยาม, มาตรา 1 |
| `exception`    | 10% | ยกเว้นความผิด / เหตุลดหย่อน (ป้องกัน, จำเป็น) |
| `unanswerable` | 5%  | out-of-scope → must **refuse**, not hallucinate |

Start with **50 items** end-to-end (all types) to debug the harness, then scale
to 200–300. Two annotators on a 20% overlap → report inter-annotator agreement
on `gold_sections`.

## Rules of thumb
- Section IDs use the parser's canonical form: `288`, `335/1`, `33 ทวิ`.
- Don't paste section text into `reference_answer` verbatim — write the answer a
  lawyer would give, then cite the มาตรา.
- If the answer depends on which amendment is in force, set
  `applicable_version_date` and note it.
- `unanswerable` items: `gold_sections` MUST be empty; `reference_answer` is the
  expected refusal wording.

## Metrics computed from this set
- Retrieval: **Recall@k, MRR** over `gold_sections`; multi-hop coverage over
  `gold ∪ supporting`.
- Generation: **claim grounding rate** (answer_claims supported by retrieved
  sections), correctness vs `reference_answer`, **hallucinated-section rate**
  (cited มาตรา not in corpus), and refusal accuracy on `unanswerable`.
