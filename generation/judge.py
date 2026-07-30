"""LLM-as-judge (§6b) — protocol, prompts, and a ready-to-use Anthropic judge.

Judge model MUST differ from the generator (self-preference bias), temperature 0,
fixed prompt. Two judgements:
  ground_claims(claims, context)  -> per-claim binary entailment  => claim-grounding rate
  score_correctness(answer, ref)  -> rubric 1-5 + binary correct   => answer correctness

AnthropicJudge uses Claude Opus 4.8 (!= the Thai open generator) with structured
output. It's optional — the deterministic metrics (hallucination, refusal) never
need it. `pip install anthropic` + credentials to enable.
"""

from __future__ import annotations

from typing import Protocol, Sequence


class NotConfigured(RuntimeError):
    pass


class JudgeBackend(Protocol):
    def ground_claims(self, claims: Sequence[str], context: str) -> list[bool]: ...
    def score_correctness(self, answer: str, reference: str) -> dict: ...


# --- prompts (pre-registered wording) --------------------------------------
GROUND_SYS = (
    "คุณเป็นกรรมการตรวจการอ้างอิง (entailment) พิจารณาว่าข้อความแต่ละข้อ "
    "'ได้รับการสนับสนุน' จากบริบทตัวบทที่ให้หรือไม่ ตัดสินเฉพาะจากบริบทที่ให้เท่านั้น"
)
GROUND_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {"type": "boolean"}}},
    "required": ["verdicts"], "additionalProperties": False,
}
CORRECT_SYS = (
    "คุณเป็นกรรมการให้คะแนนความถูกต้องของคำตอบกฎหมายอาญาไทย เทียบกับคำตอบอ้างอิง "
    "ให้คะแนน 1-5 (5=ถูกต้องครบถ้วน, 1=ผิด) และธง correct (true ถ้าถูกต้องในสาระสำคัญ)"
)
CORRECT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "correct": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["score", "correct", "rationale"], "additionalProperties": False,
}


class AnthropicJudge:
    """Claude Opus 4.8 as judge. model MUST differ from the Thai generator."""

    def __init__(self, model: str = "claude-opus-4-8"):
        import anthropic  # lazy
        self.client = anthropic.Anthropic()
        self.model = model

    def _json(self, system, schema, user):
        r = self.client.messages.create(
            model=self.model, max_tokens=1500,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium",
                           "format": {"type": "json_schema", "schema": schema}},
            system=system, messages=[{"role": "user", "content": user}],
        )
        import json
        return json.loads(next(b.text for b in r.content if b.type == "text"))

    def ground_claims(self, claims, context):
        if not claims:
            return []
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        user = f"[บริบทตัวบท]\n{context}\n\n[ข้อความที่ต้องตรวจ]\n{numbered}\n\nตอบ verdicts เรียงตามลำดับข้อ"
        v = self._json(GROUND_SYS, GROUND_SCHEMA, user)["verdicts"]
        return (v + [False] * len(claims))[:len(claims)]

    def score_correctness(self, answer, reference):
        user = f"[คำตอบอ้างอิง]\n{reference}\n\n[คำตอบที่ต้องประเมิน]\n{answer}"
        return self._json(CORRECT_SYS, CORRECT_SCHEMA, user)


def default_judge():
    class _Stub:
        def __getattr__(self, _):
            raise NotConfigured(
                "Judge not configured. Set generation/config.json backends.judge to "
                "'generation.judge:AnthropicJudge' (needs `pip install anthropic` + creds), "
                "or your own JudgeBackend. Judge MUST differ from the Thai generator.")
    return _Stub()
