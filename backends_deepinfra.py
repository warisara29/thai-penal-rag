"""DeepInfra-hosted backends — fills every model seam via one provider so the whole
pipeline runs without local GPUs. OpenAI-compatible API (chat + embeddings) plus the
inference endpoint for the reranker.

Wire in the config files (module:Class):
  retrieval/config.json  backends.embedder  = "backends_deepinfra:DeepInfraEmbedder"
                         backends.reranker  = "backends_deepinfra:DeepInfraReranker"
                         backends.navigator = "backends_deepinfra:DeepInfraNavigator"
  generation/config.json backends.generator = "backends_deepinfra:DeepInfraGenerator"
                         backends.judge      = "generation.judge:AnthropicJudge"   # != generator

Env: export DEEPINFRA_API_KEY=…   (pip install openai requests)

Models (override via env):
  DEEPINFRA_GEN_MODEL     Qwen/Qwen3.6-35B-A3B   (generator + PageIndex navigator; §4c same model)
  DEEPINFRA_EMBED_MODEL   BAAI/bge-m3
  DEEPINFRA_RERANK_MODEL  Qwen/Qwen3-Reranker-4B

Note: same model drives generation AND PageIndex navigation (design §4c, no confound),
and it is DIFFERENT from the judge (AnthropicJudge) — no self-preference bias.
Determinism: temperature 0. Qwen3 'thinking' blocks (<think>…</think>) are stripped.
"""

from __future__ import annotations

import json
import os
import re

from retrieval.backends import NotConfigured
from retrieval.base import Corpus
from retrieval.pageindex import LLMNodeSelector, PageIndexNavigator
from generation import prompts

BASE_URL = os.environ.get("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
GEN_MODEL = os.environ.get("DEEPINFRA_GEN_MODEL", "Qwen/Qwen3.6-35B-A3B")
EMBED_MODEL = os.environ.get("DEEPINFRA_EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("DEEPINFRA_RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")
# Judge + eval drafter — MUST differ from GEN_MODEL (different vendor, no self-preference bias)
JUDGE_MODEL = os.environ.get("DEEPINFRA_JUDGE_MODEL", "deepseek-ai/DeepSeek-V4-Pro")
DRAFT_MODEL = os.environ.get("DEEPINFRA_DRAFT_MODEL", "deepseek-ai/DeepSeek-V4-Pro")

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _key() -> str:
    k = os.environ.get("DEEPINFRA_API_KEY")
    if not k:
        raise NotConfigured("set DEEPINFRA_API_KEY (https://deepinfra.com/dash/api_keys) "
                            "and `pip install openai requests`")
    return k


class _Client:
    """Lazy OpenAI client so import/instantiation needs no key until first call."""
    _c = None

    @classmethod
    def get(cls):
        if cls._c is None:
            key = _key()  # NotConfigured if unset
            try:
                from openai import OpenAI  # lazy
            except ImportError as e:
                raise NotConfigured("pip install openai") from e
            cls._c = OpenAI(base_url=BASE_URL, api_key=key)
        return cls._c


def _chat(messages, max_tokens: int) -> tuple[str, int]:
    # Qwen3 is a reasoning model — disable thinking so temp-0 output is the final
    # answer (not an all-<think> reply that strips to empty). Two levers for
    # portability: vLLM chat_template_kwargs + the Qwen3 "/no_think" soft switch.
    msgs = list(messages)
    if msgs and msgs[0]["role"] == "system":
        msgs[0] = {**msgs[0], "content": msgs[0]["content"] + " /no_think"}
    try:
        r = _Client.get().chat.completions.create(
            model=GEN_MODEL, messages=msgs, temperature=0, max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception:  # server rejects the kwarg -> rely on /no_think alone
        r = _Client.get().chat.completions.create(
            model=GEN_MODEL, messages=msgs, temperature=0, max_tokens=max_tokens,
        )
    text = _THINK.sub("", r.choices[0].message.content or "").strip()
    tokens = getattr(r, "usage", None).total_tokens if getattr(r, "usage", None) else 0
    return text, tokens


class DeepInfraGenerator:
    """Generator backend: Qwen via DeepInfra, temperature 0."""

    def answer(self, question: str, context_ids: list[str], corpus: Corpus) -> tuple[str, dict]:
        user = prompts.build_user_prompt(question, context_ids, corpus)
        text, tokens = _chat(
            [{"role": "system", "content": prompts.GEN_SYSTEM},
             {"role": "user", "content": user}], max_tokens=1024)
        return text, {"llm_calls": 1, "tokens": tokens, "model": GEN_MODEL}


class DeepInfraNodeSelector(LLMNodeSelector):
    """PageIndex branch selector: the SAME Qwen model picks relevant children."""

    def _choose(self, query: str, options: list[str], max_choose: int) -> tuple[list[int], int]:
        numbered = "\n".join(f"{i}. {o[:200]}" for i, o in enumerate(options))
        user = (f"คำถาม: {query}\n\nรายการหัวข้อกฎหมาย (มีหมายเลขกำกับ):\n{numbered}\n\n"
                f"เลือกหมายเลขที่เกี่ยวข้องกับคำถามมากที่สุด ไม่เกิน {max_choose} หมายเลข "
                f"ตอบเป็นรายการหมายเลขเท่านั้น เช่น [0, 3]")
        text, tokens = _chat(
            [{"role": "system", "content": "คุณช่วยนำทางค้นหามาตราในประมวลกฎหมายอาญาโดยเลือกหัวข้อที่เกี่ยวข้อง"},
             {"role": "user", "content": user}], max_tokens=200)
        nums = [int(n) for n in re.findall(r"\d+", text)]
        seen, idxs = set(), []
        for n in nums:
            if 0 <= n < len(options) and n not in seen:
                seen.add(n); idxs.append(n)
        return idxs[:max_choose], tokens


class DeepInfraNavigator(PageIndexNavigator):
    """TreeNavigator backend for A3/A4 = PageIndex descent driven by Qwen."""

    def __init__(self, beam_width: int = 3):
        super().__init__(DeepInfraNodeSelector(), beam_width=beam_width)


class DeepInfraEmbedder:
    """Embedder backend: BGE-M3 via DeepInfra embeddings (unlocks R1/B2 + A1 dense leg)."""

    def encode(self, texts, batch: int = 100) -> list[list[float]]:
        out: list[list[float]] = []
        texts = list(texts)
        for i in range(0, len(texts), batch):
            r = _Client.get().embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch])
            out.extend(d.embedding for d in r.data)
        return out


def _chat_json(messages, max_tokens: int, model: str) -> tuple[dict, int]:
    """Structured JSON via OpenAI response_format=json_object (forces valid JSON,
    suppresses free-form reasoning). Used by the judge + eval drafter."""
    r = _Client.get().chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    text = _THINK.sub("", r.choices[0].message.content or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)  # be forgiving if wrapped in prose
    data = json.loads(m.group() if m else text)
    tokens = getattr(r, "usage", None).total_tokens if getattr(r, "usage", None) else 0
    return data, tokens


class DeepInfraJudge:
    """Judge backend on DeepInfra (DeepSeek by default) — DIFFERENT vendor from the
    Qwen generator, so no self-preference bias. Implements JudgeBackend."""

    def ground_claims(self, claims, context):
        if not claims:
            return []
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        user = (f"[บริบทตัวบท]\n{context}\n\n[ข้อความที่ต้องตรวจ]\n{numbered}\n\n"
                f"ตัดสินว่าแต่ละข้อ 'ได้รับการสนับสนุน' จากบริบทหรือไม่ ตอบ JSON: "
                f'{{"verdicts": [true/false เรียงตามลำดับข้อ]}}')
        try:
            v = _chat_json([{"role": "system", "content":
                             "คุณเป็นกรรมการตรวจการอ้างอิง ตัดสินจากบริบทที่ให้เท่านั้น ตอบ JSON"},
                            {"role": "user", "content": user}], 800, JUDGE_MODEL)[0]["verdicts"]
        except Exception:
            return [False] * len(claims)
        return [bool(x) for x in (v + [False] * len(claims))[:len(claims)]]

    def score_correctness(self, answer, reference):
        user = (f"[คำตอบอ้างอิง]\n{reference}\n\n[คำตอบที่ต้องประเมิน]\n{answer}\n\n"
                f'ให้คะแนน 1-5 (5=ถูกครบถ้วน) ตอบ JSON: '
                f'{{"score": 1-5, "correct": true/false, "rationale": "เหตุผลสั้น ๆ"}}')
        try:
            d = _chat_json([{"role": "system", "content":
                             "คุณเป็นกรรมการให้คะแนนความถูกต้องของคำตอบกฎหมายอาญาไทยเทียบคำตอบอ้างอิง ตอบ JSON"},
                            {"role": "user", "content": user}], 800, JUDGE_MODEL)[0]
            return {"score": int(d.get("score", 1)), "correct": bool(d.get("correct", False)),
                    "rationale": str(d.get("rationale", ""))}
        except Exception as e:
            return {"score": 1, "correct": False, "rationale": f"judge error: {e}"}


def draft_json(system: str, user: str, max_tokens: int = 2000) -> dict:
    """Structured draft for eval/generate_eval.py when using the DeepInfra drafter."""
    return _chat_json([{"role": "system", "content": system},
                       {"role": "user", "content": user}], max_tokens, DRAFT_MODEL)[0]


class DeepInfraReranker:
    """Reranker backend: bge-reranker-v2-m3 via the DeepInfra inference endpoint.
    VERIFY the request/response shape against current DeepInfra docs for this model —
    adjust the two marked lines if the schema differs."""

    def rerank(self, query: str, candidate_ids: list[str], corpus: Corpus) -> list[str]:
        import requests
        docs = [corpus.by_id[c].content for c in candidate_ids if c in corpus.by_id]
        ids = [c for c in candidate_ids if c in corpus.by_id]
        if not ids:
            return candidate_ids
        url = f"https://api.deepinfra.com/v1/inference/{RERANK_MODEL}"
        payload = {"queries": [query] * len(docs), "documents": docs}          # <-- verify
        r = requests.post(url, headers={"Authorization": f"Bearer {_key()}"}, json=payload, timeout=60)
        r.raise_for_status()
        scores = r.json()["scores"]                                            # <-- verify
        return [i for _, i in sorted(zip(scores, ids), reverse=True)]
