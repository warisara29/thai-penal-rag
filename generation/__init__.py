"""Generation + LLM-judge layer (design §6b).

Deterministic metrics (hallucination, refusal) run with no LLM. Generation needs
a Generator backend (the Thai open model); the LLM-judge metrics need a
JudgeBackend (a DIFFERENT model — AnthropicJudge ships ready to use).
"""
