"""Primary inferential model (design §7): mixed-effects logistic regression for
the 2x2 core, per-item binary outcome:

    outcome ~ base_retriever * kg_expansion + (1 | item)

The item random intercept models the paired design; the interaction term tests
whether KG helps more on PageIndex than on hybrid. Needs the four factorial arms
(A1 HYB, A2 HYB+KG, A3 PI, A4 PI+KG) and statsmodels+pandas. For the pure-Python
descriptive alternative the design also sanctions, use bootstrap.ci_interaction.

  A1: base=hybrid, kg=0   A2: base=hybrid, kg=1
  A3: base=pageindex, kg=0   A4: base=pageindex, kg=1
"""

from __future__ import annotations

ARM_FACTORS = {
    "A1": ("hybrid", 0), "A2": ("hybrid", 1),
    "A3": ("pageindex", 0), "A4": ("pageindex", 1),
}


def fit_2x2(outcomes_by_arm: dict[str, dict[str, float]]):
    """outcomes_by_arm: {arm: {item_id: 0/1}} for A1..A4. Returns a fitted model
    summary dict, or raises RuntimeError with guidance if prerequisites missing."""
    missing = [a for a in ARM_FACTORS if a not in outcomes_by_arm]
    if missing:
        raise RuntimeError(f"GLMM needs all four factorial arms; missing {missing}. "
                           f"Wire the model backends and run those arms first.")
    try:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError as e:
        raise RuntimeError("GLMM needs pandas+statsmodels (`pip install statsmodels pandas`). "
                           "Meanwhile use bootstrap.ci_interaction (pure Python).") from e

    rows = []
    ids = set.intersection(*[set(outcomes_by_arm[a]) for a in ARM_FACTORS])  # paired items
    for arm, (base, kg) in ARM_FACTORS.items():
        for iid in ids:
            rows.append({"item": iid, "base_retriever": base, "kg_expansion": kg,
                         "outcome": int(outcomes_by_arm[arm][iid])})
    df = pd.DataFrame(rows)
    model = BinomialBayesMixedGLM.from_formula(
        "outcome ~ base_retriever * kg_expansion", {"item": "0 + C(item)"}, df)
    res = model.fit_map()
    return {"n_items": len(ids), "n_obs": len(df),
            "params": dict(zip(res.model.exog_names, [float(x) for x in res.params])),
            "summary": str(res.summary())}
