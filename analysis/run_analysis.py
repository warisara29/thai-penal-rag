"""Analysis harness (design §7). Computes, for one outcome metric:
  - per-arm mean + paired-bootstrap 95% CI
  - pairwise Δ + CI + significance (McNemar / permutation) over the pre-registered family
  - Holm-Bonferroni adjustment across that family
  - the 2x2 KG×base interaction (bootstrap DiD; GLMM if --glmm and A1-A4 present)
Writes analysis/report.<metric>.json.

Runs today on whatever arms exist (R0 alone -> per-arm CI). Contrasts activate as
more arms are produced by retrieval/generation.

Usage:
  python -m analysis.run_analysis --metric hit --arms R0
  python -m analysis.run_analysis --metric hit --arms all
  python -m analysis.run_analysis --metric correct --arms all --glmm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import bootstrap as BS
from . import contrasts as C
from . import outcomes as O
from . import significance as S

ALL = ["A1", "A2", "A3", "A4", "R0", "R1", "C0", "C1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="hit", choices=["hit", "coverage", "correct",
                                                        "claim_grounding", "hallucination_hard"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--results-dir", type=Path, default=Path("retrieval/results"))
    ap.add_argument("--verdicts-dir", type=Path, default=Path("generation/verdicts"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis"))
    ap.add_argument("--glmm", action="store_true", help="also fit the mixed-effects 2x2 (needs statsmodels + A1-A4)")
    args = ap.parse_args()

    wanted = ALL if args.arms == "all" else [a.strip() for a in args.arms.split(",")]
    src = args.results_dir if args.metric in O.RETRIEVAL_METRICS else args.verdicts_dir
    arms = {}
    for a in wanted:
        if (src / f"{a}.jsonl").exists():
            o = O.load_arm(a, args.metric, args.k, args.results_dir, args.verdicts_dir)
            if o:
                arms[a] = o
    if not arms:
        print(f"no arm outputs for metric '{args.metric}' in {src}/ — run the upstream harness first")
        return

    report = {"metric": args.metric, "k": args.k, "per_arm": {}, "contrasts": {}, "interaction": None}
    print(f"metric={args.metric}@{args.k}  arms present: {sorted(arms)}\n")

    print("=== per-arm mean [95% bootstrap CI] ===")
    for a in sorted(arms):
        ci = BS.ci_mean(list(arms[a].values()))
        report["per_arm"][a] = ci
        print(f"  {a:4s} n={ci['n']:>3}  {ci['mean']:.3f}  [{ci['lo']:.3f}, {ci['hi']:.3f}]")

    fam = C.family_for(args.metric)
    pvals = {}
    print("\n=== pre-registered contrasts (Δ [CI], test p) ===")
    any_contrast = False
    for ct in fam:
        if ct["a"] not in arms or ct["b"] not in arms:
            continue
        any_contrast = True
        av, bv = O.paired(arms[ct["a"]], arms[ct["b"]])
        d = BS.ci_delta(av, bv)
        if ct["test"] == "mcnemar":
            t = S.mcnemar_exact(av, bv)
        else:
            t = S.permutation(av, bv)
        pvals[ct["label"]] = t["p"]
        report["contrasts"][ct["label"]] = {"rq": ct["rq"], "delta_ci": d, "test": ct["test"], **t}
        print(f"  [{ct['rq']}] {ct['label']}: Δ={d['delta']:+.3f} "
              f"[{d['lo']:+.3f}, {d['hi']:+.3f}]  p={t['p']:.4f}  (n={d['n']})")
    if not any_contrast:
        print("  (need >=2 of the paired arms present — produced as arms are run)")

    if pvals:
        holm = S.holm_bonferroni(pvals)
        report["holm_bonferroni"] = holm
        print("\n=== Holm-Bonferroni (FWER) ===")
        for label, h in holm.items():
            print(f"  {'REJECT' if h['reject'] else '  ns  '}  p_adj={h['p_adj']:.4f}  {label}")

    if all(a in arms for a in ("A1", "A2", "A3", "A4")):
        a1, a2, a3, a4 = (arms["A1"], arms["A2"], arms["A3"], arms["A4"])
        ids = [i for i in a1 if i in a2 and i in a3 and i in a4]
        did = BS.ci_interaction([a4[i] for i in ids], [a3[i] for i in ids],
                                [a2[i] for i in ids], [a1[i] for i in ids])
        report["interaction"] = did
        print(f"\n=== interaction (RQ2) (A4-A3)-(A2-A1): DiD={did['did']:+.3f} "
              f"[{did['lo']:+.3f}, {did['hi']:+.3f}]")
        if args.glmm:
            from . import mixed_effects as ME
            try:
                report["glmm"] = ME.fit_2x2(arms)
                print("  GLMM params:", report["glmm"]["params"])
            except RuntimeError as e:
                print("  GLMM skipped:", e)

    out = args.out_dir / f"report.{args.metric}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n✓ wrote {out}")


if __name__ == "__main__":
    main()
