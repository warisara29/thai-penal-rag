# Controlled comparison — isolating the KG contribution (paper §4.4)

Two generation+judge runs on the SAME generator/judge as the main run, to separate
the two changes bundled in HYBRID+PI+KG − HYBRID+PI (KG added AND 5→4 primary).

- `hybrid_pi_k4.verdicts.jsonl` — HYBRID+PI fed its top-4 (context-size control).
  correct% = 0.668 vs 0.673 at k=5  → Δ = −0.005 (context size inconsequential).
- `full_primary4_kgoff.verdicts.jsonl` — the full system's OWN 4 primary sections,
  KG slot removed (item-wise paired KG-off). correct% = 0.695 vs 0.718 with KG on
  → KG Δ = +0.023, McNemar exact p = 0.46 (n.s.); per type: multi_hop +0.178,
  lookup +0.027, definition 0.000, exception −0.040, penalty −0.073.

KG is multi-hop-specific, not a uniform lift. Regenerate rankings by truncating
runs/2026-08-07/retrieval/{A5,A4B}.jsonl to top-4 and re-running generation+judge.
