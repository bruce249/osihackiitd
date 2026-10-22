# Best LLM reasons

Each line is a reason from `results/llm_reasons.jsonl` whose chosen experiment was then measured as a hit.

- **BH_t1, round 1** (llm, seed 0). "This compares the related strong guanidine base MTBD against BTMG under the best ligand-additive combination." Outcome: yield 44.14, a hit, and the best yield on that task.
- **BH_t4, round 1** (llm, seed 1). "Keeps the promising A13 additive and L4 ligand while testing whether B3 improves on B2." Outcome: yield 85.89, a hit, and the best yield on that task.
- **BH_t5, round 3** (llm_anon, seed 2). "Checks whether the strong A13 result transfers to L2, which performed well with A15." Outcome: yield 99.62, a hit.
- **DA, round 2** (llm, seed 2). "Tests whether the strong yield at 0.153 M and 105 °C is retained or improved at the higher temperature." Outcome: yield 99.98, a hit.
- **DA, round 3** (hybrid, seed 2). "Evaluates whether polar DMAc can replace butyronitrile while preserving the strongest base, ligand, and operating conditions." Outcome: yield 99.22, a hit.
- **DA, round 5** (llm, seed 2). "This tests a small temperature reduction in a solvent already giving near-maximum yield with the strongest base, ligand, and concentration." Outcome: yield 100, a hit, and the best yield on that task.
