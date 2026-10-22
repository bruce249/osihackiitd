# Which experiment next? LLMs versus Bayesian optimisation

**DRAFT.** This page is a slot for the final numbers. Anything produced from `writeup/fixture_runs.csv` is a schema check. It is not a finding.

## Problem

A reaction screen holds hundreds or thousands of conditions, and the practical budget is about 20 experiments. The decision is which condition to run next, given only the results so far. In this benchmark, running a condition is a lookup in a fully measured public dataset. That is the usual way to compare experiment planners, and it is not a live lab.

## Method

Every strategy sees the same five random starting conditions for a given task and seed, then proposes batches of five. Buchwald–Hartwig budgets are 20 experiments. Direct arylation, if it is in the final runs, is 30. Statistical arms use 10 seeds. LLM arms use 3. A hit is a condition in the top 1% by yield, or the top 3, whichever set is larger. A strategy never sees the yield of a condition it has not picked.

```
history of (condition, yield)
        |
        v
strategy proposes the next 5
(random, GP-EI, TabPFN or RF q90, LLM, or hybrid)
        |
        v
table lookup returns the yields
        |
        +----> updated history
```

The arms are random search; Gaussian-process expected improvement (`gp_ei`); TabPFN or a random-forest fallback scored by the 0.9 quantile (`tabpfn_ucb` or `rf_ucb`); an LLM chemist (`llm`); a hybrid that shortlists with the LLM and ranks with TabPFN (`hybrid`); and an anonymised-name ablation (`llm_anon`).

## Result

At the budget, mean normalised best-so-far (`metrics.overall` in `results/summary.json`) is 0.9049131137755609 for hybrid, 0.8872951257469133 for llm, 0.871984199600428 for gp_ei, 0.8102740745742515 for random, and 1.0 for oracle. Found-hit rate at that budget is 0.3888888888888889 for hybrid, llm, and llm_anon, 0.4444444444444444 for llm_zeroshot, 0.3442622950819672 for gp_ei, 0.3064516129032258 for random, and 1.0 for oracle.

**Figure 1 slot.** Insert `results/figures/figure1_norm_best.png` here only after `results/summary.json` has `"fixture": false`. The chart is mean normalised best-so-far against the number of experiments, with a 95% bootstrap band and a dashed line at n = 10.

No "X% versus Y%" headline belongs in this draft. If GP-EI is ahead in the final file, the sentence should say so. If the hybrid or the LLM is ahead only in the first 10 experiments, the sentence should say that, and only with the numbers from `summary.json`.

## What the LLM adds

The named LLM can use chemical names plus the yields observed so far. `llm_anon` hides the names, which is the check on chemistry knowledge versus pattern-matching on labels. `llm_zeroshot`, if that arm was run, commits to a batch with no feedback. A quoted reason will be copied from `results/best_reasons.md` when that file exists. None is invented here.

## Limitations

The datasets are retrospective lookup tables; a live lab adds noise and failed runs. These datasets are well known, so the LLM may have seen them in training. The anonymised ablation addresses this partly but cannot rule it out. The batch rule (top q by acquisition) is simple. There are few tasks and seeds, so treat effect sizes with caution.

## Related work

Shields et al., Nature 2021, showed Bayesian optimisation beating expert chemists on reaction optimisation. Ramos et al., 2023, studied Bayesian optimisation with LLM in-context learning (BO-ICL). Liu et al., ICLR 2024, introduced LLAMBO. Kristiadi et al., ICML 2024, gave a critical look at LLMs for Bayesian optimisation over molecules. Hollmann et al., Nature 2025, introduced TabPFN. Those results are theirs. The comparison assembled here is TabPFN as a surrogate on these reaction screens, an LLM-shortlist plus TabPFN-rank hybrid, a head-to-head across tasks, and the anonymised-names ablation.

## Links

- Repository: TBD
- Moment submission: TBD
- Video: TBD
