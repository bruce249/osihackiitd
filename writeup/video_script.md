# Video script

About four minutes at a normal speaking pace. Do not speed the video up.

DRAFT. Do not read a chart whose title says FIXTURE. Do not invent a metric. The INSERT lines are replaced only from `results/summary.json` and `results/best_reasons.md`.

NOTE (do not speak): if Figure 1 is still the fixture chart, leave it off the recording.

## 0:00–0:30

A chemist improving a yield cannot try every condition. A Buchwald–Hartwig plate already holds hundreds of ligand, base, and additive combinations, and the direct-arylation screen has one thousand seven hundred twenty-eight. Each run costs time and starting material, so the working budget is about twenty experiments. The decision that matters is which experiment to run next, given only the results so far.

## 0:30–1:20

We compare planners under one protocol. Five random experiments open every run, identical across strategies for the same task and seed. Each method then picks batches of five. Buchwald–Hartwig stops at twenty experiments. Direct arylation stops at thirty.

Random search is the floor. Gaussian-process expected improvement is standard Bayesian optimisation. TabPFN, or a random-forest fallback, ranks the remaining conditions by the ninetieth percentile and takes the top five. The LLM reads the history and the chemical names and writes a reason for each pick. The hybrid asks the LLM for a shortlist of twenty and lets TabPFN rank it. An anonymised run hides those names, so we can see what the labels alone are worth.

## 1:20–2:30

The figure on screen is mean normalised best yield against the number of experiments. Normalised best yield is the best yield found so far, divided by the best yield available in that task. The dashed line marks experiment ten. Shading is a ninety-five percent bootstrap confidence interval over task and seed pairs.

INSERT THE REAL NUMBERS FROM summary.json.

Read the found-hit rate and the early point at ten experiments for the hybrid, for Gaussian-process expected improvement, and for random search. If standard Bayesian optimisation wins, say so, and then say what happens in the first ten runs.

The reasons on screen are copied from the log, not written for the camera. INSERT FROM best_reasons.md. INSERT FROM best_reasons.md.

A hit is a condition in the top one percent of that task, or the top three conditions, whichever set is larger. The median experiments to first hit counts the budget plus one when a run never hits.

## 2:30–3:15

The anonymised arm is how we ask whether the language model is using chemistry or memory. Hiding the names removes the route by which a model would recall a published table. If the anonymised planner falls behind the named one, the names were carrying information. If the two stay close, feedback from the yields may be doing the work. Neither outcome clears the training-set question. These datasets are famous, and a model may still have seen the yields. We treat the gap as a partial check, and we do not call it proof.

## 3:15–4:00

The notebook is the copilot. Choose a reaction, see five placeholder rows for the opening design, and press Suggest next five. You get names and reasons when a cache or a key exists. Otherwise the page tells you to set a key or wait for the cache. Nothing is billed just by opening the app.

Next is a live self-driving lab and reactions that have not been published. These screens are retrospective lookups, the batch rule is a simple top five, and there are few tasks and seeds, so treat the effect sizes with caution.
