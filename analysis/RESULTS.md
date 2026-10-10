**POST-HOC / EXPLORATORY**

## Task 0 — reproduction gate

Gate: **PASS**. 97/97 anchors pass. All estimates recomputed from actions and original reliability flags. Cooperation tolerance 0.01 pp, D 0.05 pp, counts exact; other values checked against their published rounding intervals. Welch episode sample variances use ddof=1; eight-cell weights and separate-mode OLS agree.

Llama D=14.01994 pp; Q(5)=18.22517000, p=0.00267705, I²=72.56541%.

No Tasks 1–4 may run following a failed gate without explicit acceptance of discrepancies. All subsequent work remains exploratory.

## Task 1 — repair/fallback robustness

Original repair/fallback flags define episode filters. Four N cells receive equal weight; retained-cell counts and <2 flags are in 01_retained_per_cell.csv. D is not estimable when any of eight cells has <2 episodes. First-attempt estimates are conditional and may be selection-biased. Spearman CIs resample episodes within N; undefined draws are counted and reported.

Qwen F late fallbacks: 0/280 = 0.00%; nonfallback late cooperation = 0.00% (280 decisions). Thus late zero cooperation is present among nonfallback decisions too.

**Llama: WEAKENED.** Baseline early/late 23.41/21.94%. Filtering produces inconsistent decline signs.
unfiltered: 23.41/21.94% (n=56); no_fallback: 24.24/30.20% (n=21); repair_le_25: 23.73/22.03% (n=55); repair_le_10: 24.73/21.79% (n=31); first_attempt_valid: 23.40/21.44% (n=56).

**Qwen: NOT ESTIMABLE.** Baseline early/late 22.57/0.00%. At least one filtered mode summary lacks an N cell; observed conditional estimates cannot resolve robustness across all filters.
unfiltered: 22.57/0.00% (n=16); no_fallback: nan/nan% (n=1); repair_le_25: nan/nan% (n=0); repair_le_10: nan/nan% (n=0); first_attempt_valid: 60.42/0.00% (n=16).

**Gemma: DOES NOT SURVIVE.** Baseline early/late 13.53/17.02%. No baseline early-then-decay pattern exists.
unfiltered: 13.53/17.02% (n=32); no_fallback: 13.53/17.02% (n=32); repair_le_25: 13.53/17.02% (n=32); repair_le_10: 13.53/17.02% (n=32); first_attempt_valid: 13.53/17.02% (n=32).

**Mistral: DOES NOT SURVIVE.** Baseline early/late 78.08/80.30%. No baseline early-then-decay pattern exists.
unfiltered: 78.08/80.30% (n=12); no_fallback: 78.08/80.30% (n=12); repair_le_25: 78.08/80.30% (n=12); repair_le_10: 78.08/80.30% (n=12); first_attempt_valid: 77.98/80.38% (n=12).

**DeepSeek: SURVIVES.** Baseline early/late 39.87/2.79%. The late-minus-early sign remains negative under every estimable filter; this is descriptive sign robustness, not equivalence of magnitudes.
unfiltered: 39.87/2.79% (n=64); no_fallback: 39.40/3.06% (n=57); repair_le_25: 39.87/2.79% (n=64); repair_le_10: 39.87/2.79% (n=64); first_attempt_valid: 39.91/2.68% (n=64).

**GPT-OSS: NOT ESTIMABLE.** Baseline early/late 12.54/0.00%. At least one filtered mode summary lacks an N cell; observed conditional estimates cannot resolve robustness across all filters.
unfiltered: 12.54/0.00% (n=8); no_fallback: 0.67/0.00% (n=6); repair_le_25: nan/nan% (n=5); repair_le_10: nan/nan% (n=4); first_attempt_valid: 13.51/0.00% (n=8).

Interpretation: no-fallback and low-repair restrictions select episodes and can change their composition. A sign-only SURVIVES verdict does not establish a mechanism; NOT ESTIMABLE is not evidence against decay. Constant late outcomes make several dose-response correlations undefined. For Gemma and Mistral, DOES NOT SURVIVE denotes absence of baseline decay, not a filtering-induced reversal.

