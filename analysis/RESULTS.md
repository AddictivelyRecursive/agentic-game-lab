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

## Task 2 — heterogeneity robustness

Inverse-variance leave-one-model-out analysis: **Heterogeneity robust to leaving out Llama? no.** Excluding Llama gives Q(4)=5.41083, p=0.24768, I²=26.07%. The more defensible claim is that one model differs; omnibus significance depends on Llama.

excluded         Q        p        I2  significant
   Llama  5.410833 0.247679 26.074229        False
    Qwen 15.456945 0.003842 74.121664         True
   Gemma 17.492055 0.001551 77.132475         True
 Mistral 16.614876 0.002296 75.925190         True
DeepSeek 15.003393 0.004694 73.339364         True
 GPT-OSS 17.776652 0.001364 77.498575         True

Parametric null simulated p: Normal 0.00245 (MC SE 0.00035), t-scale 0.08510 (MC SE 0.00197); 20,000 draws each. The t version uses scale=observed SE and Welch df floored at 2, so it deliberately has heavier variance. Observed Q is 18.22517000 (rounded 18.23).

                      method  draws  valid_draws  invalid_draws  observed_Q        p    mc_se                                                                                                           note       low       high  fraction
           parametric_normal  20000        20000              0    18.22517 0.002450 0.000350                                           Fixed observed SE; t uses SE as scale and df>=2, not unit-variance t       NaN        NaN       NaN
                parametric_t  20000        20000              0    18.22517 0.085096 0.001973                                           Fixed observed SE; t uses SE as scale and df>=2, not unit-variance t       NaN        NaN       NaN
         episode_bootstrap_Q   5000         4855            145         NaN      NaN      NaN Percentile CI CONDITIONAL on all six positive SEs; unconditional interval not estimable if invalid draws occur  9.719821 219.123148       NaN
        episode_bootstrap_I2   5000         4855            145         NaN      NaN      NaN Percentile CI CONDITIONAL on all six positive SEs; unconditional interval not estimable if invalid draws occur 48.558714  97.718178       NaN
episode_bootstrap_Llama_Holm   5000         4855            145         NaN      NaN      NaN    fraction conditional on estimable six-model Holm; low/high bound unconditional fraction by unresolved draws  0.631400   0.660400  0.650257

**Bootstrap limitation:** 145/5000 resamples have at least one zero/nonfinite model SE. No floors or silent deletions were used. Listed finite Q/I² intervals are conditional sensitivity summaries, not unconditional 95% intervals. Six-model Holm is estimable in 4855/5000 draws; conditional Llama significance fraction 0.6503, unconditional bounds [0.6314, 0.6604]. Per-draw results are saved.

Literal endpoints and four-point contrasts have these estimates (pp):
   model  D_fourpoint  low_fourpoint  high_fourpoint  D_endpoint  low_endpoint  high_endpoint
   Llama    14.019940       4.674071       23.365810   14.517857      4.823182      24.212533
    Qwen    -3.907292      -8.543097        0.728513   -4.000000     -9.173986       1.173986
   Gemma    -5.794271     -16.105064        4.516522  -14.531250    -24.473938      -4.588562
 Mistral    -6.812500     -16.933474        3.308474   -5.277778    -16.577539       6.021984
DeepSeek     5.728906      -3.317602       14.775414    6.239583     -3.234912      15.714078
 GPT-OSS    -7.870833    -116.986400      101.244734   -8.791667   -130.168059     112.584725

All six endpoint and four-point signs agree. Gemma illustrates a material difference: endpoint −14.53 pp, CI [−24.47, −4.59], marginal p=0.01061 (Holm p=0.05303), versus four-point −5.79 pp, CI [−16.11, 4.52]. Thus a marginal endpoint interval excluding zero should not be presented as six-model Holm significance.

Verdict: distinguish model sensitivity of the omnibus test from an assertion that all models differ. Endpoint and four-point definitions need not agree in magnitude because intermediate N cells contribute only to the latter. The pooled forest marker uses a normal fixed-weight CI and is descriptive; six estimated model SEs and sparse GPT-OSS cells limit chi-square and bootstrap calibration.

## Task 3 — persistence and summary grid

All uncertainty resamples episodes within model-mode-N cells (5,000 draws), averaging four N cells equally. Slopes are episode OLS slopes on rounds 1–30, then averaged. Zero-late means every late-round seat action is 4. Crossing is strictly below half the opening mean; zero opening yields a zero threshold and never. Never bootstrap outcomes are reported separately from finite-crossing quantiles.

   model  level_gap_pp         D  late_minus_early  late_minus_early_low  late_minus_early_high  fraction_zero_late half_opening_round
   Llama     13.913442 14.019940         -1.473214             -6.171763               3.656399              0.2500              never
    Qwen      6.105035 -3.907292        -22.572917            -26.312760             -18.843229              1.0000                  2
   Gemma     13.018663 -5.794271          3.489583             -3.424805              10.912109              0.5625              never
 Mistral     73.489583 -6.812500          2.215278             -4.639063               8.264063              0.0000              never
DeepSeek     17.044922  5.728906        -37.075521            -41.519596             -32.618457              0.6250                  6
 GPT-OSS      2.454861 -7.870833        -12.541667            -24.458333              -0.500000              1.0000                  5

           x                y  n      rho                                                          note
level_gap_pp       absolute_D  6 0.028571 n=6, descriptive only; no generalization to model populations
level_gap_pp late_minus_early  6 0.142857 n=6, descriptive only; no generalization to model populations

Verdict: level, N scaling and temporal persistence describe different behavior. Mistral has high sustained cooperation; Qwen and GPT-OSS have zero late cooperation in every episode, and DeepSeek shows a large mean decline. Llama has a small descriptive decline, while Gemma rises on average. The figure classes use only the sign of late-minus-early and do not imply a significant change; consult bootstrap CIs. Rank correlations involve only six named models and are descriptive only.

## Task 4 — explanation mismatch audit

Frozen four-rule detector (SHA256 828a912f5d83158318cdc9bbec68e313325b151d04c904da13187e5a845bdf2f) audited 19,466 nonempty explanations from 19,740 N=2/5 decisions. The 274 empty explanations were excluded and reported separately. Flags: 2,230/19,466 (11.46%). Both known examples were caught, as documented in 04_known_examples.csv. Export identity, explanation text hashes and canonical decision linkage all verified.

   model       mode            scope  decisions  flagged   percent  wilson_low  wilson_high  cluster_low  cluster_high
DeepSeek  free_form     all_nonempty       3317       66  1.989750    1.567089     2.523485     1.506024      2.501526
DeepSeek  free_form valid_unrepaired       3285       64  1.948250    1.528689     2.480062     1.459410      2.462025
DeepSeek structured     all_nonempty       3343       53  1.585402    1.214154     2.067790     1.138406      2.090800
DeepSeek structured valid_unrepaired       3329       52  1.562031    1.193190     2.042532     1.141141      2.040204
 GPT-OSS  free_form     all_nonempty        415        1  0.240964    0.042549     1.352122     0.000000      0.487805
 GPT-OSS  free_form valid_unrepaired        351        1  0.284900    0.050310     1.595907     0.000000      0.625000
 GPT-OSS structured     all_nonempty        420        2  0.476190    0.130686     1.719407     0.000000      0.952381
 GPT-OSS structured valid_unrepaired        388        1  0.257732    0.045511     1.445260     0.000000      0.534759
   Gemma  free_form     all_nonempty       1680      715 42.559524   40.214845    44.938151    39.702381     45.358631
   Gemma  free_form valid_unrepaired       1680      715 42.559524   40.214845    44.938151    39.761905     45.476190
   Gemma structured     all_nonempty       1680      680 40.476190   38.153358    42.842478    37.083333     43.870536
   Gemma structured valid_unrepaired       1680      680 40.476190   38.153358    42.842478    37.023810     43.869048
   Llama  free_form     all_nonempty       2839      278  9.792180    8.752632    10.940393     7.722144     12.008536
   Llama  free_form valid_unrepaired       2594      258  9.946029    8.852886    11.157629     7.771374     12.371565
   Llama structured     all_nonempty       2907      390 13.415893   12.225098    14.703248    11.712022     15.104802
   Llama structured valid_unrepaired       2823      376 13.319164   12.115462    14.622559    11.444595     15.072410
 Mistral  free_form     all_nonempty        630        3  0.476190    0.162077     1.390592     0.158730      0.793651
 Mistral  free_form valid_unrepaired        625        3  0.480000    0.163375     1.401640     0.160256      0.797448
 Mistral structured     all_nonempty        630       28  4.444444    3.092622     6.348455     2.698413      6.349206
 Mistral structured valid_unrepaired        624       28  4.487179    3.122532     6.408769     2.720000      6.420546
    Qwen  free_form     all_nonempty        773        6  0.776197    0.356209     1.683005     0.511509      1.158338
    Qwen  free_form valid_unrepaired         55        0  0.000000    0.000000     6.528490     0.000000      0.000000
    Qwen structured     all_nonempty        832        8  0.961538    0.488016     1.885815     0.241255      1.682692
    Qwen structured valid_unrepaired        258        1  0.387597    0.068453     2.162461     0.000000      1.158301

Verdict: these are **rates of detectable inconsistencies under a rule-based detector**, not misunderstanding. Prevalence is exploratory and limited to N=2/5. Explicit local assertions improve specificity but miss paraphrases and complicated semantics; no aggregate outcome tuning was performed. R1 may flag an incorrect action-map assertion even if it describes prior play. Wilson intervals are requested descriptive binomial intervals and do not account for dependence; episode-stratified bootstrap intervals accompany them. Repairs and fallback exclusions yield a conditional, potentially selected subset.

Validation: 150 blinded decisions, 75 flagged and 75 unflagged, stratified across models, exported with blank labels. The private key records unequal sampling fractions. No labels or precision/recall results were fabricated. score_validation.py requires real supplied labels; weighted recall corrects the case-control sampling ratio, with approximate effective-sample Wilson intervals. Human validation is pending. Optional LLM judge skipped: no API access or explicit judge authorization was supplied.

## PAPER INSERTS

### (a) Repair/fallback robustness

In post-hoc sensitivity analyses, Qwen's late free-form cooperation remained 0.00% among all 280 nonfallback decisions in rounds 26–30; none of the 280 late decisions used fallback. Thus fallback did not directly produce its late zero cooperation. High repair rates nevertheless made strict episode-filter estimates unestimable in some cells. DeepSeek's decline remained negative under every filter, while Llama's small baseline decline changed sign under some filters. Conditional first-attempt estimates can be selection-biased and do not identify a causal pipeline effect.

### (b) Heterogeneity robustness

The post-hoc heterogeneity result depended on Llama: excluding it gave Q(4)=5.41, p=0.248, I²=26.07%; every other single-model exclusion retained significance, supporting a claim that one model differs rather than broad robust heterogeneity.

### (c) Summary table caption

POST-HOC / EXPLORATORY comparison of free-form minus structured cooperation level, four-point group-size interaction D, free-form late-minus-early cooperation, and the fraction of episodes with exactly zero late cooperation. Levels average four N cells equally. D intervals use eight-cell Welch uncertainty; temporal intervals resample whole episodes within N cells (5,000 draws). Late and early cover rounds 26–30 and 1–5. Level, scaling and persistence measure different features of behavior; intervals containing zero do not establish equivalence.

### (d) Explanation audit

A post-hoc four-rule detector flagged 2,230/19,466 nonempty N=2/5 explanations (11.46%). Gemma's structured/free-form rates were 40.48%/42.56%; Llama's were 13.42%/9.79%. The 274 empty explanations were excluded. Rates among valid unrepaired decisions are reported separately. These estimates measure detectable inconsistencies under the stated rule-based detector, not misunderstanding, and apply only to N=2/5. Wilson intervals are descriptive; episode-bootstrap intervals address clustering. A blinded 150-case validation sample awaits human labels, so detector precision and recall remain unknown.


## Sharanga verification

All six short CPU jobs completed successfully (388781–388786), each requesting the compute/cpulimit minimum of four CPUs, 4 GB RAM and ten minutes; zero GPUs. Task wall times were 18, 32, 9, 9, 17 and 4 seconds. All 29 downloaded CSV tables match the local outputs at rtol=atol=1e-10. Both environments passed all 97 reproduction anchors. Remote statsmodels is 0.14.5 versus local 0.15.0; other pinned numerical/plotting packages match. See server_verification.json and SHARANGA.md for receipts and deployment issues. Scheduler dependencies handled CPU availability without any GPU polling loop.

## Reproducibility and execution notes

No new episodes, inference, GPU allocation, or judge calls were performed. **Saved-call anomaly:** 238 error-only call records have no output/parser-version field: 118 DeepSeek repair calls, 96 GPT-OSS initial calls and 24 GPT-OSS repair calls. These are explicitly logged call failures, not unobserved submitted actions; submitted actions and original reliability flags remain available. An error-only first call is not first-attempt-valid. Parser-version None in the manifest refers to these records. The original explanation-export hashes serialize strings as JSON; a raw-text hash check initially failed, was diagnosed against sample_for_cot_coding.py, and was corrected to the documented hash convention before audit results were calculated. All hashes and text links then passed. The optional judge was omitted because access/authorization was not supplied; no requested rank correlations were skipped. Human labeling remains outstanding. Scripts can be rerun in small sequential CPU jobs on Sharanga, with Task 0 failure preventing all later tasks. Local task commits preserve completed checkpoints; final packaging does not overwrite the manuscript.

