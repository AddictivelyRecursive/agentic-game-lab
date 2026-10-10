**POST-HOC / EXPLORATORY**

## Task 0 — reproduction gate

Gate: **PASS**. 97/97 anchors pass. All estimates recomputed from actions and original reliability flags. Cooperation tolerance 0.01 pp, D 0.05 pp, counts exact; other values checked against their published rounding intervals. Welch episode sample variances use ddof=1; eight-cell weights and separate-mode OLS agree.

Llama D=14.01994 pp; Q(5)=18.22517000, p=0.00267705, I²=72.56541%.

No Tasks 1–4 may run following a failed gate without explicit acceptance of discrepancies. All subsequent work remains exploratory.

## PAPER INSERTS

### (a) Repair/fallback robustness

In post-hoc sensitivity analyses, Qwen's late free-form cooperation remained 0.00% among all 280 nonfallback decisions in rounds 26–30; none of the 280 late decisions used fallback. Thus fallback did not directly produce its late zero cooperation. High repair rates nevertheless made strict episode-filter estimates unestimable in some cells. DeepSeek's decline remained negative under every filter, while Llama's small baseline decline changed sign under some filters. Conditional first-attempt estimates can be selection-biased and do not identify a causal pipeline effect.

### (b) Heterogeneity robustness

The post-hoc heterogeneity result depended on Llama: excluding it gave Q(4)=5.41, p=0.248, I²=26.07%; every other single-model exclusion retained significance, supporting a claim that one model differs rather than broad robust heterogeneity.

### (c) Summary table caption

POST-HOC / EXPLORATORY comparison of free-form minus structured cooperation level, four-point group-size interaction D, free-form late-minus-early cooperation, and the fraction of episodes with exactly zero late cooperation. Levels average four N cells equally. D intervals use eight-cell Welch uncertainty; temporal intervals resample whole episodes within N cells (5,000 draws). Late and early cover rounds 26–30 and 1–5. Level, scaling and persistence measure different features of behavior; intervals containing zero do not establish equivalence.

### (d) Explanation audit

A post-hoc four-rule detector flagged 2,230/19,466 nonempty N=2/5 explanations (11.46%). Gemma's structured/free-form rates were 40.48%/42.56%; Llama's were 13.42%/9.79%. The 274 empty explanations were excluded. Rates among valid unrepaired decisions are reported separately. These estimates measure detectable inconsistencies under the stated rule-based detector, not misunderstanding, and apply only to N=2/5. Wilson intervals are descriptive; episode-bootstrap intervals address clustering. A blinded 150-case validation sample awaits human labels, so detector precision and recall remain unknown.


## Reproducibility and execution notes

No new episodes, inference, GPU allocation, or judge calls were performed. **Saved-call anomaly:** 238 error-only call records have no output/parser-version field: 118 DeepSeek repair calls, 96 GPT-OSS initial calls and 24 GPT-OSS repair calls. These are explicitly logged call failures, not unobserved submitted actions; submitted actions and original reliability flags remain available. An error-only first call is not first-attempt-valid. Parser-version None in the manifest refers to these records. The original explanation-export hashes serialize strings as JSON; a raw-text hash check initially failed, was diagnosed against sample_for_cot_coding.py, and was corrected to the documented hash convention before audit results were calculated. All hashes and text links then passed. The optional judge was omitted because access/authorization was not supplied; no requested rank correlations were skipped. Human labeling remains outstanding. Scripts can be rerun in small sequential CPU jobs on Sharanga, with Task 0 failure preventing all later tasks. Local task commits preserve completed checkpoints; final packaging does not overwrite the manuscript.

