# Parser v3 saved-log sensitivity — POST-HOC / EXPLORATORY

Protocol candidate `v3_literal_whitespace_only_20261011`. v3 accepts only complete decision objects, additionally permitting literal newline, carriage-return and tab in strings. It does not complete truncated JSON, infer missing actions, or select among multiple decisions. Every v2-valid action is verified unchanged.

These are re-extracted first attempts on unchanged observed histories, not fresh trajectories under v3. Recovered actions may differ from historically executed repaired actions. Neither primary outcomes nor repair/fallback flags are overwritten.

## gpt-oss-120b

6/195 first-attempt failures recovered. v2 validity 88.39%; v3 validity 88.75%.

Decision: v3 does not resolve the dominant failures. Do not enroll the full cohort yet; inspect missing-response errors and run a fully instrumented budget/transport diagnostic under a new protocol. Saved outputs alone cannot certify token truncation.

## llama-3.1-8b-instruct

800/871 first-attempt failures recovered. v2 validity 92.59%; v3 validity 99.40%.

Decision: adopt the explicitly versioned v3 candidate in a separately registered new cohort, reporting prospective reliability during its first balanced block and retaining those episodes. The dominant literal-whitespace problem is fixable without inventing an action. This completes the CPU-only parser branch of Stage 0b; no extra Llama model calls were made.

## Conditional cooperation sensitivity

00b_parser_conditional_outcomes.csv reports v2-eligible and v3-eligible first-attempt cooperation alongside unchanged executed cooperation, with early/late summaries and 5,000 episode-bootstrap intervals within N cells. These are conditional estimates on the original observed histories, not an intervention rerunning those histories. Cells with no eligible episodes remain not estimable; they are not dropped from equal-N aggregation. Two old GPT-OSS seeds per cell make its bootstrap uncertainty particularly weak.