# Stage 0b decision memo — POST-HOC / EXPLORATORY

Source SHA256: `149ac53f8a19286b429d1de83939ae91028845b0b35a18e1dc8dd5cc0468b778`. All first attempts for Llama and GPT-OSS were inspected. Seed 20261010; 5,000 episode-cluster bootstrap draws. No new inference.

Failure categories are mutually exclusive. A logged error without a saved response is classified as other, not an invented empty response. Errors attached to an available output are classified from that output. Unterminated thinking and empty output are identifiable, but the saved call schema lacks token usage and finish reasons; a token-limit diagnosis is therefore unconfirmed. Incomplete JSON is classified as malformed, not silently called truncation. Nested complete objects inside an incomplete action object do not establish a missing action field. v2 already accepts markdown fences and surrounding prose.

## gpt-oss-120b

195 first-attempt failures; 99 format/missing-field, 0 invalid-action/empty/unterminated-thinking; 0 ambiguous multiple-object outputs.

Parser v3 investigation warranted by the prespecified majority rule; do not enroll until a conservative extractor is specified and saved-log sensitivity is complete. Ambiguous multiple actions must remain failures.

- free_form: parse 83.33%; valid 83.33%; repair 16.67%; fallback 0.71%.
- structured: parse 93.45%; valid 93.45%; repair 6.55%; fallback 0.00%.

## llama-3.1-8b-instruct

871 first-attempt failures; 869 format/missing-field, 2 invalid-action/empty/unterminated-thinking; 0 ambiguous multiple-object outputs.

Parser v3 investigation warranted by the prespecified majority rule; do not enroll until a conservative extractor is specified and saved-log sensitivity is complete. Ambiguous multiple actions must remain failures.

- free_form: parse 89.68%; valid 89.64%; repair 10.36%; fallback 1.31%.
- structured: parse 95.54%; valid 95.54%; repair 4.46%; fallback 0.36%.

## Follow-up decision

The completed conservative-parser investigation and final enrollment recommendations are in PARSER_V3_SENSITIVITY.md. They supersede the initial investigation branches above; the historical data and primary outcomes remain unchanged.

## Caveats

Rates describe the original tested pipeline. Bootstrap intervals cannot correct two-seed uncertainty for GPT-OSS. Diagnostic categories do not identify a model’s intent. Any future cohort has a new protocol ID and is analyzed independently. No downstream cooperation values were used to assign categories or make the decision.

Observed saved first-call keys: ['error', 'json_extraction_version', 'output', 'stage'].
