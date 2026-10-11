# Study configuration

`design.json` is the overview: six model allocations, 30-round horizon, group sizes, scaffolds, generation settings, estimands, and original-data digest.

`main_study/config.json` gives environment and agent settings. `main_study/generation_overrides.json` specifies model-specific output budgets. The other main-study JSON files are templates requiring measured deployment evidence, power documentation, and an explicit run approval before collection. They are not evidence that a new run is approved. `study_config.json` supports the pilot entry point; `local_smoke.json` and `local_episode.json` support local simulator/client checks. Root example JSON files demonstrate the corresponding input contracts.

The design used dated internal amendments, not external preregistration. Their original digests are retained in `design.json` and the validation code for provenance; author planning documents and scheduler settings are excluded. Reorganization changes code fingerprints, so historical approvals must not be reused to launch new collection.
