# Anonymous review release

Run `python tools/export_review.py` to create a code-and-manuscript ZIP using the explicit allowlist. The archive contains no `.git` directory, original remote URL, raw logs, private validation keys, local environments, or internal deployment receipts. `MANIFEST.json` records SHA-256 hashes. Exporting does not publish anything.

Upload the archive contents to a new anonymous repository or an anonymous artifact host. Do not fork or copy the original Git history. Use anonymous account details for any new commits, and inspect hosting profiles, links, PDF metadata, and the final downloadable artifact before sharing. A normal link to the existing owner repository is identifying even after file cleanup.

The cleanup removes local-only material from Git tracking without deleting it from disk. Originals of replaced overview documents and a list of paths removed from tracking are preserved in the ignored `.local_backups/acl_cleanup_20261012/` directory. Historical commits still contain removed files; do not publish this repository's history as an anonymous artifact.

The release intentionally excludes raw data. Therefore saved-log results cannot be independently reproduced using this archive alone. Do not change the manuscript to claim public code or data availability until the actual anonymous artifact is accessible and verified. No license grant has been invented; choose a license as the rights holder before a later general public release.
