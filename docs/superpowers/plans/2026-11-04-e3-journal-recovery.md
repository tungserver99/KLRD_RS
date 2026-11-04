# Partial E3 clean-generation journal recovery

Authority: user's resume request and reported `JSONL corrupt before final
incomplete line` failure. Local work remains source-only, with read-only artifact
inspection for diagnosis; no Python, tests, inference, model or API execution.

Evidence: the synchronized22,451,121-byte train_clean journal contains1967 lines;
1965 parse as JSON, lines1010/1106 do not. All parseable prompt IDs belong to the
8000-request manifest and are unique. JSON parsing alone does not establish
record hash/seed/run validity. Cause of the malformed bytes is unconfirmed;
interrupted write/sync is plausible, not proven.

Cause in code: the copied reader can truncate only an incomplete final line and
correctly stops on malformed interior lines. E3 generation also assumed cached
records formed a prefix, so merely skipping two malformed lines would either
reject later records or resubmit already-generated responses.

- [x] Supply server-only tests for middle corruption, exact backups, final JSON
  without LF, bad hashes/identity/duplicates, completed-data protection and sparse
  ID-based resume. They are not executed locally.
- [x] Add an E3-owned recovery module; do not edit copied SCW source. Quarantine
  malformed JSON lines only after all parseable rows pass original prompt,
  identity, seed and record-hash checks. Back up original bytes and save a repair
  receipt before atomic replacement. Detect a source changing during recovery.
- [x] Enable recovery only for unconsumed clean-training train/validation journals,
  never overwrite completed journal/data/checkpoint authorities. Preserve old
  behavior for other generation paths.
- [x] Select pending requests by missing IDs, not cached row count. Preserve every
  validated row after a gap; return completed results in original prompt order.
- [x] Static source/whitespace checks and comparison of protected numerical source
  hashes. No local journal repair or experiment execution.

Ruling: actual recovery occurs on the server when rerunning the same launcher.
Do not modify the user's synchronized journal locally. If a parseable record is
inconsistent, stop without discarding it. Recovery preserves the fixed cohort;
only missing responses are regenerated, without score-based selection.

Static validation: whitespace checks clean for seven edited/new source/doc files;
all five numerical training source hashes still match the pilot manifest;
tracked E2/math/verifier files unchanged. The affected journal remains exactly
22,451,121 bytes with its original last-write timestamp. No local repair, Python,
tests, inference, evaluation or API calls were performed. Actual record-hash
validation and backed-up repair remain server operations on the next run.
