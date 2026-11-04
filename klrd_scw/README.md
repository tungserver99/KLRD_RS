# Directly copied KLRD SCW support

scw_paraphrase.py, artifacts.py and paraphrase_report.py are unmodified byte-for-
byte copies from the user-supplied neighboring KLRD project. See
../docs/superpowers/references/e2b_scw_protocol.json for original paths/hashes.

The original SCW attribution in scw_paraphrase.py is retained. The pinned upstream
source and complete LICENSE_CODE are included under
../docs/superpowers/references/scw_15bc1929569357130f2dbc0b09f91bbf4f4bd947/.

This root-level package layout intentionally preserves the original .env lookup
at the KLRD_RS project root. No original KLRD checkout, .env or historical
paraphrase text is copied or used as new observations. The E2B integration calls
run_paraphrases directly and does not implement another API loop.
