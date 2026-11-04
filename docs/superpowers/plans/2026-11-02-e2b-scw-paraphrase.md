# E2B GPT-5/SCW Robustness Implementation Plan

> For agentic workers: execute inline with superpowers:executing-plans. Tests are
> prepared first and deferred to server under the explicit user restriction.

Goal:3600 fixed-protocol paraphrases and paired spherical verification of the E2
pilot, using original KLRD API code without rewriting it.
Spec: docs/superpowers/specs/2026-11-02-e2b-scw-paraphrase-design.md.

## Global constraints
No local API, tests, inference, evaluation, model download or installation.
No .env/key read or persistence. Exact copied model/prefix/API defaults.
Read-only E1/E2/KLRD artifacts. Fixed400 IDs, owners0..7+clean, Q20/50/100,
alpha0.01/0.0001, existing verifier and no new scientific gates.

## Tasks
- [x] Copy original transport, atomic helper and quality helper; record hashes,
  historical manifest evidence, pinned SCW reference/license and SDK version.
- [x] Write server-only contracts for deterministic shared selection, immutable
  source/checkpoint checks, independent per-condition receipts and payload privacy.
- [x] Implement source selection/preparation using validated E2 job and score
  receipts; freeze3600 rows, groups, original vectors and code/source provenance.
- [x] Integrate unchanged run_paraphrases calls, per-condition resume/failure
  reports and complete original-to-paraphrase mapping, without extra API calls.
- [x] Implement unchanged-verifier paraphrase scoring with batch checkpoints,
  source/runtime checks, original prompt retention and explicit long-text failure.
- [x] Implement paired group summaries, wrong/clean controls, cosine/p-values,
  confidence intervals, original quality helper and manual semantic review sample.
- [x] Add server launcher, separate requirements, README and artifact ignore rules.
- [x] Source/diff review and copy hash verification; independent packet review;
  fix important findings. Do not claim runtime/tests pass.

## Review focus
1. Duplicate shared prompt IDs across owners must not collide in original SCW cache.
2. Source metadata must never enter the GPT input or leak credentials.
3. Failed/incomplete API coverage cannot select a reduced cohort or trigger resampling.
4. Source cache, prompt alignment and before/after group membership must be bound.
5. Scoring resume must validate text, vectors, tokenizer/readout and runtime identity.
6. Lexical rewrite metrics must not be presented as semantic-preservation proof.

## Rulings / progress
- User specified direct-copy implementation; new wrapper code covers E2B data,
  spherical analysis and verification only. It does not duplicate API logic.
- Interpret GPT-5 as the exact GPT-5-family snapshot recovered from source and
  historical manifests (gpt-5-mini-2025-08-07); disclose this explicitly.
- Continue in the shared checkout without altering existing tracked E2 sources;
  no local execution or automatic git commit/push.

## Source implementation ledger / final review

All checked tasks above mean source/tests/docs are prepared, not executed.

- Direct-copy verification: all7 original transport/helper/reference/license files match their recorded SHA256. None of the three copied Python implementation files was edited.
- Historical-source link: SHA256 of the concatenated sorted25 original klrd/*.py files is 19c18f8fbaa8726888c79654e3ea8e7a15c833e00ba8b7fa6cdb09041110ba9f, exactly the analysis_code_sha256 in the historical SCW manifest. Source hashes/evidence recorded in e2b_scw_protocol.json.
- Ruling: backend-resolved defaults for omitted API generation parameters were not persisted historically; do not invent values. Preserve the original model/input-only request and document this limit.
- Ruling: use unchanged source score caches for the original baseline, with full job/record/cache/readout/runtime binding. Only rewritten text requires new verifier inference on the server.
- Ruling: supply45 deterministic human-review cases (five shared selected prompts per condition). Lexical metrics and the existing0.95 edit diagnostic do not establish semantic preservation; report manual-review status without a new scientific acceptance rule.
- Ruling: all3600 accepted paraphrases are required for metrics. The copied transport's terminal failed results remain cached and are not resampled or silently excluded. In-flight calls without durable received output retain the original retry/resume limitation.
- Ruling: keep nonempty incomplete API outputs under the unchanged original acceptance policy, report API status and quality; do not alter the transport to truncate/resample/replace them.
- Fresh independent review: limited source-packet review, no confirmed Critical/Important issue. Reviewer did not access repository files, upstream tools or execute code. No full independent repository/runtime validation is claimed.
- Reviewer question resolved by source inspection: encode_response builds original chat-template prefix plus full response IDs without truncation. verifier_rows uses that exact encoding for context checks. A server-only contract covers prompt3+response5 tokens against limits7/8. Test not executed.
- Cross-platform byte preservation: new .gitattributes disables text conversion only for copied SCW Python/reference files and fixes LF for the new launcher; existing E2 files are not targeted.
- Existing E2 Python, verifier, launcher and dependency files remain unchanged. .env and all experiment2_paraphrase output remain Git-ignored; no API key value was read or copied.
- Final checks are source/hash inspection, read-only whitespace scanning and git diff --check only. No Python/tests/API/inference/evaluation, download, install, commit or push performed locally. No E2B output run was generated locally.

## Startup visibility follow-up — 2026-11-03

Source diagnosis: the first normal preparation print occurs after frozen E1
hash loading, E2 job/cache reads and source snapshot checks. No startup log was
emitted before imports or these potentially slow Drive reads. A notebook busy
indicator alone cannot prove whether those reads are advancing.

Add tools/run_e2b_with_progress.py, a standard-library same-process runpy launcher
with immediate startup text and a20s daemon wall-time heartbeat. The shell now
uses python -u for unbuffered output. No hash-bound E2B pipeline/transport/verifier
file is changed by this follow-up; wrapper/shell/tests/README are outside the
integration_hashes list. Prepared manifests and existing API journals can retain
the same identity. Transfer both the shell and wrapper before restarting.

Heartbeat reports liveness, not an ETA or verified I/O/API progress. Existing
CLI arguments, SystemExit and transport interruption handling are retained in the
same process. A server-only mocked runpy contract is prepared, not executed.
All copied-source hashes still match. No local API/test/inference execution.

## 2026-11-03: simplify startup and resume at user request

Removed repeated whole-E1/E2 artifact and integration-code scans at preparation,
API generation, verification and aggregation. First preparation reads only the
selected source jobs/canonical score caches; subsequent stages use the saved E2B
population and original vector cache. Existing manifests are preserved verbatim
so paid API journals and scoring identities survive this operational refactor.
Verifier geometry/key loading is deferred until verification/reporting and checks
only the two frozen geometry artifacts plus the original key bank/receipt.
Startup, per-condition preparation and resume messages are flushed immediately.
Regression contracts were added for source-independent resume and rejection of
changed selected texts/vectors. No tests, API calls or model code were executed
locally, per the user's explicit constraint. Static review only.
## 2026-11-03: resume after exhausted API credit

User authorized retrying failed/missing calls without regenerating accepted
paraphrases. E2B now validates each condition's source/protocol mapping, archives
the exact previous journal, and atomically retains its successful records before
calling the byte-identical copied transport. Failed/empty calls are retried once
per launch under the original per-call policy; no unlimited outer loop. Existing
successful outputs, population manifests and score cache identities stay intact.
Added server-only mocked contracts for selective requests, zero calls when fully
cached, backup-before-publish interruption, mapping rejection and incomplete-tail
recovery. No tests/API/inference were run locally under the user's constraint.
## Runtime version policy update

At the user's request, differences from the historical E2 package/CUDA versions
are warnings, not a scoring gate. Expected and actual versions are recorded in
verification_environment_comparison.json. Actual runtime values remain in the
existing scoring identity; no schema changes invalidate previously scored caches.
The numerical verifier, FP32/FP64 policy, geometry, model and thresholds are
unchanged. Added a server-only mocked contract; no tests/API/model code executed
locally. Existing within-E2B score-cache identities still prevent mixing versions.
## 2026-11-03: user-requested high-fidelity paraphrase instruction

Replace the instruction only with the user's seven-rule text, preserving model,
API parameters, selected originals and scientific scoring. New default run_002
isolates new rewrites from completed historical run_001. Record the full prompt
and high_fidelity_v1 variant in provenance and reports; explicitly do not claim
historical SCW protocol equivalence. The copied source bytes remain unchanged;
the wrapper temporarily sets/restores its instruction constant around each
joined transport call. Resume reads each run's recorded prompt and validates
journal bindings. Added server-only mocked contracts; no local tests/API/inference.