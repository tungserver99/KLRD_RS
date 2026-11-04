# Experiment 2B — fixed GPT-5/SCW output-paraphrase robustness

Authority: user's detailed E2B specification and explicit instruction to copy the
original KLRD paraphraser directly. Local work is source reads/writes only: no
API calls, tests, inference, evaluation, model downloads or dependency installs.

## Protocol recovered from source and historical artifacts

Original KLRD root: sibling KLRD. Historical run:
outputs/klrd_vllm/paraphrase_scw_table2_v1. api_manifest.json, manifest.json and
summary.md agree with the pinned SCW reference commit
15bc1929569357130f2dbc0b09f91bbf4f4bd947.

Exact model: gpt-5-mini-2025-08-07 (the GPT-5 family snapshot actually used in
KLRD, not the distinct full-size gpt-5 model). Exact prefix, including its trailing
space: "Please rewrite the following text and return only the rewritten text: ".
Request: client.responses.create(model=SCW_MODEL,input=prefix+source_text).
No explicit temperature, top_p, output-token limit, reasoning effort, system
instruction, chunking or cleanup. Read response.output_text verbatim. Whitespace
is used only to recognize an empty result. Nonempty output with incomplete API
status is accepted by the legacy code; preserve that behavior, report statuses
and do not claim semantic completeness automatically.

Original transport: OpenAI SDK2.54.0 in saved manifest; official API endpoint,
180s timeout, SDK retries0, at most5 application attempts, backoff1/2/4/8s for
transient failures. Fatal auth/model/quota errors stop without fallback. Workers
1..128, historical default128. Durably journal each terminal response/failure;
resume never resamples completed entries. An interrupted in-flight call without
a received/durable result may be sent again; exactly-once billing cannot be
promised by this legacy protocol. No .env contents are inspected locally.

## Approach and components

Chosen by user: direct source copy. Do not reimplement or replace the API logic.
Copy klrd/scw_paraphrase.py, its klrd/artifacts.py dependency and the existing
klrd/paraphrase_report.py quality helper byte-for-byte into a root-level
klrd_scw package. The unchanged PROJECT_ENV_PATH then resolves to KLRD_RS/.env.
Copy pinned upstream reference and full LICENSE_CODE; retain provenance hashes.
Importing the neighboring repository at runtime would make server deployment
depend on an external checkout; copying keeps the integration portable.

Use one call to the unchanged transport per owner/clean condition, each with
400 rows, because the original transport forbids duplicate prompt IDs within a
split. All calls send only original response text with the identical prefix.
Transport metadata never sends original prompts, owner keys or verifier scores.

New integration lives in klrd_rs/e2b. Do not edit E2 verifier, mathematics,
thresholds, generation, frozen config, checkpoints or original KLRD source.
Only new E2B source/docs/tests, dependency declaration and ignore rules change.
No commits/push/API execution are required by this request.

## Population and immutable source

Read source run_012 as immutable. Require its frozen group0 Q1000 cohort and all
nine committed/scored populations. Deterministically hash-rank those1000 prompt
IDs with domain KLRD_RS_E2B_SCW_400_V1, choose the first400; no score/detection-based
selection and no replacement on failures. The same ordered400 IDs are used for
owners0..7 and clean, before/after. Read original text and vectors from checked
job records/score-cache receipts, not unverified merged JSONL exports. Bind
prompt manifest, prompt pool hash, job/score hashes, source code, frozen E1
receipt, model revision/tokenizer/readout hashes and E2 runtime provenance.

Freeze selected IDs, group definitions and source mappings before paid API work.
Q20=20 disjoint groups, Q50=8, Q100=4 per owner. The same groups are reused across
before/after, owners and clean. Groups are disjoint within each Q; budgets and
shared clean/control populations are not additional independent samples.

## Server stages and resume

Output fixed under experiment2_paraphrase/run_001, separate from all E1/E2/KLRD
sources. CLI stages prepare, paraphrase, verify, report, all (default). Launcher:
!bash run_experiment2_paraphrase.sh. Existing E2 launcher stays unchanged.
Use an exclusive run lock. API receipts live under private/api/{condition}.
Persist selected population and code/source identity atomically; changed inputs,
protocol or scoring runtime refuse resume. Preserve terminal API failures and
report incomplete coverage; never silently evaluate a reduced cohort.

Original score vectors are reused from the checksummed E2 text-only verifier
cache, whose verifier/readout/math implementation is pinned. Only paraphrases
need new inference. Use the unchanged SphericalBlackBoxVerifier.score_text_batch
with the exact original prompt, frozen P/Q/L and all64 E1 key hashes. Keep the
existing FP32 boundaries/FP64 Linear accumulation and alpha0.01/0.0001. No
calibration, teacher generation, local paraphraser or new fingerprint method.
Checkpoint paraphrase score batches by exact text/ID/runtime hashes. Do not
silently truncate long rewrites; if context capacity is exceeded, report a
technical incomplete state rather than silently changing the text/cohort.

## Reports and quality

Paired original/para detection counts and TPR changes for every owner/Q/alpha;
clean FPR, twenty unchanged wrong-key controls per owner, cosine/p-values,
Clopper-Pearson intervals and closed-set identification. Preserve per-group
scores and private per-response vectors. No pooled independent-trial CI or new
GO/NO_GO/LoRA authorization.

Reuse the original pair_diagnostics for token/character length ratios, word
Jaccard, word-sequence similarity and unchanged/very-small-edit fractions.
Report API statuses, empty/failed output and length/structure diagnostics.
Provide a deterministic balanced human-review subset and a blank rating sheet
for meaning preservation/real rewriting. Lexical metrics cannot certify meaning;
semantic-quality status remains unreviewed until humans supply ratings. Do not
select or regenerate paraphrases based on either quality or detection results.

## Validation boundary

Write server-only contract tests (including the original mocked transport tests
with only their import path adapted), but do not execute them locally. Verify
copies with file hashes and inspect source/diff only. Use a fresh source review
on supplied source packets if the reviewer cannot access the Windows drive.
Explicitly report that API execution, GPU scoring and tests remain unvalidated.
