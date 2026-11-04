# Experiment 2B — GPT-5/SCW paraphrase robustness

**Source prepared only. No local API calls, tests, inference, evaluation, model
downloads or dependency installation have been performed.** Execute the commands
below yourself on the server. Existing E2 pilot and launcher remain unchanged.

## Current default: user-requested high-fidelity instruction (run_002)

The launcher now uses the exact seven-rule high-fidelity instruction supplied by
the user, defined in `klrd_rs/e2b/protocol.py`. The input is that instruction,
ending in `Text to paraphrase:` and two newlines, followed by the full original
response. Delimiters separating the user's chat instructions are not included.

Run the usual command:

```bash
bash run_experiment2_paraphrase.sh
```

New output: `experiment2_paraphrase/run_002/`. The completed `run_001/` and its
team-review ZIP remain unchanged. This new prompt requires new paraphrases for
all3,600 originals; it does not reuse rewrites made with the old instruction.
Subsequent launches resume run_002's successful results and retry failed/missing
calls as before. The 400 shared IDs, source responses, model identifier, copied
API/retry loop, verifier, query groups and thresholds remain the same.

The vendored source files remain byte-identical. A wrapper supplies the recorded
run instruction to the original transport while it executes, then restores its
historical constant even on interruption. Receipts/reports label this variant
`high_fidelity_v1`, with `historical_protocol_equivalent=false`; it must not be
presented as the original SCW prompt. Explicitly selecting an old run via --output
continues to use that run's recorded instruction, including for report-only use.

The historical protocol notes below describe run_001, not the new instruction.

## Recovered original protocol — copied, not reimplemented

The original KLRD source and historical
`outputs/klrd_vllm/paraphrase_scw_table2_v1/{api_manifest.json,manifest.json,summary.md}`
agree on **gpt-5-mini-2025-08-07**, the GPT-5-family snapshot actually used in
that study. This is distinct from full-size `gpt-5`; this implementation preserves
the original snapshot and does not substitute another model/provider.

The prefix, including its trailing space, is exactly:

```json
"Please rewrite the following text and return only the rewritten text: "
```

The sole request is the original copied code:

```python
client.responses.create(model=SCW_MODEL, input=PARAPHRASE_PROMPT + source_text)
```

The original25-file KLRD source aggregate SHA256 matches the historical manifest's
analysis_code_sha256 exactly (19c18f8fbaa8726888c79654e3ea8e7a15c833e00ba8b7fa6cdb09041110ba9f).
The copy receipt records this link and each original file hash.

Only model/input were explicitly sent. The historical artifacts do not persist
the API backend's resolved defaults for omitted generation parameters; those
values are not reconstructed or claimed equivalent.

No extra system instructions, original prompt, key, score, temperature, top_p,
reasoning effort or output-token limit is passed. No source text splitting or
truncation; `response.output_text` is retained verbatim. `.strip()` only detects
empty output. The original code accepts nonempty text even when the API status
is incomplete; E2B preserves and reports that status rather than changing the
old protocol or silently claiming all content is complete.

Copied byte-for-byte into the portable `klrd_scw/` package:

- `KLRD/klrd/scw_paraphrase.py`: complete API client, .env loader, retries and journals.
- `KLRD/klrd/artifacts.py`: its atomic artifact dependency.
- `KLRD/klrd/paraphrase_report.py`: the existing `pair_diagnostics` quality function.

`docs/superpowers/references/e2b_scw_protocol.json` records hashes of these copies,
the historical manifests and the pinned SCW commit
`15bc1929569357130f2dbc0b09f91bbf4f4bd947`. The pinned source and full upstream
`LICENSE_CODE` are retained next to that receipt. Copies are hash-checked at
runtime. No sibling KLRD checkout is required on the server.

Historical SDK: `openai==2.54.0`. The copied client uses the official endpoint,
SDK retries0, timeout180s, at most5 application attempts, transient backoff
1/2/4/8s, and historical default concurrency128. `--workers` can reduce concurrency
within the original supported1..128 range without changing request parameters.
Auth/model/access/quota failures stop without fallback. Model/account availability
has not been tested locally.

## Fixed population and query groups

Source is read-only `experiment2_rs/run_012/`. From its frozen1000-prompt group0,
hash-rank IDs using domain `KLRD_RS_E2B_SCW_400_V1`, then select exactly400 shared
IDs. Selection does not inspect text, scores, p-values or detections. The same
order/IDs are used for owners0..7 and clean: **3600 original responses,3600
accepted paraphrases**, one per original. No teacher generation or historical
paraphrase reuse occurs.

During first preparation, original text and original score vectors come from selected E2 batch
jobs and canonical text-only score caches, not unchecked merged exports. The
population, group definitions and original response/vector mappings are
frozen before the first paid API request. Original scores are reused rather than
re-running the original verifier over already-scored teacher responses.

|Q|Disjoint groups per key|
|---:|---:|
|20|20|
|50|8|
|100|4|

These groups are disjoint **within each Q**. Budgets share selected responses;
owners share prompts and clean responses. No pooled independent-trial CI is made.
Wrong controls reuse the existing twenty cyclic E1 keys per owner. Spherical
thresholds remain alpha0.01 and0.0001, with no calibration or new decision rule.

## Server setup and launch

Use the same Python/PyTorch/Transformers/NumPy/SciPy/CUDA environment as the E2
source run to keep cached-original and newly computed scores comparable. Add the
historical OpenAI SDK on the server; do not upgrade the frozen E2 numerical stack:

```bash
python -m pip install -r requirements-paraphrase.txt
```

Keep `OPENAI_API_KEY` in the project-root `.env` or process environment. The
unchanged copied loader gives the project `.env` entry priority, reads only that
variable without interpolation/execution, and never writes/logs the key. The local
agent did not inspect the key. The root-level copied package preserves the exact
`.env` path behavior; `.env` files remain ignored by Git.

From the project root, in your notebook:

```python
!bash run_experiment2_paraphrase.sh
```

Default is `--stage all`: prepare -> GPT-5/SCW paraphrase -> GPU verifier -> report.
This is a NEW launcher. `!bash run_experiment2.sh` still runs the E2 pilot.
No automatic LoRA/training, numerical gates, extra judge API, alternative
paraphraser or expansion beyond3600 source responses is implemented.

Optional separate stages, all on the server:

```bash
python main_exp2_paraphrase.py --stage prepare
python main_exp2_paraphrase.py --stage paraphrase --workers 128
python main_exp2_paraphrase.py --stage verify
python main_exp2_paraphrase.py --stage report
```

Prepare does not call the API/load the HF model. Paraphrase requires the API but
not GPU inference. Verify loads the pinned HF verifier on CUDA and also writes
automatic quality diagnostics; report uses completed vector/quality caches without
model inference or API requests. `verify` and `report` never initialize the API
client or read its key. A run lock prevents duplicate concurrent writers.


### Startup visibility

The shell launcher uses `python -u tools/run_e2b_with_progress.py`. This small
wrapper prints once at startup while running the entry point in the same process.
Preparation logs each owner/clean condition; resume loads only the saved3600-response
E2B population. There is no periodic elapsed-time or liveness output.

Repeated full E1/E2 source/code checksum scans have been removed from all stages.
Initial preparation still reads selected jobs and score caches once to bind texts
to original vectors. Resume checks the small E2B population/mapping/cache only.
The copied protocol files are checked once per launch. Frozen geometry and keys
are loaded only for verification/reporting; two geometry files are checked against
the existing E1 receipt. There is no full-tree scan or new numerical gate.

Transfer the updated `klrd_rs/e2b/` directory, `main_exp2_paraphrase.py`, shell
launcher and progress wrapper together. Existing population manifests (including
their historical source/code hashes), paid API journals and score caches are kept
without rewriting their identities. Stop the existing notebook
cell before relaunching; do not run two API processes for the same output.

## Resume, failures and long responses

Current default output is `experiment2_paraphrase/run_002/`. Repeating the same command
reuses terminal API journals and completed score batches. The original transport
is invoked separately for each400-response owner/clean condition because it
forbids duplicate prompt IDs within a split; every call sends the same instruction.
Responses are checked against full original text/model/IDs, not only row order.

After restoring account credit, rerun the SAME command with the SAME output:

```bash
bash run_experiment2_paraphrase.sh
```

The E2B resume wrapper retains every valid `ok` paraphrase and requeues only
`failed`/empty calls and identities without a durable result. It first archives
the exact old journal under `private/api/<condition>/retry_history/`, then atomically
publishes the accepted rows for the unchanged copied transport to resume. The
original model, instruction, request parameters and per-call retry policy stay
unchanged. This is an explicit extension of historical terminal-failure resume:
failed calls can now be retried across separate launches, with their history kept.
No successful paraphrase is resampled or selected using verifier results.

Log lines show `already OK`, `failed to retry` and `missing` per condition;
`provenance/api_resume.json` records the launch plan. A new failure stops this
launch; no unbounded retry loop or reduced-cohort analysis is performed. Repeated
empty outputs still require inspection. Do not delete the output folder or journals.
Interrupted in-flight calls without a durable received result can be sent again;
exactly-once billing for those interrupted calls cannot be guaranteed.

Full paraphrase text is scored using the **original prompt** through unchanged
`klrd_rs/e2/verifier.py`, frozen P/Q/L/key bank and FP32 boundaries/FP64 Linear
accumulation. No token clipping or sentence selection is performed. A rewrite
exceeding the pinned model context stops verification as technically incomplete;
it is never silently truncated, filtered or replaced. Completed API results stay
available, so verification retries do not regenerate paraphrases.

Differences between E2 and the current Torch/Transformers/NumPy/SciPy/CUDA
versions produce warnings, not a stop. Expected/actual versions are recorded in
`provenance/verification_environment_comparison.json`; actual scoring versions
are retained in `verification_runtime.json`. Existing score batches within this
E2B run still retain their own runtime identity to avoid mixing cached scores
from different environments. Original/paraphrase mappings and protocol remain fixed. Logging/integrity refactors do not invalidate paid API results.
Source E1/E2 trees are not re-audited on resume; the saved E2B population is the input.
Scientific request fields and group definitions remain fixed.
No E1/E2/KLRD source/artifact is overwritten, and all generated E2B output is
ignored by Git.

## Main artifacts

- `provenance/population_manifest.json`, `protocol.json`: deterministic selection,
  selected-population digest, copied-protocol evidence and unchanged groups/configuration.
  Older manifests retain source/code hashes as provenance without rechecking them.
- `private/population.jsonl`, `original_paraphrase_pairs.jsonl`: exact original
  prompt/text/response ID and paraphrase mapping. Keep private.
- `private/api/{owner_00..owner_07,clean}/`: original API metadata, append-only
  terminal journals and failure views, including attempts/model/status/usage.
- `private/original_vectors.npz`, `paraphrase_vectors.npz`, `paraphrase_scores/`:
  matched32-dimensional vectors, eligible-token counts and granular score caches.
- `scores/per_key_before_after.csv`: per-key/Q/alpha TPR before/after, drop in
  percentage points, paired lost/gained detections, clean/wrong controls and95% CI.
- `scores/group_scores.csv`: every group cosine, p-value and both detection flags.
- `scores/per_wrong_key_before_after.csv`, `closed_set_identification.csv`:
  all20 wrong-key controls and unchanged21-way identification.
- `report/summary.json`, `experiment2b_report.md`: paired results and limitations;
  inherited E2 cohort-provenance concerns are retained.

The report is descriptive robustness evidence; it adds no GO/NO_GO threshold and
cannot establish student/LoRA learning or a uniform fixed-key FPR guarantee.

## Did GPT-5 really rewrite while preserving meaning?

`quality/per_response_quality.csv` and `rewrite_summary.json` use the original
KLRD quality function: token/character length ratios, word Jaccard, word-sequence
similarity, exact copies and very-small edits (similarity>=0.95). Additional numeric
literal/structure flags and API status counts help identify cases needing review.
These metrics do not prove semantic equivalence.

Inspect45 deterministic pairs (five per condition, selected independently of
paraphrase content/detection) in `private/blinded_quality_pairs.md`. Fill
`quality/human_reviews.csv` with yes/no/uncertain for meaning preservation,
completeness and substantial rewriting, then rerun `--stage report`. The report
shows PENDING_HUMAN_REVIEW until all cases are rated, and still describes the
review as a subset, not a guarantee for all3600 outputs. Ratings never select,
filter or regenerate paraphrases. Preserve the private mapping if unblinding is
needed after review.

## Validation status

Only copy hashes, source inspection and Git diff/whitespace checks are performed
locally. Server-only tests are prepared, **not run**:

```bash
python -m pytest -q tests/test_e2b.py tests/test_e2b_scw_copied.py
```

The copied transport tests use a fake client. They do not certify real account
availability, paid API behavior, semantic preservation or GPU scoring. Real
server execution remains the user's next step; no speed/robustness PASS is claimed.
