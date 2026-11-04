# Experiment 3 full: 10k LoRA, 512/1024, ARC-C and paraphrase

Run on the GPU server from this project, with the existing E1/E2/E2B and pilot
artifacts available in their recorded locations:

```bash
!bash run_experiment3.sh
```

In a terminal omit `!`. This launcher runs **all six stages**, in order:
fresh full training, evaluation512, evaluation1024, ARC-C, paid GPT-5 paraphrase,
then the six comparison tables, Markdown report and review ZIP. It stops on a
failed stage. It does not install dependencies. Ordinary clean/student generation
reuses the pilot's recorded backend (vLLM in the current pilot); the new exact
teacher generation uses an isolated vLLM worker. Training and verification retain
their existing Hugging Face paths. No HF/vLLM sampled-output parity test is added.

## FP32 generation throughput

E3 clean/student generation and the full-study exact teacher now use the KLRD
queue pattern:128 requests submitted per call, up to64 concurrent sequences,
8192 scheduled tokens per engine iteration and85% GPU memory budget. The token
budget is independent of request queue size; chunked prefill handles long
prefixes without truncation. FP32, IEEE, disabled TF32, original sampling and
per-prompt seeds are retained. Clean/student enable prefix caching. The exact
teacher keeps prefix caching disabled because its rotation state is mutable.
Decode-only CUDA Graph capture sizes now extend through64 sequences.

Each batch is converted/validated before one durable journal append/fsync,
reusing the copied KLRD/SCW batch writer. Logs show request queue size, concurrency,
saved responses and measured visible tokens/second; ordinary vLLM also shows its
native request progress. These are execution changes, not an E3Config or training
checkpoint identity change. Historical successful responses retain their hashes;
new responses record their actual execution settings. Fixed seeds do not imply
identical sampled tokens across batch/cache configurations.

The standard command is unchanged. If the server cannot accommodate64 FP32
sequences, reduce execution limits without changing precision, e.g.:

```bash
KLRD_RS_VLLM_REQUEST_BATCH_SIZE=64 KLRD_RS_VLLM_MAX_NUM_SEQS=32 bash run_experiment3.sh
```

Optional execution variables (defaults):

| Environment variable | Default |
|---|---:|
| `KLRD_RS_VLLM_REQUEST_BATCH_SIZE` |128|
| `KLRD_RS_VLLM_MAX_NUM_SEQS` |64|
| `KLRD_RS_VLLM_MAX_NUM_BATCHED_TOKENS` |8192|
| `KLRD_RS_VLLM_GPU_MEMORY_UTILIZATION` |0.85|

Concurrency is capped by the submitted request count. No automatic BF16 fallback,
new calibration gate or HF/vLLM parity test is introduced. On interruption,
committed newline-complete rows remain reusable; up to the last uncommitted
request batch can repeat. Server VRAM fit and speedup are not measured locally.
The historical E2 launcher/backend is not modified by this E3 tuning.

## Server setup and preserved inputs

Use the working CUDA environment, model cache and project `.env` on the server.
If ARC dependencies are missing, install them **on the server only**:

```bash
python -m pip install -r requirements-exp3-full.txt
```

The original ARC harness is pinned to `lm_eval==0.4.13`. Keep the working
Torch/Transformers/vLLM versions; this source supports exact-teacher vLLM0.32.0
and0.33.0. No general E2 Torch-version equality gate is introduced. Optimizer
resume still requires a coherent numerical training environment.

Defaults:

| Purpose | Directory |
|---|---|
| New full study | `experiment3_full/run_001/` |
| Read-only pilot reference | `experiment3_rs/run_001/` |
| Frozen E2 | Pilot's recorded `e2_run`, normally `experiment2_rs/run_012/` |
| High-fidelity prompt and400 IDs | Pilot's recorded `e2b_run`, normally `experiment2_paraphrase/run_002/` |
| Historical KLRD | Pilot's recorded `legacy_root`, normally `../KLRD` |

The default full run never writes into those historical run directories.
`run_experiment3_pilot.sh` preserves the old2k pilot launcher. To select another
new full output, set `KLRD_RS_FULL_OUTPUT=experiment3_full/run_002` before running
the shell; do not change output when trying to resume the same run.

## Training and evaluation

Full training starts from the pinned clean base and a **new adapter**, not the
pilot adapter. The total is10,000 train and1,000 validation sequences, using the
pilot's already frozen full prompt partitions. The verified2,000/200 clean
pilot contexts can be reused, leaving8,000/800 clean completions to prepare.
These are contexts for full-vocabulary forward KL, not SFT targets or copied
LoRA weights. Training response length remains512; LoRA settings, exact rotation,
temperature, masking, precision and minimum-validation-KL checkpoint selection
reuse the existing training implementation.

The512 phase reuses verified pilot responses/scores and E2 clean/teacher caches,
and generates only the new full student. The1024 phase generates **new** clean,
exact-teacher and merged full-student responses for the same1,000 frozen group0
prompts. It preserves each model's512 seed policy, original prompt order,
native EOS, tau=.8, top_p=1 and unrestricted top_k. Context capacity is calculated
from full chat prefixes plus requested output length; prompts are not truncated.
The ordinary merged student needs no secret key or runtime rotation.

Verification reuses the existing returned-text spherical verifier with the
frozen geometry/key bank,20 wrong keys, Q20/50/100/1000 and alpha .01/.0001.
Reports retain per-group cosine/p-values, controls, confidence intervals and
warnings about limited independent groups and nested query budgets. Generation
utility separates native generated length, visible length and verifier tokens;
missing historical timing is reported as unavailable rather than invented.
Ten fixed private examples are retained outside the public review ZIP.

## ARC-C source reuse

The four files in `klrd_arc/` are direct byte copies from `../KLRD/klrd/`.
`SOURCE_RECEIPT.json` records the original hashes. See
`docs/ARC_C_SOURCE_REUSE.md` for recovered protocol evidence.

The original evaluator controls ARC-Challenge zero-shot answer-choice
loglikelihood, chat formatting, batch1, max context4096, all test examples,
`acc_norm,none`, seed20261021 and bootstrap1000. No generated-answer scorer or
free-generation temperature is substituted. Clean, exact teacher, pilot student
and full student share the same loaded dataset and exact request hashes.

**Precision difference:** the historical KLRD loader used BF16. This matched
four-model comparison uses FP32 to preserve the frozen exact teacher precision;
the protocol/report disclose this difference and make no numerical equivalence
claim to historical BF16 results. The original evaluator did not explicitly pin
a dataset commit. The new run records actual dataset fingerprints, task config,
ordered IDs and content hashes; it does not invent a historical commit.

## Paraphrase and resume

The all-stage shell explicitly invokes the paid paraphrase phase. It uses the
frozen GPT-5 model identifier and **`high_fidelity_v1` prompt** from E2B run002
through the existing copied SCW transport. There are400 new1024 originals for
each of clean, teacher and full student: **1,200 new accepted paraphrases**.
Historical512 paraphrase texts are never substituted. Keys and verifier scores
are not sent to the API. Retries may add requests/charges; there is no local
paraphraser fallback or automatic cohort expansion. The server supplies the key
through its existing environment/`.env`; no key is copied into source or reports.

Rerun the same shell and output to resume. Completed training reuses its selected
checkpoint/export; response journals keep committed generations; verification
reuses completed score batches; ARC uses per-model native likelihood caches;
paraphrase submits only missing/failed responses, preserving accepted successes.
An interruption can repeat the last uncommitted unit. Conflicting source data,
protocol or checkpoint identities require investigation rather than silently
mixing results. Performance-only graph differences are recorded in teacher
runtime metadata. No new numerical calibration gate is introduced.

If an uncommitted clean-training JSONL has malformed interior lines, the launcher
can now recover it before generation: it validates every parseable response's
prompt/run/seed/hash, saves an exact `.corrupt.<id>.bak` beside the private
journal, writes a `.recovery.<id>.json` receipt, then atomically retains all valid
lines. Resume selects missing IDs even when valid responses occur after a gap,
and returns the finished corpus in the original prompt order. Do not delete the
journal or change output to work around corruption. A parseable row with a bad
binding/hash or a completed/consumed corpus is not silently discarded or replaced.
This recovery changes no cohort, precision, sampling or scientific threshold.

Individual phase commands:

```bash
python main_exp3_full.py --mode train
python main_exp3_full.py --mode evaluate --max-new-tokens 512
python main_exp3_full.py --mode evaluate --max-new-tokens 1024
python main_exp3_full.py --mode arc-c
python main_exp3_full.py --mode paraphrase --max-new-tokens 1024
python main_exp3_full.py --mode report
```

The report command is offline: it does not train, generate or call an API. It can
report missing/partial phases after an interruption. Individual train/evaluate/
ARC commands never call the next phase or the paraphrase API themselves.

`run_experiment3_full.pbs` runs one selected phase per PBS job, e.g.
`qsub -v E3_PHASE=eval1024 run_experiment3_full.pbs`. Adjust queue/account/resource
names and activate your environment for your cluster. Allowed phases are
`train`, `eval512`, `eval1024`, `arc-c`, `paraphrase`, `report`; submit sequentially
after checking the previous job's status. No job is submitted by this source.

## Results and review package

Outputs under `experiment3_full/run_001/` include:

- `training/`, `checkpoints/`, `merged_student/`: full training and selected export.
- `eval512/`, `eval1024/`: separate original generation, scoring and utility.
- `arc_c/`: per-model benchmark results and matched comparison.
- `eval1024/paraphrase/`: attack pairs, usage, quality diagnostics and scores.
- `report/tables/`: all six requested comparison CSVs.
- `report/full_experiment_report.md`: evidence-based discussion and phase status.
- `KLRD_RS_experiment3_full_review.zip`: allowlisted public tables, metadata,
  training logs and source, with a package manifest.

The ZIP excludes weights, adapters, optimizer state, private prompts/responses,
API journals, credentials and frozen key/basis arrays. Raw original/paraphrase
pairs remain private for authorized human quality review. Lexical/numeric quality
checks are diagnostics; they do not prove semantic fidelity or filter the cohort.

This delivery is **source only**. Runtime regression tests are supplied for the
server but have not been executed locally. No local training, inference,
evaluation, model download, dependency install, API call or PBS submission was
performed. GPU memory use, installed-library compatibility and actual scientific
results remain to be verified on the server.
