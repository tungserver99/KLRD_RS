# Experiment 2 Implementation Plan

Goal: implement the team's predeclared exact-teacher free-generation study and one Python-command shell launcher.
Authority: gpt_guide/KLRD_RS_EXPERIMENT_2_CODEX_SPEC.md. The user explicitly authorized executing this detailed protocol; no new research-design approval gate is needed.
Architecture: independent klrd_rs/e2 package and main_exp2.py; immutable hash-pinned E1 inputs; cached HF FP32 sampler; deterministic prompt/group/job manifests; chunk-atomic generation; canonical black-box verification; descriptive per-key reporting and public review ZIP.
Execution: native inline, test first, final fresh reviewer. This directory is not a Git repository; do not modify prior experiment artifacts or legacy modules.

## Frozen invariants
Qwen2.5-3B pinned revision, tau .8, m32, PCA128, loaded P/Q/L, lambda .09271215438842773, exact 64 E1 key hashes. Primary keys0..7, 20 cyclic wrong keys, stress23/59 separate. 5000 fresh unique prompts, five groups1000 and nested50 groups100. Exactly45000 primary/clean responses; no LoRA, PSS, paraphrase calls, reranking, threshold tuning or key exclusion.

## Tasks
- [x] Test/implement frozen source receipts, E1 key reconstruction, historical prompt exclusions and deterministic fresh cohort/group manifests. Fail on changed inputs, insufficient prompts or undocumented overlap.
- [x] Test/implement batched cached HF exact-rotation sampler and real-runtime prefill/decode/zero-rotation/KV/EOS/reproducibility/resume gates before primary generation.
- [x] Test/implement atomic fixed-shape jobs, deterministic per-request RNG, canonical text-only vector scoring once per response, privileged-token diagnostic and sharded merge.
- [x] Test/implement pilot/full/stress scheduling, complete per-key owner/clean/wrong/identification tables, utility/free-prefix KL, blinded examples, conservative operational decisions and packaging.
- [x] Verify all tests including tiny real Qwen CPU, immutable source checks, compile and independent review. Deliver run_experiment2.sh with only python main_exp2.py. Full Qwen GPU parity and scientific outcomes remain pending server execution because this connected machine is CPU-only, not a configured GPU worker.

## Predeclared implementation choices
FP32 HF backbone/readout, TF32 disabled; cache is mandatory. Generation batches8, fixed job shape and no pending-only regrouping on resume. Independent streams per owner/prompt/phase; clean is one shared bank. KL sampled at a fixed hashed 1/16 response-position mask independent of key. Utility sample100 promptIDs, examples10 preselected. Stress keys each group0 (not primary). Full-vocabulary parity max .0002/p99 .00005, KV max .0002/p99 .00005. No adaptation to observed errors.

Operational qualitative terms are declared before generation: wrong-key concern if any owner has >5% mean wrong-key detection at either alpha, clean concern if any key has conservative detections in >=2 Q1000 groups, severe degeneration if looping/empty+very-short rate exceeds clean by >10 percentage points, widespread distortion if >=2 primary keys have p99>0.05 or any owner has >5% visited-prefix KL samples above0.05. Concerns prevent the promising label. Full-cohort conservative weak power (<8/40 detections) with no meaningful nominal signal is E2_NO_GO; otherwise unresolved concerns produce E2_PARTIAL. Missing/invalid/incomplete data are E2_INCONCLUSIVE, never scientific failure.

## Review focus
Cache position/padding correctness; native rotation exactness and temperature once; all responses including EOS/empty/low-quality retained; frozen source and cohort hash validation; no counting shared clean or wrong-key scores as independent trials; no partial-run scientific GO; raw private outputs/key material excluded from ZIP.
