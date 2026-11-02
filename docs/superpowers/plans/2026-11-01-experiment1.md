# KLRD-RS Experiment 1 Implementation Plan

Goal: implement the approved frozen Experiment 1 and a one-command launcher.
Architecture: immutable portable inputs, public low-rank rotation geometry, chunked frozen-model replay, explicit stage gates and atomic resume caches.
Tech stack: Python, NumPy, SciPy, PyTorch, Transformers, pytest, mpmath.
Spec: KLRD_RS_FINAL_METHOD_AND_EXPERIMENT_1_SPEC.md; approved design in ../specs/2026-11-01-klrd-rs-experiment1-design.md.

## Global constraints
Pinned Qwen revision and tokenizer; tau .8; PCA128; m32; public seed 2026102801; calibration/evaluation key seeds 2026102802/2026102803; 32/64 keys; no training or paraphrase calls; no threshold fitting; one pooled global lambda; input artifacts immutable.

## Review focus
Missing or mismatched historical cohorts must never be silently substituted.
Text token prediction positions include the first response token and exclude special tokens.
Fisher/lambda/local partitions must be independent at prompt level.
Resume caches must refuse input/code/config changes and incomplete checkpoints.
Toy checks cannot establish actual model/scientific success.

## Tasks
- [x] 1. Write failing mathematical tests; implement rotation, scores, whitening, spherical and sign null, pooled strength and local expectation calculations; run CPU tests.
- [x] 2. Write failing artifact/indexing/resume tests; implement immutable input adapter and copied historical data with hash manifest; validate exact cohorts and provenance.
- [x] 3. Write failing integration tests; implement replay and stages A-H plus optional legacy, pipeline gates, required outputs and reports; run synthetic integration.
- [x] 4. Review complete code independently, fix material issues with regression tests, run full suite and compile check; document server execution and deliver run_experiment1.sh containing only python main.py.

Execution: inline implementation, fresh reviewer at the end. This directory is not a Git repository; no worktree or commit operations apply. The approved design is the implementation authority; no second approval gate is needed.
