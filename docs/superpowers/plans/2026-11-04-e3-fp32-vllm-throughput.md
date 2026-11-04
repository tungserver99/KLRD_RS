# E3 FP32 vLLM throughput implementation plan

**Goal:** Apply KLRD's queued batching and batch journal writes to E3 ordinary
generation and the new full-study exact teacher, retaining FP32 and all science.
**Architecture:** execution-only tuning outside E3Config/checkpoint identities;
128 requests per submit, 64 concurrent sequences, 8192 scheduled tokens and .85
GPU memory budget. Keep original E2 and copied KLRD/SCW sources unchanged.
**Authority:** user accepted KLRD-style optimization with FP32 preserved.

- [x] Add server-only contracts for independent queue/concurrency/token limits,
  unchanged seeds/sampling, short/resumed batches, durable batch commits and
  historical-identity reuse. Do not execute them locally.
- [x] Add lightweight execution tuning with environment overrides. Ordinary
  clean/student prefix caching enabled; teacher prefix caching remains disabled
  because its rotation is mutable. Retain FP32/IEEE/TRITON and decode graphs.
- [x] Prepare every batch's records before one append/fsync. Add start/end token
  throughput logs and ordinary engine progress; preserve completed journals.
- [x] Apply the same execution tuning to the dedicated E3 full teacher wrapper,
  overriding E2 engine options in that subprocess only. Keep exact rotation,
  EOS, key/readout checks, verifier, ARC, training and API behavior unchanged.
- [x] Update docs and perform static whitespace/hash/source checks only.

Ruling: full test-first runtime cycle is prohibited by the user's local-source-
only restriction. Tests are supplied before implementation but remain unrun.
Ruling: no dtype fallback, no automatic OOM retry that changes precision, no new
calibration/parity gates. Manual execution overrides permit smaller batches.
Ruling: batch identity is execution metadata; do not change E3Config or training
numerical-source files. Changes can alter sampled tokens despite fixed seeds;
record actual execution and make no bitwise equivalence/speedup promise.
Ruling: a preemption can repeat up to the last uncommitted128-request batch;
committed newline-complete rows are retained and never generated again.

Static evidence: reviewed all callback call sites after switching to batch
commits; whitespace checks clean for all ten edited/new source/doc files.
SHA-256 of five numerical training sources matches the saved pilot manifest.
Tracked E2/math/verifier sources and vendored ARC hashes are unchanged. The
launcher still contains the same six explicit phase commands. Tests, Python,
models, API and GPU workloads were not run locally. Runtime speed and64-sequence
FP32 VRAM fit are unverified; manual execution overrides are documented.
