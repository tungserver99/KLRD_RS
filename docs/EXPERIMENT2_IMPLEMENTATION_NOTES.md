# Experiment 2 implementation ledger

Authority: gpt_guide/KLRD_RS_EXPERIMENT_2_CODEX_SPEC.md. The user authorized this
protocol and requested the server launcher. The project is not a Git repository;
implementation was in place with no branch/commit/merge operations.

## Choices fixed before any E2 output generation

- New package klrd_rs/e2, independent main_exp2.py and run_experiment2.sh. Existing
  E1 runtime modules and all E1 output artifacts were left unchanged.
- FP32 HF cached backbone, exact E1 finite rotation readout, TF32 off, deterministic
  algorithms and math SDPA. No validated vLLM pre-final-norm integration existed;
  no approximate logits processor was substituted.
- Frozen E1 receipt derives expected hashes from the prior reviewed ZIP. Its digest
  is separately pinned in e2/anchor.py. Model readout parameter hashes are compared
  to E1 before generation. P/Q/L are loaded, never regenerated or refit. All64 key
  hashes are reconstructed from the original E1 implementation/seed and checked.
- Exclusion index incorporates37 KLRD prompt manifests/selections, with222227 IDs,
  exact and normalized text hashes, and no unparsed manifests or IDs without text.
  It is an eligibility input, not an already selected E2 cohort or proof of5000
  fresh prompts. Actual eligibility and source-split counts are resolved on server.
- Batch8 fixed-shape atomic jobs with per-owner/prompt/phase streams. Whole batch
  repeat on pre-commit interruption; committed samples are never regrouped or
  resampled. Primary5000, owners0..7, wrong20, groups5x1000/50x100 are fixed.
- Stress23/59 each use group0 and remain supplementary. Stage0 outputs use separate
  seeds and are excluded from primary statistics. Pilot never selects keys or
  adjusts lambda, significance thresholds, budgets or sampling.
- Operational qualitative-concern cutoffs are recorded in E2Config/README before
  any E2 GPU data: >5% mean wrong-key activation at either alpha; >=2/5 conservative
  clean detections for any primary key; >10pp excess degeneration; >=2 primary
  p99-KL violations or >5% visited-prefix KL samples above.05 for a primary key.
  These prevent a promising label and do not rescale/drop keys. No E2 outcome
  was inspected in deciding these implementation choices.

## Test-first evidence and independent review

New tests failed before the E2 modules existed. Tiny real Qwen2 tests then verified
native exact-rotation parity in prefill/decode, zero rotation/temperature once,
KV vs full prefix, padded vs individual requests, EOS, seeded reproduction and
the canonical text verifier. Other tests cover source/key hashes, exact5000
grouping, shortages, disjoint denominators, atomic commits, ties/wrong-key selection,
incomplete labels and a partial-owner report/ZIP without private data.

A fresh independent reviewer found one Important concurrency issue: identity and
shared provenance initialization used a shared temporary filename without a lock.
A regression failed first; identity creation is now write-once under a file lock
and shared provenance writes are serialized. Concurrent initialization tests pass.
No other Important/Critical issue or deferred minor was reported by that review.

Additional regressions guard frozen nested groups in analyze-only mode and a
single-owner shard report whose first owner is missing; mixed CSV schemas now
serialize without losing fields. The partial report remains E2_INCONCLUSIVE.

## Boundary of completed work

The local topmost environment has CPU-only PyTorch and no pinned Qwen3B weights
cached. No real-Qwen CUDA parity numbers,5000-prompt fresh selection or45000-response
study result is fabricated. The full server launcher performs these gates and the
entire smoke/pilot/full/stress schedule, saves all owner-key results, exports the
review ZIP and stops before any later experiment. CPU tests are implementation
evidence only. See verification/experiment2_local_validation.json for final checks.
