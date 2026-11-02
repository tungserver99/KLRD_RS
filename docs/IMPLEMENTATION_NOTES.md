# Implementation ledger

Plan: docs/superpowers/plans/2026-11-01-experiment1.md.
Approved design: docs/superpowers/specs/2026-11-01-klrd-rs-experiment1-design.md.
Scientific authority: KLRD_RS_FINAL_METHOD_AND_EXPERIMENT_1_SPEC.md.

## Rulings

- This project directory is not a Git repository. Implementation is in place;
  no branch, worktree, commit, merge or push was created.
- Native inline execution follows the approved frozen design; the final code
  was independently reviewed using a fresh agent.
- Use the existing KLRD clean public PCA128, whose calibration activation hash
  and fitting source are retained. The 819 MB activation tensor is unnecessary
  for replay once its basis is verified, so only its hash/size are bundled.
- Original KLRD clean_final and the later SCW clean paraphrases do not match.
  Exact matching uses the existing followup_raw_llr_v1 clean-final cohort and
  its selection/manifest; calibration remains original train_gen calibration.
  No new test cohort was selected. All diagnostic data remain development data.
- KSRF was inspected, but its different cohort, sampling defaults and semantic
  features are unsuitable here. Reuse comes directly from KLRD token encoding,
  residual capture and rotation conventions. Runtime code has no sibling imports.
- Public geometry uses factorized generators A_j=q_j p_j^T-p_j q_j^T.
  Exact RMSNorm invariance permits projected readout formulas; tests compare
  with matrix exponentials, generic automatic derivatives, independent finite
  differences, and native logits of a tiny real Qwen2 architecture.
- Fisher ridge is fixed at 0.001*trace(F)/m. Strength search requires pooled mean
  within 5% below 0.005 and pooled p99 <=0.05; infeasibility is NO_GO.
- Known unrecoverable historical calibration responses are ineligible for
  context sampling. Retain their cohort/partition IDs, report exclusions, and
  never fill them with other prompts. Diagnostic text-only canonical scoring
  retains original visible texts and reports its token-recovery limitation.
- All mathematical gates must pass, the raw/effective Fisher must be finite
  and full-rank, lambda must freeze before final keys, and >=95% of validation
  keys must have positive local Delta. This last threshold operationalizes the
  specification's suggested diagnostic target. It does not establish power.

## Test-first verification

- Mathematical tests failed with absent implementation, then passed after
  exact rotations, tangent derivatives, Fisher, beta tails and strength search.
- Artifact/readout tests failed before their modules existed, then passed.
- Pipeline integration failed before orchestration existed, then passed on
  a deterministic synthetic fixture. A real portable bundle test validates
  2000 calibration and 2500 matched original/paraphrase records.
- A corruption test demonstrated that changing a frozen public lambda was
  incorrectly accepted; a checksummed strength commit now rejects it before
  local caches can be used.
- No Qwen3B model download or full GPU experiment was run locally. The user's
  topmost environment contains CPU-only torch 2.5.1; GPU execution remains to
  be performed on the server.

## Independent final review

Two Important findings, no Critical findings:

1. Optional missing legacy/paraphrase files were incorrectly treated as required
   by the bundle loader. Loader-level regression failed first, then passed after
   explicit unavailable-optional provenance and skip/missing handling.
2. Known nonroundtrip calibration responses were used to reconstruct historical
   prefixes. Regression failed first, then passed with the fixed eligibility rule
   and per-partition exclusion IDs/counts.

Reviewer found no additional material issue in tangent algebra, indexing,
partitions, key/strength ordering, cache integrity, confidence units or claim
boundaries. Full GPU numeric tolerances and peak VRAM remain unverified.
No deferred minor findings.

Final verification in topmost: 19 tests passed, including a no-download tiny
Qwen2 native-logit/replay integration test. Real-input audit passed with 2000
calibration records and 2500 exact clean/SCW pairs. Final entrypoint smoke passed
as SYNTHETIC_CHECKS_PASSED, scientific_success=false. Python compile check passed.
Launcher bytes verified as exactly b'python main.py\n'; no other shell launcher.
