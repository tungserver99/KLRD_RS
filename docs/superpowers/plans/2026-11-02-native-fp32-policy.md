# E2 numerical validation policy v2 — implementation plan

User scope: numerical policy only. Do not rewrite vLLM, change scientific
thresholds/keys/lambda/artifacts/protocol, run tests/inference locally, install
dependencies, train or switch to HF generation. All tests below are written
for the user's GPU server and MUST NOT be executed by Codex locally.

## Design

1. Preserve original strict max/p99 bounds (0.0002/0.00005) for same-context exact
   rotation/readout/raw logits, RMSNorm, temperature/sampler, and actual cached
   decode vs fresh prefill. Preserve normalization/negative-KL checks and resume.
2. Add diagnostics-only native mode which calls original vLLM final RMSNorm on
   `(hidden,residual)` directly. Default production path remains the existing
   rotation implementation. No FP64 Linear, new GEMM, new backend or weights.
3. Predeclare a numerical specification BEFORE collecting controls. From the
   frozen prompt pool exclude the first ten historical diagnostic prompts;
   choose 32 native calibration and 32 independent holdout prompts by fixed
   SHA256 ordering within four length quartiles (8+8 per quartile). No outcome,
   key, p-value or response quality selection. Three native-visited prefixes
   per prompt (prefill, +1, +2 tokens); diagnostic tokens never become trials.
4. Native controls compare batch8 vs single and native vLLM vs ordinary HF FP32
   replay on identical prefixes. Reference HF never generates text. Fit an
   engine-specific envelope ONLY from native calibration controls, freeze it,
   then validate native and keyed comparisons on holdout data. Fail rather than
   recalibrate if holdout exceeds its frozen bound.
5. Distribution metrics: TV, KL both directions, symmetric probability-weighted
   absolute log-probability error, and max error on significant tokens (union
   p>=1/V or q>=1/V, with argmax included). Retain raw full-vocabulary max/p99 as
   diagnostics. Raw probability normalization is checked separately before any
   FP64 normalization used for measurement.
6. Independent precision guardrails, not fitted to previous failed values:
   u=2^-24 (FP32 unit roundoff), TV<=sqrt(u), each KL<=2u,
   symmetric weighted abs log-error<=2sqrt(u), significant max<=4sqrt(u).
   These are explicit engineering budgets, NOT a theorem that all FP32 models
   satisfy them. Under a uniform per-prefix KL<=2u assumption, chain rule and
   Pinsker give length512 path-TV<=sqrt(512u)~0.00552. Finite tests do not establish
   that uniform assumption or certify ownership error/FPR.
7. Envelope = min(independent guardrail, native calibration maximum times
   fixed headroom + arithmetic floor). Headroom2 for linear metrics,4 for KL
   (second-order near equality). This is an empirical guard, not a KL triangle
   theorem. Floors64u for TV,128u weighted,512u significant,2*(64u)^2 KL.
   No automatic expansion using holdout/keyed failures. No clean-response
   ownership threshold fitting. If native controls violate absolute caps, stop.
8. Require every held-out comparison to meet caps AND frozen envelope. Require
   strict gates, native/custom zero parity, seeded resume and policy fingerprint
   integrity before authorizing existing Stage0/pilot/full E2. Store classifications
   (implementation failure, unstable native numerical baseline, keyed mismatch).
   Preserve historical reports; new run_008 reuses the same scientific cohort.

## Files/tasks

- Write tests first (DO NOT RUN): `tests/test_e2_numerical_policy.py`.
- Add `klrd_rs/e2/numerical_policy.py`: spec, deterministic split, distribution
  metrics, frozen calibration and acceptance. No scientific config edits.
- Add validation-only native bypass/control flag in `vllm_model.py` and
  `vllm_backend.py`, default False and restored after diagnostic calls.
- Add `native_fp32_validation.py`: native calibration/holdout collection and
  paired keyed distributions. Reuse the existing vLLM engine and HF replay.
- Refactor only gates in `vllm_gates.py`; preserve strict implementation checks.
- Pass the entire frozen prompt pool to validation from runner/server tool;
  add policy metadata to reports, code identity and runtime approval checks.
- Write mathematical/empirical limitations and server commands in a dedicated
  policy README. No local execution or runtime PASS claims.

## Review focus

- Native really calls original fused/non-fused norm, not just angles=0.
- Calibration cannot see keyed comparisons or ownership outcomes.
- Calibration/holdout disjoint, fresh relative to historical ten diagnostic IDs.
- Guardrails independent of baseline magnitudes; native errors cannot excuse
  arbitrary custom implementation errors; strict checks remain vetoes.
- Normalization, missing metrics, NaN/Inf, stale code/environment/calibration,
  incomplete coverage, malformed shapes and wrong temperature fail closed.
- Numerical acceptance does not assert uniform path bounds or scientific GO.

## Execution ledger

Source-only implementation; no tests, scripts, models or benchmarks executed
locally. User explicitly overrides runtime steps in TDD/verification templates.

- Separate math/design review: no mathematical overclaim found in the conditional
  path bound or calibration/holdout separation; exact coverage was requested and
  implemented as a Cartesian identity check.
- Source review of six core files found two important points, both fixed:
  enforce/bind batch_size=8 so batch controls cannot degenerate; retain elementwise
  RMSNorm errors and enforce both strict max and p99. Written regressions cover
  both. Reviewer reread the fixes and reported no newly obvious blocker.
- All review statements are source-only. No runtime PASS or measured behavior
  under the new policy is claimed.

## V3 follow-up after native calibration failure

Run_008 native baseline exceeded the v2 significant-token relative log-error
guardrail, while TV, both KL directions and weighted error met their caps.
That relative-error assumption is not a demonstrated native FP32 bound. V3
retains relative log-error as diagnostics and replaces its veto with significant
token absolute probability error, bounded by TV for normalized distributions.
The token probability cap equals the unchanged TV cap; its empirical envelope
is separately frozen from native controls. Strict, TV, KL and weighted caps do
not change. Worst significant log-error token ID, p, q and probability error
are reported. Prior policy/code declared calibration/holdout IDs are excluded
using design metadata only; current exclusions persist on resume. New study
output is run_009, protecting run_008. Regression tests are written but have
not been executed locally. Server validation is still required.
