# Numerical continuation implementation record

The team's supplied debugging protocol is the authority for this change. The
original scientific specification and Config.resolved() remain unchanged.

## Preserved evidence

- Historical `experiment1_rs` remains NO_GO at C. A frozen inventory covers all
  24 files, including checksums of every original context-cache payload.
- `continuation_inputs/step_c_exact_contexts.npz` retains the exact two residuals
  used by C (float32, shape 2 x 2048), IDs, positions 90/105, observed target IDs,
  public PCA128 basis/mean and the saved 32 plane factors.
- Original code/config/environment/status/checks are snapshotted. No source-run
  result, tolerance, private cache or completion marker was edited.
- Final local byte-for-byte audit of all 24 historical files passed, including
  the large NPZ caches. Receipt: verification/continuation_source_integrity.json.

## Implemented gate

The diagnostic promotes the exact native FP32 readout parameters and plane values
to FP64 without refitting or reorthogonalizing. It compares full_scores against an
independent JVP of log_softmax(tail(rotate(...))) and central differences at every
predeclared h=.002,.005,.01. Original FD tolerances stay .003/.03. Independent strict
autograd and Richardson checks are predeclared in numerical.policy(). Both exact
contexts and the full vocabulary are tested for every direction. Errors include
max/p99/median raw/original-scaled values, actual extrema context/token IDs and
pooled quantiles. No diagnostic method calls the analytic implementation as its
reference derivative.

Roundoff classification additionally requires a >=10x error improvement from FP32
to FP64 FD at h=.002 and small FP32 analytic disagreement. Inconclusive/failed
diagnostics stop before Fisher/strength/keys/teacher/cohort scoring. Accepted
diagnostics continue in a new output directory with the saved geometry/context
caches, original statistical gates and every validation key reported.

The source run contains no frozen lambda or final keys because it stopped before
those stages. Continuation performs the original calibration-only global-lambda
freeze once, then draws the original predeclared final bank. It never replaces an
existing frozen continuation lambda or adapts the key bank to results.

## Validation and review

- New precision tests failed before numerical.py existed, then passed. A
  deliberately corrupted analytic formula is refused.
- Frozen config changes, replaced contexts, changed native planes, source-cache
  corruption and private-cache leakage into the review ZIP are tested.
- Independent final review identified two Important preservation issues. Temporary
  fixtures reproduced each failure before the fix: historical output descendants
  could receive error files; same-size old-cache corruption could escape resumed
  audits. Early protected-path checks and full source-cache hashing fixed them.
- Final suite: **32 tests passed** in the local topmost CPU environment. Python
  compile checks passed. The launcher is the sole project-root .sh file and its
  exact bytes are `python main.py --continue-experiment1` followed by LF.
- No deferred minor findings.

## Evidence boundary

No pinned Qwen weights are cached on this machine and torch has no CUDA here.
Actual Qwen FP64 classification, the cause conclusion and all remaining E1
scientific results are **PENDING SERVER EXECUTION**. Toy-test agreement is not
reported as a Qwen result. The launcher performs the requested numerical
diagnostic, conditional E1 continuation and review-ZIP export on the server.
