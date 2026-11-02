# Numerical diagnostic and Experiment 1 continuation

Source correction: the intended reuse project is **KLRD**. Historical data,
geometry and reused code in this implementation already come from KLRD; the
initial KSRF reference was a copied-path mistake. This clarification changes no
frozen input, scientific configuration, cache, or result.

Run from this project directory, using your already prepared Python environment:

```bash
bash run_experiment1.sh
```

The single shell launcher contains only `python main.py --continue-experiment1`.
No environment activation, install command, external configuration or machine-specific
Python path is used. It stays within Experiment 1 and stops after producing review artifacts.

## Files to retain on the server

Keep the updated code, `inputs/`, `continuation_inputs/`, and the existing
`experiment1_rs/` directory, including `private/cache/contexts_*.npz` and their
completion markers. Do not delete or replace the old run. `continuation_inputs/`
contains the exact two historical residual rows, public planes, frozen configuration,
old code snapshot and a hash inventory of the original run. It is required input,
not a generated alternative cohort.

The previous run used Qwen2.5-3B-Instruct at revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1` on an A100 40 GB. Use that pinned model
and tokenizer and a CUDA environment supporting FP64. The original model and
production readout remain FP32. Only a separate readout copy is promoted to FP64;
parameter values, residual values, directions, seeds and scientific configuration
are unchanged. No PCA/QR refit, key selection, KL increase or threshold calibration.

## What the command does

1. Validate the exact-context pack, frozen configuration, historical input identity
   and old source-run file inventory/checksums. The two contexts are covariance
   rows 0 and 1: prediction positions 90 and 105 from the same recorded prompt ID.
2. Recompute full-vocabulary analytic scores and independent PyTorch forward-mode
   automatic derivatives. Central differences use **all** h=.002,.005,.01 and all
   32 directions. No favorable step size, token subset or direction is selected.
3. Keep the original FD atol=.003 and rtol=.03. Additionally require analytic/JVP
   agreement at atol=1e-9, rtol=1e-7 and both Richardson-extrapolated FD pairs at
   atol=1e-7, rtol=1e-5. The separate numerical policy is saved; Config.resolved()
   must equal the old frozen configuration. No original tolerance is increased.
4. Attribute the failure to FP32 finite-difference roundoff only if these gates
   pass, FP64 reduces the h=.002 FD error by at least 10x, and native FP32 analytic
   error is at most one tenth of the FP32 FD error. Otherwise write NO_GO and stop.
5. If accepted, inherit only the four hash-verified context caches into a **new**
   run, revalidate the remaining geometry/readout gates, and execute the rest of
   the original E1 pipeline. Old files are never opened for writing. The source
   global lambda and final key bank did not exist: the old run stopped before
   their stages. The new run therefore freezes lambda once using the original
   calibration-only bank/procedure, then draws the predeclared validation bank;
   it does not recalibrate an existing lambda or choose new key seeds.
6. Report every validation key, all KL violations and local advantages, spherical
   null/fixed-key diagnostics and optional legacy sign-randomization results.
   All original statistical success/failure criteria and independence units remain.
7. Save GO/NO_GO, verify old results again and automatically create a review ZIP.

The original specification's stage IDs are retained: D Fisher, E global strength,
F/G spherical null and descriptive fixed-key pilot, H local teacher, I optional
legacy. These cover the team's requested continuation sections D–G. No Experiment 2,
full teacher sequence generation, LoRA/training or new paraphrase API calls occur.

## Outputs

Everything new goes under `experiment1_rs_continuation/`:

- `numerical/fp64_diagnostic.{md,json}`: cause classification, policy, all checks,
  readout hashes, historical failed metrics and direction-23 detail.
- `numerical/error_by_context_direction_method.csv`: max/p99/median absolute and
  original-scaled errors, context/prompt/position and worst vocabulary-token IDs/text.
- `numerical/error_by_direction_method_pooled.csv`: exact pooled quantiles over both
  contexts per direction/method. `context_index=-1` denotes a pooled row; the
  worst-context fields identify the actual extremum. These are numerical errors,
  not independent statistical ownership trials.
- `numerical/current_FP32_geometry_checks.json`: the original FP32 gate remains
  visible, even when a validated FP64 tangent check permits continuation.
- `geometry/`, `null/`, `teacher/`, `legacy/`: original E1 outputs when those stages
  were reached. `teacher/local_capacity_by_key.csv` and `teacher/KL_by_key.csv`
  report all validation keys, never just an average.
- `report/summary.json`, `experiment1_report.md`: final GO/NO_GO and scope.
- `KLRD_RS_experiment1_continuation_review.zip`: public reports, exact two-context
  numerical reproduction data, source/tests, old NO_GO evidence and checksums.
  It omits private caches, raw datasets, model weights and the legacy secret key.

GO is an Experiment 1 implementation/local-capacity gate only. It is not proof of
free-generation TPR, fixed-key reliability, implanted weights or removal robustness.
Missing later-stage files after NO_GO mean NOT_RUN and must not be replaced with
the unrelated synthetic smoke results.

Run the same command again after interruption to resume the new run. Diagnostic
evidence is recomputed, not accepted from an editable PASS file. Code, policy,
readout values or input changes invalidate continuation caches; use a new separate
`--output` directory rather than altering old results.

## Local verification boundary

Tests run with the local topmost CPU environment; they cover FP64/JVP/FD agreement,
deliberately corrupted formulas, exact anchors, preserved planes, read-only cache
inheritance and packaging. This machine has neither CUDA nor pinned Qwen weights
cached. Full Qwen FP64 classification and remaining E1 results are **pending server
execution** until the command above actually produces them. No diagnostic cause or
scientific GO has been claimed from a toy fixture.
