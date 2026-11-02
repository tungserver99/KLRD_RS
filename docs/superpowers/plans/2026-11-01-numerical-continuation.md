# Numerical Diagnostic and Experiment 1 Continuation Plan

Goal: diagnose the frozen Step C FP32 failure using FP64 analytic/autodiff/FD, then continue only if independently validated.
Authority: team continuation request and KLRD_RS_FINAL_METHOD_AND_EXPERIMENT_1_SPEC.md.
Execution: inline, test-first, one independent final review. No new research design or approval gate is needed for the explicitly specified debugging task. Not a Git repository.

## Invariants
Preserve experiment1_rs byte-for-byte. Use exactly historical covariance rows 0 and 1, the same native FP32-rounded plane factors, pinned model/readout parameter values, tokenizer, seeds, dimensions and Config.resolved(). No tuning to downstream outcomes. No training, generation or paraphrase calls. Freeze lambda once by the original procedure because source run stopped before it existed. No per-key changes.

## Tasks
- [x] Freeze/hash source artifacts and exact context anchor before code changes.
- [x] RED: test FP64 analytic/autodiff/FD comparisons, independent error statistics and deliberate formula corruption; implement numerical diagnostic and explicit acceptance policy.
- [x] RED: test context/config/plane tampering refusal, immutable source/cache inheritance, gate-controlled continuation and result packaging; implement continuation orchestration.
- [x] Connect the single shell launcher and main flag; keep normal scientific configuration unchanged. Report old FP32 failure separately from validated FP64 gate.
- [x] Run full CPU suite, compile checks and fresh reviewer; preserve source hash checks and deliver reproduction README. Full Qwen/remaining scientific results execute on server only and are never fabricated locally.

## Diagnostic policy declared before Qwen computation
Compare analytic FP64 to independent PyTorch JVP (atol 1e-9, rtol 1e-7). For FD h in .002,.005,.01 retain the original atol .003, rtol .03. Additionally require Richardson-extrapolated FD (pairs .002/.005 and .005/.01) to match analytic at atol 1e-7, rtol 1e-5. Report raw and original-scaled max/p99/median for every context/direction/method, plus worst token IDs. All checks finite; no h or vocabulary/context/direction selection. A clear >=10x FP32-vs-FP64 FD error gap and accurate FP32 analytic scores are required to classify roundoff. Otherwise stop, do not waive the numerical gate.

## Review focus
Prevent changed contexts/planes, model values or frozen scientific config. No broad cache migration. Keep prior NO_GO intact. Differentiate pending server evidence from actual results. Bootstrap units and scientific criteria stay as originally implemented. Exact-run continuation must not accept a user-supplied PASS report without recomputing diagnostic.
