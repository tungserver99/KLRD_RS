# Llama 3.2 E1/E2 Portable Path Design

## Intent

Add a reproducible KLRD-RS Experiment 1 and Experiment 2 path for `meta-llama/Llama-3.2-3B-Instruct` while preserving the existing Qwen workflow, frozen artifacts, defaults, launchers, and historical outputs. The deliverable is source code, CPU-safe tests, operator documentation, and a server launcher; this work produces no experimental results.

## Goals

- Add a validated model profile and a family-aware HF backbone adapter that obtains dimensions, RMSNorm parameters, output head, tokenizer identity, and immutable model revision from the loaded checkpoint.
- Build fresh Llama clean calibration responses, residual activations, public mean-orthogonal PCA basis, rotation planes, Fisher whitening, calibration-only global lambda, and an E1 receipt under `runs/llama32_3b/e1/run_001/`.
- Load E2 geometry only from a valid frozen Llama E1 receipt. Generate with a new vLLM Llama-derived exact-rotation teacher and verify visible returned text with the clean HF model.
- Reuse existing math, readout, text encoding, key scoring, cohort, atomic IO, and diagnostics where their contracts are model-independent.
- Provide an offline `inspect` mode, CPU synthetic contract tests, server-only parity checks, and one operator launcher that runs E1 prepare/run/report, E2 smoke, then E2 full run/report.
- Keep private keys, raw responses, caches, and prompts in isolated ignored run directories; record profile/model/data/code/config hashes and per-stage status for resume and reporting.

## Non-goals and anti-goals

- No changes to Qwen defaults, old entrypoints, launchers, artifacts, frozen receipts, or historical results.
- No E3, E3-full, LoRA, E2B, paraphrase experiment, ARC-C, Experiment 4, benchmark, ZIP packaging, or local model/data download.
- No change to KLRD-RS mathematics, tau, PCA dimension, plane count, Fisher ridge rule, KL budgets, spherical p-values, alpha values, or text-only primary verifier.
- No claims of scientific success before GPU server evidence.

## Constraints

- Local work is limited to static checks and tiny CPU synthetic tests. Do not load model weights, download datasets, run generation/inference, initialize CUDA/vLLM, run benchmarks, submit jobs, or call external APIs.
- Llama uses a separately resolved immutable model/tokenizer revision and fresh geometry, keys, calibration strength, and response caches. Never reuse Qwen P/Q/L/lambda or Qwen E1 response artifacts.
- E1 uses 2,000 clean Llama calibration responses. The clean-original descriptive null set targets 2,500 fresh Llama responses; clean paraphrase diagnostics are `NOT_AVAILABLE` unless created in a separately authorized task.
- E2 targets 5,000 distinct eligible prompts, eight predeclared owner keys, 20 deterministic wrong keys per owner, Q=100/Q=1000 disjoint groups, and primary returned-text re-encoding.
- Final normalization receives the combined residual exactly once; apply the orthogonal rotation before native RMSNorm and preserve the native LM head. Temperature is applied once in sampler settings.
- The new Llama class derives from vLLM's native `LlamaForCausalLM`; the installed vLLM version is recorded, not hard-rejected by an exact version check. Actual parity remains a server-side acceptance item; do not alter Qwen registration.
- The requested `.sh` contains only `python` invocations, uses the environment's `python` from `PATH`, installs nothing, and runs E1 prepare/run/report, E2 smoke, and E2 full run/report in sequence.

## Approaches

1. **Parallel portable E1/E2 path (selected).** Add a small profile registry, generic HF backbone contracts, model-specific frozen-receipt loader, thin E1/E2 orchestration, and a new vLLM Llama class. This follows the guide's architecture, isolates model-specific artifacts, and leaves the proven Qwen path stable.
2. **Genericize the existing Qwen pipeline and runner.** This would reduce some new orchestration code, but broadens changes to legacy defaults, receipts, runner identity, and Qwen compatibility surface. It is a poor fit for the guide's requirement to preserve the Qwen path.
3. **Copy the existing full E1/E2 pipelines and replace Qwen-specific strings.** This appears locally simple but duplicates hundreds of lines and retains hidden Qwen assumptions in the copies. It is expressly discouraged by the guide and is the least maintainable option.

The selected path reuses `klrd_rs/math.py` (`planes`, `rotate`, `keys`, `sphere_p`, `fit_fisher`, `choose_lambda`), `klrd_rs/readout.py::RotationReadout`, `klrd_rs/data.py` encoding, `klrd_rs/e2/verifier.py` scoring contracts where generic, and E2 cohort/IO helpers where identity checks permit. It does not reuse `pipeline.run_experiment`, `continuation.py`, or `e2/frozen.py` as orchestration/loaders: those bind historical Qwen inputs, receipts, dimensions, lambda, or Qwen-only continuation rules.

## Compatibility findings and boundaries

- **Must be new for portable E1/E2:** profile resolution; Llama clean generation/calibration; activation/PCA/Fisher/lambda and key freeze; model-aware E1 receipt; Llama E2 loading; vLLM architecture selection and Llama teacher class; new entrypoints, report paths, and launch instructions.
- **Qwen-only; leave untouched:** `Config` and `E2Config` defaults; `ModelAdapter` and `E2ModelAdapter`'s `model_type == 'qwen2'` guard; continuation anchor/config checks; E2 `frozen.py`'s source receipt, Qwen calibration metadata and `(2048,32)` geometry check; Qwen vLLM class and `ARCHITECTURE` registration; `main.py`, `main_exp2.py`, and existing launchers/artifacts.
- **Future E3/E2B; leave untouched:** E2B scoring and E3 modules that instantiate `Config()`/`E2Config()` directly. They remain Qwen-bound until a separately scoped port.
- `RotationReadout.tail` models RMSNorm plus linear head, including optional head bias. Existing E1 `geometry_checks` uses finite-difference tolerances from the old Qwen config; portable Llama tests must add independent exact/JVP and FP64 diagnostic checks without weakening those gates.
- Existing `pipeline.py` already factors mathematical stage calls, but its input identity/report and continuation path encode the Qwen historical experiment. A thin new orchestrator can reuse the math/readout and numerical stage helpers while building fresh Llama calibration and provenance.
- The repository's current E2 vLLM runtime is pinned to v0.32.0 and Qwen registration. Upstream v0.32.0 Llama has the final fused contract `hidden_states, _ = self.norm(hidden_states, residual)`, native `LlamaForCausalLM`, and a model registry API that supports lazy string registration. This source contract informs implementation but does not replace actual server parity evidence.

## Proposed source layout

- `configs/model_profiles/llama32_3b.yaml`: profile ID, model/dataset identity, scientific constants, seeds, and runtime policy. Model revision starts unresolved and is frozen by server-side prepare.
- `klrd_rs/portable/profile.py`, `hf_backbone.py`, `calibration.py`, `e1.py`, `frozen.py`, `e2.py`, `vllm_adapter.py`, `vllm_llama.py`: compact portable implementation split by profile, model contract, calibration, stage orchestration, receipt validation, and vLLM teacher.
- `main_portable_e1.py`, `main_portable_e2.py`: offline help/inspect and explicit server modes. E2 must fail with `MISSING_FROZEN_E1` if receipt is absent or mismatched.
- `run_llama_e1_e2.sh`: invokes E1 prepare/run/report followed by E2 smoke and E2 full run/report using only `python` commands. `run_llama_e1.sh` and `run_llama_e2.sh` may remain as phase-specific entrypoints if needed for resume and cluster execution.
- `README_LLAMA_E1_E2.md`: exact server setup assumptions, per-phase commands, outputs, resume identity, error handling, and GPU-only checks.
- CPU-only tests for profiles, synthetic RMSNorm/readout parity, mean-zero tangent and geometry, separate key streams, receipts, text-only verifier, mocked lazy registration, offline inspect, and Qwen defaults/paths.

## Server-only validation and evidence

Server evidence is required for model/tokenizer identity resolution; activation capture; native HF/readout parity; Llama vLLM zero-angle and rotated-logit parity at prefill and KV decode; EOS/chat-template/text roundtrip behavior; graph buffer A→B→A behavior; numerical gates; E1 fit feasibility; E2 smoke and resume. Reports label these checks `NOT_RUN` until actual evidence files are produced. No expected PASS is inferred from source inspection.

## Open implementation decisions

- Exact profile schema field names and whether optional YAML parsing can rely on the existing server environment.
- How the current vLLM 0.32.0 Llama base class's weight loader interacts with replacing only its final norm module; verify through the implementation and mocked registry contract, then leave actual weight-load and logits checks to server.
- Exact deterministic source prompt selection and clean generation seed namespace, to be pinned before any response generation and recorded in the run manifest.

