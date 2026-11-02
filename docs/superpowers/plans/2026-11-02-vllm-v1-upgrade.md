# vLLM V1 Upgrade Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, task by task.

**Goal:** Move exact E2 generation to vLLM 0.32.0/V1, keeping `!bash run_experiment2.sh`.
**Architecture:** Shared pinned runtime options, lazy Qwen2 subclass, one spawned resident engine. Restart that worker in eager mode for diagnostics; never retain two teacher engines. Preserve existing numerical and verifier changes.
**Tech Stack:** Python, vLLM 0.32.0, PyTorch, Transformers 5, CUDA/Triton.
**Spec:** docs/superpowers/specs/2026-11-02-vllm-v1-upgrade-design.md (approved 2026-11-02).

## Global Constraints
- FP32; TP=PP=1; exact rotation before final RMSNorm; temperature once.
- No local tests, inference, benchmarks, installs or dataset reads: source review only, server execution deferred.
- Preserve dirty user changes and current runs; use run_012, protecting runs 001 through 011.
- Keep run_experiment2.sh content `python main_exp2.py` and notebook command `!bash run_experiment2.sh`.
- No automatic commit, push, environment mutation or running-server operation.

## Review Focus
- Captured graph must read updated angles after A -> B -> A owner switching.
- Padded/reordered V1 batches must yield correctly associated actual diagnostic traces.
- Failed worker initialization/switch must release the old worker before allocating another.
- Changed engine/runtime provenance must refuse old resumes.
- Optional numerical diagnostics must remain non-blocking; genuine runtime failures raise.

### Task 1: Runtime contract and model API
**Files:** create klrd_rs/e2/vllm_runtime.py, tests/test_vllm_v1_migration.py; modify klrd_rs/e2/vllm_model.py, tests/test_exp2_vllm_contract.py.
**Interfaces:** engine_options(settings,eager=False)->dict; runtime_identity()->dict; model.compute_logits(hidden_states); trace query token IDs; rotation buffer addresses remain fixed.
- [x] Write server-only tests first for runtime identity/options, logits API, padded trace boundaries, trace ordering and buffer updates.
- [x] Implement pinned TRITON_ATTN FP32 options, NONE compilation with FULL_DECODE_ONLY graphs, eager diagnostics, tagged Qwen2 signature.
- [x] Source-review against upstream v0.32.0; defer pytest to GPU server.

### Task 2: Worker lifecycle, provenance and output isolation
**Files:** modify klrd_rs/e2/vllm_backend.py, klrd_rs/e2/vllm_gates.py, klrd_rs/e2/config.py, main_exp2.py.
**Interfaces:** sampler.diagnostic_mode() context; _start_worker/_stop_worker/_ensure_worker; raw_generate preserves existing response fields plus execution mode. Use V1 engine_core.shutdown.
- [x] Add tests first for eager routing, context restore, version/resume identity, and launcher/output protection.
- [x] Replace V0 initialization/runner access; pin V1 in-process engine core inside our spawned worker; restart for eager diagnostics with readout/tokenizer identity checks.
- [x] Use V1 query metadata to exclude padding, reorder prefill traces by full token-prefix identity, retain finite-logit checks.
- [x] Update resolved config and route default output to run_012; protect all previous run directories.

### Task 3: Server validation, dependencies and instructions
**Files:** modify tools/check_vllm_exp2.py, requirements-vllm.txt, README_VLLM_EXPERIMENT2.md, README_EXPERIMENT2.md; update tests.
- [x] Add bounded server checks for direct CUDA graph owner-buffer A/B/A replay and actual engine owner switching, engine/eager reproducibility and KV numerical diagnostics.
- [x] Pin vllm==0.32.0, align Transformers with tagged upstream >=5.16.1,<5.20.0, remove obsolete XFormers dependency.
- [x] Document separate server environment and unchanged notebook launcher; clearly report server execution unvalidated.
- [x] Review diff/whitespace and dependency/source consistency, dispatch one fresh source reviewer, resolve important findings. Do not claim tests pass.

## Execution ledger
- Ruling: continue in approved shared checkout, retaining all dirty changes, without branch/worktree/commit operations. User approved modifying this project; worktree creation would not carry the existing dirty implementation and .git is read-only in the workspace policy.
- Ruling: user approval authorizes continuous inline implementation; no extra approval checkpoint for this plan.
- Ruling: server-only tests are written first but not executed locally, as required by approved design and prior recorded user restriction; no RED/GREEN claim.
- Pre-flight: runtime options feed both model worker and config identity; changing options must change resume identity. Diagnostic trace shape feeds checked_trace; full-prefix association is required.
- Task 1: source implemented; server-only tests prepared first, not executed locally. Tagged API confirms compute_logits(hidden_states), RMSNorm dispatch base mode, TRITON_ATTN float32 support and full graphs without compilation.
- Task 2: source implemented; explicit worker replacement, eager diagnostics, V1 engine_core.shutdown, version/runtime identity and run_012 isolation. Launcher file untouched.
- Task 3: source implemented; migration tool, dependency constraints and server instructions prepared. Limited source-packet review completed; server validation deferred.
- Ruling: pin Model Runner V2 explicitly in the spawned worker, because the reviewed query/logits metadata is the v0.32.0 V2 layout; record it in runtime identity.
- Ruling: request processed_logprobs explicitly. Tagged V1 defaults to raw_logprobs before temperature, while existing probes compare log_softmax(raw_logits/tau). Sampling still applies temperature exactly once; only returned diagnostic values change.
- Source validation: git diff --check passed after LF normalization. No Python/test/experiment execution locally; no runtime PASS or speedup claim.
- Ruling: explicitly set the worker-only Triton FP32 dot default to ieee in both environment and cached knob. The tagged attention kernel uses tl.dot without input_precision; torch TF32=False alone cannot establish this policy. Parent verifier settings remain untouched. Cost: IEEE attention may trade throughput for preserving the approved FP32 policy.

## Final review and handoff

- Independent reviewer could not access the repository: the file-read approval was aborted without content. No full independent repository review is claimed.
- A subsequent source-packet review found no confirmed Critical/Important issue in the supplied runtime/model/lifecycle packet. This does not validate omitted code, upstream APIs or GPU execution. The implementer separately inspected the actual files and tagged primary sources.
- No Critical/Important fix requests or deferred minor findings were returned by the packet reviewer.
- Final precision ruling: force VLLM_FLOAT32_MATMUL_PRECISION=highest before initialization, reject conflicting NVIDIA/PyTorch TF32 environment overrides, and verify actual torch dtype/matmul flags after initialization. This prevents inherited environment settings from silently changing the approved numerical policy.
- All plan checkboxes above describe prepared source/tests/docs and source review, not executed server validation. Server tests, startup, generation, CUDA graph dispatch, numerical behavior and speed remain unvalidated.
- Launcher verified unchanged with git diff --quiet -- run_experiment2.sh (exit 0), contents python main_exp2.py. Notebook command remains !bash run_experiment2.sh.
- Final source whitespace check: git diff --check. No model/dataset loading, tests, Python execution, installs, benchmarks, experiments, commits or pushes performed locally.
