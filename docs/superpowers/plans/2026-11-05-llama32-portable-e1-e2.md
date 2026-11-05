# Llama 3.2 Portable E1/E2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately versioned, reproducible KLRD-RS E1/E2 workflow for Llama 3.2 while preserving every existing Qwen default, launcher, frozen input, and historical artifact.

**Architecture:** Add a small profile and family-aware HF adapter, a fresh Llama E1 calibration/freeze path, and an E2 path that loads only the frozen Llama receipt. Reuse model-independent math/readout/text scoring/cohort helpers; add a new vLLM class derived from vLLM's native Llama implementation without changing Qwen registration. The shell workflow runs E1, E2 smoke, then E2 full and report.

**Tech Stack:** Python, PyTorch, Transformers, vLLM 0.32.0 target, NumPy/SciPy, YAML profile parsing through the existing server environment, pytest CPU/synthetic tests, Bash launcher.

**Spec:** `docs/superpowers/specs/2026-11-05-llama32-e1-e2-portable-design.md`; detailed scientific contract: `gpt_guide/KLRD_RS_Llama32_MultiBackbone_Codex_Guide.md`.

## Global Constraints

- Preserve all Qwen paths and defaults, including `main.py`, `main_exp2.py`, `klrd_rs/config.py`, `klrd_rs/e2/config.py`, Qwen frozen loaders/classes, existing launchers, manifests, and experiment artifacts.
- Implement only E1/E2 for `meta-llama/Llama-3.2-3B-Instruct`; no E3, E3-full, E2B, LoRA, paraphrase, ARC-C, Experiment 4, benchmark, ZIP packaging, or API calls.
- Keep `tau=0.8`, `pca_dim=128`, `m=32`, plane formula, Fisher ridge fraction `0.001`, target mean KL `0.005`, p99 safeguard `0.05`, alphas `0.01` and `0.0001`, full-vocabulary exact rotation, and key-independent spherical test unchanged.
- Never reuse Qwen geometry, lambda, keys, clean responses, or frozen receipt for Llama. Resolve and freeze immutable model/tokenizer identity and dataset revision before generation; resume rejects identity drift.
- E1 uses 2,000 fresh clean Llama calibration responses and targets 2,500 separate clean-original diagnostic responses. E2 targets 5,000 distinct eligible prompts and eight predeclared owner keys.
- Rotate the combined pre-final-normalization residual exactly once before native final RMSNorm; keep the native head and apply temperature once. Primary verification re-encodes visible returned text; EOS is sampled but excluded from observed verifier positions.
- The requested `.sh` contains only `python` commands using PATH, installs nothing, and runs E1 prepare/run/report, E2 smoke, then E2 full run/report.
- Local validation is limited to static checks, tiny CPU/synthetic tests, help, and offline inspect. No model/dataset download, model inference, CUDA, vLLM worker, benchmark, PBS/cloud submission, or scientific result claim.
- Server-only runtime/parity/evaluation results must be recorded as `NOT_RUN` until real server evidence exists.

## Review Focus

- **Wrong profile/revision or resume identity drift:** reject before model/data load or artifact reuse; tests pin unresolved revision handling and identity mismatch.
- **RMSNorm/readout formula, optional bias, or residual combined twice:** exact synthetic logits equality and fused/non-fused residual tests pin the final tail contract.
- **Calibration leakage or Qwen geometry/key reuse:** split, seed namespace, order, and receipt tests pin disjoint stages and cross-profile rejection.
- **Primary verifier using privileged sampled IDs or wrong token positions:** fake-tokenizer tests pin `prefix_len-1`, special/EOS exclusion, and visible-text re-encoding.
- **Accidental Qwen regression or local heavyweight work:** Qwen default/path regression tests and subprocess/import guards pin legacy config and offline CLI behavior.

---

## File Structure

- Create `configs/model_profiles/llama32_3b.yaml` for scientific/runtime/data defaults; model revision is explicitly unresolved until server-side freeze.
- Create `klrd_rs/portable/{__init__.py,profile.py,hf_backbone.py,calibration.py,e1.py,frozen.py,e2.py,vllm_adapter.py,vllm_llama.py}` for profile validation, supported HF family contracts, clean-data stages, receipts, E1/E2, and lazy vLLM registration/teacher.
- Create `main_portable_e1.py` and `main_portable_e2.py` with import-safe `--help` and offline `--mode inspect`, plus explicit prepare/run/report and smoke modes.
- Create the user-facing `run_llama_e1_e2.sh`; it uses `python` from PATH only and runs E1 through E2 full report.
- Create `README_LLAMA_E1_E2.md` documenting server environment prerequisites, exact commands, expected outputs, stop/resume conditions, and GPU-only checks.
- Modify `.gitignore` only as needed to exclude `runs/llama32_3b/` private/output data. Do not edit existing Qwen source/config/launcher/artifact files.
- Create CPU-only tests `tests/test_portable_profiles.py`, `test_portable_hf_contract.py`, `test_portable_geometry_and_keys.py`, `test_portable_e2_contract.py`, `test_portable_qwen_compat.py`, and server-only acceptance instructions in the README.

## Implementation Tasks

### Task 1: Profile schema and offline CLI contract

**Files:**
- Create: `configs/model_profiles/llama32_3b.yaml`
- Create: `klrd_rs/portable/__init__.py`
- Create: `klrd_rs/portable/profile.py`
- Create: `main_portable_e1.py`
- Create: `main_portable_e2.py`
- Create: `tests/test_portable_profiles.py`

**Interfaces:**
- Produce `ModelProfile.load(profile_id: str, profile_dir: Path | None = None) -> ModelProfile` and `ModelProfile.validate() -> None`.
- Produce immutable fields for `profile_id`, `model_id`, optional resolved `revision`, `model_type`, supported HF/vLLM architectures, data identity, math constants, seeds, E1 sizes, E2 sizes, and runtime settings.
- `inspect` loads this profile using local files only; it must not import Transformers model loaders, datasets, vLLM, or CUDA.

- [ ] **Step 1: Write profile tests** covering required Llama profile values; unresolved revision accepted only before prepare; malformed SHA, unknown architecture, invalid dimensions/alphas, unknown fields, and attempted Qwen run-path overlap are rejected.
- [ ] **Step 2: Run `pytest -q tests/test_portable_profiles.py`** and confirm the tests fail because the portable profile module does not exist.
- [ ] **Step 3: Implement YAML loading and validation** with a small frozen dataclass; keep all scientific values from the guide and never fill a model SHA by guess.
- [ ] **Step 4: Add import-safe CLI skeletons** whose `--help` and `--mode inspect` only load/validate profile metadata and print intended run paths.
- [ ] **Step 5: Run the targeted tests and CLI checks**; expected result is pass with no imports of `transformers`, `datasets`, `vllm`, or `torch.cuda` initialization in inspect mode.

### Task 2: Generic HF backbone and exact readout contract

**Files:**
- Create: `klrd_rs/portable/hf_backbone.py`
- Create: `tests/test_portable_hf_contract.py`

**Interfaces:**
- Produce `load_backbone(profile: ModelProfile, device: str = "cuda") -> HFBackbone`.
- `HFBackbone` exposes `model`, `tokenizer`, `hidden_size`, `vocab_size`, `max_position_embeddings`, final norm, output head, `readout(P, Q, tau)`, `capture_pre_norm(encoded_rows)`, and `native_tail_equivalence_check(residuals)`.
- Explicit registry supports only `qwen2` and `llama` family contracts; this portable loader does not change legacy `ModelAdapter`.

- [ ] **Step 1: Write tiny synthetic RMSNorm+linear tests** for tied and untied weights, optional head bias, arbitrary tiny hidden/vocab dimensions, fused residual input, zero and nonzero angles; assert native tail logits equal `RotationReadout.tail` and rotation log-probs equal direct native `RMSNorm(rotate(r)) -> head` log-softmax.
- [ ] **Step 2: Run `pytest -q tests/test_portable_hf_contract.py`** and confirm failure before adapter implementation.
- [ ] **Step 3: Implement the explicit family registry and loader** with `trust_remote_code=False`, FP32, TF32 disabled, native config-derived sizes/epsilon/gain, full context-length checks, optional bias, and no implicit prompt truncation.
- [ ] **Step 4: Implement pre-final-norm capture and direct-tail parity helpers** without assuming arbitrary future model `.norm` structure.
- [ ] **Step 5: Run the targeted CPU tests**; actual checkpoint load and GPU tail parity remain server-only `NOT_RUN`.

### Task 3: Fresh Llama calibration data and frozen model identity

**Files:**
- Create: `klrd_rs/portable/calibration.py`
- Create or extend: `klrd_rs/portable/profile.py`
- Create: `tests/test_portable_geometry_and_keys.py`

**Interfaces:**
- Produce `resolve_model_identity(profile, tokenizer, model_config, output) -> dict` that records immutable commit/revision, tokenizer revision, vocab hash, chat-template hash, shapes, generation config, and environment before calibration.
- Produce `prepare_clean_calibration(profile, backbone, output, mode) -> dict` that selects deterministic eligible UltraChat prompts and generates ordinary clean Llama responses; it must not import Qwen artifacts.
- Produce `fit_public_basis(residuals, mean, pca_dim, seed) -> tuple[basis, mean]` with explicit mean-orthogonality and orthonormality checks.
- Keep separate declared partitions for PCA/covariance, Fisher, strength, and local diagnostics, and separate the 2,500 clean-original diagnostic cohort from the 2,000 calibration responses.

- [ ] **Step 1: Write synthetic calibration tests** for deterministic selection, split disjointness, response metadata/hash fields, mean-orthogonal basis, orthonormality, and hard failure when basis assertions fail.
- [ ] **Step 2: Run `pytest -q tests/test_portable_geometry_and_keys.py`** and verify the missing calibration functions fail.
- [ ] **Step 3: Implement model identity resolution and fail-closed data selection** pinned to the guide's UltraChat dataset revision, native chat template, response IDs, seed namespace, EOS/termination, roundtrip, and source hashes.
- [ ] **Step 4: Implement ordinary clean generation and activation input serialization**; E1 calibration generations must be fresh Llama output and carry exact model/tokenizer identity.
- [ ] **Step 5: Implement the new clean-only PCA/public basis routine** and assert mean orthogonality rather than dropping the legacy sanity gate.
- [ ] **Step 6: Run targeted CPU tests** using synthetic activations/tokenizer only; do not call dataset/model loaders or generate any responses locally.

### Task 4: E1 portable geometry, strength, keys, receipt, and report

**Files:**
- Create or modify: `klrd_rs/portable/e1.py`
- Create or modify: `klrd_rs/portable/frozen.py`
- Modify: `main_portable_e1.py`
- Extend: `tests/test_portable_geometry_and_keys.py`

**Interfaces:**
- Produce `run_e1(profile, output, mode) -> dict` with explicit `prepare`, `run`, and `report` phases and identity-bound resumability.
- Produce `load_frozen_e1(path, expected_profile) -> FrozenE1` that verifies schema, profile, immutable model/tokenizer identity, artifact hashes, geometry shapes, key IDs/hashes, and frozen lambda; any mismatch raises `MISSING_FROZEN_E1` or a specific artifact error.
- Produce E1 artifacts under `runs/llama32_3b/e1/run_001/` and freeze geometry/lambda before drawing 64 final keys.

- [ ] **Step 1: Write E1 synthetic tests** for correct reuse of `planes`, `fit_fisher`, `choose_lambda`, separate calibration/final key streams, keys generated only after lambda freeze, zero-vector `sphere_p=1`, and receipt corruption/cross-profile rejection.
- [ ] **Step 2: Run the targeted tests** and verify they fail on absent portable E1 APIs.
- [ ] **Step 3: Implement thin stage orchestration** using existing `klrd_rs/math.py`, `readout.py`, `data.py`, `diagnostics.py`, and `io.py` math helpers without changing their formulas or tolerances.
- [ ] **Step 4: Implement receipt, checksum manifest, stage status, diagnostic report, and resume checks**; never overwrite frozen run_001 or generate keys before geometry and lambda are frozen.
- [ ] **Step 5: Add `prepare`, `run`, and read-only `report` CLI routing**; missing or changed inputs fail closed.
- [ ] **Step 6: Run targeted CPU/synthetic tests and offline inspect**; all real E1 feasibility and GPU checks remain `NOT_RUN`.

### Task 5: New vLLM Llama exact-rotation teacher and lazy architecture factory

**Files:**
- Create: `klrd_rs/portable/vllm_adapter.py`
- Create: `klrd_rs/portable/vllm_llama.py`
- Create: `tests/test_portable_e2_contract.py`

**Interfaces:**
- Produce `register_profile_architecture(profile) -> None`, lazily registering only `profile.vllm_architecture` to the portable Llama model import path.
- Produce `KLRDRSLlamaForCausalLM`, derived from the pinned native `LlamaForCausalLM`, overriding only the final norm path to combine residual once, rotate before native RMSNorm, and retain native LM head/weight-loading behavior.
- Produce import-safe `runtime_identity(profile) -> dict` and `engine_options(profile, settings, eager=False) -> dict`; no vLLM import at module import/inspect time.

- [ ] **Step 1: Write mocked registration and rotation-contract tests** verifying architecture mapping is profile-specific, registration is lazy, Qwen registration is untouched, and `hidden + residual` is rotated exactly once before native norm.
- [ ] **Step 2: Run `pytest -q tests/test_portable_e2_contract.py`** and confirm expected failure before implementation.
- [ ] **Step 3: Implement the new native-Llama-derived class and profile-specific runtime** against vLLM 0.32.0 source contract; do not instantiate or inherit Qwen model classes.
- [ ] **Step 4: Add static checks for architecture, weight tying, optional bias, full vocabulary, temperature-once, native EOS list, and no HF-generation fallback.**
- [ ] **Step 5: Run mocked CPU contract tests**; actual weight-load, prefill/KV parity, CUDA graphs, and sampler checks are server-only `NOT_RUN`.

### Task 6: E2 portable cohort, frozen loader, text verifier, smoke and report

**Files:**
- Create or modify: `klrd_rs/portable/e2.py`
- Modify: `main_portable_e2.py`
- Extend: `tests/test_portable_e2_contract.py`

**Interfaces:**
- Produce `run_e2(profile, e1_path, output, mode) -> dict` supporting `smoke`, `run`, and read-only `report`; `smoke` never auto-starts `run`.
- Produce `prepare_e2_cohort(...)` using the existing cohort eligibility and grouping rules, but a Llama-specific exclusion/receipt identity.
- Reuse `SphericalBlackBoxVerifier`'s vector score and `sphere_p` contract where generic; primary path must tokenize visible response text and exclude EOS/special targets.
- Require a verified Llama E1 receipt before any E2 model/dataset initialization; no E1 auto-fit or fallback.

- [ ] **Step 1: Write fake-tokenizer and tiny-model tests** for prefix position `prefix_len-1`, multi-EOS/special exclusion, visible-text re-encoding, wrong-key scoring from one score vector, disjoint group IDs, 20 deterministic wrong-key mapping, and missing/corrupt E1 stop.
- [ ] **Step 2: Run targeted tests** and confirm failures before E2 implementation.
- [ ] **Step 3: Implement explicit frozen E1 load, Llama cohort selection, exact teacher sampling, clean control, text-only scoring, metadata, checksums, atomic resume, and status reporting** using generic helpers without calling Qwen runner/frozen classes.
- [ ] **Step 4: Add smoke-only output isolation and explicit full-run/report behavior**; do not tune lambda, keys, thresholds, cohort, or sampling from smoke results.
- [ ] **Step 5: Run targeted CPU tests and offline `inspect`**; real generation, KV decode, GPU parity, and E2 smoke remain `NOT_RUN` on this host.

### Task 7: User launcher, operations README, ignores, and Qwen regression gate

**Files:**
- Create: `run_llama_e1_e2.sh` (single full workflow launcher, per user's request)
- Create: `README_LLAMA_E1_E2.md`
- Modify if needed: `.gitignore`
- Create: `tests/test_portable_qwen_compat.py`

**Interfaces:**
- Phase launchers use only `python` CLI commands from PATH; no environment activation/install and no absolute Python path.
- Combined launcher runs E1 `prepare -> run -> report`, then E2 `smoke -> run -> report`.
- README documents required server environment, user-managed gated HF access, output paths, stage prerequisites, exact resume/new-run rules, and server-only validation evidence paths.

- [ ] **Step 1: Write Qwen regression tests** pin `Config()` and `E2Config()` model/revision/strength/path defaults, old launcher command contents, old frozen shape contract, and untouched Qwen vLLM architecture identifiers.
- [ ] **Step 2: Run Qwen compatibility tests** before changes; record current baseline as passing or identify pre-existing failures without altering old behavior.
- [ ] **Step 3: Write launcher contract test** asserting no `D:/anaconda3`, no other machine-specific absolute path, no install/environment command, and E1-to-smoke-only command sequence.
- [ ] **Step 4: Implement launchers, README, and narrowly scoped `.gitignore` entry** for private Llama run artifacts.
- [ ] **Step 5: Run the full portable CPU/static suite, `--help`, and offline `inspect` for both entrypoints.** Do not run E1 `prepare/run`, E2 `smoke/run`, or any model/dataset/GPU path locally.
- [ ] **Step 6: Review `git diff` and `git status`** to confirm no Qwen source/launcher/artifact files or user-provided guide were changed.

## Self-review

- Spec coverage: E1 calibration/provenance/geometry/Fisher/lambda/keys/report are Tasks 1–4; E2 vLLM teacher, frozen receipt, cohort/verifier/smoke/report are Tasks 5–6; launchers/docs and Qwen compatibility are Task 7; local/server boundaries are global constraints and verification steps.
- Step granularity: each task adds a contract or cohesive stage with its own tests; no task runs real model behavior locally.
- Type consistency: `ModelProfile`, `HFBackbone`, `FrozenE1`, and `run_e1`/`run_e2` are introduced before downstream use and have explicit contracts.
- Review focus: each of the five high-risk input/error classes maps to focused CPU/static assertions in Tasks 1, 2, 3–4, 6, and 7.
- Proportion: tasks describe decisions/interfaces and checkable outputs without implementing full function bodies in this plan.

