# Experiment 4 Prefix KV Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Avoid repeating the shared calibration-context prefix computation during mapping while preserving the 128 contexts, full seed-word set, model precision, and official SaTML scorer.

**Architecture:** Pre-tokenize the exact prompt set to find a prefix shared across every seed word for each context, then choose one common prefix length across contexts. Prefill each model once and route suffix token IDs through a project-owned cache-aware `run_inference` adapter; keep the vendored scorer and processor unchanged. Add runtime parity checks against the existing full-prompt adapter before any shard is written.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-exp4-prefix-kv-cache-design.md`

## Global Constraints

- Keep `experiment4_satml/vendor/mlprints/bigram_suppression.py` byte-identical.
- Keep 128 calibration contexts and all valid seed-word IDs.
- Keep attacked-model FP32 and calibration-model FP16.
- Do not write cache tensors into checkpoints or persistent mapping outputs.
- Fail closed if cached and full-prompt logits exceed the stated parity tolerance.
- Preserve mapping shard identity and resume behavior across batch-size changes.

## Review Focus

- Tokenizer boundary changes between context text and the seed word: verify exact full-prompt token IDs are split at a shared prefix and suffix.
- Variable suffix lengths and padding: verify attention masks, position IDs, and selected last-token logits against full inference.
- Transformers Cache API differences: support legacy tuple and Cache objects without mutating the reusable prefix cache.
- Shard resume: reject stale shard identities and do not recompute completed shards.
- GPU memory pressure during prefill: bound cache preparation by contexts and fail with a clear message before creating shards.

---

### Task 1: Cache-aware next-token inference primitives

**Files:**
- Modify: `experiment4_satml/core.py`
- Modify: `tests/test_exp4_satml.py`

**Interfaces:**
- Produces `prefill_prefix_cache(model, prefix_ids, device)` returning reusable model cache and prefix length.
- Produces `run_cached_next_token_logits(model, cache, prefix_length, suffix_ids, pad_token_id, device)` returning one final-position logit row per suffix.

- [x] **Step 1: Write failing cache parity tests**
  - Add a tiny randomly initialized causal LM test comparing cached logits to logits from complete token sequences.
  - Cover different suffix lengths in one batch and verify invalid empty suffixes fail.
- [x] **Step 2: Run the focused tests and confirm they fail because the cache helpers are absent.**
- [x] **Step 3: Implement the two cache primitives**
  - Prefill equal-length prefixes under inference mode.
  - Convert reusable cache state to a non-mutating representation where needed.
  - Pad suffix batches with an explicit attention mask and position IDs; select logits at each row's final non-padding token.
- [x] **Step 4: Run the focused tests and confirm cached/full logits match within `atol=1e-5, rtol=1e-5`.**

### Task 2: Reuse context-prefix caches in mapping

**Files:**
- Modify: `experiment4_satml/pipeline.py`
- Modify: `experiment4_satml/core.py`
- Modify: `tests/test_exp4_satml.py`

**Interfaces:**
- Mapping builds per-context prefix IDs and per-word suffix IDs from the exact strings already passed to the upstream attack class.
- The adapter preserves the upstream `run_inference` contract: raw prompt list in, `{"scores": [logits]}` out.

- [x] **Step 1: Add failing tests for common-prefix construction**
  - Verify each selected full token sequence equals its cached prefix concatenated with its suffix.
  - Verify a changed prefix triggers full-inference fallback.
- [x] **Step 2: Run those tests to confirm they fail for the expected missing helper.**
- [x] **Step 3: Add prefix discovery and one-time per-model prefill**
  - Tokenize every exact `BOS + context + space + word` prompt with the attacked tokenizer.
  - Store shared prefix IDs and suffix token IDs without changing the token sequence.
  - Prefill attacked and calibration models once, logging preparation progress.
- [x] **Step 4: Route the unchanged official scorer through the cache-aware adapter**
  - Return correctly ordered logits for each official batch.
  - Before the first mapping shard, compare cached and full-prompt logits for representative contexts/words from both models; abort on mismatch.
  - Retain the existing 64-context diagnostic replay from cached logits.
- [x] **Step 5: Test prompt ordering, model selection, fallback, and resume identity using small fixtures.**
- [x] **Step 6: Run all Experiment 4 lightweight tests.**
  - Includes a mock Transformers 5 `ddp_cache_data` constructor-path check.

### Task 3: Document cache behavior and verification

**Files:**
- Modify: `experiment4_satml/README.md`

- [x] **Step 1: Document prefix-cache preparation, GPU memory use, parity gate, and fallback behavior.**
- [x] **Step 2: Verify README instructions match the implementation and run `git diff --check`.**
