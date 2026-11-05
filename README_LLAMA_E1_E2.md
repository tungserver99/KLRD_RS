# KLRD-RS Llama 3.2 E1/E2 handoff

This is an additive portable path for `meta-llama/Llama-3.2-3B-Instruct`. The existing Qwen entrypoints, profiles, launchers, frozen artifacts, and E2 vLLM registration remain Qwen-only and unchanged. Llama uses fresh clean generations, a separately fitted public basis/Fisher/whitener/lambda, and an E1 receipt under `runs/llama32_3b/`.

No experiment was run while preparing this handoff. All GPU, gated-checkpoint, UltraChat, vLLM, generation, and scientific outputs remain **NOT_RUN** until the commands below are run on the GPU server.

## Server environment prerequisites

- Use a Linux GPU server with a working CUDA stack and a Python environment that has compatible `torch`, `transformers`, `datasets`, `numpy`, `scipy`, `PyYAML`, and vLLM installed. The shell launcher does not create or activate an environment and uses `python` from `PATH`. The installed vLLM version is recorded in run metadata and is not rejected solely for differing from the reference version `0.33.0`.
- The run requires an authorized Hugging Face account with access to the gated Llama checkpoint. The code reads `HF_TOKEN` from the process environment or the repository-root `.env` file (and falls back to the local `hf auth login` token cache). The `.env` file is Git-ignored; never put access tokens in the profile, shell script, run manifests, or Git. The token is passed to Hugging Face loaders and exposed to vLLM child workers through `HF_TOKEN`, but is not logged or written to run artifacts.
- vLLM's `max_model_len` is derived from the longest selected native chat-template prompt plus the configured 512 new-token limit. Prompts are not truncated; the full model context window is not reserved when the experiment uses short prompts.
- The pinned UltraChat revision is `8049631c405ae6576f93f445c6b8166f76f5505a`. The initial model profile revision is `null`; E1 prepare resolves the official repository commit from Hugging Face and freezes its model, tokenizer, vocabulary, chat-template, generation-config, shape, and runtime identity before clean generations.
- Keep the repository's `e2_inputs_v2/exclusion_index.json` available. E2 reads it as a prior-exposure exclusion source and does not modify it.
- Review vLLM worker, CUDA graph, native stop-token, tokenizer-template, and HF/vLLM parity requirements in the guide before interpreting any result. The server's vLLM version is recorded in runtime metadata; runtime compatibility and parity are still established by the actual run.

## Offline inspection

These commands load only profile metadata and do not load Transformers, datasets, vLLM, model weights, or CUDA:

```bash
python main_portable_e1.py --profile llama32_3b --mode inspect
python main_portable_e2.py --profile llama32_3b --mode inspect
```

## Full workflow: E1, E2 smoke, then E2 full

The single user-facing shell file runs E1 prepare/run/report, then E2 smoke, then E2 full run/report. It invokes only `python` commands, and `&&` stops the chain when a phase fails. E2 full starts automatically after smoke succeeds:

```bash
bash run_llama_e1_e2.sh
```

The equivalent phases are:

```bash
python main_portable_e1.py --profile llama32_3b --mode prepare --output runs/llama32_3b/e1/run_002
python main_portable_e1.py --profile llama32_3b --mode run --output runs/llama32_3b/e1/run_002
python main_portable_e1.py --profile llama32_3b --mode report --output runs/llama32_3b/e1/run_002
python main_portable_e2.py --profile llama32_3b --mode smoke --e1 runs/llama32_3b/e1/run_002 --output runs/llama32_3b/e2/run_002
python main_portable_e2.py --profile llama32_3b --mode run --e1 runs/llama32_3b/e1/run_002 --output runs/llama32_3b/e2/run_002
python main_portable_e2.py --profile llama32_3b --mode report --e1 runs/llama32_3b/e1/run_002 --output runs/llama32_3b/e2/run_002
```

E1 prepare resolves and records the checkpoint identity, selects 2,000 eligible `train_gen` prompts, generates one ordinary clean Llama response per prompt, and separately prepares 2,500 clean-original `test_gen` diagnostic responses. E1 run builds pre-final-RMSNorm activations from visible response text, fits mean-orthogonal PCA128 and the existing KLRD-RS Fisher/lambda math, freezes geometry and lambda, and only then draws the 64 owner keys. E2 smoke validates the frozen receipt first, freezes the full 5,000-prompt cohort before sampling, then runs a small diagnostic-only clean/owner-0 generation slice in its own `smoke/` directory.

## Running only E2 full later

If E1 and smoke have already succeeded and you want to resume directly at E2 full, run these commands. The full phase uses the frozen 5,000-prompt cohort, ordinary clean control, eight predeclared owner keys (indices 0–7), and 20 deterministic wrong keys per owner. It does not fit geometry, lambda, thresholds, or keys from smoke outcomes.

```bash
python main_portable_e2.py --profile llama32_3b --mode run --e1 runs/llama32_3b/e1/run_002 --output runs/llama32_3b/e2/run_002
python main_portable_e2.py --profile llama32_3b --mode report --e1 runs/llama32_3b/e1/run_002 --output runs/llama32_3b/e2/run_002
```

Primary results use the visible returned response text re-encoded with the pinned clean HF tokenizer. EOS and other special targets are excluded from score observations; native sampled IDs are retained only as generation provenance. Groups are disjoint within each query budget (`Q=100` and `Q=1000`). The analytic spherical p-value and alpha levels `0.01` and `0.0001` remain unchanged. Report key-wise counts and the number of independent groups; do not claim fixed-key FPR guarantees from a handful of groups.

## Outputs and resume rules

- E1: `runs/llama32_3b/e1/run_002/`
  - `model_tokenizer_identity.json`, `clean_calibration.jsonl`, `clean_diagnostic.jsonl`, prompt manifests and `calibration_partitions.json`
  - `geometry/` basis, planes, Fisher, whitening, and covariance checks
  - `teacher/global_lambda.json`, `keys/*_key_ids.json`, private key vectors, `frozen_e1.json`, stage status, and report
- E2: `runs/llama32_3b/e2/run_002/`
  - `smoke/` contains diagnostic-only smoke cohort/generation/scores and never triggers the full run
  - `full/` contains the full frozen cohort, generation batches, canonical text-score caches, provenance, and report
- The `runs/llama32_3b/` tree is Git-ignored. Keep prompts, response text, key vectors, and caches private on the server. Do not copy credentials or model weights into the repository.
- Resume only with the same profile, immutable model/tokenizer revision, dataset revision, source files, seeds, artifacts, and phase output directory. A changed identity or checksum fails closed. Once `frozen_e1.json` exists, E1 run will use that freeze only when its input/source hashes still match; never overwrite it. Use the next unused `run_NNN` directory when model, source, code, or scientific settings change.
- E2 requires `MISSING_FROZEN_E1` to stop if the receipt is absent, corrupt, cross-profile, or changed. It does not auto-run or refit E1.

## Server-only acceptance evidence

The code reports source artifacts and local status, but model-level acceptance is still **NOT_RUN** until recorded on the GPU server. Preserve separate evidence for:

1. Frozen model/tokenizer commit, vocabulary/chat-template hashes, shapes, native generation config, and environment in `model_tokenizer_identity.json`.
2. HF capture at multiple response positions and native tail parity for zero/nonzero rotations.
3. vLLM native Llama weight loading, zero-angle clean parity, rotated-logit/log-prob parity at prefill and KV-cache decode, exact EOS handling, and one-time temperature application.
4. CUDA graph owner-buffer A→B→A validation, mutable key isolation, no cross-request state leakage, and full-vocabulary logits.
5. E1 basis mean-orthogonality, plane orthonormality, Fisher rank/conditioning, global lambda mean/p99 KL gates, all 64 key hashes, and local/null diagnostic outputs.
6. E2 smoke output, text-only verifier checks, resume after interruption, clean/wrong-key controls, and full-run group summaries.

An E1 local teacher result is not an E2 detection claim. No test here establishes GPU parity, deployment fixed-key FPR, or scientific success.
