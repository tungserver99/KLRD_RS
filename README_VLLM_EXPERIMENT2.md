# Experiment 2: exact latent-rotation teacher with vLLM

**Current full run: run_010, numerical diagnostics OFF by default.**
`bash run_experiment2.sh` loads the frozen configuration and existing exact vLLM
teacher, then follows the existing smoke/pilot/generation/evaluation protocol.
No native calibration, numerical holdout or HF parity is required. Numerical
reports do not determine `sampler_valid`; that field covers frozen runtime
configuration. Artifact/configuration and genuine runtime failures still raise.
All calibration/parity acceptance descriptions below are historical.
The generation backend, verifier and scientific thresholds remain unchanged.

The separate server tool's `--mode parity` explicitly opts into legacy numerical
diagnostics. Numerical tolerance failures are recorded as non-blocking diagnostics;
genuine runtime errors propagate. `--mode benchmark` skips numerical diagnostics
unless `--numerical-diagnostics` is explicitly supplied. Neither mode is required
before running E2. No tests or experiments were run locally for this change.

Implementation status: **SOURCE IMPLEMENTED; SERVER EXECUTION NOT VALIDATED**.
No tests, model inference, benchmark, download or installation were executed on
the local machine for this migration. Do not interpret this document as PASS or
as evidence of a speedup. Only your server results establish those facts.

## What to synchronize

Copy `main_exp2.py`, the whole `klrd_rs/e2/` directory, `tools/check_vllm_exp2.py`,
`tests/test_exp2_vllm_contract.py`, `requirements-vllm.txt` and the READMEs. Keep
`e2_inputs_v2/`, pinned E1 artifacts, and the previous E2 frozen prompt files:
`experiment2_rs/run_005/provenance/prompt_manifest.json` and
`experiment2_rs/run_005/private/prompts.jsonl`. The code also accepts the same
frozen cohort from runs 004/003/002 if 005 has not reached cohort freeze.

`run_010` is the current vLLM output tree. All runs 001–009 remain read-only. The
exact previous prompts, order, nested groups, owner assignments, quality IDs and
seeds are reused. Only backend metadata changes in the new manifest. No old
responses or score caches are imported. Same per-request seed does not imply
bitwise identical tokens between HF and vLLM. Within one frozen vLLM runtime,
atomic batch8 commit/resume still refuses changed identities and reuses jobs.

## Target server environment

- Linux x86_64 CUDA GPU server; initial target is the existing A100 40GB, one GPU.
- **vLLM exactly 0.10.2**, V0, XFormers, FP32; TP=PP=1, no quantization/FP8.
- `transformers>=4.55.2,<5`, torch 2.8.0 / XFormers 0.0.32.post1 as specified
  by the upstream vLLM 0.10.2 CUDA requirements; use matching CUDA wheels.
- Do not reuse the earlier torch 2.13 / transformers 5 environment unchanged.
- The launcher installs nothing. Prepare an isolated compatible environment on
  the server yourself. `requirements-vllm.txt` declares the target dependencies.

The integration deliberately refuses unreviewed vLLM versions. Version 0.11+
changes/removes the V0 APIs used here; installing the latest package is not a
compatibility fix. No package is patched in site-packages.

vLLM has its own spawned resident process and reserves 40% of device memory,
with an FP32 HF reference/verifier resident in the parent. KV scheduling and
continuous batching stay in vLLM; request/job batch size stays 8. Eager execution
and disabled compilation make the initial integration independent of graph
capture and changing diagnostic flags. Peak allocator usage is recorded for
each process separately; use `nvidia-smi` for total device memory. The initial
settings target 40GB, not an unverified low-memory GPU fallback.

## Exact intervention

`KLRDRSQwen2ForCausalLM` inherits upstream Qwen2 weight loading, attention and
decoder. Its final RMSNorm subclass preserves `model.norm.weight`. For a fused
call, it computes `r = hidden + residual`, applies the E1 rotation to `r`, calls
the original RMSNorm with **no residual argument**, and then uses the original
LM head. Non-fused calls use `r = hidden`. It never adds residual twice.

No tangent generation, logit bias, filtering, reranking, LoRA or model training.
No FP64 Linear in the vLLM process. Temperature is applied **only** by
`SamplingParams(temperature=0.8)`. All key settings, model revision, P/Q/L,
lambda, max512 new tokens, native EOS IDs and per-request seeds remain frozen.
Only one owner is active during a synchronous generation batch; the engine is
idle before changing owners. Prefix caching is disabled in the initial target.

HF remains exclusively for the existing verifier and measurement. Native-token
full-vocabulary KL is replayed once per returned batch at the same hashed 1/16
visited positions, including EOS. It uses the exact sampled token IDs, not
re-tokenized text. Primary ownership scores retain the existing canonical
returned-text verifier. The existing verifier precision policy is unchanged.
There is **no HF generation fallback**, including when vLLM gates fail.

## Commands to run yourself on the GPU server

If batch/reference parity fails, the gate now performs one extra prefill-only
batch as a native FP32 Linear shape diagnostic. It returns each original Linear
output unchanged, but compares its last row with the same vLLM Linear applied
to that identical input row/weight in isolation. Report:
`tests/fp32_linear_shape_diagnostic.json`, including module names, matrix shapes,
dtype, precision flags and local output errors. The probe restores every method
in a finally block. No FP64 Linear, new weights, tolerance changes or acceptance
override is introduced. This identifies local arithmetic shape dependence;
it does not certify accumulated model accuracy. Only run the parity command
while these discrepancies remain unresolved; do not launch the full E2 study.

From the project root in your compatible environment:

```bash
# Contract/unit tests (written, not executed locally)
python -m pytest -q tests/test_exp2_vllm_contract.py

# Actual pinned Qwen/vLLM numerical tests; no primary E2 trials
python tools/check_vllm_exp2.py --mode parity

# Same gates, then 2 batch8 jobs each for clean and key0, max512 tokens
python tools/check_vllm_exp2.py --mode benchmark --jobs 2
```

Diagnostic reports go into fresh `verification/vllm_server/<unique-id>/` folders.
Benchmark outputs are not E2 primary observations and never select keys/lambda.
The script records native token throughput including the KL audit, engine-only
time, KL audit time and per-process GPU allocator peaks. It does not include
ownership-verifier time in its generation benchmark; use normal E2 job timings
for end-to-end study costs. No experiment-result ZIP is created by this script.

The actual-model gate requires real vLLM logits at ten prefill and twenty
incremental decode states; validates clean/native rotation, fused residual+norm,
normalization, sampler log probabilities and temperature, fresh vLLM prefill vs
KV decode, batch8 vs single prefill, explicit EOS prefix, fixed-seed reproduction
and atomic resume. It also compares fixed prefixes with ordinary E1 FP32 Linear
arithmetic and the FP32 readout, temporarily restoring FP32 Linear methods for
this reference probe and restoring the existing verifier policy afterwards.
The strict same-context/KV max/p99 thresholds 0.0002/0.00005 are unchanged.
Current cross-shape/backend acceptance uses the native policy described in
README_NUMERICAL_POLICY.md. Strict, native/held-out distribution, native-zero
and resume reports must all PASS with a matching checksum approval marker. The historical
run_006 report mixed batch-vs-single errors into the KV summary. The corrected
report keeps these separate; it does not omit or relax either test.
An unresolved discrepancy is `SAMPLER_INVALID`, not a scientific NO_GO.

### Observed run_006 failure and current diagnostic change

The executed server report had rotation max/p99 0.00003052/0.00001144, but the
combined KV/batch max/p99 were 0.00184631/0.00053406. Individually listed cached
decode comparisons had maximum 0.00012589; the larger combined maximum therefore
came from the appended batch-vs-single prefill checks. HF reference max was
0.00017166 (under max limit), while p99 was 0.00009155 (over its limit).

This identifies failing comparison families; it does NOT establish their
numerical root cause or certify cache correctness. Current code separates all
four statuses, records per-prompt batch residual/logprob errors, and includes
total variation/KL/probability-weighted errors for diagnosis ONLY. Those extra
metrics never replace the original max/p99 thresholds. Worker TF32/matmul flags
are recorded from actual runtime state rather than hardcoded. Distributed groups
are explicitly destroyed during worker teardown. No FP64 Linear was added to
generation and no numerical threshold was changed.

Synchronize the code and run only `--mode parity` first. These diagnostic and
cleanup edits have NOT been executed locally, as required by the user. A repeated
failure remains blocked until the numerical cause is established.

After you inspect correctness and throughput, run the existing single-command
launcher for the full predeclared study:

```bash
bash run_experiment2.sh
```

It still contains only `python main_exp2.py`. No environment setup/config/path.
It rechecks actual-server correctness before Stage0/pilot/full study. Output
schema, job files, scoring, evaluation, fixed-key controls and success criteria
remain those of Experiment 2. It stops for review after E2; no Experiment3/LoRA.

## Reviewed upstream API references

- [Qwen2 v0.10.2](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/model_executor/models/qwen2.py)
- [RMSNorm v0.10.2](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/model_executor/layers/layernorm.py)
- [LLM/apply_model v0.10.2](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/entrypoints/llm.py)
- [Sampler v0.10.2](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/model_executor/layers/sampler.py)
- [Custom registration](https://docs.vllm.ai/en/stable/contributing/model/registration/)
