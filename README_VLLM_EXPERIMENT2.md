# Experiment 2: exact latent-rotation teacher on vLLM 0.32.0 / V1

**Source migrated; GPU server execution is not yet validated.** No local tests,
model loading, experiments, installations or benchmarks were run. No measured
speedup or end-to-end runtime is claimed.

## Unchanged launch command

In the notebook, from the project directory and the new environment:

```python
!bash run_experiment2.sh
```

In a terminal, use `bash run_experiment2.sh`. The launcher remains exactly
`python main_exp2.py`; it installs nothing and adds no required diagnostic step.
It writes to `experiment2_rs/run_012/`. Repeating the same command resumes that
run only when its code, runtime, input and configuration identities still match.
Runs 001 through 011 and frozen input/E1 directories are protected. No old
responses, score caches or job checkpoints are imported into run_012.

## Prepare a separate GPU server environment once

Target: Linux/CUDA, the existing A100 40GB, Python >=3.11,<3.15 (3.12 suggested).
Keep the current running experiment and its environment intact. Prepare this
migration in a separate code copy/environment before selecting it in Jupyter.

```bash
python3.12 -m venv .venv-vllm032
source .venv-vllm032/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-vllm.txt
python -m pip check
```

The pinned release is `vllm==0.32.0`; its tagged common requirements specify
`transformers>=5.16.1,<5.20.0`. Let its distribution resolve matching torch/CUDA
packages, and use a server driver compatible with those wheels. Do not carry
forward the old torch 2.8 / XFormers pin or install with `--no-deps` to bypass
compatibility. The general `requirements.txt` range is compatible with this
Transformers range. No package is patched inside site-packages.

For a notebook, launch Jupyter from the activated environment (install/register
its kernel as needed) and ensure the notebook shell's `python` resolves to that
environment. `!bash` inherits PATH; selecting a kernel alone does not necessarily
change which `python` the shell finds. This setup is separate from the unchanged
experiment command above. An old/missing vLLM version fails before HF weights load.

## Runtime and scientific contract

- V1 with Model Runner V2, TRITON_ATTN, FP32 weights/activations/KV, TP=PP=1.
  The tagged Triton backend explicitly supports `torch.float32`. The worker pins
  TRITON_F32_DEFAULT=ieee and its cached Triton knob, independently of torch TF32.
- Custom lazy Qwen2 registration preserves upstream weight loading, decoder,
  attention and LM head. Combine the fused residual once, apply the frozen
  exact rotation before final RMSNorm, preserve norm.weight, then normalize
  without adding the residual again. No FP64 Linear in generation.
- Temperature remains exclusively in SamplingParams. Diagnostic logprobs use
  processed_logprobs (after temperature), not the V1 raw-logprobs default. Preserve frozen P/Q/L,
  strength, keys, prompt groups, quality IDs, seeds, max512 tokens, native EOS,
  full-vocabulary sampling, fixed batch8 commits and resume rules.
- Keep 40% GPU memory utilization, max_model_len=2048, max_num_seqs=8 and
  max_num_batched_tokens=8192. Prefix caching is disabled. Production enables
  chunked prefill and synchronous scheduling.
- Request FULL_DECODE_ONLY CUDA graphs with capture sizes 1 through 8 and
  compilation mode NONE. Full graphs do not require torch.compile; mutable
  P/Q/angle buffers are updated in place after capture. Owner changes occur
  only between completed synchronous batches. No precision fallback.
- A spawned process owns one resident V1 engine. Its engine core and uni
  executor run inside that process, so worker exit releases the teacher.
  No HF generation fallback is available.
- Numerical tracing/native-baseline/linear probes use an explicitly eager
  engine with chunked prefill disabled. The old worker exits before an eager
  or production replacement starts; two teacher engines are never resident
  together. V1 attention query boundaries exclude padding. Full token-prefix
  matching restores prefill trace order after scheduler reordering.
- Existing HF verifier, FP64 Linear accumulation, minimum execution chunk64,
  NumPy key projection, spherical scoring and native-prefix KL replay remain
  unchanged. No fingerprint formula, scientific threshold or dataset changes.
  Numerical calibration/parity remains opt-in and non-blocking. Genuine runtime,
  configuration, artifact and NaN/Inf failures still raise.

`E2Config.resolved()` and the engine manifest record the new version/backend,
runner, graph/compilation and scheduling settings. Runtime/configuration validity
is separate from numerical diagnostic PASS. Metadata records graph requests;
it does not pretend whole-engine graph dispatch has been observed. The same seed
does not guarantee identical tokens across old/new engines, devices or shapes.
Upgrading generation does not remove HF verifier or KL audit costs.

## Synchronize source

Synchronize `main_exp2.py`, the whole `klrd_rs/e2/` directory (including the new
`vllm_runtime.py`), `tools/check_vllm_exp2.py`, `requirements-vllm.txt`, both E2
READMEs and migration/contract tests. Keep the pinned E1 artifacts,
`e2_inputs_v2/` and the frozen cohort files from run_005 (or the existing
004/003/002 fallback). Do not copy old observations into run_012.

## Optional checks on the GPU server

These commands are prepared for the server; none were executed locally.
They do not modify the normal E2 launch command or create primary observations.

```bash
python -m pytest -q tests/test_exp2_vllm_contract.py tests/test_vllm_v1_migration.py
python tools/check_vllm_exp2.py --mode migration
python tools/check_vllm_exp2.py --mode benchmark --jobs 2
```

`--mode migration` captures the actual final-norm rotation in a CUDA graph,
checks A -> B -> A replay against eager outputs and stable buffer addresses,
then checks actual engine fixed-batch owner switching, seeded reproducibility,
engine switching and three real eager prefill/KV-decode trace states. The small
norm graph proves mutable rotation-buffer behavior; it does not independently
prove whole-model vLLM graph dispatch or numerical parity. Sampled A/B tokens
may coincide; exact norm outputs must respond to the distinct angles.

`--mode benchmark` measures clean/key0 batch8 generation up to512 tokens and
records engine time, KL audit time and per-process allocator peaks. It excludes
ownership-verifier time, so it does not predict the full study runtime. Inspect
normal E2 timings and device-wide GPU memory separately.

For detailed existing numerical/KV/HF-reference diagnostics, explicitly run:

```bash
python tools/check_vllm_exp2.py --mode parity
# Or append --numerical-diagnostics to --mode benchmark.
```

These use eager execution; they do not certify numerical behavior of graph
generation. A tolerance result is recorded without vetoing production; genuine
runtime errors propagate. All diagnostic output goes to fresh directories under
`verification/vllm_server/`. Historical V0 calibration/tolerance failures are
not approval evidence for the migrated engine.

## Reviewed upstream sources (tagged v0.32.0)

- [Release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)
- [Qwen2](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/models/qwen2.py)
- [RMSNorm](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/layers/layernorm.py)
- [Triton attention](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/triton_attn.py)
- [Compilation/graph configuration](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/compilation.py)
- [V1 engine lifecycle](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/engine/core_client.py)
- [Dependencies](https://github.com/vllm-project/vllm/blob/v0.32.0/requirements/common.txt)
