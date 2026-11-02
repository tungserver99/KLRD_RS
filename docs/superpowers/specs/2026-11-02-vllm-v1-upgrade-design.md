# vLLM 0.32.0 / V1 migration for Experiment 2

Date: 2026-11-02
Status: approved by user 2026-11-02; unchanged launch command !bash run_experiment2.sh.

## Intent and scope

Upgrade the exact FP32 rotation teacher from vLLM 0.10.2/V0 to the latest
verified stable release, 0.32.0/V1. Exp2 currently takes roughly 25 hours.
The migration enables current inference optimizations; it makes no speedup claim.
Only generation integration, runtime provenance, dependencies, tests and operating
instructions change. Preserve fingerprint mathematics, verifier precision,
scientific thresholds, frozen artifacts, prompt groups, owner keys, seeds, EOS,
temperature, batch8 jobs and native-prefix KL measurement. Preserve all existing
uncommitted edits, including non-blocking numerical diagnostics.

## Approaches

The user's supplied direction favors a direct V1 migration, which is recommended.
Keeping 0.10.2 does not meet the upgrade request. Supporting both V0 and V1 would
require two internal runtime integrations and duplicate validation; this is out
of scope. Target exactly 0.32.0, rather than an unconstrained latest dependency,
so execution and resume provenance stay reproducible.

## Integration

- Keep lazy out-of-tree Qwen2 registration and upstream weight loading. Update
  forward/compute_logits and RMSNorm integration to the tagged upstream API.
  Continue adding the residual once, rotating before final RMSNorm, preserving
  norm.weight, and applying temperature only through SamplingParams.
- Keep TP=PP=1, FP32, no quantization and no HF generation fallback. Choose an
  attention backend that actually supports FP32 on the target A100; do not
  silently substitute FP16/BF16 or assume FlashAttention supports this dtype.
  The exact supported backend/configuration is a builder decision after inspecting
  tagged v0.32.0 source. Fail clearly if no compatible backend exists.
- Retain a resident spawned worker and synchronous fixed-owner batches. Use
  LLM.apply_model for configuration and in-place updates of fixed-shape rotation
  buffers, whose storage addresses must remain stable after graph capture.
  Do not compile owner angles as Python constants or replace their buffers.
- Remove forced V0/XFormers settings, obsolete constructor arguments and traversal
  of the old UniProcExecutor runner. Review shutdown against the V1 API.
- Normal generation uses a V1-supported CUDA Graph configuration where compatible
  with the exact rotation. Validate owner switching under graph replay, including
  nonzero owner A -> B -> A. Record requested runtime settings truthfully; do not
  label graph replay as observed without server evidence.
- Numerical traces, native-baseline and monkeypatched linear diagnostics require
  eager execution. Use a dedicated eager diagnostic engine lifecycle, created
  only on explicit diagnostic requests, with the generation engine unloaded
  before it starts; never keep two teacher engines resident. Recreate generation
  configuration after diagnostics if needed. Do not toggle inaccessible V1 runner
  internals or assume Python hooks execute during graph replay. Existing numerical
  diagnostics remain opt-in and non-blocking; genuine runtime errors propagate.
- Keep prefix caching disabled across key changes for this migration. Chunked
  prefill may be enabled if compatible with tracing separation and FP32 backend.
  Keep the initial 40% memory budget, length2048, seq8 and token8192 limits; tune
  only with measured server evidence. No verifier optimization in this scope.

## Runtime identity and outputs

Update both dependency checks and E2Config.resolved provenance to 0.32.0/V1,
including attention/compilation settings. Refuse mixed-version resumes. Route new
execution to a distinct output tree (next unused run, selected during implementation)
without editing or importing observations, caches or jobs from current runs.
Reuse the established frozen cohort mechanism. Do not install packages, stop the
current server run, edit its environment or execute experiments locally.

## Dependencies and documentation

Pin vllm==0.32.0 in requirements-vllm.txt. Remove obsolete XFormers pin and
Transformers<5 restriction; reconcile with tagged upstream dependencies, letting
the vLLM distribution select compatible torch/CUDA wheels. Keep general project
requirements consistent without inventing unverified exact torch pins. Document
an isolated Linux server environment and preserve the single-command E2 launcher.
Replace contradictory historical runtime instructions with clearly labeled
historical notes or current instructions. State that changing engines does not
promise bitwise-identical tokens or a particular end-to-end speedup.

## Validation and completion

The existing project handoff records a user restriction against local tests,
model loading, installs, benchmarks and dataset reads. Honor it: local work is
source review only. Prepare server-only contract tests for updated signatures,
residual/rotation/norm ordering, version checks/provenance, fixed seeds/EOS,
owner-buffer storage, graph replay owner switching and eager diagnostic routing.
Update the existing server check tool to exercise actual v0.32.0 Qwen2 traces,
KV decode, clean/native behavior, fixed-key batches, finite logits and benchmark
engine-only versus KL audit time. Preserve diagnostic/non-blocking policy.
No executed-server PASS or speedup claim until server evidence exists.

## Reviewed primary sources

- https://github.com/vllm-project/vllm/releases/tag/v0.32.0
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/vllm/model_executor/models/qwen2.py
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/vllm/model_executor/layers/layernorm.py
- https://docs.vllm.ai/en/latest/api/vllm/entrypoints/llm/

## Builder decisions

Resolve exact FP32 backend, V1 compilation settings, engine shutdown/lifecycle API,
trace indexing for the current runner, and next unused output directory through
source inspection. Do not alter the scientific protocol to work around an API.
If tagged source reveals a blocking incompatibility, report it with concrete
source evidence before changing precision or scientific semantics.
