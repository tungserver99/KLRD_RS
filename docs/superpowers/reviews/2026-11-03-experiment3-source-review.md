# Experiment 3 source review — 2026-11-03

Scope: revised E3 spec, new main_exp3.py, klrd_rs/e3 modules, launcher,
requirements, README and server-only contract sources. Existing E1/E2/E2B logic
and frozen artifacts are not modified. All checks below are static only.

## Findings resolved

- Linear warmup starts at LR0. The first-update diagnostic now waits for the
  first positive-LR update, retains the zero-LR event and persists verification
  state on resume. No scheduler/objective change was made.
- Merge smoke normalizes chat template output through existing prompt_ids,
  including Transformers versions returning BatchEncoding.
- Initial training environment provenance is preserved; later sessions receive
  separate environment records rather than overwriting it.
- Original FP32 KL(q||p0) is separate from untrained student BF16 numerical drift.
- Epoch selection, genuine optimizer/scheduler/RNG/permutation/cursor resume,
  best-adapter/merged integrity, EOS/response causal masking and streamed exact
  full-vocab tail gradients were inspected. The student residual graph is
  traversed once per microbatch; gradients are weighted by whole-window tokens.
- Paraphrase uses only explicit opt-in mode, rejects absent original student
  signal before API spending, uses the frozen400-ID high-fidelity protocol,
  reuses cached teacher0/clean controls and never replaces accepted outputs.
- Missing historical E2B paired JSONL now fails before any helper could create
  a derived file in the read-only historical run.
- Utility uses frozen E2 visible-token/repetition rules. Group CIs and nested
  dependence, zero vectors, clean false detections and21-way identification
  remain explicit. Training-only/failed merge cannot be called EVAL_COMPLETE.
- Legacy corpus hashes are computed once/file, not once/row. Historical held-out
  discovery now covers explicit artifact families/roles, including phase-B and
  implantation eval/dev; missing or unknown layouts remain reported limitations.
- Merged weight hashes verified before evaluation are reused within that locked
  process, avoiding a duplicate GB-scale read merely to form generation identity.
- Review ZIP uses named public summaries/source allowlists. Raw private
  prompts/responses, key/basis/Fisher vectors, weights/adapters and API receipts
  are excluded; private examples remain on the server.

## Verification limits and server handoff

No Python, tests (including CPU/mocks), compile, models, training, inference,
evaluation, dependency installation/download or GPT API call was executed locally.
Static source/whitespace/diff and unchanged historical copied-code hashes are the
available evidence. The tests are authored server-only contracts, not passing
results. No experiment output or fake scientific results were created.

GPU memory, PEFT/Transformers compatibility, actual gradients, checkpoint resume,
ordinary merged generation and all scientific performance remain unverified until
server execution. The existing legacy corpus is not automatically accepted as
FP32-equivalent when its provenance records BF16/missing hash authority; fallback
clean generation cost and historical coverage limitations are documented.

Primary and scoped independent source review were used. Late historical-role
coverage additions received primary static review; no independent runtime verdict
is claimed. No commit, push or model run was performed as part of this delivery.

## Generation throughput follow-up

Static review of the later optimization covers: per-prompt seed/params with
batched ordinary vLLM requests, input/output mapping validation before commit,
short final batches, retained cached identities, isolated worker lifetime, and
explicit engine failure records. New CLI runs default to vLLM while existing
backend choices and training-code fingerprints are preserved. Standard FP32
TRITON_ATTN/IEEE execution and decode-only CUDA Graphs are configured; resolved
worker runtime is recorded. No measured speedup, CUDA Graph execution, memory
fit, bit-identical outputs or passing tests are claimed without server execution.
