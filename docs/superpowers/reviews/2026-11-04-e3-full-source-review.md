# E3 full source review, 2026-11-04

Scope: implement the full10k/1k, matched512 comparison, new1024 three-condition
evaluation, original ARC-C reuse, three-condition high-fidelity paraphrase and
six-table report/ZIP. The user's latest instruction authorizes all stages in
the default shell, including the explicit paid paraphrase phase. Independent
CLI phases and the PBS template remain single-phase.

## Checks actually performed

- Read integration source for common state, data preparation, generation,
  exact-teacher adapter, ARC, paraphrase, reporting and launcher dispatch.
- PowerShell SHA-256 comparison: all four vendored ARC sources match both their
  receipt and the original sibling KLRD files byte for byte.
- PowerShell SHA-256 comparison against the saved pilot input manifest: unchanged
  `e3/train.py`, `e3/teacher.py`, `e3/config.py`, `readout.py` and `data.py`.
- `git diff --exit-code` confirms tracked E2, mathematical and verifier sources
  remain unchanged.
- `git diff --no-index --check` reports no whitespace errors for reviewed new or
  edited implementation files. Vendored ARC files were excluded from newline
  normalization so their byte receipts remain valid.
- The default shell contains six explicit commands in the required order.
- The saved pilot configuration records vLLM/FP32, E2 run012 and E2B run002.
- No local `experiment3_full/` output directory was created.

## Integration issues addressed

- A partial three-file data-selection transaction is reconstructed on resume,
  instead of treating the first selection file as a completed transaction.
- Full training only reuses pilot clean contexts, never pilot adapter weights.
- Evaluation namespaces distinguish512 and1024 and bind checkpoint/cohort/length.
- Frozen per-response E2 seeds can be passed to ordinary1024 clean generation;
  historical default seed behavior remains available for old callers.
- Later paraphrase scoring cannot invalidate an evaluation receipt by rewriting
  its shared environment-comparison scratch file; per-condition runtime evidence
  remains bound separately.
- Teacher compilation/graph/chunked-prefill deviations are recorded as execution
  notes; true FP32/readout/tokenizer/attention invariants remain checked.
- The review ZIP includes the actual full dependency filename and root PBS
  template, and uses literal public artifact allowlists for run outputs.
- Historical pilot instructions point to the preserved pilot launcher.

## Explicit limits

No Python interpreter, test, mock, compile, shell test, model, training,
evaluation, benchmark, API, dependency installation, download or PBS submission
was run locally. Delivered regression tests are **not claimed to pass**.
The exact-teacher adapter supports the recorded0.32/0.33 versions by reusing E2
worker/model source, but server API and custom-model compatibility, CUDA graph
execution and memory use are not established by this source inspection.

ARC precision is matchedFP32 for all four models, unlike the original BF16
loader; the source receipt, protocol and report disclose that difference.
The original ARC dataset commit was not explicitly pinned; actual server task,
dataset and request hashes are recorded without inventing a historical revision.
Paraphrase quality diagnostics require human semantic review and do not filter
the cohort. Scientific success is determined from actual saved measurements,
not from a completed program or a lower validation KL.
