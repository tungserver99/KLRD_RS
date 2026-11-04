# Experiment 3: exact-teacher KL LoRA implantation

This document covers the historical **2k pilot**. `run_experiment3.sh` now runs
the new full study, including paid paraphrasing; see `README_EXPERIMENT3_FULL.md`.
Use `run_experiment3_pilot.sh` for the pilot workflow described below.

This code implements `gpt_guide/KLRD_RS_Experiment3_LoRA_Codex_Spec_REVISED.md`.
Local delivery is source only: no test, model, training, evaluation, download or
API execution has been performed locally. All commands below are for the GPU server.

## What the default command does

```bash
bash run_experiment3_pilot.sh
```

The launcher first invokes `python main_exp3.py --mode pilot`. It prepares/reuses
2,000 clean train sequences plus 200 clean validation sequences, trains key0
LoRA for two epochs, selects the epoch with minimum validation forward KL, exports
`checkpoints/best_adapter/` and an ordinary `merged_student/`, then writes a
training-only report, then automatically invokes `python main_exp3.py --mode evaluate`
to generate/score the 1,000-output student evaluation and write the updated report.
It stops if training/merge fails. It never calls GPT-5 or installs dependencies.
Direct `--mode pilot` remains training/merge only. The default output is
`experiment3_rs/run_001/`; rerun the same command there to resume.

## One-time server setup

Use a CUDA environment compatible with the current pinned Qwen model and PEFT.
Keep the existing E2/E2B environment available; do not change their artifacts.
On the GPU server only:

```bash
python -m pip install -r requirements-lora.txt
```

PyTorch, Transformers and PEFT versions are recorded. No exact match to E2's
Torch version is required merely to begin E3. Numerical training checkpoints
retain the actual runtime/source identity for coherent optimizer resume. Do not
change the environment halfway through an optimizer-state run.

New runs started by the launcher/CLI default to **standard vLLM**. Execution
now queues128 requests with up to64 concurrent sequences, while saved pilot
configuration fields remain unchanged. HF remains an explicit alternative:

```bash
python main_exp3.py --mode pilot --generation-backend hf
```

The initial selection is recorded; existing runs keep their saved backend,
including previous HF runs. The historical E3Config constructor is deliberately
unchanged: the new default is applied by main_exp3.py only when creating a run,
so a generation-default edit cannot invalidate numerical training checkpoints.

vLLM submits up to128 missing prompts per call, with a separate SamplingParams/seed
for each prompt and at most64 concurrent sequences, native KV cache and decode-only
CUDA Graph configured. FP32 uses TRITON_ATTN/IEEE and disables TF32. There is no
custom rotated model, quantization or silent BF16/HF fallback. Per-output records
are saved durably in original prompt order. The last short batch is retained.
HF keeps its existing per-prompt path. Model/tokenizer/revision, tau=.8,
top_p=1, unrestricted top_k, max512, native EOS and single response stay fixed.

Batch size is execution metadata. Compatible old vLLM journal bindings/digests
are retained without rewriting successful rows; new row execution and worker
runtime metadata record the actual batch/graph/backend. A failed engine batch
stops with a private failure record; it is not turned into fake empty outputs.
Resume submits only IDs without committed responses. A new backend is not
silently mixed into an existing run. Identical sampled tokens across HF/vLLM or
different batch shapes are not asserted: floating-point/RNG paths can differ.
The training objective, dataset selection, thresholds and verifier are unchanged.
Generation and training release/isolate model memory between phases. Default
student evaluation generation is FP32; explicitly selecting bfloat16 is labeled
in provenance and is not claimed numerically equivalent.

## Required full server workspace

When running in this existing project directory, E3 resolves these paths from
the location of main_exp3.py automatically; no separate copying is needed.
The synchronization list below is for a fresh server checkout or missing files.

Required actual artifacts (not only team-review ZIPs):

- `experiment1_rs_continuation/`, including geometry, frozen lambda/key ID receipts,
  final configuration and FP64 readout diagnostic; `e2_inputs_v2/source_receipt.json`.
- `inputs/calibration_manifest.json` with the pinned dataset revision/chat template.
- `experiment2_rs/run_012/`: private prompts, jobs/identity, canonical score caches,
  generation records and group/environment manifests.
- `experiment2_paraphrase/run_002/`: protocol, fixed400-ID selection, private original/
  paraphrase mappings and vector caches for frozen teacher0/clean controls.
- The old sibling `../KLRD/` if trusted original clean train/validation corpora are
  available. For another layout, choose the path on the first invocation:

```bash
python main_exp3.py --mode pilot --legacy-root /server/path/to/KLRD
```

The data stage first audits reusable ordinary clean train/validation provenance.
If absent, insufficient or incompatible, it selects deterministic eligible
`HuggingFaceH4/ultrachat_200k/train_gen` prompts and generates clean completions
once. It never substitutes dataset assistant answers, E2 outputs or paraphrases
for clean-generated training contexts. Full partitions are fixed at10,000/1,000;
pilot is their predetermined2,000/200 prefix. ID/exact/NFKC overlap with frozen
E2/E2B and discovered historical held-out cohorts is excluded. Calibration reuse,
eligibility exclusions and any missing historical evidence are reported in
`provenance/data_selection.json` and `historical_heldout_audit.json`. Known original
KLRD artifact layouts are filtered by held-out role; unused reserve inventories
and historical training rows are not treated as held-out cohorts. Unknown layouts
remain an explicit limitation.

The currently inspected old clean corpus records BF16 generation and lacks the
explicit sampling/hash evidence required by this implementation for FP32 reuse.
Unless a fully eligible corpus is supplied on the server, expect a one-time
fresh FP32 clean generation of2,200 pilot responses; this is recorded, not hidden.

Missing required artifacts produce an explicit error identifying what to sync.
No fake frozen files, private results or scientific PASS claims are generated.

## Fixed objective and memory policy

Key0 is the predeclared first E1 validation key, not a key selected using TPR.
The exact teacher is `RotationReadout.log_probs(r0, lambda*(L@key0))`, where `r0`
is the clean adapter-disabled HF residual immediately before final RMSNorm.
HF already includes the residual; nothing is added again. Tau=.8 is applied
once. EOS remains a full-vocabulary candidate but is not added as a response
prediction time step. Eligible causal positions use existing `encode_response`.

The student is ordinary LoRA-on Qwen with an unrotated frozen native RMSNorm/head.
Loss is only `sum(exp(log_q)*(log_q-log_p))`, summed over full vocabulary and
normalized by all eligible response positions in the whole accumulation window,
including the final short window. It is not CE/SFT, reverse KL, hidden-state MSE,
verifier score or a paraphrase loss. No tau-squared factor is introduced.

One resident FP32-master backbone is used. Teacher forward is FP32/no-grad with
LoRA disabled; the student backbone uses BF16 autocast by default, with both
final tails/log-softmax in FP32. LoRA r16/alpha32/dropout0 trains only the seven
backbone projection modules. Base, embeddings, final norm and head remain frozen.
Chunked full-vocab tail gradients are accumulated back into the attached student
residual, traversing the transformer backward once per microbatch. Gradient
checkpointing is enabled; vocab or dataset size is not reduced to avoid OOM.

Execution-only options, fixed when creating a run, include `--logit-chunk 16`,
`--checkpoint-steps 10`, and `--student-backbone-fp32`. These do not change the
full-vocab KL target. Batch1/accumulation16, LR1e-5, max2 epochs, warmup.03,
weight decay0 and clipping1 follow the revised specification.

## Separate evaluation, merge, paraphrase and packaging

Pilot completion already exports the best adapter and merged model. If export
was interrupted, resume training or run merge explicitly:

```bash
python main_exp3.py --mode merge --output experiment3_rs/run_001
```

Generate ordinary merged-student responses on exactly the1,000 frozen E2 group0
prompts, evaluate a fresh untrained/zero-LoRA-equivalent clean baseline, and score
returned decoded text using the unchanged E2 spherical verifier:

```bash
python main_exp3.py --mode evaluate --output experiment3_rs/run_001
```

No generation-time rotation, key, custom model, logits processor or detector is
loaded by ordinary student inference. Teacher0 and clean E2 canonical scores are
reused. Q100 has10 fixed disjoint groups, Q1000 has one descriptive group, and
Q20/Q50 subdivisions are secondary/nested. Original sampled IDs never substitute
for the returned-text primary score. Clean/wrong controls, zero vectors, failures
and both fixed alphas .01/.0001 remain visible. There is no threshold fit.

Only after reviewing original-text student transfer, explicitly opt into400
student high-fidelity rewrites:

```bash
python main_exp3.py --mode paraphrase --output experiment3_rs/run_001
```

This uses the exact frozen run002 instruction and `gpt-5-mini-2025-08-07`.
It calls only for student responses. Existing E2B teacher0/clean paraphrases and
vectors are reused after identity checks. Valid completed student paraphrases
are not requested again; failed/missing calls can resume with their history kept.
Interrupted in-flight requests without durable results can still be billed twice.
No API keys are embedded in source or results; API initialization occurs only in
explicit paraphrase mode. The existing copied transport reads the project .env.

Rebuild reports and the allowlisted review ZIP without model/API execution:

```bash
python main_exp3.py --mode report --output experiment3_rs/run_001
```

The ZIP includes code/spec/config, public tables, charts and diagnostics. It
excludes model weights, adapters, secret key/basis/Fisher vectors, raw private
train/eval prompts/responses and credentials. Private full artifacts stay on the
server. Reports keep TRAINING_ONLY, EVAL_COMPLETE, PARAPHRASE_COMPLETE or
INCOMPLETE separate; lower training KL alone never establishes student transfer.

Full10k/1k is a separate explicit decision/run, never chained after the pilot:

```bash
python main_exp3.py --mode full --output experiment3_rs/run_002
```

## Resume and validation evidence

Use the same command/output after preemption. Generated rows have deterministic
IDs and saved responses; completed rows are not sampled again. Training commits
adapter, optimizer, scheduler, Python/Torch/CUDA RNG, permutation and cursor.
An incomplete in-flight optimizer window may repeat from the last committed
checkpoint; committed step state is not discarded. Best selection uses only
validation teacher KL, never E2 or paraphrase TPR. Merge integrity is verified
before evaluation; missing merge does not silently trigger training.

Source tests under `tests/test_exp3*.py` are provided ONLY for server execution.
After reviewing server configuration, the owner may run:

```bash
python -m pytest -q tests/test_exp3_contract.py tests/test_exp3_training.py tests/test_exp3_data.py tests/test_exp3_evaluation.py
```

No such command has been run locally. A tiny real GPU bug catch is supplied as
`klrd_rs.e3.train.server_smoke(state,dataset)` for1-2 examples and a real update;
it checks masking, gradients, frozen weights and adapter-disabled clean behavior.
It does not impose a new calibration/parity campaign. Normal training also checks
finite losses and LoRA gradients. PEFT/Transformers compatibility, GPU memory,
resume behavior and throughput still require server execution.

Main costs: missing clean-corpus generation, two sequential backbone forwards plus
one backward per training microbatch, full-vocab tails, and FP64-linear black-box
verification. Reuse valid clean corpus/caches; use the batched vLLM default,
decrease tail chunk size or save optimizer state more frequently
as needed. Optional student paraphrase adds about400 GPT-5 mini requests.
No timing or scientific result is claimed before server measurements.

API references checked during implementation:
[PEFT adapter disable and merge](https://huggingface.co/docs/peft/main/package_reference/peft_model),
[LoRA export](https://huggingface.co/docs/peft/main/package_reference/lora),
[PyTorch checkpointing](https://docs.pytorch.org/docs/stable/checkpoint).

vLLM generation API: [per-prompt SamplingParams and batch ordering](https://docs.vllm.ai/en/v0.32.0/api/vllm/entrypoints/llm/),
[sampling parameters](https://docs.vllm.ai/en/v0.32.0/api/vllm/sampling_params/).
