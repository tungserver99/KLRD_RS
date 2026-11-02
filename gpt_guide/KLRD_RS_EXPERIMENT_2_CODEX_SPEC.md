# KLRD-RS — Experiment 2: Exact-Teacher Free Generation & Black-Box Verification

**Status:** Implementation specification / predeclared feasibility protocol. **Not experimental results.**  
**Prior stage:** Experiment 1 continuation, `VALID_IMPLEMENTATION`, `GO` for teacher-generation feasibility only.  
**Next stage authorized by this document:** Experiment 2 **only**. **Do not** train LoRA, implement PSS, run SaTML suppression, or launch a paraphrase API.

## 0. Task for Codex (read this first)

Extend the **existing KLRD-RS codebase** from Experiment 1. Using the **exact frozen Experiment 1 artifacts**, build an **exact autoregressive rotation-teacher sampler**, generate ordinary open-ended responses, and assess whether the resulting texts contain a reliably detectable **key-specific ownership fingerprint** under the existing spherical verifier.

In contrast with Experiment 1, which measured only **local conditional expectations on pre-existing clean prefixes**, Experiment 2 must sample **new entire sequences from the rotation teacher**. Do not generate with the clean model and then score those outputs as if they were teacher-generated. Do not rerank candidate responses. Do not use a semantic-response preference or any hidden prompt. The key affects **only the teacher's next-token distribution during this offline experiment**; a future trained student will not need the key at inference.

**Main decision:** Is the *ideal rotation teacher*, before any LoRA training, detectable at query budgets `Q=100` and `Q=1000` with the **same** frozen spherical verification rule at significance levels `0.01` and `1e-4`, while maintaining reasonable open-ended text quality?

**Deliverable:** reproducible code, private data artifacts, a concise report with tables for **every owner key**, unit/integration tests, per-output audit trails, and an independently reviewable ZIP **without private key vectors or full raw private datasets**. Run the experiment on the configured GPU environment, rather than supplying only a script or toy results. Do not declare a scientific GO from a smoke test or partial run.

---

## 1. Background and immutable source of truth

Use the actual code/artifacts from **`KLRD_RS_experiment1_continuation_review.zip`** and the corresponding **on-server Experiment 1 workspace**. The review ZIP contains summaries, public geometry and code, but intentionally excludes original raw datasets, full context caches and private key material. Locate those missing items in the existing KLRD/KLRD-RS workspace; do **not** fabricate replacements.

Reference files inside the continuation artifact (relative to `continuation/`):

```text
geometry/basis.npz
geometry/fisher.npz
geometry/whitening.npz
geometry/fisher_checks.json
keys/calibration_key_ids.json
keys/validation_key_ids.json
teacher/global_lambda.json
teacher/KL_by_key.csv
teacher/local_capacity_by_key.csv
provenance/resolved_config.json
provenance/input_manifest.json
provenance/sha256_manifest.json
report/experiment1_report.md
report/summary.json
null/fixed_key_pilot.json
```

Existing implementation modules to reuse/adapt:

```text
code/klrd_rs/model.py
code/klrd_rs/readout.py  # RotationReadout.log_probs, kl, full_scores
code/klrd_rs/math.py     # planes, keys, fit_fisher, sphere_p
code/klrd_rs/data.py     # prompt_ids / canonical text re-encoding
code/klrd_rs/io.py, inputs.py, config.py
```

**Frozen configuration (do not silently change):**

| Setting | Value |
|---|---|
| Base model and tokenizer | `Qwen/Qwen2.5-3B-Instruct` |
| Exact revision | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| Teacher/verifier temperature | `tau = 0.8` |
| PCA basis dimension | `128` |
| Rotation/Fisher score dimensions | `m = 32` |
| Rotation basis seed | `2026102801` (already materialized; **load**, don't regenerate/refit) |
| Fisher whitener | E1 `geometry/whitening.npz` (**load**; do not refit) |
| Validation key bank | E1 **64 keys**, `key_evaluation_seed=2026102803` |
| Shared global rotation strength | `lambda = 0.09271215438842773` |
| Original calibration mean-KL target | `0.005` nats/token |
| Original p99-KL safeguard | `0.05` nats/token |
| Spherical verifier significance levels | `alpha_nominal=0.01`; `alpha_conservative=0.0001` |
| Primary query budgets | `Q=100` and `Q=1000` |

`lambda` was fixed on the calibration key bank; do not rescale it by owner key or test results. The E1 validation bank contains **one observed p99 KL violation at key index 2** (`~0.0534`); **keep that key and report it**. Prior E1 fixed-key pilot found anomalous **nominal-alpha** false-positive behavior for key indices **23** and **59** on paraphrased-clean text; these may be used as **pre-labelled supplementary stress controls**, never secretly substituted into the primary key cohort.

Before any new generation:

1. Verify SHA-256 checksums of loaded artifacts and pinned model/tokenizer revision.
2. Recreate the exact key vectors **only through the frozen E1 key-generation implementation and seed**, and verify all 64 vector hashes against `validation_key_ids.json`; never use a new key RNG for the same key IDs.
3. Verify loaded planes, Fisher whitening and global strength against frozen provenance. Never recompute PCA, QR, Fisher, strength or old E1 diagnostics just to make them pass.
4. Treat earlier KLRD original/SCW-paraphrased cohorts as **development diagnostics**, not an independent final set.
5. Preserve all previous run folders read-only; give Experiment 2 a fresh versioned output directory.

**Fail-fast:** If a required frozen input is missing or hashes disagree, report `BLOCKED_MISSING_OR_CHANGED_ARTIFACT`, with exact file/expected hash. Do not silently replace with synthetic data, rerun E1 fitting, change model version or select a different key.

---

## 2. Exact teacher and verifier definitions

### 2.1. Exact rotation teacher

Let `c_t` be the entire actual prefix: owner-provided prompt + generated response tokens up to position `t-1`. Let `r(c_t)` be the residual vector **immediately before** the final model normalization, `G` the native frozen normalization and vocabulary head, and `p0` the clean next-token distribution at the frozen temperature.

Let `P,Q` denote the **32 fixed orthonormal rotation planes** of E1, and `L = F_ridge^{-1/2}` the **frozen E1 whitening matrix**. For an E1 evaluation key `k` (`||k||_2=1`), compute the **32 angles**:

```text
angles_k = lambda_frozen * (L @ k)
```

Apply **exact orthogonal rotations** in these planes to `r(c_t)`; the distribution from which the next token is *actually sampled* is:

\[
q_k(w\mid c_t) = \operatorname{softmax}\!\left(G(R(\mathrm{angles}_k)r(c_t))/0.8\right)_w.
\]

**No tangent approximation for generation**. The tangent-score approximation is for the verifier only. The sampler must be mathematically equivalent to the existing `RotationReadout.log_probs(r, angles)` / native `G(Rr)` implementation, including RMSNorm, the tokenizer, logit temperature, vocabulary and stopping tokens.

### 2.2. Spherical verifier — unchanged from Experiment 1

On the **text actually returned by the generator**, construct the canonical original user prompt/chat template plus **re-tokenized returned text**. Teacher-force through the **clean frozen reference model**, obtaining clean-prefix pre-final-norm residuals and token targets. For every eligible observed response-token position, compute E1's Fisher-whitened tangent-score vector `psi(c_t, y_t)`; aggregate:

\[
V(D) = \sum_{(i,t)\in\mathcal I(D)}\psi(c_{it},y_{it}),\quad
T_k(D)=\frac{k^\top V(D)}{\lVert V(D)\rVert_2}.
\]

If `V(D)=0`, use `p=1` and report the zero-vector event (do not silently omit the batch). Use the **existing** `sphere_p(V,k)` implementation:

\[
p_k(D)=\mathrm{BetaCDF}_{(m-1)/2,(m-1)/2}\!\left(\frac{1-T_k(D)}{2}\right),\quad m=32.
\]

**Same p-value formula and thresholds** for clean, owner-key teacher and wrong keys; no clean-response percentile calibration, re-centering, paraphrase detection or key-specific threshold.

Only count tokens recoverable from the returned text. Exclude EOS/hidden special-control tokens from the primary verifier. Preserve exact provenance of the chat template and text-to-token roundtrip; the **canonical black-box verification path** must not peek at generation-time hidden residuals or the teacher's original token IDs. For diagnostic purposes, it may additionally score the original generated token IDs and report the difference, but it cannot use that path as the primary result.

### 2.3. Statistical interpretation (must appear in the report)

The analytic spherical p-value is valid under the **random-key null** `D independent of k`, with basis and whitener chosen independently of the evaluation key. It is not a blanket 1% false-positive guarantee for any one frozen owner key under arbitrary negative-text distributions. The conservative `1e-4` threshold gives a distribution-over-keys bound for a specified key-independent null distribution; **do not** claim measured 1% fixed-key FPR from a handful of disjoint batches.

Do not infer “TPR=100%” from the positive local expectations in Experiment 1. Only **free-generated responses** can establish practical teacher detection power.

---

## 3. Generation engine — exactness takes priority

### 3.1. Implementation constraints

Implement an efficient **batched incremental sampler with KV cache**, replacing only the final readout distribution by the exact rotated teacher. A valid implementation may use:

- **vLLM**, if the installed version provides a tested hook/modified model runner that applies the **exact pre-final-normalization residual rotation before the readout**, or a demonstrably equivalent exact logits computation; **not** a text postprocessor, green-token proxy or approximate arbitrary logits processor.
- Otherwise, a **Hugging Face / PyTorch** GPU sampler with `past_key_values`, `torch.no_grad()/inference_mode()` and the previously validated E1 readout is acceptable. Document why ordinary vLLM sampling could not faithfully implement this specific teacher intervention. An HF exact sampler is preferred to a faster but incorrect vLLM approximation.

**No** full-sequence replay from the first token for every step; KV cache is mandatory. Do not train or edit model weights. Use the pinned model, frozen tail, and exact KLRD-RS score/readout. May use BF16/FP32 backbone only after comparing generated logits to the E1 FP32 exact reference on fixed contexts; use FP32 for log-softmax, probability normalization, KL and score accumulation. Do not silently enable TF32 if it breaks the comparison tolerance.

When sampling, compute next-token log probabilities **once** with `tau=0.8`; do not divide logits by temperature twice. **Do not** add post-hoc `top_k`, `top_p < 1`, repetition penalties, beam search, reranking or hidden prompting; none are part of `q_k`.

Suggested explicit sampling config, matching the original KLRD family:

```yaml
sampling: true
sampler: categorical_from_exact_qk
temperature_in_teacher: 0.8
top_p: 1.0
top_k: 0           # no truncation; implement directly if API uses another sentinel
num_beams: 1
repetition_penalty: 1.0
max_new_tokens: 512
stop_on_eos: true
```

The EOS token **must** be included in the teacher's categorical distribution for generation, even though EOS is excluded from black-box verification. If EOS appears, stop naturally. Do not trim, repair, paraphrase or regenerate empty/short/low-quality outputs to improve the metrics. Preserve them and flag them.

Clean reference output generation must use the **same sampling configuration** and frozen `p0` distribution, and the **same prompt pool**, with `angles=0` / no rotation. Save pairing indices and seeds. For paired comparability, use aligned per-prompt random streams where feasible; independence assumptions must still be reported correctly.

### 3.2. Mandatory sampler correctness tests before long generation

1. At `angles=0`, sampling-logits equal native clean model logits / 0.8 and the E1 readout within an explicit numerical tolerance.
2. For several frozen owner keys (including key 2), `teacher_log_probs` match **full native `G(rotate(r))`** on at least 10 actual prefill and 10 incremental-KV decode contexts; record max and p99 absolute log-prob error. Do not validate on prefill alone.
3. Per-step returned log probabilities sum to 1 (within floating-point tolerance), have no NaN/Inf, and sampled tokens belong to the model vocabulary.
4. KV-cache incremental residual/logit results agree with a non-cached full-prefix reference at selected steps/tokens, including after EOS-sensitive tokens; no off-by-one label shift.
5. Verify prompt/chat template and attention masks (padding + batch decoding), absence of KV-cache cross-request contamination, and deterministic reproduction of one response per fixed `(owner_key, prompt_id, generation_seed)` on the same runtime.
6. Clean-vs-keyed KL computed from full-vocabulary probabilities is nonnegative (up to tiny numerical rounding) and is zero at zero rotation.
7. For a few outputs, compare native generated token-ID scoring versus **text-only re-tokenization**. Record response roundtrip rate and the effect on spherical p-values. No privileged IDs in primary verification.
8. Test interruption/resume by deliberately stopping and rerunning a tiny job. Completed records must not be duplicated or silently regenerated.

If an exact sampler fails any test: **STOP**, output `SAMPLER_INVALID`, do not run a large experiment, and do not substitute synthetic results.

---

## 4. Prompt cohort and immutable evaluation design

### 4.1. Prompt source and eligibility

Use the **same broad UltraChat prompt distribution** as KLRD, but a **fresh, deterministic, non-overlapping cohort** where possible. First enumerate eligible prompts from official `test_gen` that were **not** in E1 calibration, previous KLRD diagnostic, legacy null/final, or any other explicitly loaded development manifest. Verify both prompt IDs and exact/normalized prompt-text hashes. If `test_gen` has fewer than the required number of fresh prompts, draw the shortfall deterministically from unused `train_gen` **after excluding all prior calibration/training/development IDs/text hashes**; record source split per row. Do **not** quietly call a mixed-split pool a pristine `test_gen` final set. If strict nonoverlap cannot be verified, write an explicit `COHORT_PROVENANCE_LIMITATION` and treat the result as exploratory.

Keep original KLRD input eligibility: **prompt length 16–512 tokens** under the pinned tokenizer; no model-scored quality filtering, semantic filtering or paraphrase-based filtering. One user message per prompt with the pinned Qwen chat template and generation suffix. Freeze the prompt pool **before** examining fingerprint response scores. Do not pick easy prompts or regenerate prompts to improve detection.

**Frozen new design constants (Experiment 2 only):**

```yaml
experiment2_prompt_seed: 2026110101
experiment2_generation_seed: 2026110102
experiment2_group_seed: 2026110103
primary_owner_key_indices: [0, 1, 2, 3, 4, 5, 6, 7]
wrong_key_count_per_owner: 20
supplementary_stress_key_indices: [23, 59]
primary_Q: [100, 1000]
primary_alpha: [0.01, 0.0001]
max_new_tokens: 512
primary_disjoint_Q1000_batches_per_key: 5
```

Use a common, deterministically ordered **5,000 unique prompts** for all eight primary keys. Each key generates a fresh independently sampled response on each prompt. Generate **one clean response per prompt** (5,000 clean outputs shared for paired comparisons); do **not** count reused clean outputs across keys as independent negative examples. Cross-key comparisons are paired by prompt and correlated; treat key and prompt as distinct analysis levels. If 5,000 truly fresh prompts cannot be sourced, **report the shortage and stop with `INSUFFICIENT_ELIGIBLE_FRESH_PROMPTS`**, rather than reducing Q or the number of disjoint batches after looking at results.

Construct **five disjoint groups of 1,000 prompt IDs** in one deterministic permutation. Within each 1,000-prompt group create **ten disjoint groups of 100**. Thus each primary key has exactly **5 trials for Q=1000** and **50 trials for Q=100**; all groups are predetermined and independent of outputs. Note: Q=100 groups are nested within Q=1000 groups, so results across budgets are **not independent**. For each group, predeclare IDs/order, hash manifest and `group_id` before sampling. Do not bootstrap outputs into apparent “independent trials”.

Cost: primary experiment is **8×5,000 = 40,000** exact-teacher responses plus **5,000** clean responses; schedule/chunk across GPU jobs as needed. This is a feasibility design, not a claim that five Q1000 batches per key can precisely estimate a population TPR/FPR. All reported intervals and conclusions must reflect the small number of independent Q1000 trials. Do not quietly replace the 40k-response design with a smaller data run and label it complete.

**Stress controls** `23` and `59`: if feasible, generate **one predeclared 1,000-prompt group each**, analyzed separately. They were chosen after inspecting E1 null behavior, so their performance must **not** be pooled into the representative eight-key estimate. Existing E1 paraphrased-clean data can be reused strictly as a labelled diagnostic; **no new paraphrasing** is needed or authorized.

### 4.2. Safe, resumable execution

- **Stage 0 (implementation smoke):** 1 key × 10–20 prompts + clean controls. Correctness only; exclude from primary statistics if prompts overlap primary, or use a disjoint smoke pool.
- **Stage 1 (pilot, predeclared):** first 1,000-prompt group for **all eight** primary keys and clean model. Produce a labelled `PILOT_ONLY` report without selecting “good” keys or tuning hyperparameters.
- **Stage 2 (full study):** remaining four 1,000-prompt groups for the **same eight keys**, using the exact same frozen engine/config. Unless a severe correctness/compute blocker is detected, do not use pilot TPR to drop bad keys, tune lambda, change the threshold or alter Q. If the user/team later approves an early stop for cost, archive the partial study as `INCOMPLETE`, not as full Experiment 2.

Resume must preserve source sequence IDs, RNG seeds, prompt IDs, key IDs, EOS decisions and generated token sequences. A crash should not trigger different samples or duplicated observations. Every job should fit within cluster scheduling limits; support sharding by `(owner_key, group_id, chunk_id)` and deterministic merging. Never modify the frozen E1 artifacts while running.

---

## 5. Black-box scoring and controls

### 5.1. Primary owner-key detection

For every primary owner key `k_i` and every predeclared group `D`, run the **exact same E1 spherical verifier** on returned text:

```text
p_owner(D, k_i) = sphere_p(V(D), k_i)
owner_detect_nominal = [p_owner <= 0.01]
owner_detect_conservative = [p_owner <= 0.0001]
```

Use one response per prompt. Report at **both Q=100 and Q=1000**, preserving complete p-value and cosine-score distributions, not only detection flags. Primary evaluation is the **canonical black-box re-tokenization** path.

### 5.2. Clean FPR and fixed-key pilot

Use the same 5,000 independently clean-generated responses, partitioned into the **same groups**, scored against each of the eight frozen owner keys. Report per-key nominal/conservative detection rates for Q=100 and Q=1000, count of truly disjoint groups, binomial/Wilson intervals (descriptive), and worst/p90 across keys. Because all keys reuse the clean text, do not treat those scores as independent negative batches in pooled CIs. 5 Q1000 groups per key **cannot certify** a 1% fixed-key FPR, nor rule out rare false positives at `1e-4`.

For supplementary robustness context, report the already-observed E1 original/paraphrased-clean fixed-key pilot separately (especially 23, 59). **No clean percentile threshold may be learned or recalibrated** from either E1 or E2 clean data.

### 5.3. Wrong-key specificity

For each owner `i`, select **20 fixed wrong keys** from the remaining **E1 64-key bank** using a deterministic index rule set before seeing any scores (e.g. next 20 cyclic indices modulo 64, skipping `i`). Log all wrong-key IDs and hashes in a frozen manifest. Wrong keys must never be regenerated to improve specificity.

Score the **same owner-teacher responses** under all 20 wrong keys; never generate extra responses for wrong-key scoring. Report wrong-key detection mean, median, p90 and maximum **across keys**, plus pooled groups with correct dependence caveats. Also report **closed-set 21-way identification**: classify as the key with highest cosine score / smallest spherical p-value among `[correct + 20 wrong keys]`; ties count as incorrect. Closed-set identification is a *different metric* from false-positive-controlled ownership verification and must be labelled separately.

An apparent owner TPR increase accompanied by high wrong-key activation is **not** evidence of a strong ownership fingerprint.

### 5.4. Text-only vs privileged-token control

For a small fixed cohort and each owner key, compute an auxiliary verifier using original sampled token IDs. Compare against primary verifier using visible text only:

```text
native-token T, p
text-only T, p
roundtrip_equal
absolute score shift
whether detection flips at either alpha
```

Do not report privileged-token results as primary. If text-only reconstruction destroys the signal, this is a **real black-box deployment limitation**, not a reason to bypass the text-only path.

---

## 6. Utility, generation distribution and KL diagnostics

This is a **teacher feasibility** experiment, not a complete paper benchmark suite. Still, quality cannot be ignored.

**Required automatic diagnostics**, clean vs owner teacher, per key and overall:

- Response-length distribution in tokens and characters (mean, median, p5/p95), empty/very-short rate, EOS termination rate and max-length truncation rate.
- Repetition diagnostics: repeated n-gram rate (e.g. 3-/4-gram), distinct-1/2, obvious looping/degenerate text percentage with a frozen deterministic rule.
- Tokenization roundtrip/recoverability percentage.
- Full-vocabulary per-step or deterministically sampled **free-generation-context KL** `KL(q_k || p0)` on prefixes that the **teacher actually visits**; report per-owner mean, p95/p99, maxima, proportion above the E1 guardrail `0.05`, and compare with E1 calibration and E1 key 2. Use key-independent reservoir sampling if computing every token is too expensive, and report sampling rate/seed.
- A predeclared, reproducible text quality comparison on a small fixed set of prompt IDs (e.g. 100 across the frozen cohort), preferably with blinded human-readable original/clean vs teacher pairs and a simple objective task-following diagnostic. **Do not** interpret an LLM judge score alone as a rigorous utility proof.
- At least 10 representative **preselected** prompt/clean/teacher examples (not cherry-picked favorable cases). Redact private keys/system-local paths from reports.

Monitor GPU time/throughput and peak memory; calculate actual cost before recommending LoRA. Record EOS sampling rules, truncation, decode settings and any inference fallback. Do not postprocess responses to hide quality problems.

**Important:** An occasional p99-KL breach under frozen lambda (as for E1 key 2) must be reported rather than fixed with per-key rescaling or key exclusion. If multiple primary keys or many free-generated contexts violate the safeguard materially, mark a distortion problem and do not claim unqualified utility success.

---

## 7. Statistical reporting and scientific gates

### 7.1. Main results table — mandatory

Per owner key and for the pooled descriptive summary, at Q=100 and Q=1000, show:

| Field | Required |
|---|---|
| `n_prompts`, `n_disjoint_groups`, `n_observed_tokens` | Actual counts |
| Owner TPR @ p<=0.01 and p<=1e-4 | Counts, rates, descriptive interval |
| Clean FPR @ both alphas | Counts, rates, disjoint-group denominator |
| Wrong-key detection @ both alphas | Mean, median, p90, worst and all 20 raw rates |
| 21-way correct-key identification | Count/rate and ties |
| Score distribution | Owner/clean/wrong median and quantiles of `T` and p-value |
| Free-context KL | Mean, p95, p99, guardrail violations |
| Utility | Length/repetition/EOS/truncation + examples |

Do not use response-level or token-level bootstrap to claim many **independent** ownership trials. For Q=1000, each key has only **five** disjoint trials; per-key rates take coarse 20% increments. Report counts prominently. Correlation across keys using the same prompt pool further limits pooled confidence interpretations. If using bootstrap for uncertainty on other continuous measures, resample whole **prompt IDs** and keep within-prompt key pairing; label this resampling uncertainty, not independent deployments.

### 7.2. Decision labels (feasibility, not a final paper claim)

The target proposed in the research plan is meaningful free-generation detection at **Q=1000, p<=1e-4**, without wrong-key collapse or conspicuous utility damage. Predefine these *operational* reporting labels:

- **`E2_PROMISING_FOR_LORA_REVIEW`** (not an automatic LoRA authorization): 32/40 or more of the key×Q1000 batches detect at `1e-4` (**>=80% descriptive pooled TPR**), **at least six of eight** owner keys detect in **>=4/5** Q1000 batches, no pervasive wrong-key activation (descriptively near nominal), clean checks do not display pronounced systematic failure at `1e-4`, and no severe model-quality degeneration or widespread KL violation. Report uncertainty and any exceptions; do not claim fixed-key 1% FPR certified.
- **`E2_PARTIAL`**: meaningful ownership signal at nominal alpha or only a subset of keys at conservative alpha; or a good TPR but unresolved wrong-key, utility, black-box tokenization or fixed-key-null concern.
- **`E2_NO_GO`**: teacher has weak key-specific power even at Q=1000, or severe generation-quality/detection failures that invalidate the proposed approach.
- **`E2_INCONCLUSIVE`**: incomplete cohort, too few trials, missing source artifacts, invalid sampler/score, or unresolved numerical discrepancies. A test implementation failure must not be described as a scientific failure of the method.

**Interpretation:** Even `E2_PROMISING_FOR_LORA_REVIEW` is only a feasibility signal, not a statistically precise population-TPR estimate. Do **not** automatically launch Experiment 3. Stop and hand over a review package.

### 7.3. Prohibited mid-run choices

- No retraining, no LoRA, no hidden-state finetuning, no PSS, no proxy paraphraser, no SaTML suppression.
- No new owner key selection after seeing results; no discarding key 2 or selecting keys by E1/E2 TPR/KL/FPR.
- No refit of Fisher/rotation basis or re-tuning `lambda`; no per-key rescaling, per-key thresholds or clean percentile calibration.
- No changing sampler to top-p/top-k filtering, greedy or reranking to rescue detection.
- No replacing exact rotated-teacher sampling with clean-generation replay or offline selected candidates.
- No dropping bad generations or low-score prompts; no paraphrase regeneration or output filtering.
- No interpreting 5 Q1000 batches as reliable proof of rare false-positive rates.

---

## 8. Suggested implementation interfaces

Adapt to existing modules, without needlessly rewriting Experiment 1:

```python
class ExactRotationTeacherSampler:
    def __init__(self, model, tokenizer, rotation_readout, P, Q, L, lambda_frozen): ...
    def generate(self, prompt_ids, key, seed, max_new_tokens=512):
        """Return visible response, generated token IDs, EOS/truncation metadata,
        optional deterministic per-step KL samples, hashes and audit fields.
        Must sample autoregressively from exact q_k at every token."""

class CleanSampler:
    def generate(self, prompt_ids, seed, max_new_tokens=512): ...

class SphericalBlackBoxVerifier:
    def score_text(self, prompt, returned_response, keys):
        """Use original visible prompt plus text-only response re-tokenization;
        return V, cosine T and spherical p for ALL supplied keys."""
```

The `SphericalBlackBoxVerifier` should compute public `V(D)` **once per batch** and vectorize key scoring. Do not rerun the clean backbone 21 times for each wrong key. Reuse the existing efficient readout/tangent implementation and chunking. Keep the exact key vectors private; key hashes/IDs can be included in the review ZIP.

**Testing checklist**: sampler zero-rotation parity, exact native-rotation parity (prefill + KV decode), normalized probabilities, temperature once, EOS handling, text-only scoring, wrong-key score equivalence, p-value Beta reference, hash validation, interruption/resume, disjoint group IDs, no duplication, seeded reproducibility, toy/CPU smoke clearly labelled vs real Qwen.

---

## 9. Required outputs and packaging

Create a new independent root, e.g.:

```text
experiment2_rs/
  provenance/
    input_manifest.json              # source E1 paths + verified hashes
    resolved_config.json             # all new E2 settings + frozen E1 settings
    prompt_manifest.json             # source split, ID/text hash, group and seeds
    key_manifest.json                # owner/wrong/stress IDs + hashes, no secret vectors
    engine_manifest.json             # HF/vLLM, precision, package versions, device
    sha256_manifest.json
  tests/
    exact_sampler_parity.json
    kv_cache_parity.json
    text_roundtrip_checks.json
    resume_integrity.json
  private/                            # DO NOT INCLUDE IN PUBLIC REVIEW ZIP
    teacher_generations_key_*.jsonl  # prompt IDs, response, token IDs, seed, quality metadata
    clean_generations.jsonl
    intermediate_score_vectors.*
    key_material_if_essential.*       # permissions controlled, never commit
  scores/
    batch_owner_scores.csv
    batch_clean_scores.csv
    batch_wrong_key_scores.csv
    per_key_summary.csv
    closed_set_identification.csv
    text_only_vs_native_control.csv
    fixed_key_stress_controls.csv
  utility/
    generation_quality_by_key.csv
    free_context_KL_by_key.csv
    examples_redacted.md
  report/
    experiment2_report.md
    summary.json
    findings_and_limitations.md
  KLRD_RS_experiment2_review.zip
```

Every row of `batch_owner_scores.csv` / `batch_clean_scores.csv` should contain:

```text
owner_key_id, group_id, Q, n_prompt_ids, n_eligible_response_tokens,
score_norm, cosine_T, p_spherical, detect_at_0_01, detect_at_1e_4,
roundtrip_mismatch_count, data_status
```

Wrong-key rows additionally need `wrong_key_id`; keep the full 20-key results. Record the exact response IDs contributing to each group in a separate manifest so an auditor can reconstruct aggregations. Report JSON must contain gate flags, count denominators, reasons for incomplete groups/outputs, and decision label. Hash the source code and private generation manifests without leaking raw text or keys.

ZIP package should include source/tests, resolved config, nonsecret manifests, tables, summary and selected redacted examples. **Exclude private key values, secret seeds if not intended for publication, full private raw outputs and caches, access tokens, HPC login details and any credentials.** Preserve primary raw artifacts on the server for reproducibility. If raw text is already non-sensitive and intended for review, provide a separate restricted attachment rather than leaking it automatically through a public ZIP.

---

## 10. Completion contract / what to report back

1. State whether exact teacher generation was implemented with a validated vLLM path or cached HF path; provide real-Qwen prefill/decode parity numbers.
2. Report exact prompt counts and five disjoint Q1000 groups per owner key, or explicitly mark incomplete.
3. Provide per-owner-key detection results at both alpha levels and Q levels (do not show only overall mean).
4. Provide correct-vs-20-wrong-key identification, clean FPR diagnostics, utility/KL and key-2 distortion details.
5. Discuss the distinction between E1 random-key null validity and E2 observed fixed-key FPR; give independence denominators, not inflated bootstrap claims.
6. Report `E2_PROMISING_FOR_LORA_REVIEW` / `E2_PARTIAL` / `E2_NO_GO` / `E2_INCONCLUSIVE` and the exact predeclared rules behind the label.
7. Archive the full versioned run and generate the review ZIP with checksum manifest.
8. **STOP after Experiment 2.** No LoRA, no PSS and no new paraphrase experiment without a separate review decision.

**Core research principle:** Prove that the *actual exact teacher* emits independently verifiable, key-specific evidence in free-generated text before attempting to distill that evidence into a production model's weights.
