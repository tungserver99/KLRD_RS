# KLRD-RS + PSS — Frozen Research Design and Experiment 1

Status: proposal / preregistration-ready design, **not experimental results**. 2026-10-28.

## 0. Research goal and decisions

Develop a weight-resident, black-box-verifiable LLM fingerprint with (i) no semantic-domain utility trade-off by design, (ii) a key-randomization null not requiring clean-response threshold calibration, and (iii) a later paraphrase-stable extension that remains in the same latent rotation-score family.

Primary core: **KLRD-RS** (Randomized Score-Space KLRD).
Extension (not implemented in Experiment 1): **PSS** (Paraphrase-Stable Subspace).

- Keep KLRD: orthogonal residual rotations before final normalization; natural training contexts; token-level KL teacher distillation; model weights only at deployment.
- Replace KLRD's empirically calibrated raw LLR/Z verifier by a spherical-key score test.
- No bigram/green-list construction; no DPO, semantic-response candidate ranking, or hand-design of lexical groups.
- In Experiment 1, do **not** use paraphrases to choose rotation basis, Fisher transform, key, or strength. Previously generated paraphrases are **null diagnostics only**.
- SaTML's published bigram statistical analysis may be less effective against full-context latent rotations; this is a hypothesis, NOT a guarantee of robustness to white-box conditional analysis.

## 1. Core mathematical definition

For a full generation prefix `c` (including visible owner prompt when available), let `r(c)` be the clean-model residual vector directly entering the final normalization, and `G` the unchanged final normalization/readout. Let `tau = 0.8`, and

`p0(w|c) = softmax(G(r(c))/tau)[w]`.

Construct **public, key-independent** skew-symmetric orthogonal-plane generators `A_j`, `j=1..m`, from clean calibration activations. Use `m=32` disjoint orthonormal 2D planes in the existing 128-D PCA activation subspace, orthogonal to the activation mean. Fix generator orientation with a deterministic public seed and record generator hashes.

Define exact rotation family

`R(a) = exp(sum_j a_j A_j)`, with `R(a)^T R(a) = I`.

`p_a(w|c) = softmax(G(R(a) r(c))/tau)[w]`.

Tangent score vector:

`phi_j(c,w) = partial_{a_j} log p_a(w|c) at a=0`.

The score identity holds exactly at each context:

`sum_w p0(w|c) phi_j(c,w) = 0`.

Estimate Fisher on independent, ordinarily sampled clean calibration contexts:

`F = E_{c, w~p0(.|c)} [phi(c,w) phi(c,w)^T]`.

**Important:** An empirical `F` can be regularized; record the effective matrix `F_eff = F_hat + eps*I`. Define `L = F_eff^(-1/2)` and `psi(c,w) = L phi(c,w)`. This is approximately, not necessarily perfectly, Fisher-whitened. `L` must be frozen before owner-key sampling.

Owner key `k ~ Uniform(S^(m-1))`, obtained by normalizing an independent isotropic Gaussian vector. The owner key is sampled **after** freezing basis, Fisher and strength. Do not reject keys or choose a good key based on any outcome. All final owner keys must be independent from keys used to set global strength.

Teacher:

`a_k = lambda * L k`

`q_k(w|c) = p_{a_k}(w|c)` (exact finite orthogonal rotation, not first-order approximation).

Choose **one global lambda independent of final owner keys** using a separate calibration-key bank and independent clean calibration contexts. Target approximately 0.005 mean KL nats/token with p99 target <= 0.05; if the uniform key distribution cannot meet these at a viable common lambda, report the failure rather than per-key rescaling or rejection.

Student training (future work, NOT Experiment 1):

`min_theta E_c KL(q_k(.|c) || p_theta(.|c))`.

During deployment, only student weights remain; no runtime rotation, sampling intervention, or key.

### 1.1. Black-box verifier

For observed prompt/response text `D`, use a fixed, key-independent selection of recoverable response token positions `I(D)` (all visible response prediction positions; exclude EOS, padding, special/hidden positions), and compute

`V(D) = sum_(i,t) psi(c_it, y_it)`.

`T_k(D) = k^T V(D)/||V(D)||` when `||V||>0`, otherwise p-value 1.

For any **key-independent** data distribution `D` (including output transformations independent of the owner's key), conditional on any fixed nonzero `V(D)` and frozen geometry:

`(1+T_k(D))/2 ~ Beta((m-1)/2, (m-1)/2)` over the random owner key.

Thus exact one-sided p-value:

`p_k(D) = BetaSF((1+T_k(D))/2; (m-1)/2, (m-1)/2)`.

Use stable incomplete-beta survival function. No clean-response calibration, no paraphraser identification, and no threshold changes conditional on observed transformation.

**Scope and caveats:**

- This is exact for the key-randomization null; it does NOT show FPR <= alpha for every *fixed* key or every clean distribution.
- For fixed null distribution `P0`, if `E_k[FPR_k(alpha0)]<=alpha0`, then `P_k[FPR_k(alpha0)>alpha] <= alpha0/alpha` (Markov). Setting `alpha0=1e-4`, `alpha=.01` gives at least 99% of randomly drawn keys with FPR <= 1% for that fixed null population; this is a **key-probability** statement, not a guarantee for a particular sampled key.
- For multiple predeclared null populations, apply a union bound and adjust `alpha0` accordingly.
- If prompts are selected using key-dependent probing procedures, if the null source has learned about the key, or if the geometry is refit using the owner key, the independence premise can fail.
- No method based solely on key independence can guarantee nontrivial power **and** fixed-key FPR against arbitrary malicious, key-aware text generators.
- Owner side uses known prompt text for score computation; for **input rewrite**, unseen internal prompt may differ. Key-randomized null survives if the transformation is key-independent, but statistical power is not assured.

For m=32, analytic cosine cutoffs: one-sided alpha=.01 corresponds to approximately T>=0.4032; alpha=1e-4 to T>=0.6036. These are numerical illustrations, not experimentally calibrated thresholds.

### 1.2. Why a score-space test aligns with a rotation teacher

At small lambda,

`log(q_k(w|c)/p0(w|c)) = lambda * k^T psi(c,w) + O(lambda^2)`.

And `E_{q_k} psi(c,W) - E_{p0} psi(c,W)` initially aligns with `k` in whitened Fisher coordinates. These are local first-order statements; they do not prove free-running student detectability.

## 2. Later extension PSS (out of Experiment 1)

First show a functioning core checkpoint, including exact SaTML bigram-suppression evaluation and a stronger context-conditioned/statistical baseline if feasible. Then diagnose original, output paraphrase and input paraphrase using the same verifier.

For output paraphrase proxy `A_out` independent of key and clean responses `Y~P0(.|x)`, define the sequence score vector `V(x,Y)` and

`C_out = E[V(x,A_out(Y)) V(x,Y)^T]`.

The score-function identity says that the derivative at zero fingerprint strength of expected detector evidence after output paraphrase along direction `u` equals `u^T C_out u`, when `V` truly is the score of the teacher response distribution and the paraphraser is fixed/key-independent.

Choose a predeclared d-dimensional subspace U (e.g. d=16, columns orthonormal) of the *same whitened* m-dimensional score space to preserve positive first-order evidence transfer and sufficient original Fisher signal. Use fresh spherical keys in `R^d`; teacher parameters become `a_k=lambda L U k`; detector score becomes `k^T U^T V/||U^T V||` and the Beta distribution uses d, **never** infer d or channel identity from the suspect text. The statistical test retains the same rule, not literally the same numerical Beta parameters when d differs.

Input paraphrasing requires a different paired operator: the generator receives transformed prompt `x'`, the owner may only know `x`; the transfer is between teacher score at `(x',Y')` and detector score computed with `(x,Y')`. Do not silently treat this as output paraphrase. Evaluate input and output separately.

Proxy is for selection only; use held-out prompts and an independent final paraphraser. Do not include proxy generation or recalibration at verification.

## 3. Experiment 1 — Spherical-null & local-teacher feasibility, no training

### 3.1. Primary scientific questions

1. Are residual rotation tangent features implemented correctly and nondegenerate in Fisher geometry?
2. Does the analytic spherical p-value equal numerical key randomization on key-independent clean original **and already available** paraphrased data?
3. Does a new exact finite-rotation teacher with **uniform spherical keys and one global lambda** induce positive correct-key tangent-score signal on ordinary clean prefix contexts at acceptable KL distortion?
4. On the **existing historical KLRD checkpoint only as diagnostic**, does a tangent projection align with the original keyed likelihood-ratio evidence on held-out original responses? This is not a new-method TPR claim.

**Out of scope:** new LoRA/SFT/DPO, fine-tuning, new paraphrase API calls, SaTML attack evaluation, key selection by outcomes, empirical threshold fitting, held-out free generation from new teacher, final fixed-key-FPR certification.

### 3.2. Frozen model and data

- Model: `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, same tokenizer revision.
- Sampling: model chat template; do_sample=true, temperature=.8, top_p=1, top_k=0, max_new_tokens=512. Same as original KLRD.
- Calibration: reuse **clean-only** `train_gen` calibration activations/responses from KLRD, excluding final evaluation prompts; if exact artifacts absent, recover the split manifest and rerun clean model only (do not silently change cohort).
- Diagnostic validation clean sets: KLRD's previously saved `test_gen` clean-original and clean-output-paraphrased pairs, preserving exact prompt IDs and text; optional old fingerprint original/paraphrased pairs strictly for historical compatibility diagnostics.
- These diagnostic cohorts have already influenced method development. They are **development data**, not a new confirmatory final test.
- Do NOT fit PCA, Fisher or lambda using these diagnostic validation cohorts.
- Don't claim that 2,500 responses give precise fixed-key FPR at Q=100. At most 25 disjoint Q=100 batches per such cohort.

### 3.3. Exact experimental parameters

```yaml
model: Qwen/Qwen2.5-3B-Instruct
revision: aa8e72537993ba99e69dfaafa59ed015b17504d1
tau: 0.8
pca_dim: 128
m: 32
rotation_planes: 32
geometry_seed: 2026102801
key_calibration_seed: 2026102802
key_evaluation_seed: 2026102803
calibration_keys: 32
validation_keys: 64
fisher_context_sample_cap: 100000
mean_KL_target_nats: 0.005
p99_KL_safeguard_nats: 0.05
nominal_alpha: 0.01
conservative_alpha: 0.0001
query_budgets_diagnostic: [20, 100, 1000]
```

A fixed regularization rule for Fisher must be defined using **calibration data only** and saved to the resolved config, not chosen by downstream FPR. Example: fixed ridge `eps = 1e-3 * trace(F_hat)/m`. Report original eigenvalues, shrinkage and condition number. If the effective tangent space has insufficient rank, report NO-GO; do not silently change m.

### 3.4. Procedure

**Step A — Resolve artifacts and freeze provenance.** Locate exact clean model, KLRD calibration splits, existing original/paraphrased responses, original checkpoint/key for optional compatibility. Hash all manifests. Keep all historical artifacts immutable. Missing diagnostic artifact must be reported, never replaced with a different test cohort.

**Step B — Public geometry.** Use existing KLRD clean calibration activation subspace (PCA128, centered mean) to construct 32 disjoint skew-rotation generators with public seed. Validate generator skew-symmetry, disjoint orthonormal planes and norm preservation numerically. Do not use any fingerprinted text.

**Step C — Tangent-score implementation.** Compute phi_j using exact derivative of `log softmax(G(R(a)r)/tau)[w]` at zero. Verify with central finite differences on small independent synthetic/calibration sample for all directions (tolerance adjusted for FP32 finite differences). Check conditional score identity using a full-vocabulary probability-weighted sum on a small sample. Do not evaluate only positive observed tokens for this identity.

**Step D — Fisher fit.** Estimate `F_hat` from independently sampled next-token observations under `p0` on held-out clean calibration contexts; compute fixed ridge, L and whitened psi. Validate covariance eigenvalues and approximate isotropy on separate clean calibration subset. Avoid leakage from later validation text.

**Step E — Global teacher lambda.** Draw 32 calibration-only uniform spherical keys, independent of later evaluation keys. Choose one lambda using only calibration contexts so that mean KL is about .005 and p99 KL <= .05 across predefined pooled key/context calibration measurements. Save lambda; do not rescale per final key or filter final keys. After lambda is frozen, draw 64 fresh validation keys with their own seed.

**Step F — Exact spherical-null unit tests.** On nonzero fixed V vectors drawn from clean-original, paraphrased-clean and synthetic sources, sample many *independent diagnostic random keys* and compare empirical distribution of p-values with uniform using KS/quantile checks. At m=32 compare numeric beta survival to high-precision references and Monte Carlo tail. This validates implementation of the key-randomization null; it does **not** prove fixed-key FPR empirically.

**Step G — Fixed-key pilot (descriptive only).** Freeze 64 keys. For existing clean-original and clean-paraphrased cohorts, construct disjoint chronological or preseeded Q=20, Q=100 groups, avoiding overlapping response IDs within a group and counting only independent groups for confidence intervals. Report per-key p-value histograms and rates at alpha=.01 and 1e-4, plus worst/p90 across keys; flag that the sample is too small to certify 1% fixed-key FPR. Q=1000 may be reported as a tiny-sample illustration only.

**Step H — Local teacher capacity.** On held-out clean prefix contexts (without generating new full sequences), compute `q_k` for frozen lambda and 64 validation keys. Estimate `Delta(k) = E_c E_(w~q_k)[k^T psi(c,w)] - E_c E_(w~p0)[k^T psi(c,w)]` and compare with the finite-difference/first-order prediction. Record mean/p99 KL for every key and any violations without excluding keys. This is an **oracle local-context expectation**, NOT a free-generation TPR or student result.

**Step I — Legacy checkpoint diagnostic (optional, especially useful).** If exact historical KLRD original/SCW-paraphrased responses and key can be loaded, evaluate its matching original 8-plane tangent-score direction and compare with its historical raw LLR on matched prompts. Because its original private rotation signs were Rademacher signs, one can ALSO enumerate the `2^8=256` counterfactual sign assignments over the same frozen eight planes and compute an **exact conditional sign-randomization p-value** for a linear score `s^T V_old(D)`, provided the public/secret plane basis was fixed independently of those signs. Its finest possible p-value is `1/256`, so it cannot support the conservative `1e-4` operating point. This diagnostic is a very cheap direct test of whether the old trained KLRD checkpoint retains key-direction evidence without clean threshold calibration. **Do not** use a spherical p-value for the old discrete sign key, reclassify historical outcomes as a success of the new core, or tune the new basis or lambda based on old checkpoint performance.

### 3.5. Required outputs

```text
experiment1_rs/
  provenance/{input_manifest.json,sha256_manifest.json,resolved_config.json}
  geometry/{basis.npz,fisher.npz,whitening.npz,geometry_checks.json}
  keys/{calibration_key_ids.json,validation_key_ids.json}
  null/{sphere_math_tests.json,clean_original_key_pvalues.csv,
        clean_paraphrase_key_pvalues.csv,fixed_key_pilot.json}
  teacher/{global_lambda.json,local_capacity_by_key.csv,KL_by_key.csv}
  legacy/{compatibility_diagnostic.json} # optional
  report/{experiment1_report.md,summary.json}
```

Do not save secret key material to a public repo; local experiment artifacts should be access-controlled. Report key IDs and hashes instead.

### 3.6. Predeclared gates

**VALID IMPLEMENTATION** only if:

- all analytical/unit tests pass (skew symmetry/orthogonality, numerical derivatives, token indexing, score mean-zero, Beta tail);
- geometry and effective Fisher matrix are finite, reproducible and not degenerate;
- one fixed global lambda is obtained without inspecting evaluation keys or outcomes;
- teacher local signal is positively aligned for a strong majority of the 64 independent validation keys, with KL violations fully reported.

Suggested diagnostic target: >=95% keys with `Delta(k)>0` measured on fresh calibration-heldout contexts; the sign alone is a weak gate, also report absolute magnitude, Monte Carlo CI and ratio to noise level.

**INCONCLUSIVE / NO GO** if mathematical implementation fails, Fisher rank is insufficient, KL constraints are infeasible at meaningful lambda, or local teacher advantage is not consistently positive. Do not patch by selecting favorite keys or tuning on paraphrase results.

**Not yet established by Experiment 1 even if it passes:** free-running teacher ownership detection, sufficient power at alpha=1e-4, practical fixed-key FPR confidence bounds, LoRA implantation, SaTML robustness, or paraphrase robustness. These require later independent experiments.

### 3.7. Next experiments (precommitted ordering)

- **Experiment 2 — Core teacher full-generation:** fresh prompts, ordinary generation under exact `q_k`, evaluate correct/wrong keys with same p-values at Q=100 and 1000; utility/stealth; no proxy.
- **Experiment 3 — Core LoRA:** use one fixed key not selected by TPR, same teacher KL, matched clean/self-KL control, held-out prompts, original FPR/power, direct SaTML official bigram_suppression plus optional stronger context-aware statistical analysis. Assess key leakage from weights/logits as a distinct threat.
- **Experiment 4 — PSS development:** first quantify core paraphrase weakness. Fit output/input transfer matrices on a predeclared proxy and clean development data; select subspace entirely within the frozen core score space. Use different held-out paraphraser for final validation.
- **Experiment 5 — PSS teacher and LoRA:** fresh keys, frozen subspace, same verifier rule with dimension d, original/input/output paraphrase plus SaTML and utility. Include random-d-subspace control.

## 4. Integrity rules

- Do not use final-stage or old Stage-2B selected fingerprint paraphrases to pick E1 geometry, keys or lambda.
- Never use clean responses to fit a detection threshold; calibration data is allowed solely for **public feature geometry/strength**, not to choose owner-key detection threshold.
- No conditioning on chosen key's TPR, fixed-key FPR, or robustness in key selection.
- No changing query budget, alpha, key bank, dimensionality, proxy model or evaluation cohort after outcomes are seen.
- No claims of protection against all white-box statistical fingerprint removal from absence of a bigram green list.
- Clearly separate (a) mathematical null validity across keys, (b) fixed-key deployment reliability, (c) conditional local-teacher signal, and (d) actual generated-text/weight-level detectability.
