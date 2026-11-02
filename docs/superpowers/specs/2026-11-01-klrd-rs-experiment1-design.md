# KLRD-RS Experiment 1 — Implementation design

Date: 2026-11-01. Status: proposed implementation; not code or experimental results.
Authority: KLRD_RS_FINAL_METHOD_AND_EXPERIMENT_1_SPEC.md. Its frozen research design takes precedence.

Source clarification: the user corrected the initial KSRF directory reference to
KLRD. Actual copied code and all historical experiment inputs already originate
from KLRD. KSRF mentions below record the initial inspection only; they do not
authorize or describe a runtime dependency or an input cohort.

## Intent and scope
Implement Steps A–H, with optional Step I, as a portable Python pipeline.
Experiment 1 tests mathematical null validity and conditional local teacher capacity.
It does not train, generate teacher evaluation sequences, invoke paraphrase APIs,
implement PSS, certify fixed-key FPR, or claim robustness.

## Approach
The requested frozen design already fixes the scientific approach.
Use PyTorch/Transformers for residual replay and exact readout derivatives.
Reuse audited KLRD I/O, tokenization and batching utilities by copying necessary
functions into this project with source hashes; no runtime import from sibling projects.
A wholesale KSRF fork would carry incompatible semantic features, sampling settings
and cohorts. A vLLM-only implementation cannot provide the required residual derivatives.
vLLM is appropriate only for recovering missing clean generations from an exact historical manifest.

## Deliverables and invocation
- main.py: default full Experiment 1 orchestration; optional explicit audit/test stages.
- klrd_rs/: provenance, data adapter, model replay, geometry, scores, Fisher,
  strength, null diagnostics, local capacity, optional legacy, reporting.
- tests/: numerical, token-boundary, provenance, leakage, resume and pipeline checks.
- inputs/: copied, validated, necessary historical data and public geometry artifacts.
  All private artifacts and run outputs excluded from version control.
- requirements.txt and README.md document dependencies; no installation in launcher.
- Exactly one shell file, run_experiment1.sh, containing only:
  python main.py
No shebang, activation, config file requirement, machine Python path, or shell setup.
Defaults are Python constants. Paths resolve relative to the project, not Windows drive names.
Local Python verification uses D:/anaconda3/envs/topmost/python.exe only outside the launcher.
Preserve historical files read-only; copy data required to run on the server into project inputs.
Do not bundle model weights or secrets into a public deliverable.

## Observed source inventory
KLRD_RS currently contains only the supplied specification.
KSRF has data/batched replay/I/O utilities, but its default clean sampling uses 256 tokens,
different seeds and newly selected cohorts; those defaults cannot be reused for E1.
Candidate historical KLRD run: outputs/klrd_vllm.
Observed filenames include private/manifest.json, selected_prompts.json,
active_basis.pt, calibration_activations.pt, clean_train.jsonl,
clean_calibration.jsonl, clean_validation.jsonl, clean_final.jsonl,
and paraphrase_scw_table2_v1/paraphrases.jsonl.
These are candidate artifacts, not yet certified compatible by filename alone.
Model revision, source prompt IDs, split membership, basis provenance and sampling
identity must be validated before accepting them.
Do not expose contents of .env or historical secret keys in logs.

## A — Artifacts and provenance
Recover exact train_gen/test_gen selection from historical selected_prompts and manifests.
Validate prompt IDs/text and matching clean/SCW pairs; fail on duplicates, ambiguity,
revision mismatch, incompatible tokenization, or calibration/evaluation overlap.
Missing diagnostic populations get explicit MISSING status, never replacement cohorts.
If exact calibration responses/activations are absent, recover only from the exact
manifest and pinned clean model; without that manifest report BLOCKED_TECHNICAL.
Hash every consumed input, resolved defaults, copied source and stage dependency.
Save outputs under experiment1_rs/ using the exact required subdirectories.
Atomic stage completion and dependency hashes allow safe reruns after interruption.
A changed input/configuration never silently reuses a completed incompatible stage.

## B–C — Geometry and score
Pinned model/tokenizer:
Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1.
Temperature 0.8. Public PCA128 comes solely from clean calibration activations.
Validate the historical constrained basis and mean; generate 32 disjoint orthonormal
planes using public seed 2026102801, deterministic QR orientation and generator hashes.
Do not allocate full hidden_dim by hidden_dim generators: represent each skew
generator as q*p^T - p*q^T and apply exact paired sin/cos rotations.
Capture input of the final normalization using a pre-hook, not post-norm hidden states.
Compute tangent scores through the unchanged normalization/readout.
Validate all 32 derivatives against central differences on FP32/FP64 samples, including
full-vocabulary probability-weighted zero mean and exact rotation norm preservation.
Exclude special/EOS/pad/hidden response positions with a fixed key-independent rule.
Replay text with the pinned chat template; residual at each prediction position must
predict the following response token, including the first response token.
Record unrecoverable text/token boundaries rather than guessing.

## D — Fisher and partitioning
Split clean calibration at prompt level into disjoint Fisher fit, covariance validation,
lambda calibration and local-capacity subsets, using a frozen public manifest.
The builder chooses fixed partition sizes before inspecting outcomes and saves them.
Use up to 100000 ordinarily sampled clean next-token contexts for Fisher estimation.
Sample W from the complete clean p0 at each selected prefix, without top-p truncation.
F_hat is the empirical uncentered score second moment.
Fix eps = 1e-3 * trace(F_hat)/32, and L = (F_hat + eps*I)^(-1/2).
Report raw/effective eigenvalues, ridge, numerical rank, condition and independent
whitened covariance. Reject nonfinite or rank-deficient geometry; never change m.
Predeclare numerical rank/isotropy tolerances in resolved config before computation.

## E — Strength and keys
Draw 32 normalized Gaussian calibration keys with seed 2026102802.
Search a single lambda on pooled predefined calibration key/context KL(q||p0).
Record the search trace, achieved mean and p99; target mean 0.005 nats and p99 <= 0.05.
Declare target-approach tolerance before observing results; report infeasibility instead
of calling a near-zero distortion teacher viable.
Freeze basis, Fisher and lambda before drawing the 64 validation keys with seed
2026102803. No key rejection, individual rescaling or downstream retuning.
Public outputs contain key IDs/hashes only. Deterministic seed specifications are already
public research parameters; do not present seeded evaluation keys as deployment secrets.
Any persisted key vectors belong only to a private excluded output area.

## F–G — Null and fixed-key pilot
Compute V as a sum of whitened response scores and T = dot(k,V)/norm(V).
For zero V return p=1; otherwise use stable incomplete-beta survival with a=b=15.5.
Reject NaNs and invalid dimension/key norms; clamp roundoff at cosine boundaries only.
Test high-precision references, analytic tails and independent diagnostic Gaussian keys.
Use fixed V from synthetic and available clean-original/clean-paraphrase populations.
Persist Monte Carlo seeds, sample counts, KS/quantile tests and uncertainty.
Use chronological disjoint complete response groups Q=20,100,1000; no overlapping
response IDs within a population/budget. Report remainders and number of groups.
Same groups may be evaluated under all keys, but never counted as independent replicas
across keys. Report per-key rates, histograms, binomial intervals and worst/p90 summaries.
Q=1000 is tiny-sample illustration. No fitted clean threshold or fixed-key certification.

## H — Local teacher
Use held-out clean prefixes and exact q_k from finite rotation for every validation key.
Compute full-vocabulary Delta using the same frozen whitened score.
Compare with lambda * k^T L F_holdout L k and directional finite differences.
Report each key's mean/p99 KL, Delta, absolute size and signal/noise.
Estimate uncertainty by prompt-cluster resampling, not by treating dependent prefix
tokens as independent samples. Save sample counts and bootstrap seeds/replicates.
Report all keys and every distortion violation without filtering.

## I — Optional legacy
Load only matching historical 8-plane basis/sign direction and response/LLR pairs.
Compare tangent direction with matched historical raw LLR.
Enumerate all 256 sign assignments, include ties in the one-sided p-value,
only when plane/sign independence is supported by provenance.
Never apply spherical p-values to discrete legacy signs.
If provenance/key/matched data is unavailable, record an explicit skipped reason.
Do not copy a full legacy model if saved paired evidence suffices.

## Gates and reporting
Write all required files in spec section 3.5 and a per-stage status ledger.
VALID IMPLEMENTATION requires every mathematical gate and >=95% of 64 validation
keys with positive held-out local Delta, with absolute signal/uncertainty disclosed.
Use INCONCLUSIVE/NO_GO for numerical/geometry/strength/local-signal failures,
and BLOCKED_TECHNICAL for absent or unverifiable required inputs.
A partial diagnostic cohort never becomes a complete original+paraphrase result.
A synthetic smoke run is labeled synthetic and cannot establish research success.
Keep null validity, fixed-key reliability, local expectations and generated/weight-level
detectability separate in every report.

## Verification
Test first: exact rotation vs matrix exponential; analytic derivatives vs autodiff and
central differences; score mean-zero over full vocabulary; Fisher whitening/rank;
beta high-precision values/tails/zero vector; discrete legacy enumeration.
Integration tests: toy frozen model exercising A–H, first-token and special-token
indexing, duplicate/unmatched/cohort leakage refusal, immutable source artifacts,
resume invalidation, optional missing legacy, missing paraphrase status and launcher.
Run the entire CPU test suite and compile check using local topmost.
Inspect pinned-model behavior only if the required model/GPU is available; otherwise
state clearly that full Qwen/GPU execution remains for the server.

