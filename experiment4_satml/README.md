# Experiment 4: SaTML statistical bigram suppression

This independent module evaluates the KLRD-RS **Full LoRA Student 10k** from
`experiment3_full/run_001/merged_student`. It does not train, rewrite, or merge
the student. It never writes into Experiment 3. The statistical score and
generation processor use the upstream source copied into
`vendor/mlprints/bigram_suppression.py`. Mapping shards add resumability around
the upstream scorer; no alternate scoring rule or top-k approximation is used.
The vendored file is retained verbatim from [Sentient AGI's MLPrints
repository](https://github.com/sentient-agi/mlprints/blob/main/src/mlprints/attack/bigram_suppression.py);
its AGPL-3.0-or-later license is included alongside it.

The attack is labeled **SaTML statistical bigram suppression with reconstructed
public calibration assets**. Calibration texts are reconstructed by this
project, so this is not an exact-data reproduction.

## Server inputs

The defaults assume the finished E3 full and pilot runs exist at their
configured paths. The full checkpoint must contain the verified merged model
and E3 full must contain its completed 1024-token evaluation artifacts. The
frozen prompts, grouping and clean responses are read from E3's frozen E2 group0
and full run. Set the relative paths in `configs/experiment4.yaml` if the server
layout differs. The public calibration model is `Qwen/Qwen2.5-1.5B`; its weights
are fetched by Transformers on the server if they are not already cached.
The E3 initialization helper reads the E3 pilot run as protocol metadata; the
attacked checkpoint and evaluation population are always from E3 **full**.

Use the server environment already prepared for E3, including its PyTorch,
Transformers and `lm_eval==0.4.13` packages. This project executes the vendored
upstream bigram attack classes directly and supplies a simple batched Hugging
Face forward pass for next-token logits. The adapter avoids importing unrelated
MLPrints modules and does not change the score formulas. The project can be
copied to the server without the separate MLPrints checkout.

The only external data asset downloaded when absent is the official
first20hours/google-10000-english list. Only the first 6,000 source rows are
retained; empty and duplicate entries are removed without backfilling later
rows. Tokenizer filtering can leave fewer than 6,000
valid preceding tokens. No WikiText fallback or alphabetic sorting is used.

## Run

From the project root, activate the server's prepared environment and run:

```bash
bash run_experiment4_satml.sh
```

The shell file contains only `python -m ...` commands. It does not install
packages, activate an environment, or encode a Python executable path. Its
single command runs all stages, stops on error and can be rerun. `--resume` preserves complete calibration
rows, word-level mapping shards and per-prompt responses. Do not change the
output directory or scientific configuration when resuming a partial run.

To inspect paths without downloading assets or loading a model:

```bash
python -m experiment4_satml.cli --dry-run
```

Run individual stages with `--stage prepare|calibration|mapping|generation|
verification|arc|report`. Preparation loads only the already-existing local
E3 student tokenizer. Calibration and mapping require a CUDA GPU and use the
full 128 contexts. Mapping finds the exact common token prefix of the calibration
prompts, prefills each model's KV cache once, and runs a cached-versus-full-logit
parity gate for three seed words across representative contexts before writing
any mapping shard. It then reuses the prefix KV state for each word while the
unchanged vendored SaTML scorer computes the mapping. If no shared prefix exists,
prompt order differs, or token IDs no longer match the scanned prompts,
full-prompt inference is used. Cache data is adapted to the installed
Transformers Cache API. Prefix cache tensors stay on the GPU only during
mapping. The 64-context mapping is a diagnostic
replay of captured forward logits. Mapping is built once and reused at δ=2,4,8,12.
The runtime parity gate uses FP32 `atol=rtol=1e-4`; FP16/BF16 uses
`atol=5e-2, rtol=5e-3` to allow low-precision kernel roundoff while still
aborting when cached logits exceed those bounds.

## Fixed protocol

- Full LoRA Student 10k, E3 checkpoint and E3 frozen Qwen2.5-3B-Instruct prompt
  ordering; independent calibration LM `Qwen/Qwen2.5-1.5B`.
- 6,000 top-frequency seed words, 128 generated continuations, seed 42, 128 new
  tokens, sampling temperature .8 and top-p .95 for calibration data.
- Official SaTML threshold `1e-4`, z/max aggregation, `.999` clipping, signed
  score and special-token exclusion; default mapping batch size 128.
- E3 student sampling settings for response generation: temperature .8, top-p
  1, top-k 0, native EOS, max 1,024 new tokens, same E3 per-prompt student
  seeds. Attacked generation uses Transformers `generate()` so that the
  original Hugging Face SaTML processor is called; E3's configured backend and
  this backend are recorded separately. No HF/vLLM sampled-token parity claim
  is made.
- E3 returned-text verifier and frozen geometry/key bank, Q=20/50/100/1000,
  alpha .01/.0001, owner key 0 and the same 20 wrong keys. Alpha is nominal
  random-key-null significance, not empirical clean FPR. Q=1000 is a single
  descriptive group.
- E3 ARC-Challenge `acc_norm,none`, FP32, batch size 1, 0-shot, task chat
  template and E3 fixed seeds. For attacked conditions, HFLM logits are adjusted
  at every candidate continuation position using the preceding input token.

Tokenizer compatibility requires identical vocabulary sizes and every token
string-to-ID mapping. Special-token metadata differences are recorded for audit;
they do not block the run because calibration prompts and mapping inputs use the
attacked tokenizer for both models. An actual vocabulary mismatch fails loudly.
Calibration and response journals are append-only JSONL with flush/fsync;
mapping is stored as atomic per-token shards and then materialized for the
official processor. An incomplete final JSONL row is preserved as a `.bak`
before resuming; malformed interior rows stop the run. A failure does not
silently reduce calibration N. Mapping metadata records the prefix length and
logit-parity results used by the runtime gate.

## Outputs

`experiment4_satml/outputs/run_001/` contains seed source and filtered words,
the calibration manifest/texts/raw generations, the saved mapping and
diagnostics, per-delta response journals and diagnostics, E3 verifier tables,
ARC metrics, and `reports/experiment4_report.md` plus JSON results. Response
journals include generated token IDs and decoded text. Do not share the `private`
E3 source artifacts or generated response texts without checking project policy.
