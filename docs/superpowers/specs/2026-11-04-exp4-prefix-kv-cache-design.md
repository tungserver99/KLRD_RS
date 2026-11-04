# Experiment 4 Prefix KV Cache Design

## Intent

Reduce repeated model computation in Experiment 4 mapping. Every seed-word score
uses the same 128 calibration contexts with only the final word changed. The
current adapter recomputes each full prompt for every seed word.

## Goals

- Preserve the 128 calibration contexts, all valid seed words, model precision,
  token-ID inputs, and official SaTML score/processor code.
- Reuse model KV state for the common token prefix of calibration prompts.
- Keep mapping resumable by its existing per-token atomic shards.
- Show cache preparation and per-word progress in logs.

## Non-goals

- Do not edit `experiment4_satml/vendor/mlprints/bigram_suppression.py`.
- Do not reduce the seed-word or context counts or change model precision.
- Do not claim the optimization is valid unless cached and full-prompt logits
  pass parity checks.

## Approaches

1. Increase batch size only. This keeps inference simple and changes no logits,
   but does not remove repeated context-prefix computation. Batch 128 is already
   the selected low-risk setting.
2. Cache exact shared prefixes in the project-owned Hugging Face inference
   adapter. The unchanged vendored scorer calls this adapter and still computes
   the mapping weights. This is the recommended approach because it targets the
   repeated computation without reducing contexts or changing precision.
3. Reduce contexts/words or use lower precision. This may run faster, but changes
   the experiment inputs or numerical results and is out of scope.

## Design

For each calibration context, tokenize the exact full prompts formed with every
distinct valid seed word. Find the token prefix common to all those prompts and
use the minimum common-prefix length across contexts as a shared prefill length.
Precompute the prefix KV cache once per model. During mapping, the project-owned
`run_inference` adapter feeds each prompt's remaining token IDs with the matching
cached state, then returns next-token logits in the same shape/order expected by
the vendored `BigramSuppressionAttackModel`.

The original upstream score class and logits processor remain byte-identical.
The cache adapter falls back to full-prompt inference if a prompt's tokenized
prefix does not match its cached prefix. Cache tensors are held on the model
device only for the mapping stage and are not written into checkpoints.

## Validation

- Unit-test cached versus full-prompt logits with a small local causal model,
  including variable suffix lengths and left/right masking as used by the
  adapter.
- Verify exact token-prefix matching and fallback behavior.
- On the GPU run, perform a small parity gate over representative contexts and
  seed words for both loaded models before computing mapping shards; fail closed
  on tolerance violations.
- Retain the existing model/tokenizer vocabulary checks and all 128-context
  scoring diagnostics.

## Implementation scope

- Modify `experiment4_satml/core.py` for cache prefill and cached next-token
  inference helpers.
- Modify `experiment4_satml/pipeline.py` to prepare per-model caches once and
  route the unchanged official scorer through the cache-aware adapter.
- Extend `tests/test_exp4_satml.py` with parity and fallback tests.
- Update `experiment4_satml/README.md` to describe cache preparation, memory use,
  and the parity gate.
