# Experiment 3 exact-teacher LoRA implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. No tests, Python execution, inference, installs or APIs on local.

**Goal:** Implement the user-approved revised E3 spec as server-run source, with exact forward-KL LoRA pilot, ordinary merged-model black-box evaluation, opt-in high-fidelity paraphrase, and private-safe packaging.
**Architecture:** New klrd_rs/e3 modules; frozen E1/E2/E2B code/artifacts read-only. Shared state is dict(root:Path,out:Path,cfg:E3Config); get_frozen(state) lazily provides authoritative frozen inputs. No automatic inference/API after training.
**Tech Stack:** PyTorch FP32 master weights, PEFT LoRA, Transformers pinned Qwen, standard vLLM optional inference, unchanged E2 verifier and E2B transport.
**Spec:** gpt_guide/KLRD_RS_Experiment3_LoRA_Codex_Spec_REVISED.md

## Global constraints
- Exact q from RotationReadout.log_probs(clean pre-final-norm residual, lambda*(L@bank[0])); tau=.8, m32, frozen strength .09271215438842773.
- Student unrotated native FP32 norm/head, full vocab incl EOS; response positions from encode_response only; no CE, score reward, Fisher or threshold changes.
- Pilot 2000/200, full 10000/1000, seed2026110301; pilot deterministic prefix of full partitions. Exclude frozen/historical held-out by ID/text/NFKC hash.
- One resident FP32-master LoRA backbone, teacher adapters disabled/no-grad/eval, student adapters enabled/grad/backbone BF16 autocast optional. Only backbone LoRA r16/alpha32/dropout0.
- Microbatch1, accumulation16, LR1e-5, 2epochs, warmup.03, weight_decay0, clipping1, chunk32. Normalize sum KL by valid positions across entire accumulation window.
- State/optimizer/scheduler/RNG/permutation resumable atomic checkpoints; best by validation KL only; automatic best-adapter + clean merged model export after training.
- Primary eval1000 E2 group0 IDs; untrained/clean controls, exact teacher cached vectors; returned-text score, Q100/Q1000 +nested20/50.
- Paraphrase opt-in400 student outputs, run002 frozen high-fidelity prompt/model; reuse teacher0/clean E2B controls; no APIs in pilot/full/evaluate/report.
- Review ZIP excludes weights, adapters, key vectors and raw private prompts/responses.

## Interfaces and tasks
- [x] Integration/provenance: E3Config(profile='pilot', ...), resolved(); initialize_run(state,create=False), get_frozen(state), bind_inputs(state), atomic_directory(destination,writer), file_manifest(directory), append_event(path,row). CLI modes pilot/full/train/merge/evaluate/paraphrase/report/smoke; shell pilot only. README and server-only contracts.
- [x] Data/generation: prepare(state)->{'train':list[record],'validation':list[record]}; load_eval_prompts(state)->list[1000 original rows]; generation.generate(state,rows,output_path,model_path=None,kind='student')->list[records], no key/frozen access, standard HF/vLLM. Training records prompt_id,prompt,response,source,split,response_id plus generated_ids/metadata. Persist train_manifest/val_manifest and corpus provenance, legacy discovery then deterministic completion.
- [x] Exact teacher/training: teacher.py functions capture residual and full-vocab summed KL; train.run(state,dataset) and train.merge(state), server_smoke(state,dataset). Genuine stateful resume and validation-best selection; untrained baseline preserved; FP32 tail and frozen base invariants.
- [x] Evaluation/paraphrase/report: evaluate.run(state), paraphrase.run(state), report.run(state,package=True). Use existing canonical score caches, controls + error reporting, high-fidelity transport wrapper; no repeated E1/E2 audit per batch. Reports explicit TRAINING_ONLY/EVAL_COMPLETE/PARAPHRASE_COMPLETE/INCOMPLETE.
- [x] Static integration review and independent spec/code review; prepare tests but do not execute locally. No commits/push requested for this task.

## Review focus
1. Variable lengths and final incomplete accumulation window: token-weighted exact objective, EOS vocabulary retained but EOS time step not added.
2. Resume after checkpoint publication before log append, partial generation journals, merged artifact integrity; never regenerate committed output IDs.
3. Legacy absent/ineligible/overlapping data: honest counts and deterministic clean completion, no E2 training leakage.
4. Gradient capture through frozen tail/checkpoint recomputation, teacher adapter isolation after updates, one transformer backward per microbatch.
5. API opt-in and accepted-output reuse, no secret artifacts in ordinary student export or public review ZIP.

## Rulings
- The user's revised spec is the approved design; no redundant design approval is requested.
- Preserve existing uncommitted E2B work in the current workspace; create only additive E3 sources and targeted .gitignore entries. No source overwrite, git checkout, merge or push.
- Static checks only under the explicit local-execution prohibition; all GPU/API/tests remain server tasks.

## 2026-11-03: generation throughput update

- New CLI/launcher runs choose vLLM by default; saved runs retain their backend.
  Apply this default in main_exp3.py without editing the historical E3Config or
  training/teacher files, preserving numerical-training resume fingerprints.
- vLLM schedules up to8 missing prompts/call, independently seeded SamplingParams,
  bounded KV/continuous batching, FP32 TRITON_ATTN/IEEE and decode CUDA Graphs.
- Preserve frozen sampler/model/dataset/objective/verifier settings. Do not claim
  identical sampled tokens across backend/batch numerical paths.
- Compatible old batch1 vLLM bindings and accepted rows remain unchanged. Runtime
  execution metadata is separate; engine failures are logged and stop, rather
  than committing fake empty responses or silently falling back to HF.
- Add server-only contracts for batching/remainders/seeds, output binding, failure
  publication and compatible resume. No local Python/tests/models/API execution.

## User override: launcher automatically evaluates after pilot

The user now explicitly requests run_experiment3.sh to train/merge and then run
student evaluation automatically. This supersedes the revised spec's earlier
launcher-only-training instruction. Direct --mode pilot stays training/merge
only; its saved scientific configuration and checkpoint identities are unchanged.
The shell uses set -euo pipefail so a failed training command cannot proceed to
evaluation. No automatic paraphrase/API invocation is added. Source contracts and
README updated; no local tests/inference executed.
