"""Frozen-receipt-only Llama E2 cohort, exact teacher, and text verifier."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..data import encode_response, prompt_ids
from ..e2.cohort import make_groups, text_hash, generation_seed
from ..e2.exclusions import legacy_text_hash, source_aliases
from ..io import digest, read_json, read_jsonl, save_npz, sha256, write_json
from ..math import sphere_p
from .frozen import FrozenE1, load_frozen_e1
from .hf_backbone import load_backbone
from .profile import ModelProfile
from .auth import get_hf_token
from .vllm_adapter import engine_options, register_profile_architecture, runtime_identity


def encode_visible_response(tokenizer: Any, prompt: str, response: str) -> dict[str, list[int]]:
    """Canonical text-only causal encoding; special/EOS targets are excluded."""
    return encode_response(tokenizer, prompt, response)


def wrong_key_indices(profile: ModelProfile, owner_index: int, key_count: int | None = None) -> list[int]:
    count = int(key_count if key_count is not None else profile.scientific["final_keys"])
    total = int(profile.scientific["wrong_keys_per_owner"])
    if not 0 <= int(owner_index) < count or total > count - 1:
        raise ValueError("WRONG_KEY_DESIGN_INVALID: owner or wrong-key count exceeds frozen key bank")
    seed = int(profile.seeds["e2_groups"])
    offset = int.from_bytes(hashlib.sha256(f"{profile.profile_id}|{seed}|{owner_index}".encode()).digest()[:8], "big") % (count - 1)
    candidates = [index for index in range(count) if index != int(owner_index)]
    return [candidates[(offset + step) % len(candidates)] for step in range(total)]


def prepare_e2_cohort(
    datasets: Mapping[str, Any],
    tokenizer: Any,
    profile: ModelProfile,
    excluded: Mapping[str, set],
    count: int | None = None,
) -> list[dict]:
    """Select deterministic eligible UltraChat prompts under the Llama exclusion identity."""
    profile.validate()
    wanted = int(count if count is not None else profile.scientific["e2_prompt_count"])
    ids = {str(value) for value in excluded.get("ids", set())}
    exact = set(excluded.get("exact", set()))
    normalized = set(excluded.get("normalized", set()))
    legacy = set(excluded.get("legacy_normalized", set()))
    selected = []
    for split_offset, split in enumerate(profile.data["e2_split_order"]):
        if split not in datasets:
            raise ValueError(f"E2_DATASET_SPLIT_MISSING: {split}")
        rows = datasets[split]
        order = np.random.default_rng(int(profile.seeds["e2_prompts"]) + split_offset).permutation(len(rows))
        for raw_index in order:
            index = int(raw_index)
            source = dict(rows[index])
            prompt = source.get("prompt")
            if not isinstance(prompt, str):
                messages = source.get("messages")
                if isinstance(messages, (list, tuple)):
                    user_messages = [item.get("content") for item in messages if isinstance(item, Mapping) and item.get("role") in ("user", "human")]
                    prompt = user_messages[-1] if user_messages else None
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            source_id = source.get("prompt_id", source.get("id"))
            prompt_sha = text_hash(prompt)
            normalized_sha = text_hash(prompt, normalized=True)
            if source_id is None:
                prompt_id = f"{split}:{index:08d}:{prompt_sha[:12]}"
            else:
                prompt_id = str(source_id)
            aliases = source_aliases(prompt_id)
            if (aliases & ids or prompt_sha in exact or normalized_sha in normalized
                    or legacy_text_hash(prompt) in legacy):
                continue
            token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
            if not 16 <= token_count <= 512:
                continue
            selected.append({
                "prompt_id": prompt_id,
                "prompt": prompt,
                "source_split": split,
                "source_index": index,
                "prompt_text_sha256": prompt_sha,
                "normalized_prompt_sha256": normalized_sha,
                "prompt_length": token_count,
                "source_row_sha256": digest(source),
            })
            ids.update(aliases)
            exact.add(prompt_sha)
            normalized.add(normalized_sha)
            legacy.add(legacy_text_hash(prompt))
            if len(selected) == wanted:
                return selected
    raise ValueError(f"INSUFFICIENT_ELIGIBLE_FRESH_PROMPTS: {len(selected)} eligible, need {wanted}")


def build_e2_groups(rows: list[dict], seed: int) -> list[dict]:
    """Reuse Qwen-independent grouping rules while guaranteeing disjoint IDs per Q."""
    if not rows or len({row.get("prompt_id") for row in rows}) != len(rows):
        raise ValueError("E2_GROUP_INPUT_IDS_INVALID")
    if len(rows) % 1000 == 0:
        groups = make_groups(rows, int(seed))
    elif len(rows) % 100 == 0:
        ids = [row["prompt_id"] for row in rows]
        order = np.random.default_rng(int(seed)).permutation(len(rows))
        groups = []
        for group_index, start in enumerate(range(0, len(rows), 100)):
            selected = [ids[int(index)] for index in order[start:start + 100]]
            groups.append({
                "group_id": f"smoke_g{group_index:03}_Q100",
                "group_index": group_index,
                "Q": 100,
                "prompt_ids": selected,
                "parent_group_id": None,
                "sha256": digest(selected),
            })
    else:
        raise ValueError("E2 primary cohort must form complete disjoint Q=100 or Q=1000 groups")
    for budget in (100, 1000):
        seen = set()
        for group in groups:
            if group["Q"] != budget:
                continue
            current = set(group["prompt_ids"])
            if seen & current:
                raise ValueError(f"E2_GROUP_OVERLAP: Q={budget}")
            seen.update(current)
    return groups


class LlamaTextVerifier:
    """Primary score from visible returned text using the clean HF Llama reference."""

    def __init__(self, backbone: Any, frozen: FrozenE1, tau: float = 0.8, chunk_size: int = 64):
        self.backbone = backbone
        self.frozen = frozen
        self.readout = backbone.readout(frozen.P, frozen.Q, tau)
        self.chunk_size = max(1, int(chunk_size))

    @torch.inference_mode()
    def score_rows(self, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros((len(rows), self.frozen.whitener.shape[0]), dtype=np.float64)
        counts = np.zeros(len(rows), dtype=np.int64)
        for start in range(0, len(rows), 4):
            batch = rows[start:start + 4]
            encodings = [encode_visible_response(self.backbone.tokenizer, row["prompt"], row["response"]) for row in batch]
            for encoded in encodings:
                if len(encoded["input_ids"]) > self.backbone.max_position_embeddings:
                    raise ValueError("E2_CONTEXT_TOO_LONG: no silent prompt/response truncation")
            residuals = self.backbone.capture_pre_norm(encodings)
            for local, (encoded, values) in enumerate(zip(encodings, residuals)):
                counts[start + local] = len(encoded["targets"])
                raw = np.zeros(self.frozen.whitener.shape[0], dtype=np.float64)
                for offset in range(0, len(encoded["targets"]), self.chunk_size):
                    x = torch.as_tensor(values[offset:offset + self.chunk_size], device=self.backbone.device, dtype=torch.float32)
                    target = torch.as_tensor(encoded["targets"][offset:offset + self.chunk_size], device=self.backbone.device, dtype=torch.long)
                    raw += self.readout.observed(x, target).double().cpu().numpy().sum(0)
                scores[start + local] = self.frozen.whitener @ raw
        return scores, counts

    @staticmethod
    def score_keys(vector: np.ndarray, bank: np.ndarray) -> dict[str, Any]:
        vector = np.asarray(vector, dtype=np.float64)
        bank = np.asarray(bank, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        cosines = bank @ vector / norm if norm else np.zeros(len(bank), dtype=np.float64)
        return {"cosines": cosines, "p_values": np.asarray(sphere_p(vector, bank)), "norm": norm, "zero_vector": norm == 0}


def _history_exclusions(root: Path, e1_path: Path) -> tuple[dict[str, set], dict]:
    history_path = root / "e2_inputs_v2" / "exclusion_index.json"
    if not history_path.is_file():
        raise ValueError(f"MISSING_PRIOR_EXPOSURE_INDEX: required read-only history index {history_path}")
    history = read_json(history_path)
    for field in ("prompt_ids", "exact_text_hashes", "normalized_text_hashes"):
        if not isinstance(history.get(field), list):
            raise ValueError(f"PRIOR_EXPOSURE_INDEX_INVALID: missing {field}")
    excluded = {
        "ids": {str(value) for value in history.get("prompt_ids", [])},
        "exact": set(history.get("exact_text_hashes", [])),
        "normalized": set(history.get("normalized_text_hashes", [])),
        "legacy_normalized": set(history.get("legacy_normalized_text_hashes", [])),
    }
    for manifest_name in ("clean_calibration_prompts.json", "clean_diagnostic_prompts.json"):
        path = e1_path / manifest_name
        if not path.is_file():
            continue
        for row in read_json(path).get("prompts", []):
            excluded["ids"].update(source_aliases(row["prompt_id"]))
            excluded["exact"].add(row["prompt_text_sha256"])
            excluded["normalized"].add(row["normalized_prompt_sha256"])
            excluded["legacy_normalized"].add(legacy_text_hash(row["prompt"]))
    audit = {
        "schema": "klrd_rs.portable.e2_exclusion_audit.v1",
        "historical_index": str(history_path.relative_to(root)),
        "historical_index_sha256": sha256(history_path),
        "historical_ids": len(history.get("prompt_ids", [])),
        "combined_excluded_ids": len(excluded["ids"]),
        "combined_exact_hashes": len(excluded["exact"]),
        "combined_normalized_hashes": len(excluded["normalized"]),
        "policy": "read-only prior exposure index plus fresh Llama E1 calibration and clean-original diagnostic prompts",
    }
    return excluded, audit


def _dataset_rows(profile: ModelProfile) -> dict[str, Any]:
    from datasets import load_dataset

    return {
        split: load_dataset(profile.data["dataset"], revision=profile.data["dataset_revision"], split=split)
        for split in profile.data["e2_split_order"]
    }


def _atomic_batch(path: Path, identity: dict, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = read_json(path)
        if saved.get("identity") != identity or digest(saved.get("records")) != saved.get("records_sha256"):
            raise ValueError(f"E2_RESUME_BATCH_MISMATCH: {path}")
        return
    write_json(path, {"identity": identity, "records": records, "records_sha256": digest(records)})


def _load_or_select_cohort(profile: ModelProfile, e1_path: Path, frozen: FrozenE1, output: Path, tokenizer: Any, count: int, root: Path) -> tuple[list[dict], list[dict], dict]:
    private_path = output / "private" / "cohort.jsonl"
    manifest_path = output / "provenance" / "cohort_manifest.json"
    excluded, exclusion_audit = _history_exclusions(root, e1_path)
    if private_path.exists() or manifest_path.exists():
        if not private_path.is_file() or not manifest_path.is_file():
            raise ValueError("E2_COHORT_PARTIAL_COMMIT: use a new output directory")
        manifest = read_json(manifest_path)
        rows = read_jsonl(private_path)
        if sha256(private_path) != manifest.get("private_cohort_sha256"):
            raise ValueError("E2_COHORT_CHECKSUM_MISMATCH")
        if manifest.get("count") != count or manifest.get("dataset_revision") != profile.data["dataset_revision"]:
            raise ValueError("E2_COHORT_IDENTITY_MISMATCH")
        if manifest.get("profile_id") != profile.profile_id or manifest.get("frozen_e1_sha256") != digest(frozen.receipt):
            raise ValueError("E2_COHORT_FROZEN_E1_MISMATCH")
        if manifest.get("exclusion_audit") != exclusion_audit:
            raise ValueError("E2_COHORT_EXCLUSION_HISTORY_DRIFT")
        if len(rows) != count or len({row.get("prompt_id") for row in rows}) != count:
            raise ValueError("E2_COHORT_ROW_COUNT_OR_IDENTITY_MISMATCH")
        groups = manifest["groups"]
        return rows, groups, {"manifest": manifest, "exclusion_audit": exclusion_audit}
    datasets = _dataset_rows(profile)
    rows = prepare_e2_cohort(datasets, tokenizer, profile, excluded=excluded, count=count)
    groups = build_e2_groups(rows, int(profile.seeds["e2_groups"])) if count % 100 == 0 else []
    manifest = {
        "schema": "klrd_rs.portable.e2_cohort.v1",
        "profile_id": profile.profile_id,
        "frozen_e1_sha256": digest(frozen.receipt),
        "dataset": profile.data["dataset"],
        "dataset_revision": profile.data["dataset_revision"],
        "count": count,
        "seed": int(profile.seeds["e2_prompts"]),
        "group_seed": int(profile.seeds["e2_groups"]),
        "source_split_counts": {split: sum(row["source_split"] == split for row in rows) for split in profile.data["e2_split_order"]},
        "groups": groups,
        "exclusion_audit": exclusion_audit,
        "private_cohort_sha256": None,
    }
    private_path.parent.mkdir(parents=True, exist_ok=True)
    temp = private_path.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, private_path)
    manifest["private_cohort_sha256"] = sha256(private_path)
    write_json(manifest_path, manifest)
    write_json(output / "provenance" / "exclusion_audit.json", exclusion_audit)
    return rows, groups, {"manifest": manifest, "exclusion_audit": exclusion_audit}


def _native_eos_ids(identity: dict) -> list[int]:
    value = identity.get("generation_config", {}).get("eos_token_id")
    if isinstance(value, int):
        result = [value]
    elif isinstance(value, list) and value:
        result = [int(item) for item in value]
    else:
        raise ValueError("NATIVE_EOS_UNRESOLVED: frozen model identity has no EOS IDs")
    return sorted(set(result))


def _engine_tokenizer_check(engine_tokenizer: Any, identity: dict) -> None:
    vocab = engine_tokenizer.get_vocab()
    canonical = json.dumps({str(token): int(index) for token, index in vocab.items()}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest() != identity["vocab_sha256"]:
        raise ValueError("VLLM_TOKENIZER_VOCAB_MISMATCH")
    template = getattr(engine_tokenizer, "chat_template", None)
    if not isinstance(template, str) or hashlib.sha256(template.encode()).hexdigest() != identity["chat_template_sha256"]:
        raise ValueError("VLLM_TOKENIZER_CHAT_TEMPLATE_MISMATCH")


def _configure_engine_model(engine: Any, frozen: FrozenE1, angles: np.ndarray) -> None:
    import functools

    def configure(model, P, Q, angle):
        if not hasattr(model, "configure_rotation"):
            raise RuntimeError("PORTABLE_LLAMA_ARCHITECTURE_NOT_LOADED")
        model.configure_rotation(P, Q, angle)
        return {"rotation_configured": True}

    results = engine.apply_model(functools.partial(configure, P=frozen.P, Q=frozen.Q, angle=angles))
    if not results or not all(item.get("rotation_configured") for item in results):
        raise RuntimeError("PORTABLE_LLAMA_ROTATION_CONFIGURATION_FAILED")


def _run_generation_condition(engine: Any, profile: ModelProfile, frozen: FrozenE1, rows: list[dict], output: Path,
                              owner: str, key_index: int | None, phase: str, settings: dict) -> list[dict]:
    from vllm import SamplingParams

    if key_index is None:
        angles = np.zeros(int(profile.scientific["m"]), dtype=np.float32)
    else:
        angles = float(frozen.lambda_value) * (frozen.whitener @ frozen.key_bank[key_index])
    _configure_engine_model(engine, frozen, angles)
    condition_dir = output / "private" / "generation" / phase / owner
    result = []
    eos_ids = _native_eos_ids(frozen.model_identity)
    batch_size = int(settings.get("batch_size", 8))
    for batch_index, start in enumerate(range(0, len(rows), batch_size)):
        batch = rows[start:start + batch_size]
        request_ids = [row["prompt_id"] for row in batch]
        identity = {
            "profile_id": profile.profile_id,
            "frozen_e1_sha256": digest(frozen.receipt),
            "settings_sha256": digest(settings),
            "owner": owner,
            "owner_key_index": key_index,
            "phase": phase,
            "prompt_ids": request_ids,
            "max_new_tokens": int(profile.scientific["sampling_max_new_tokens"]),
            "temperature": float(profile.scientific["tau"]),
            "top_p": float(profile.scientific["sampling_top_p"]),
            "top_k": -1,
        }
        batch_path = condition_dir / f"batch_{batch_index:05d}.json"
        if batch_path.exists():
            saved = read_json(batch_path)
            if saved.get("identity") != identity or saved.get("records_sha256") != digest(saved.get("records")):
                raise ValueError(f"E2_GENERATION_RESUME_MISMATCH: {batch_path}")
            result.extend(saved["records"])
            continue
        prompts = []
        params = []
        seeds = []
        for row in batch:
            token_ids = prompt_ids(engine.get_tokenizer(), row["prompt"])
            if len(token_ids) + int(profile.scientific["sampling_max_new_tokens"]) > int(frozen.model_identity["shapes"]["max_position_embeddings"]):
                raise ValueError("E2_PROMPT_PLUS_RESPONSE_EXCEEDS_CONTEXT: truncation is forbidden")
            seed = generation_seed(int(profile.seeds["e2_generation"]), owner, row["prompt_id"], phase)
            prompts.append({"prompt_token_ids": token_ids})
            seeds.append(seed)
            params.append(SamplingParams(
                n=1,
                temperature=float(profile.scientific["tau"]),
                top_p=float(profile.scientific["sampling_top_p"]),
                top_k=-1,
                min_p=0.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                seed=seed,
                max_tokens=int(profile.scientific["sampling_max_new_tokens"]),
                min_tokens=0,
                stop_token_ids=eos_ids,
                ignore_eos=False,
                include_stop_str_in_output=True,
                detokenize=False,
            ))
        generated = engine.generate(prompts, params, use_tqdm=True)
        if len(generated) != len(batch):
            raise RuntimeError("E2_GENERATION_BATCH_INCOMPLETE")
        batch_records = []
        tokenizer = engine.get_tokenizer()
        for row, request, seed, item in zip(batch, prompts, seeds, generated):
            if list(item.prompt_token_ids) != request["prompt_token_ids"] or len(item.outputs) != 1:
                raise RuntimeError("E2_VLLM_PROMPT_OR_OUTPUT_ORDER_MISMATCH")
            completion = item.outputs[0]
            token_ids = [int(token) for token in completion.token_ids]
            if item.stop_reason in eos_ids and (not token_ids or token_ids[-1] != item.stop_reason):
                raise RuntimeError("VLLM_NATIVE_STOP_TOKEN_MISSING: refusing to fabricate sampled EOS IDs")
            text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            visible = [token for token in token_ids if token not in set(tokenizer.all_special_ids)]
            roundtrip = tokenizer.encode(text, add_special_tokens=False)
            record = {
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "source_split": row["source_split"],
                "response": text,
                "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "generated_token_ids": token_ids,
                "generation_seed": int(seed),
                "owner": owner,
                "owner_key_index": key_index,
                "phase": phase,
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
                "eos_seen": any(token in eos_ids for token in token_ids),
                "max_length_truncated": len(token_ids) == int(profile.scientific["sampling_max_new_tokens"])
                                        and bool(token_ids) and token_ids[-1] not in eos_ids,
                "roundtrip_exact": visible == list(roundtrip),
                "sampling_temperature_applications": 1,
                "generation_backend": "vllm",
            }
            record["record_sha256"] = digest(record)
            batch_records.append(record)
        _atomic_batch(batch_path, identity, batch_records)
        result.extend(batch_records)
    return result


def _load_generation_batches(output: Path, phase: str, owner: str) -> list[dict]:
    directory = output / "private" / "generation" / phase / owner
    rows = []
    if not directory.is_dir():
        return rows
    for path in sorted(directory.glob("batch_*.json")):
        saved = read_json(path)
        records = saved.get("records")
        if not isinstance(records, list) or digest(records) != saved.get("records_sha256"):
            raise ValueError(f"E2_GENERATION_BATCH_CHECKSUM_MISMATCH: {path}")
        rows.extend(records)
    return rows


def _score_and_commit(backbone: Any, frozen: FrozenE1, records: list[dict], output: Path,
                      phase: str, owner: str) -> tuple[np.ndarray, np.ndarray]:
    verifier = LlamaTextVerifier(backbone, frozen)
    vectors, counts = [], []
    score_dir = output / "private" / "scores" / phase / owner
    for batch_index, start in enumerate(range(0, len(records), 32)):
        batch = records[start:start + 32]
        identity = {"phase": phase, "owner": owner, "records_sha256": digest(batch),
                    "model_revision": frozen.model_identity["model_revision"],
                    "canonical_path": "visible returned text; native generated IDs are not verifier targets",
                    "scoring_code_sha256": sha256(Path(__file__))}
        path = score_dir / f"batch_{batch_index:05d}.npz"
        marker = path.with_suffix(".json")
        if path.exists() or marker.exists():
            if not path.is_file() or not marker.is_file():
                raise ValueError("E2_SCORE_PARTIAL_COMMIT")
            receipt = read_json(marker)
            if sha256(path) != receipt.get("sha256") or receipt.get("identity") != identity:
                raise ValueError("E2_SCORE_RESUME_MISMATCH")
            with np.load(path, allow_pickle=False) as saved:
                values, tokens = saved["vectors"], saved["visible_tokens"]
        else:
            values, tokens = verifier.score_rows(batch)
            save_npz(path, vectors=values, visible_tokens=tokens,
                     prompt_ids=np.asarray([row["prompt_id"] for row in batch], dtype=str))
            write_json(marker, {"identity": identity, "sha256": sha256(path)})
        vectors.extend(values)
        counts.extend(tokens.tolist())
    return np.asarray(vectors, dtype=np.float64), np.asarray(counts, dtype=np.int64)


def _aggregate_groups(rows: list[dict], vectors: np.ndarray, groups: list[dict], frozen: FrozenE1,
                      profile: ModelProfile, condition: str) -> dict:
    by_id = {row["prompt_id"]: vectors[index] for index, row in enumerate(rows)}
    owner_index = None if condition == "clean" else int(condition)
    wrong_indices = [] if owner_index is None else wrong_key_indices(profile, owner_index)
    summary = {"condition": condition, "budgets": {}}
    group_outputs = []
    for budget in profile.scientific["primary_query_budgets"]:
        evaluated = []
        for group in groups:
            if group["Q"] != budget:
                continue
            vector = np.sum([by_id[prompt_id] for prompt_id in group["prompt_ids"]], axis=0)
            result = LlamaTextVerifier.score_keys(vector, frozen.key_bank)
            for key_index, (p_value, cosine) in enumerate(zip(result["p_values"], result["cosines"])):
                evaluated.append({
                    "group_id": group["group_id"], "group_index": group["group_index"],
                    "query_budget": budget, "key_index": key_index,
                    "vector_norm": result["norm"], "cosine": float(cosine), "p_value": float(p_value),
                })
        group_outputs.extend(evaluated)
        rejection_counts = {
            str(alpha): [sum(row["key_index"] == key and row["p_value"] <= alpha for row in evaluated)
                         for key in range(len(frozen.key_bank))]
            for alpha in profile.scientific["alphas"]
        }
        group_count = len({row["group_id"] for row in evaluated})
        summary["budgets"][str(budget)] = {
            "disjoint_groups": group_count,
            "evaluations": len(evaluated),
            "per_key_rejection_counts_at_alpha": rejection_counts,
            "per_key_rejection_rates_at_alpha": {
                alpha: [count / group_count if group_count else None for count in counts]
                for alpha, counts in rejection_counts.items()
            },
            "owner_key_index": owner_index,
            "wrong_key_indices": wrong_indices,
            "clean_condition": condition == "clean",
        }
    summary["group_rows"] = group_outputs
    return summary


def run_e2(profile: ModelProfile, e1_path: str | Path, output: str | Path, mode: str) -> dict:
    """Run explicit smoke/full E2 or read a completed report after verifying frozen E1."""
    profile.validate()
    if mode not in ("smoke", "run", "report"):
        raise ValueError("E2 mode must be smoke, run, or report")
    frozen = load_frozen_e1(e1_path, profile)
    root = Path(__file__).resolve().parents[2]
    output_path = profile.validate_run_path(root, "e2", Path(output))
    if Path(e1_path).resolve() == output_path.resolve():
        raise ValueError("E2_OUTPUT_MUST_BE_ISOLATED_FROM_E1")
    if mode == "report":
        summary_path = output_path / "full" / "report" / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"MISSING_E2_REPORT: {summary_path}")
        summary = read_json(summary_path)
        if summary.get("frozen_e1_sha256") != digest(frozen.receipt):
            raise ValueError("E2_REPORT_FROZEN_E1_IDENTITY_MISMATCH")
        return summary

    phase = "smoke" if mode == "smoke" else "primary"
    phase_root = output_path / "smoke" if mode == "smoke" else output_path / "full"
    settings_path = phase_root / "provenance" / "identity.json"
    settings = {
        "profile_id": profile.profile_id,
        "frozen_e1_sha256": digest(frozen.receipt),
        "model_revision": frozen.model_identity["model_revision"],
        "dataset_revision": profile.data["dataset_revision"],
        "mode": mode,
        "prompt_count": int(profile.scientific["e2_prompt_count"]),
        "owner_indices": [0] if mode == "smoke" else list(profile.scientific["e2_owner_indices"]),
        "max_new_tokens": int(profile.scientific["sampling_max_new_tokens"]),
        "temperature": float(profile.scientific["tau"]),
        "top_p": float(profile.scientific["sampling_top_p"]),
        "top_k": -1,
        "generation_seed_base": int(profile.seeds["e2_generation"]),
        "cohort_seed": int(profile.seeds["e2_prompts"]),
        "group_seed": int(profile.seeds["e2_groups"]),
        "source_sha256": {
            "portable_e2": sha256(Path(__file__)),
            "vllm_adapter": sha256(Path(__file__).with_name("vllm_adapter.py")),
            "vllm_llama": sha256(Path(__file__).with_name("vllm_llama.py")),
            "profile_yaml": sha256(root / "configs" / "model_profiles" / f"{profile.profile_id}.yaml"),
        },
        "verifier": "clean HF Llama on re-encoded visible returned text",
    }
    runtime = runtime_identity(profile)
    pinned_profile = profile
    if profile.revision != frozen.model_identity["model_revision"]:
        from dataclasses import replace
        pinned_profile = replace(profile, revision=frozen.model_identity["model_revision"])
    get_hf_token(root)
    backbone = load_backbone(pinned_profile, device="cuda")
    e1_root = frozen.path
    rows, groups, cohort_info = _load_or_select_cohort(
        profile, e1_root, frozen, phase_root, backbone.tokenizer,
        int(profile.scientific["e2_prompt_count"]), root,
    )
    count = min(8, len(rows)) if mode == "smoke" else len(rows)
    run_rows = rows[:count]
    if mode == "smoke":
        groups = []
    prompt_lengths = [len(prompt_ids(backbone.tokenizer, row["prompt"])) for row in run_rows]
    max_model_len = max(prompt_lengths) + int(profile.scientific["sampling_max_new_tokens"])
    context_limit = int(frozen.model_identity["shapes"]["max_position_embeddings"])
    if max_model_len > context_limit:
        raise ValueError("E2_PROMPT_PLUS_RESPONSE_EXCEEDS_CONTEXT: truncation is forbidden")
    settings["max_model_len"] = max_model_len
    if settings_path.exists() and read_json(settings_path) != settings:
        raise ValueError("E2_RUN_IDENTITY_DRIFT: use a new output directory")
    write_json(settings_path, settings)
    batch_identity = {**settings, "phase": phase, "actual_prompt_count": count}
    run_identity_path = phase_root / "provenance" / "run_identity.json"
    if run_identity_path.exists() and read_json(run_identity_path) != batch_identity:
        raise ValueError("E2_PHASE_IDENTITY_DRIFT: use a new phase directory")
    write_json(run_identity_path, batch_identity)
    register_profile_architecture(profile)
    try:
        from vllm import LLM
        options = engine_options(profile, {
            "model_revision": frozen.model_identity["model_revision"],
            "max_model_len": max_model_len,
            "generation_seed": int(profile.seeds["e2_generation"]),
        }, eager=(mode == "smoke"))
        engine = LLM(**options)
        engine_tokenizer = engine.get_tokenizer()
        _engine_tokenizer_check(engine_tokenizer, frozen.model_identity)
        runtime["effective_max_model_len"] = max_model_len
        write_json(phase_root / "provenance" / "runtime_identity.json", runtime)
        condition_results = {}
        owners = [0] if mode == "smoke" else list(profile.scientific["e2_owner_indices"])
        for owner in ["clean", *owners]:
            key_index = None if owner == "clean" else int(owner)
            generated = _run_generation_condition(engine, profile, frozen, run_rows, phase_root,
                                                  str(owner), key_index, phase, settings)
            vectors, visible_tokens = _score_and_commit(backbone, frozen, generated, phase_root, phase, str(owner))
            condition_cache = phase_root / "private" / "scores" / phase / f"condition_{owner}.npz"
            condition_cache.parent.mkdir(parents=True, exist_ok=True)
            condition_marker = condition_cache.with_suffix(".json")
            cache_identity = {
                "phase": phase,
                "owner": str(owner),
                "prompt_ids": [row["prompt_id"] for row in run_rows],
                "frozen_e1_sha256": digest(frozen.receipt),
                "settings_sha256": digest(settings),
                "score_source": "visible returned text",
            }
            if condition_cache.exists():
                if not condition_marker.is_file():
                    raise ValueError("E2_CONDITION_SCORE_PARTIAL_COMMIT")
                marker_value = read_json(condition_marker)
                if marker_value.get("sha256") != sha256(condition_cache) or marker_value.get("identity") != cache_identity:
                    raise ValueError("E2_CONDITION_SCORE_CHECKSUM_MISMATCH")
                with np.load(condition_cache, allow_pickle=False) as saved:
                    if saved["identity_sha256"].item() != digest(cache_identity):
                        raise ValueError("E2_CONDITION_SCORE_IDENTITY_MISMATCH")
            else:
                save_npz(condition_cache, vectors=vectors, visible_tokens=visible_tokens,
                         prompt_ids=np.asarray([row["prompt_id"] for row in run_rows], dtype=str),
                         identity_sha256=np.asarray(digest(cache_identity)))
                write_json(condition_marker, {"identity": cache_identity, "sha256": sha256(condition_cache)})
            condition_results[str(owner)] = {
                "vectors": vectors,
                "visible_tokens": visible_tokens,
                "responses": len(generated),
                "response_record_sha256": digest(generated),
                "condition_score_sha256": sha256(condition_cache),
            }
            del generated
        if mode == "smoke":
            summary = {
                "status": "SMOKE_COMPLETED_DIAGNOSTIC_ONLY",
                "profile_id": profile.profile_id,
                "frozen_e1_sha256": digest(frozen.receipt),
                "prompts": count,
                "conditions": sorted(condition_results),
                "primary_cohort_changed": False,
                "full_run_started": False,
                "scientific_claim": "NONE; smoke correctness only",
            }
        else:
            results = {}
            for owner, condition in condition_results.items():
                results[owner] = _aggregate_groups(run_rows, condition["vectors"], groups, frozen, profile, owner)
            report = {
                "status": "COMPLETED_WITHOUT_AUTOMATIC_SUCCESS_CLAIM",
                "profile_id": profile.profile_id,
                "frozen_e1_sha256": digest(frozen.receipt),
                "cohort_count": len(rows),
                "owner_indices": owners,
                "wrong_key_indices": {str(owner): wrong_key_indices(profile, owner) for owner in owners},
                "alpha_levels": list(profile.scientific["alphas"]),
                "query_budgets": list(profile.scientific["primary_query_budgets"]),
                "conditions": {
                    owner: {
                        "responses": data["responses"],
                        "mean_visible_response_tokens": float(np.mean(data["visible_tokens"])) if len(data["visible_tokens"]) else 0.0,
                        "response_record_sha256": data["response_record_sha256"],
                        "condition_score_sha256": data["condition_score_sha256"],
                    }
                    for owner, data in condition_results.items()
                },
                "group_summaries": results,
                "clean_control": "ordinary native Llama generation on same prompts and seeds",
                "primary_verifier": "visible response text re-encoded by clean pinned HF tokenizer; special/EOS targets excluded",
                "server_parity": "NOT_RUN_UNLESS_SEPARATE_ACCEPTANCE_EVIDENCE_EXISTS",
                "scientific_scope": "measured E2 outputs only; fixed-key FPR uncertainty must be stated from independent groups",
            }
            write_json(phase_root / "report" / "summary.json", report)
            summary = report
        write_json(phase_root / "report" / "summary.json", summary)
        return summary
    finally:
        if "engine" in locals():
            del engine
