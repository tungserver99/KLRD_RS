"""Fresh, identity-bound Llama calibration inputs and public PCA geometry."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import gc
import unicodedata
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..data import prompt_ids
from ..io import digest, read_json, read_jsonl, write_json
from .profile import ModelProfile
from .auth import get_hf_token


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_text(encoded)


def _commit_from(value: Any) -> str | None:
    for key in ("_commit_hash", "commit_hash", "revision"):
        candidate = getattr(value, key, None)
        if isinstance(candidate, str) and len(candidate) == 40 and all(c in "0123456789abcdef" for c in candidate):
            return candidate
    init_kwargs = getattr(value, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        candidate = init_kwargs.get("_commit_hash") or init_kwargs.get("revision")
        if isinstance(candidate, str) and len(candidate) == 40 and all(c in "0123456789abcdef" for c in candidate):
            return candidate
    return None


def _plain_config(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        value = config.to_dict()
    elif isinstance(config, Mapping):
        value = dict(config)
    else:
        value = {key: item for key, item in vars(config).items() if not key.startswith("_")}
    return _json_safe(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def resolve_model_identity(
    profile: ModelProfile,
    tokenizer: Any,
    model_config: Any,
    output: str | Path,
    generation_config: Any | None = None,
) -> dict:
    """Freeze the loaded checkpoint, tokenizer, chat template, and runtime identity."""
    profile.validate()
    model_revision = _commit_from(model_config)
    if model_revision is None:
        model_revision = profile.revision
    if model_revision is None or len(model_revision) != 40:
        raise ValueError("MODEL_REVISION_UNRESOLVED: loaded config must expose an immutable commit SHA")
    if profile.revision is not None and profile.revision != model_revision:
        raise ValueError("MODEL_REVISION_MISMATCH: loaded checkpoint differs from profile revision")
    tokenizer_revision = _commit_from(tokenizer) or model_revision
    if tokenizer_revision != model_revision:
        raise ValueError("TOKENIZER_REVISION_MISMATCH: tokenizer and model must share the pinned revision")
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise ValueError("TOKENIZER_VOCAB_INVALID: tokenizer returned an empty vocabulary")
    vocab_digest = _canonical_sha({str(token): int(index) for token, index in vocab.items()})
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("CHAT_TEMPLATE_MISSING: native tokenizer chat template is required")
    config_dict = _plain_config(model_config)
    model_type = getattr(model_config, "model_type", config_dict.get("model_type"))
    architectures = tuple(getattr(model_config, "architectures", config_dict.get("architectures", ())) or ())
    if model_type != profile.model_type:
        raise ValueError("MODEL_CONFIG_FAMILY_MISMATCH: loaded checkpoint family differs from profile")
    if profile.hf_architecture not in architectures:
        raise ValueError("MODEL_CONFIG_ARCHITECTURE_MISMATCH: loaded architecture differs from profile")
    generation_values = _plain_config(generation_config or getattr(model_config, "generation_config", None))
    if not generation_values:
        generation_values = {
            key: config_dict[key]
            for key in ("bos_token_id", "eos_token_id", "pad_token_id")
            if key in config_dict
        }
    shapes = {
        "hidden_size": int(getattr(model_config, "hidden_size", 0)),
        "vocab_size": int(getattr(model_config, "vocab_size", 0)),
        "max_position_embeddings": int(getattr(model_config, "max_position_embeddings", 0)),
        "num_hidden_layers": int(getattr(model_config, "num_hidden_layers", 0)),
    }
    if min(shapes.values()) < 1:
        raise ValueError("MODEL_CONFIG_SHAPE_MISSING: loaded config dimensions are incomplete")
    identity = {
        "schema": "klrd_rs.portable.model_identity.v1",
        "profile_id": profile.profile_id,
        "model_id": profile.model_id,
        "model_type": model_type,
        "hf_architecture": profile.hf_architecture,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "vocab_sha256": vocab_digest,
        "chat_template_sha256": _sha256_text(template),
        "dataset": profile.data["dataset"],
        "dataset_revision": profile.data["dataset_revision"],
        "shapes": shapes,
        "generation_config": generation_values,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "vllm_target": profile.runtime.get("vllm_version"),
            "transformers": _package_version("transformers"),
            "vllm": _package_version("vllm"),
        },
    }
    identity["identity_sha256"] = digest(identity)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    receipt_path = output_path / "model_tokenizer_identity.json"
    if receipt_path.exists() and read_json(receipt_path) != identity:
        raise ValueError("IDENTITY_DRIFT: existing model/tokenizer receipt differs; use a new run directory")
    write_json(receipt_path, identity)
    return identity


def _prompt_from_source(row: Mapping[str, Any]) -> str | None:
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        return prompt
    messages = row.get("messages")
    if not isinstance(messages, (tuple, list)):
        return None
    users = [message.get("content") for message in messages if isinstance(message, Mapping) and message.get("role") in ("user", "human")]
    return users[-1] if users and isinstance(users[-1], str) else None


def select_calibration_prompts(
    rows: Any,
    tokenizer: Any,
    profile: ModelProfile,
    seed: int | None = None,
    count: int | None = None,
    split: str | None = None,
) -> list[dict]:
    """Select reproducible eligible prompts from an already pinned source split."""
    profile.validate()
    source_split = split or str(profile.data["calibration_split"])
    if source_split != profile.data["calibration_split"] and source_split != profile.data.get("diagnostic_split"):
        raise ValueError("UNDECLARED_CALIBRATION_SPLIT: source split is not declared in the profile")
    base_seed = int(profile.seeds["diagnostic_generation" if source_split == profile.data.get("diagnostic_split") else "clean_generation"])
    selection_seed = base_seed if seed is None else int(seed)
    wanted = int(count if count is not None else (
        profile.scientific["clean_original_diagnostic_responses"]
        if source_split == profile.data.get("diagnostic_split")
        else profile.scientific["calibration_responses"]
    ))
    if wanted < 1:
        raise ValueError("prompt count must be positive")
    order = np.random.default_rng(selection_seed).permutation(len(rows))
    selected, seen_exact, seen_normalized, seen_ids = [], set(), set(), set()
    for raw_index in order:
        index = int(raw_index)
        row = dict(rows[index])
        prompt = _prompt_from_source(row)
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if not 16 <= len(tokens) <= 512:
            continue
        exact = _sha256_text(prompt)
        normalized_prompt = " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())
        normalized = _sha256_text(normalized_prompt)
        if exact in seen_exact or normalized in seen_normalized:
            continue
        source_id = row.get("id", row.get("prompt_id"))
        prompt_id = str(source_id) if source_id is not None else f"{source_split}:{index:08d}:{exact[:12]}"
        if prompt_id in seen_ids:
            continue
        source_hash = _canonical_sha(row)
        selected.append({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "source_split": source_split,
            "source_index": index,
            "source_row_sha256": source_hash,
            "prompt_text_sha256": exact,
            "normalized_prompt_sha256": normalized,
            "prompt_length": len(tokens),
        })
        seen_exact.add(exact)
        seen_normalized.add(normalized)
        seen_ids.add(prompt_id)
        if len(selected) == wanted:
            return selected
    raise ValueError(f"INSUFFICIENT_ELIGIBLE_PROMPTS: found {len(selected)} of {wanted} in {source_split}")


def partition_calibration_rows(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    """Declare disjoint Fisher/covariance/strength/local context partitions."""
    ids = [row.get("prompt_id") for row in rows]
    if any(not isinstance(value, str) for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("calibration prompt IDs must be nonempty and unique")
    order = np.random.default_rng(int(seed)).permutation(len(rows))
    cuts = [0, int(0.50 * len(rows)), int(0.65 * len(rows)), int(0.80 * len(rows)), len(rows)]
    if min(np.diff(cuts)) < 2:
        raise ValueError("INSUFFICIENT_CALIBRATION_PARTITIONS: each role needs at least two prompts")
    names = ("fisher", "pca_covariance", "strength", "local")
    result = {name: [rows[int(index)] for index in order[lo:hi]] for name, lo, hi in zip(names, cuts[:-1], cuts[1:])}
    flat = [row["prompt_id"] for values in result.values() for row in values]
    if len(flat) != len(set(flat)) or set(flat) != set(ids):
        raise ValueError("CALIBRATION_PARTITION_OVERLAP: partitions must cover the selected pool exactly once")
    return result


def serialize_clean_generations(
    prompts: list[dict],
    generated: list[dict],
    tokenizer: Any,
    model_identity: Mapping[str, Any],
) -> list[dict]:
    """Bind ordinary clean model responses, native IDs, roundtrip, and hashes to prompts."""
    if len(prompts) != len(generated):
        raise ValueError("generation count must match selected prompts exactly")
    eos_ids = set()
    generation_config = model_identity.get("generation_config", {})
    eos_config = generation_config.get("eos_token_id") if isinstance(generation_config, Mapping) else None
    if isinstance(eos_config, int):
        eos_ids.add(eos_config)
    elif isinstance(eos_config, (list, tuple)):
        eos_ids.update(int(value) for value in eos_config)
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if tokenizer_eos is not None:
        eos_ids.add(int(tokenizer_eos))
    output = []
    for prompt, sample in zip(prompts, generated):
        ids = [int(token) for token in sample.get("token_ids", [])]
        text = sample.get("text")
        seed = sample.get("seed")
        finish_reason = sample.get("finish_reason")
        if not ids or not isinstance(text, str) or not text or not isinstance(seed, int):
            raise ValueError("CLEAN_GENERATION_METADATA_MISSING: IDs, visible text, and seed are required")
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        record = {
            **prompt,
            "response": text,
            "response_sha256": _sha256_text(text),
            "generated_token_ids": ids,
            "generation_seed": seed,
            "finish_reason": finish_reason,
            "termination": "eos" if ids[-1] in eos_ids else (str(finish_reason or "unknown")),
            "eos_seen": any(token in eos_ids for token in ids),
            "roundtrip_decoded_sha256": _sha256_text(decoded),
            "roundtrip_exact": decoded == text,
            "generation_model_kind": "clean",
            "model_revision": model_identity["model_revision"],
            "tokenizer_revision": model_identity["tokenizer_revision"],
        }
        record["record_sha256"] = digest(record)
        output.append(record)
    return output


def build_calibration_activation_rows(records: list[dict], tokenizer: Any, max_position_embeddings: int) -> list[dict]:
    """Re-encode visible clean text for causal residual replay without silent truncation."""
    from ..data import encode_response

    output = []
    for row in records:
        encoded = encode_response(tokenizer, row["prompt"], row["response"])
        if len(encoded["input_ids"]) > int(max_position_embeddings):
            raise ValueError("CALIBRATION_CONTEXT_TOO_LONG: full prompt and response exceed model context; truncation is forbidden")
        output.append({
            "prompt_id": row["prompt_id"],
            "source_split": row["source_split"],
            "response_sha256": row["response_sha256"],
            "input_ids": encoded["input_ids"],
            "positions": encoded["positions"],
            "targets": encoded["targets"],
            "model_revision": row["model_revision"],
            "tokenizer_revision": row["tokenizer_revision"],
        })
    return output


def _basis_sanity(basis: np.ndarray, mean: np.ndarray, atol: float = 2e-6) -> bool:
    identity = np.eye(basis.shape[1], dtype=np.float64)
    return (
        np.allclose(basis.T @ basis, identity, atol=atol, rtol=0)
        and np.allclose(basis.T @ mean, np.zeros(basis.shape[1]), atol=atol, rtol=0)
    )


def fit_public_basis(residuals: np.ndarray, mean: np.ndarray, pca_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit a seeded clean-only PCA basis inside the subspace orthogonal to activation mean."""
    values = np.asarray(residuals, dtype=np.float64)
    center = np.asarray(mean, dtype=np.float64)
    if values.ndim != 2 or center.shape != (values.shape[1],) or len(values) <= pca_dim:
        raise ValueError("PCA requires [samples, hidden] residuals, a matching mean, and samples > pca_dim")
    if not np.isfinite(values).all() or not np.isfinite(center).all() or pca_dim < 1:
        raise ValueError("PCA inputs and dimension must be finite and positive")
    if not np.allclose(center, values.mean(axis=0), atol=2e-8, rtol=2e-8):
        raise ValueError("PCA mean must equal the clean calibration activation mean")
    hidden = values.shape[1]
    mean_norm = np.linalg.norm(center)
    if mean_norm > 1e-12 and pca_dim > hidden - 1:
        raise ValueError("mean-orthogonal PCA dimension cannot exceed hidden_size - 1")
    centered = values - center
    if mean_norm > 1e-12:
        unit_mean = center / mean_norm
        centered = centered - (centered @ unit_mean)[:, None] * unit_mean[None, :]
    target = min(hidden, pca_dim + min(16, max(4, pca_dim // 4)))
    omega = np.random.default_rng(int(seed)).standard_normal((len(centered), target))
    sample_range = centered.T @ omega
    q, _ = np.linalg.qr(sample_range, mode="reduced")
    reduced = q.T @ centered.T
    left, _singular_values, _right = np.linalg.svd(reduced, full_matrices=False)
    basis = q @ left[:, :pca_dim]
    # Normalize signs for stable serialization across LAPACK sign conventions.
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    if basis.shape != (hidden, pca_dim) or not _basis_sanity(basis, center):
        raise ValueError("PUBLIC_BASIS_SANITY_FAILURE: mean-orthogonal basis sanity assertion failed")
    return basis, center.copy()


def _dataset_rows(profile: ModelProfile, split: str):
    from datasets import load_dataset

    dataset = load_dataset(
        profile.data["dataset"],
        revision=profile.data["dataset_revision"],
        split=split,
    )
    return dataset


def _generate_clean_vllm(profile: ModelProfile, backbone: Any, prompts: list[dict], seeds: list[int]) -> list[dict]:
    get_hf_token()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    installed = _package_version("vllm")
    from vllm import LLM, SamplingParams

    revision = backbone.profile.revision
    if revision is None:
        revision = getattr(backbone.model.config, "_commit_hash", None)
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("MODEL_REVISION_UNRESOLVED: clean generation requires a frozen model SHA")
    tokenized_prompts = [
        {"prompt_token_ids": prompt_ids(backbone.tokenizer, row["prompt"])}
        for row in prompts
    ]
    max_model_len = max(len(row["prompt_token_ids"]) for row in tokenized_prompts) + int(
        profile.scientific["sampling_max_new_tokens"]
    )
    context_limit = int(backbone.model.config.max_position_embeddings)
    if max_model_len > context_limit:
        raise ValueError("CLEAN_PROMPT_PLUS_RESPONSE_EXCEEDS_CONTEXT: truncation is forbidden")
    engine = LLM(
        model=profile.model_id,
        revision=revision,
        tokenizer=profile.model_id,
        tokenizer_revision=revision,
        dtype="float32",
        trust_remote_code=False,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
    )
    sampling = [SamplingParams(
        temperature=float(profile.scientific["tau"]),
        top_p=float(profile.scientific["sampling_top_p"]),
        top_k=-1,
        max_tokens=int(profile.scientific["sampling_max_new_tokens"]),
        seed=int(seed),
        stop_token_ids=_stop_ids(backbone.model.generation_config),
    ) for seed in seeds]
    try:
        outputs = engine.generate(tokenized_prompts, sampling_params=sampling, use_tqdm=True)
        generated = []
        for output, seed in zip(outputs, seeds):
            if len(output.outputs) != 1:
                raise ValueError("CLEAN_GENERATION_COUNT_MISMATCH: expected one completion per prompt")
            completion = output.outputs[0]
            generated.append({
                "token_ids": list(completion.token_ids),
                "text": completion.text,
                "finish_reason": completion.finish_reason,
                "seed": int(seed),
            })
        return generated
    finally:
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _stop_ids(generation_config: Any) -> list[int]:
    value = getattr(generation_config, "eos_token_id", None)
    if isinstance(value, int):
        return [value]
    if isinstance(value, (tuple, list)):
        return [int(item) for item in value]
    raise ValueError("NATIVE_EOS_UNRESOLVED: generation config must declare one or more EOS IDs")


def prepare_clean_calibration(profile: ModelProfile, backbone: Any, output: str | Path, mode: str) -> dict:
    """Prepare fresh pinned clean generations; never reads legacy Qwen calibration artifacts."""
    profile.validate()
    if mode not in ("calibration", "diagnostic"):
        raise ValueError("mode must be calibration or diagnostic")
    split = profile.data["calibration_split"] if mode == "calibration" else profile.data["diagnostic_split"]
    count_key = "calibration_responses" if mode == "calibration" else "clean_original_diagnostic_responses"
    seed_key = "clean_generation" if mode == "calibration" else "diagnostic_generation"
    count = int(profile.scientific[count_key])
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    identity = resolve_model_identity(
        profile, backbone.tokenizer, backbone.model.config, output_path,
        generation_config=getattr(backbone.model, "generation_config", None),
    )
    identity_digest = identity["identity_sha256"]
    marker = output_path / f"clean_{mode}.identity.json"
    stage_identity = {"mode": mode, "count": count, "split": split, "model_identity_sha256": identity_digest,
                      "dataset": profile.data["dataset"], "dataset_revision": profile.data["dataset_revision"],
                      "profile_id": profile.profile_id, "seed": int(profile.seeds[seed_key]),
                      "sampling": {"temperature": profile.scientific["tau"],
                                   "top_p": profile.scientific["sampling_top_p"],
                                   "top_k_unrestricted": profile.scientific["sampling_top_k_unrestricted"],
                                   "max_new_tokens": profile.scientific["sampling_max_new_tokens"]},
                      "calibration_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output_file = output_path / f"clean_{mode}.jsonl"
    if marker.exists() and read_json(marker) != stage_identity and output_file.exists():
        raise ValueError("CALIBRATION_IDENTITY_DRIFT: refusing to resume changed clean data")
    # A failed engine startup can leave identity and prompt manifests without
    # generated responses. Refresh that pre-generation metadata on retry.
    write_json(marker, stage_identity)
    source_rows = _dataset_rows(profile, split)
    selected = select_calibration_prompts(source_rows, backbone.tokenizer, profile, seed=int(profile.seeds[seed_key]), count=count, split=split)
    prompt_manifest = output_path / f"clean_{mode}_prompts.json"
    write_json(prompt_manifest, {"identity": stage_identity, "prompts": selected, "prompt_manifest_sha256": digest(selected)})
    if output_file.exists():
        existing = read_jsonl(output_file)
        expected_ids = [row["prompt_id"] for row in selected]
        actual_ids = [row.get("prompt_id") for row in existing]
        invalid_record = any(
            row.get("response_sha256") != _sha256_text(str(row.get("response", "")))
            or row.get("record_sha256") != digest({key: value for key, value in row.items() if key != "record_sha256"})
            for row in existing
        )
        if (invalid_record or len(existing) != count or actual_ids != expected_ids
                or any(row.get("model_revision") != identity["model_revision"]
                       or row.get("tokenizer_revision") != identity["tokenizer_revision"]
                       or row.get("generation_model_kind") != "clean" for row in existing)):
            raise ValueError("CALIBRATION_RESUME_MISMATCH: existing clean responses do not match current identity and prompts")
        return {
            **stage_identity,
            "record_count": len(existing),
            "prompt_manifest_sha256": digest(selected),
            "response_jsonl_sha256": hashlib.sha256(output_file.read_bytes()).hexdigest(),
            "roundtrip_exact_count": sum(bool(row.get("roundtrip_exact")) for row in existing),
            "output": str(output_file),
            "resumed": True,
        }
    seeds = [int.from_bytes(hashlib.sha256(f"{profile.seeds[seed_key]}|{row['prompt_id']}|{mode}".encode()).digest()[:4], "big") for row in selected]
    samples = _generate_clean_vllm(profile, backbone, selected, seeds)
    records = serialize_clean_generations(selected, samples, backbone.tokenizer, identity)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_file)
    return {
        **stage_identity,
        "record_count": len(records),
        "prompt_manifest_sha256": digest(selected),
        "response_jsonl_sha256": hashlib.sha256(output_file.read_bytes()).hexdigest(),
        "roundtrip_exact_count": sum(bool(row["roundtrip_exact"]) for row in records),
        "output": str(output_file),
    }
