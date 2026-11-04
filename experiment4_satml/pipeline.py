from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

from klrd_rs.data import prompt_ids as _prompt_ids

from .core import (SEED_URL, assert_tokenizer_compatible, atomic_json,
                   adjust_positionwise_logits, build_prefix_suffix_plan,
                   calibration_prompt, choose_calibration_words,
                   download_seed_source, load_official_mlprints,
                   PrefixCacheInferenceAdapter, prefill_prefix_cache, top_seed_words,
                   validate_output_vocab_sizes,
                   valid_seed_words)


def _config(path: Path) -> dict:
    import yaml
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict) or set(cfg) != {"paths", "models", "experiment"}:
        raise ValueError("Unexpected Experiment 4 configuration schema")
    cfg["config_path"] = path.resolve()
    cfg["root"] = path.resolve().parents[2]
    for key, value in list(cfg["paths"].items()):
        p = Path(value).expanduser()
        cfg["paths"][key] = p if p.is_absolute() else (cfg["root"] / p).resolve()
    return cfg


def _generation_root(cfg):
    """Keep vLLM responses separate from any existing Transformers journal."""
    name = "generation_vllm" if cfg["experiment"].get("generation_backend", "transformers") == "vllm" else "generation"
    return cfg["paths"]["output"] / name


def _e3_state(cfg):
    from klrd_rs.e3full.common import initialize
    return initialize(cfg["root"], cfg["paths"]["e3_full_output"],
                      cfg["paths"]["e3_pilot_output"], mode="report")


def _cached_hub_revision(repo_id: str, requested_revision: str) -> str:
    """Read the immutable commit from the already-cached Hub snapshot path."""
    requested_revision = str(requested_revision)
    if (len(requested_revision) == 40
            and all(char in "0123456789abcdef" for char in requested_revision.lower())):
        return requested_revision.lower()
    from huggingface_hub import snapshot_download
    try:
        snapshot = Path(snapshot_download(repo_id, revision=requested_revision,
            local_files_only=True, allow_patterns=["config.json"]))
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve cached immutable revision for {repo_id}@{requested_revision}; "
            "the calibration model must be present in the Hugging Face cache") from exc
    commit = snapshot.name
    if (len(commit) != 40
            or any(char not in "0123456789abcdef" for char in commit.lower())):
        raise RuntimeError(f"Unexpected cached snapshot path for {repo_id}: {snapshot}")
    return commit.lower()


def _estimate_mapping_eta_seconds(elapsed_seconds: float, completed_this_run: int,
                                  remaining_this_run: int) -> float | None:
    """Estimate remaining time from newly computed shards, excluding cache hits."""
    if completed_this_run <= 0:
        return None
    return max(0.0, float(elapsed_seconds) / completed_this_run * remaining_this_run)


def _merged_checkpoint(state):
    from klrd_rs.e3.evaluate import verify_merged
    verify_merged(state)
    return state["out"] / "merged_student"


def _jsonl(path: Path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n") + 1
        tail = raw[cut:]
        try:
            complete_tail = json.loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            backup = path.with_suffix(path.suffix + ".incomplete-tail.bak")
            if backup.exists() and backup.read_bytes() != tail:
                raise ValueError(f"Conflicting incomplete-tail backup already exists: {backup}")
            if not backup.exists():
                backup.write_bytes(tail)
            raw = raw[:cut]
            print(f"Recovered incomplete final journal row; preserved bytes at {backup}", flush=True)
        else:
            if not isinstance(complete_tail, dict):
                raise ValueError(f"Invalid final journal row at {path}")
            raw += b"\n"
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    rows = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Invalid JSONL row at {path}:{line_no}")
        rows.append(row)
    return rows


def _append_jsonl(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _unique_index(rows, field, context):
    result = {}
    for row in rows:
        key = row.get(field)
        if key in result:
            raise ValueError(f"Duplicate {field}={key!r} in {context}")
        result[key] = row
    return result


def _atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _record_run_config(cfg):
    output = cfg["paths"]["output"]
    paths = {key: str(value) for key, value in cfg["paths"].items()}
    current = {"paths": paths, "models": cfg["models"], "experiment": cfg["experiment"]}
    scientific = json.loads(json.dumps(current))
    scientific["experiment"].pop("calibration_batch_size", None)
    scientific["experiment"].pop("generation_backend", None)
    path = output / "run_config.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        old_scientific = json.loads(json.dumps(previous["config"]))
        old_scientific["experiment"].pop("calibration_batch_size", None)
        old_scientific["experiment"].pop("generation_backend", None)
        if old_scientific != scientific:
            raise ValueError("Experiment 4 scientific configuration changed for this output directory")
        sizes = sorted(set(previous.get("calibration_batch_sizes_used", [])) |
                       {int(cfg["experiment"]["calibration_batch_size"])})
    else:
        sizes = [int(cfg["experiment"]["calibration_batch_size"])]
    atomic_json(path, {"config": current, "scientific_config_sha256": hashlib.sha256(
        json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "calibration_batch_sizes_used": sizes,
        "batch_size_change_policy": "Allowed between mapping resumes; batching does not change score aggregation"})


def _tokenizer(cfg, checkpoint):
    from transformers import AutoTokenizer
    attacked = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True,
                                              trust_remote_code=False)
    calibration = AutoTokenizer.from_pretrained(
        cfg["models"]["calibration_model"],
        revision=cfg["models"].get("calibration_revision", "main"),
        trust_remote_code=False)
    compatibility = assert_tokenizer_compatible(attacked, calibration)
    return attacked, calibration, compatibility


def prepare(cfg):
    source_text, source_hash = download_seed_source(cfg["paths"]["seed_source"])
    words = top_seed_words(source_text, cfg["experiment"]["seed_words"])
    from transformers import AutoTokenizer
    state = _e3_state(cfg)
    checkpoint = _merged_checkpoint(state)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True,
                                               trust_remote_code=False)
    valid = valid_seed_words(words, tokenizer)
    assets = cfg["paths"]["output"] / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    _atomic_text(assets / "seed_words_6000.txt", "\n".join(words) + "\n")
    _atomic_text(assets / "seed_words_valid.txt", "\n".join(valid) + "\n")
    manifest = {"seed_url": SEED_URL, "source_revision": "google-10000-english/master",
                "source_sha256": source_hash, "source_rows_retained": len(words),
                "source_rows_considered": cfg["experiment"]["seed_words"],
                "tokenizer": str(checkpoint), "tokenizer_revision": state["cfg"].revision,
                "valid_single_token_words": len(valid),
                "tokenization": "tokenizer.encode(' ' + word, add_special_tokens=False)",
                "calibration_contexts": cfg["experiment"]["calibration_texts"],
                "calibration_context_policy": "fixed KLRD-RS E4 design; not a paper reproduction"}
    atomic_json(assets / "seed_manifest.json", manifest)
    print(f"Prepared {len(words)} top-frequency words; {len(valid)} are one token.", flush=True)


def generate_calibration(cfg):
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig
    state = _e3_state(cfg)
    checkpoint = _merged_checkpoint(state)
    tokenizer, _, compatibility = _tokenizer(cfg, checkpoint)
    e = cfg["experiment"]
    out = cfg["paths"]["output"] / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    valid = (out.parent / "assets/seed_words_valid.txt").read_text(encoding="utf-8").splitlines()
    selected_path = out / "calibration_seed_words.json"
    selected = choose_calibration_words(valid, e["calibration_texts"], e["seed"])
    selection = {"seed": e["seed"], "words": selected}
    if selected_path.exists() and json.loads(selected_path.read_text(encoding="utf-8")) != selection:
        raise ValueError("Calibration seed word selection changed; do not mix it with the existing journal")
    atomic_json(selected_path, selection)
    journal = out / "calibration_generations.jsonl"
    done = _unique_index(_jsonl(journal), "index", str(journal))
    for index, row in done.items():
        if (not isinstance(index, int) or not 0 <= index < len(selected)
                or row.get("word") != selected[index] or not row.get("decoded_text")
                or not row.get("generated_token_ids")):
            raise ValueError(f"Invalid cached calibration row: {journal} index={index!r}")
    generation_identity = {"e3_identity": state["identity"], "checkpoint": str(checkpoint),
        "seed_words": selected, "seed": e["seed"], "generation": {"do_sample": True,
            "temperature": 0.8, "top_p": 0.95, "max_new_tokens": e["calibration_new_tokens"]}}
    identity_path = out / "calibration_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != generation_identity:
        raise ValueError("Calibration generation identity changed; refusing to mix cached texts")
    if journal.exists() and not identity_path.exists():
        raise ValueError("Refusing to resume unbound calibration generation journal")
    atomic_json(identity_path, generation_identity)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.float32, local_files_only=True,
        trust_remote_code=False, attn_implementation="sdpa").to("cuda").eval()
    generation = GenerationConfig(do_sample=True, temperature=0.8, top_p=0.95,
        top_k=0, max_new_tokens=e["calibration_new_tokens"], num_beams=1,
        repetition_penalty=1.0, eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    for index, word in enumerate(selected):
        if index in done:
            continue
        prompt = f"{tokenizer.bos_token or ''}{word}"
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        torch.manual_seed(e["seed"])
        torch.cuda.manual_seed_all(e["seed"])
        with torch.inference_mode():
            generated = model.generate(input_ids=encoded, attention_mask=torch.ones_like(encoded),
                generation_config=generation, do_sample=True, temperature=0.8, top_p=0.95,
                max_new_tokens=e["calibration_new_tokens"])[0, encoded.shape[1]:].tolist()
        text = tokenizer.decode(generated, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False).replace("\r", " ").replace("\n", " ").strip()
        if not generated or not text:
            raise RuntimeError(f"Calibration continuation empty at index {index} ({word!r}); refusing to reduce N=128")
        _append_jsonl(journal, {"index": index, "word": word, "prompt": prompt,
            "generated_token_ids": [int(i) for i in generated], "decoded_text": text,
            "generated_token_count": len(generated), "seed": e["seed"],
            "generation": {"do_sample": True, "temperature": 0.8, "top_p": 0.95,
                           "max_new_tokens": e["calibration_new_tokens"]},
            "checkpoint": str(checkpoint), "checkpoint_revision": state["cfg"].revision,
            "tokenizer_compatibility": compatibility})
        print(f"Calibration text {index + 1}/{len(selected)} saved", flush=True)
    rows = _jsonl(journal)
    if len(rows) != e["calibration_texts"] or [r["index"] for r in rows] != list(range(len(selected))):
        raise ValueError("Calibration generation journal is incomplete or unordered")
    # Official MLprints reads one text per line; raw JSONL preserves exact metadata.
    _atomic_text(out / "calibration_texts.txt", "\n".join(r["decoded_text"] for r in rows) + "\n")
    atomic_json(out / "calibration_manifest.json", {"count": len(rows), "seed": e["seed"],
        "generation": {"do_sample": True, "temperature": 0.8, "top_p": 0.95,
                       "max_new_tokens": e["calibration_new_tokens"]},
        "checkpoint": str(checkpoint), "checkpoint_identity": state["identity"],
        "attacked_model_revision": state["cfg"].revision,
        "tokenizer_revisions": {"attacked": compatibility["attacked_tokenizer_revision"],
                                "calibration": compatibility["calibration_tokenizer_revision"]},
        "tokenizer_compatibility": compatibility,
        "calibration_texts_sha256": hashlib.sha256((out / "calibration_texts.txt").read_bytes()).hexdigest()})


def _save_torch(path: Path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _first_captured_batches(captured_batches, row_count):
    """Slice exactly the first row_count logits while preserving batch order."""
    remaining = int(row_count)
    selected = []
    for batch in captured_batches:
        logits = batch["scores"][0]
        take = min(remaining, int(logits.shape[0]))
        if take:
            selected.append({"scores": [logits[:take]]})
            remaining -= take
        if not remaining:
            break
    if remaining:
        raise RuntimeError(f"Cached logits contain fewer than {row_count} contexts")
    return selected


def build_mapping(cfg):
    import torch
    from transformers import AutoModelForCausalLM
    state = _e3_state(cfg)
    checkpoint = _merged_checkpoint(state)
    tokenizer, _, compatibility = _tokenizer(cfg, checkpoint)
    mlprints = load_official_mlprints(cfg)
    e = cfg["experiment"]
    root = cfg["paths"]["output"]
    words = (root / "assets/seed_words_valid.txt").read_text(encoding="utf-8").splitlines()
    texts = [r["decoded_text"] for r in _jsonl(root / "calibration/calibration_generations.jsonl")]
    if len(texts) != e["calibration_texts"] or any(not t for t in texts):
        raise ValueError("Expected exactly 128 non-empty calibration texts")
    unique = {}
    for word in words:
        ids = tokenizer.encode(f" {word}", add_special_tokens=False)
        if len(ids) == 1:
            unique.setdefault(int(ids[0]), word)
    if not unique:
        raise ValueError("Seed tokenizer filtering produced no valid words")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("Full suppression calibration requires the GPU server")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32,
        local_files_only=True, trust_remote_code=False, attn_implementation="sdpa").to(device).eval()
    calibration = AutoModelForCausalLM.from_pretrained(cfg["models"]["calibration_model"],
        revision=cfg["models"].get("calibration_revision", "main"), torch_dtype=torch.float16,
        trust_remote_code=False, attn_implementation="sdpa").to(device).eval()
    calibration_revision = _cached_hub_revision(
        cfg["models"]["calibration_model"],
        cfg["models"].get("calibration_revision", "main"))
    output_vocab = validate_output_vocab_sizes(
        model.config.vocab_size, calibration.config.vocab_size, tokenizer)
    words_by_token = list(unique.values())
    cache_plan = build_prefix_suffix_plan(tokenizer, texts, words_by_token,
                                          bos_token=tokenizer.bos_token or "")
    prefix_caches = {"attacked": [], "calibration": []}
    cache_adapter = None
    cache_parity = None
    cache_status = "full_prompt_no_shared_prefix"
    if cache_plan["prefix_length"] > 0:
        batch_size = int(e["calibration_batch_size"])
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        for side, cache_model in (("attacked", model), ("calibration", calibration)):
            print(f"Prefilling {side} KV cache: prefix_length={cache_plan['prefix_length']}, "
                  f"contexts={len(texts)}, batch_size={batch_size}", flush=True)
            for start in range(0, len(texts), batch_size):
                end = min(len(texts), start + batch_size)
                try:
                    cache, prefix_length = prefill_prefix_cache(
                        cache_model, cache_plan["prefix_ids"][start:end], device,
                        pad_token_id=pad_token_id)
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"Prefix-cache prefill ran out of GPU memory at batch_size={batch_size}; "
                        "lower experiment.calibration_batch_size and resume") from exc
                if prefix_length != max(cache_plan["prefix_lengths"][start:end]):
                    raise RuntimeError("Prefix-cache padded width changed during prefill")
                prefix_caches[side].append(cache)
                print(f"  {side} prefix contexts {end}/{len(texts)} cached", flush=True)
        official_inference = mlprints.run_inference
        cache_adapter = PrefixCacheInferenceAdapter(
            model, calibration, tokenizer, texts, words_by_token, cache_plan,
            prefix_caches, batch_size, official_inference, pad_token_id, device,
            bos_token=tokenizer.bos_token or "")
        parity_words = list(dict.fromkeys((words_by_token[0],
            words_by_token[len(words_by_token) // 2], words_by_token[-1])))
        parity = cache_adapter.verify_parity(parity_words)
        cache_parity = parity
        cache_status = "passed"
        print(f"Prefix-cache logits parity passed: {json.dumps(parity, sort_keys=True)}", flush=True)
    else:
        print("No common token prefix across seed prompts; using full-prompt inference.", flush=True)
    shard_dir = root / "mapping/shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    calibration_module = mlprints.BigramSuppressionAttackModel
    calibration_hash = hashlib.sha256((root / "calibration/calibration_texts.txt").read_bytes()).hexdigest()
    mapping_identity = {"e3_identity": state["identity"], "checkpoint": str(checkpoint),
        "calibration_model": cfg["models"]["calibration_model"],
        "calibration_revision": calibration_revision,
        "output_vocab": output_vocab,
        "calibration_texts_sha256": calibration_hash, "seed_words_sha256": hashlib.sha256(
            "\n".join(words).encode()).hexdigest(), "probability_threshold": e["probability_threshold"],
        "clip_quantile": e["clip_quantile"], "contexts": len(texts),
        "tokenizer_compatibility": compatibility}
    mapping_identity_path = root / "mapping/identity.json"
    if mapping_identity_path.exists() and json.loads(mapping_identity_path.read_text(encoding="utf-8")) != mapping_identity:
        raise ValueError("Mapping inputs/configuration changed; existing shards cannot be reused")
    if any(shard_dir.glob("*.pt")) and not mapping_identity_path.exists():
        raise ValueError("Refusing to resume unbound mapping shards")
    atomic_json(mapping_identity_path, mapping_identity)
    expected_batches = math.ceil(len(texts) / e["calibration_batch_size"])
    cached_count = sum((shard_dir / f"{previous_id:06d}.pt").is_file()
                       for previous_id in unique)
    total_new = len(unique) - cached_count
    new_completed = 0
    mapping_started = time.monotonic()
    print(f"Mapping {len(unique)} preceding tokens; {cached_count} cached shards; "
          f"{len(texts)} contexts, batch_size={e['calibration_batch_size']} "
          f"({expected_batches} batches/model/token).", flush=True)
    for index, (previous_id, word) in enumerate(unique.items(), 1):
        target = shard_dir / f"{previous_id:06d}.pt"
        if target.exists():
            existing_shard = torch.load(target, map_location="cpu", weights_only=True)
            if existing_shard.get("identity") != mapping_identity or existing_shard.get("word") != word:
                raise ValueError(f"Mapping shard identity mismatch: {target}")
            continue
        word_started = time.monotonic()
        print(f"Mapping word {index}/{len(unique)}: {word!r} (token_id={previous_id})",
              flush=True)
        captured = {"model": [], "calibration": []}
        call_index = 0
        original_inference = mlprints.run_inference
        if cache_adapter is not None:
            cache_adapter.begin_word(word)
        def capture_inference(*args, **kwargs):
            nonlocal call_index
            if cache_adapter is None:
                result = original_inference(*args, **kwargs)
            else:
                result = cache_adapter(*args, **kwargs)
            label = "model" if call_index % 2 == 0 else "calibration"
            if "scores" not in result or any(not torch.isfinite(t).all() for t in result["scores"]):
                raise FloatingPointError(f"Non-finite/missing {label} logits for preceding word {word!r}")
            captured[label].append({k: [t.detach().cpu().clone() for t in v]
                                    for k, v in result.items() if k == "scores"})
            call_index += 1
            if call_index % 2 == 0:
                batch_number = call_index // 2
                elapsed = time.monotonic() - word_started
                print(f"\r  Forward batches {batch_number}/{expected_batches} "
                      f"(attacked + calibration); {elapsed:.1f}s on this word",
                      end="", flush=True)
            return result
        mlprints.run_inference = capture_inference
        try:
            full = calibration_module(model, tokenizer, calibration, seed_words=[word],
                calibration_texts=texts, probability_threshold=e["probability_threshold"],
                clip_quantile=e["clip_quantile"], suppression_delta=1.0,
                batch_size=e["calibration_batch_size"])
            # Re-run the original class over the cached first 64 contexts. Only
            # the official forward passes above touch either model; scoring and
            # signed clipping below remain byte-for-byte algorithmically shared.
            first = {name: _first_captured_batches(batches, 64)
                     for name, batches in captured.items()}
            replay_index = 0
            replay_batch = {"model": 0, "calibration": 0}
            def replay_inference(*args, **kwargs):
                nonlocal replay_index
                label = "model" if replay_index % 2 == 0 else "calibration"
                replay_index += 1
                batch = replay_batch[label]
                replay_batch[label] += 1
                if batch >= len(first[label]):
                    raise RuntimeError("Cached 64-context replay requested missing logits")
                return first[label][batch]
            mlprints.run_inference = replay_inference
            half = calibration_module(model, tokenizer, calibration, seed_words=[word],
                calibration_texts=texts[:64], probability_threshold=e["probability_threshold"],
                clip_quantile=e["clip_quantile"], suppression_delta=1.0,
                batch_size=e["calibration_batch_size"])
        finally:
            mlprints.run_inference = original_inference
        final = full.processor.mapping.get(previous_id)
        partial = half.processor.mapping.get(previous_id)
        if final is None or partial is None or len(final[0]) == 0:
            raise RuntimeError(f"Official suppression mapping is empty for seed word {word!r}")
        if len(captured["model"]) != expected_batches or len(captured["calibration"]) != expected_batches:
            raise RuntimeError(f"Incomplete forward-logit cache for preceding word {word!r}")
        valid_counts = torch.zeros(model.config.vocab_size, dtype=torch.int64)
        log_threshold = math.log(e["probability_threshold"])
        for model_batch, calibration_batch in zip(captured["model"], captured["calibration"]):
            model_logprobs = model_batch["scores"][0].float().log_softmax(-1)
            calibration_logprobs = calibration_batch["scores"][0].float().log_softmax(-1)
            if model_logprobs.shape != calibration_logprobs.shape:
                raise ValueError("Attacked/calibration next-token logit shapes differ")
            ratios = model_logprobs - calibration_logprobs
            ratios[model_logprobs <= log_threshold] = torch.nan
            valid_counts += ratios.isfinite().sum(0).cpu()
        common = torch.isin(partial[0], final[0])
        a, b = partial[1][common].double(), final[1][torch.searchsorted(final[0], partial[0][common])].double()
        # Token IDs are sorted by the official .nonzero() call.
        correlation = float(torch.corrcoef(torch.stack((a, b)))[0, 1]) if len(a) > 1 and a.std() and b.std() else None
        if correlation is not None and not math.isfinite(correlation):
            correlation = None
        sign_agreement = float((torch.sign(a) == torch.sign(b)).double().mean()) if len(a) else 0.0
        diagnostic = {"preceding_word_index": index,
            "previous_token_id": previous_id, "word": word,
            "valid_token_coverage_64": int(len(partial[0])), "valid_token_coverage_128": int(len(final[0])),
            "score_correlation": correlation, "sign_agreement": sign_agreement,
            "vocabulary_columns_with_zero_valid_contexts": int((valid_counts == 0).sum()),
            "vocabulary_columns_with_fewer_than_two_valid_contexts": int((valid_counts < 2).sum()),
            "final_weight_nonfinite_count": int((~torch.isfinite(final[1])).sum())}
        if diagnostic["final_weight_nonfinite_count"]:
            raise FloatingPointError(f"Official source returned non-finite weights for {word!r}")
        _save_torch(target, {"identity": mapping_identity, "previous_token_id": previous_id, "word": word,
            "token_ids": final[0].cpu(), "weights": final[1].cpu(), "diagnostic": diagnostic})
        new_completed += 1
        elapsed_word = time.monotonic() - word_started
        elapsed_total = time.monotonic() - mapping_started
        eta_seconds = _estimate_mapping_eta_seconds(
            elapsed_total, new_completed, total_new - new_completed)
        eta = "n/a" if eta_seconds is None else f"{eta_seconds / 3600:.2f} h"
        print(f"\r  Saved {index}/{len(unique)} shards ({new_completed}/{total_new} new); "
              f"word={elapsed_word:.1f}s; elapsed={elapsed_total / 60:.1f} min; ETA={eta}",
              flush=True)
    # Materialize the exact mapping consumed by the official processor once.
    mapping = {}
    diagnostics = []
    for path in sorted(shard_dir.glob("*.pt")):
        row = torch.load(path, map_location="cpu", weights_only=True)
        mapping[int(row["previous_token_id"])] = (row["token_ids"], row["weights"])
        diagnostics.append(row["diagnostic"])
    diagnostics.sort(key=lambda value: value["preceding_word_index"])
    _write_jsonl(root / "mapping/diagnostics.jsonl", diagnostics)
    _save_torch(root / "mapping/suppression_mapping.pt", mapping)
    atomic_json(root / "mapping/metadata.json", {"status": "complete", "identity": mapping_identity,
        "preceding_tokens": len(mapping),
        "valid_seed_words": len(words), "calibration_texts": len(texts),
        "probability_threshold": e["probability_threshold"], "clip_quantile": e["clip_quantile"],
        "algorithm_source": str(Path(__file__).resolve().parent / "vendor/mlprints/bigram_suppression.py"),
        "algorithm_source_sha256": hashlib.sha256((Path(__file__).resolve().parent /
            "vendor/mlprints/bigram_suppression.py").read_bytes()).hexdigest(),
        "tokenizer_compatibility": compatibility,
        "prefix_cache": {"status": cache_status,
            "prefix_length_max": cache_plan["prefix_length"],
            "prefix_length_min": min(cache_plan["prefix_lengths"]), "parity": cache_parity},
        "scoring": "official BigramSuppressionAttackModel maximum/z-score, quantile clipping, signed scores",
        "diagnostics": "64 contexts replayed from cached forward logits; final mapping uses all 128"})
    del cache_adapter, prefix_caches, model, calibration
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _expected_adjustment_stats(prefix_ids, generated_ids, mapping_stats, delta):
    """Compute expected sparse adjustments from generated bigram IDs."""
    previous_ids = [int(prefix_ids[-1])] + [int(token_id) for token_id in generated_ids[:-1]]
    covered = adjusted = nonzero_values = 0
    absolute_sum = 0.0
    for previous_id in previous_ids:
        stats = mapping_stats.get(previous_id)
        if stats is None:
            continue
        covered += 1
        weight_count, nonzero_count, weight_abs_sum = stats
        expected_nonzero = nonzero_count if delta != 0 else 0
        if expected_nonzero:
            adjusted += 1
            nonzero_values += expected_nonzero
            absolute_sum += abs(float(delta)) * weight_abs_sum
    return {"steps": len(generated_ids), "covered_steps": covered,
        "expected_adjusted_steps": adjusted,
        "expected_nonzero_adjustment_values": nonzero_values,
        "expected_absolute_adjustment_sum": absolute_sum,
        "expected_adjustment_values": sum(mapping_stats.get(token, (0, 0, 0.0))[0]
            for token in previous_ids if token in mapping_stats),
        "observed_nonzero_logit_changes": None,
        "observed_absolute_logit_change_sum": None}


def _normalize_eos_token_ids(eos_token_id):
    """Normalize Transformers' scalar or multi-token EOS config to integer IDs."""
    eos_ids = ([int(eos_token_id)] if isinstance(eos_token_id, int)
               else [int(token_id) for token_id in (eos_token_id or [])])
    if not eos_ids:
        raise ValueError("Missing native model EOS configuration")
    return eos_ids


def _apply_bigram_adjustment(scores, previous_token_id, mapping, delta):
    """Apply the upstream sparse signed adjustment to one vLLM logits row."""
    pair = mapping.get(int(previous_token_id))
    if pair is None:
        return scores
    token_ids, weights = pair
    scores[token_ids] -= float(delta) * weights.to(scores.device)
    return scores


def _make_vllm_suppression_processor(mapping_path):
    """Build the vLLM 0.33 adapter for the request-level bigram adjustment."""
    import torch
    from vllm import SamplingParams
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

    class _RequestSuppression:
        def __init__(self, mapping, delta, prompt_last_token_id):
            self.mapping = mapping
            self.delta = float(delta)
            self.prompt_last_token_id = int(prompt_last_token_id)

        def __call__(self, output_ids, logits):
            previous = output_ids[-1] if output_ids else self.prompt_last_token_id
            return _apply_bigram_adjustment(logits, previous, self.mapping, self.delta)

    class SaTMLVllmLogitsProcessor(AdapterLogitsProcessor):
        @classmethod
        def validate_params(cls, params):
            extra = params.extra_args or {}
            if not isinstance(extra.get("prompt_last_token_id"), int):
                raise ValueError("SaTML requires prompt_last_token_id in vLLM extra_args")
            if not isinstance(extra.get("delta"), (int, float)):
                raise ValueError("SaTML requires numeric delta in vLLM extra_args")

        def __init__(self, vllm_config, device, is_pin_memory):
            super().__init__(vllm_config, device, is_pin_memory)
            print("E4 vLLM processor: loading sparse suppression mapping on GPU.", flush=True)
            cpu_mapping = torch.load(mapping_path, map_location="cpu", weights_only=True)
            self.mapping = {int(key): (token_ids.to(device), weights.to(device))
                            for key, (token_ids, weights) in cpu_mapping.items()}
            del cpu_mapping
            print(f"E4 vLLM processor: {len(self.mapping)} preceding-token rows ready.", flush=True)

        def new_req_logits_processor(self, params):
            extra = params.extra_args or {}
            self.validate_params(params)
            return _RequestSuppression(self.mapping, extra["delta"],
                                       extra["prompt_last_token_id"])

    return SaTMLVllmLogitsProcessor


def generate_attacks(cfg):
    import torch
    from transformers import AutoModelForCausalLM, LogitsProcessorList
    state = _e3_state(cfg)
    checkpoint = _merged_checkpoint(state)
    tokenizer, _, _ = _tokenizer(cfg, checkpoint)
    e = cfg["experiment"]
    root = cfg["paths"]["output"]
    generation_root = _generation_root(cfg)
    generation_root.mkdir(parents=True, exist_ok=True)
    source = __import__("klrd_rs.e3full.evaluation", fromlist=["load_observations"])
    e3_phase, observations, groups, _ = source.load_observations(state, 1024)
    prompts = observations["student"][0]
    mapping_path = root / "mapping/suppression_mapping.pt"
    mapping_cpu = torch.load(mapping_path, map_location="cpu", weights_only=True)
    if not mapping_cpu:
        raise ValueError("Refusing generation with an empty suppression mapping")
    from klrd_rs.e3.generation import _record, _request_seed
    mapping_stats = {}
    for previous_id, (_, weights) in mapping_cpu.items():
        cpu_weights = weights.detach().float().cpu()
        mapping_stats[int(previous_id)] = (int(cpu_weights.numel()),
            int(torch.count_nonzero(cpu_weights)), float(cpu_weights.abs().sum().item()))
    backend = str(e.get("generation_backend", "transformers")).lower()
    if backend not in {"transformers", "vllm"}:
        raise ValueError(f"Unsupported Experiment 4 generation backend: {backend}")
    execution = {"backend": "transformers.generate", "request_batch_size": 1,
                 "max_num_seqs": 1, "cudagraph_mode": "NONE"}
    model = None
    engine = None
    eos_ids = None
    prefix_ids_by_prompt = {}
    if backend == "transformers":
        from .core import load_official_mlprints
        official = load_official_mlprints(cfg).BigramSuppressionProcessor
        mapping = {int(key): (token_ids.to("cuda"), weights.to("cuda"))
                   for key, (token_ids, weights) in mapping_cpu.items()}
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32,
            local_files_only=True, trust_remote_code=False,
            attn_implementation="sdpa").to("cuda").eval()
        eos_ids = _normalize_eos_token_ids(model.generation_config.eos_token_id)
        torch.cuda.empty_cache()
    else:
        import triton
        from transformers import AutoConfig, GenerationConfig
        from vllm import LLM, SamplingParams
        from klrd_rs.e3.generation import _vllm_options
        from klrd_rs.e3.vllm_execution import tuning

        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        triton.knobs.language.fp32_default = "ieee"
        generation_config = GenerationConfig.from_pretrained(
            checkpoint, local_files_only=True)
        eos_ids = _normalize_eos_token_ids(generation_config.eos_token_id)
        prefix_ids_by_prompt = {row["prompt_id"]: _prompt_ids(tokenizer, row["prompt"])
                                for row in prompts}
        maximum_prompt_length = max(map(len, prefix_ids_by_prompt.values()), default=0)
        model_config = AutoConfig.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False)
        required_context = maximum_prompt_length + int(e["max_new_tokens"])
        native_context = getattr(model_config, "max_position_embeddings", None)
        if native_context is not None and required_context > native_context:
            raise ValueError("Prompt plus requested generation exceeds the model context")
        execution = dict(tuning(), backend="vllm", cudagraph_mode="FULL_DECODE_ONLY",
            precision_note="FP32 remains FP32; sampled token identity across backends/batch shapes is not asserted",
            max_model_len=max(2048, required_context), max_prompt_tokens=maximum_prompt_length)
        engine_settings = {"model": str(checkpoint), "revision": None,
            "tokenizer": str(checkpoint), "tokenizer_revision": None, "dtype": "float32"}
        engine_options = _vllm_options(engine_settings, execution)
        logits_processor = _make_vllm_suppression_processor(mapping_path)
        print(f"E4 vLLM: loading model and suppression processor; "
              f"queue={execution['request_batch_size']}, max concurrent={execution['max_num_seqs']}.",
              flush=True)
        engine = LLM(**engine_options, logits_processors=[logits_processor])
        print("E4 vLLM engine ready.", flush=True)
    settings = {"model": str(checkpoint), "revision": None, "tokenizer": str(checkpoint),
        "tokenizer_revision": None, "backend": "hf" if backend == "transformers" else "vllm",
        "dtype": "float32", "temperature": .8,
        "top_p": 1.0, "top_k": 0, "max_new_tokens": e["max_new_tokens"], "seed": state["cfg"].seed,
        "kind": "student", "eos_token_ids": eos_ids}
    mapping_sha256 = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    from klrd_rs.io import digest
    for delta in e["deltas"]:
        out = generation_root / f"delta_{delta}"
        journal = out / "responses.jsonl"
        done = _unique_index(_jsonl(journal), "prompt_id", str(journal))
        identity = {"e3_identity": state["identity"], "checkpoint": str(checkpoint),
            "prompts_sha256": hashlib.sha256(json.dumps([r["prompt_id"] for r in prompts]).encode()).hexdigest(),
            "backend": "transformers.generate" if backend == "transformers" else "vllm.generate",
            "e3_reference_backend": state["cfg"].generation_backend,
            "delta": delta, "settings": settings, "mapping_sha256": mapping_sha256}
        if backend == "vllm":
            identity["execution"] = execution
        identity_path = out / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Refusing to resume {journal}: generation identity/configuration changed")
        if journal.exists() and not identity_path.exists():
            raise ValueError(f"Refusing to resume unbound response journal: {journal}")
        atomic_json(identity_path, identity)
        if backend == "vllm":
            pending = [row for row in prompts if row["prompt_id"] not in done]
            batch_size = execution["request_batch_size"]
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                prefixes = [prefix_ids_by_prompt[row["prompt_id"]] for row in batch]
                sampling = [SamplingParams(n=1, temperature=settings["temperature"],
                    top_p=1.0, top_k=-1, max_tokens=settings["max_new_tokens"],
                    seed=_request_seed(settings, row), ignore_eos=False,
                    stop_token_ids=settings["eos_token_ids"], detokenize=True,
                    skip_special_tokens=True, repetition_penalty=1.0,
                    presence_penalty=0.0, frequency_penalty=0.0,
                    extra_args={"delta": delta, "prompt_last_token_id": int(prefix[-1])})
                    for row, prefix in zip(batch, prefixes)]
                batch_number = start // batch_size + 1
                print(f"E4 vLLM delta={delta}: batch {batch_number} START "
                      f"{len(batch)} requests; {len(done)}/{len(prompts)} already saved.", flush=True)
                started = time.perf_counter()
                outputs = engine.generate(
                    [{"prompt_token_ids": prefix} for prefix in prefixes],
                    sampling_params=sampling, use_tqdm=True)
                elapsed = time.perf_counter() - started
                if len(outputs) != len(batch):
                    raise ValueError("vLLM returned an incomplete batch; no batch rows committed")
                for output, prefix in zip(outputs, prefixes):
                    if list(output.prompt_token_ids) != prefix or len(output.outputs) != 1:
                        raise ValueError("vLLM returned unexpected prompt/order/sample count")
                for row, prefix, output in zip(batch, prefixes, outputs):
                    answer = output.outputs[0]
                    ids = [int(token_id) for token_id in answer.token_ids]
                    record = _record(row, ids, tokenizer, settings, identity,
                        finish_reason=answer.finish_reason, stop_reason=answer.stop_reason,
                        execution=dict(execution, actual_request_batch_size=len(batch)),
                        elapsed_seconds=elapsed / max(len(batch), 1))
                    record["attack_diagnostics"] = _expected_adjustment_stats(
                        prefix, ids, mapping_stats, delta)
                    record["record_sha256"] = digest(
                        {key: value for key, value in record.items() if key != "record_sha256"})
                    _append_jsonl(journal, record)
                    done[row["prompt_id"]] = record
                visible_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
                print(f"E4 vLLM delta={delta}: {len(done)}/{len(prompts)} saved; "
                      f"batch generation={elapsed:.1f}s, {visible_tokens / max(elapsed, 1e-9):.1f} tok/s.",
                      flush=True)
            if len(done) != len(prompts):
                raise ValueError(f"delta={delta} vLLM generation journal incomplete")
        else:
            processor = official(mapping, delta)
        for row in prompts:
            if backend != "transformers":
                break
            if row["prompt_id"] in done:
                cached = done[row["prompt_id"]]
                if (cached.get("generation_identity") != digest(identity)
                        or cached.get("record_sha256") != digest({k: v for k, v in cached.items() if k != "record_sha256"})
                        or cached.get("prompt") != row["prompt"]):
                    raise ValueError(f"Response journal identity mismatch: {row['prompt_id']}")
                continue
            prefix = _prompt_ids(tokenizer, row["prompt"])
            input_ids = torch.tensor([prefix], device="cuda", dtype=torch.long)
            seed = _request_seed(settings, row)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=e["max_new_tokens"], do_sample=True, temperature=.8, top_p=1.0,
                    top_k=0, num_beams=1, repetition_penalty=1.0, no_repeat_ngram_size=0,
                    logits_processor=LogitsProcessorList([processor]), renormalize_logits=True,
                    eos_token_id=settings["eos_token_ids"], pad_token_id=tokenizer.pad_token_id or settings["eos_token_ids"][0])
            ids = generated[0, len(prefix):].tolist()
            reason = "stop" if ids and ids[-1] in settings["eos_token_ids"] else "length"
            attack_stats = _expected_adjustment_stats(prefix, ids, mapping_stats, delta)
            record = _record(row, ids, tokenizer, settings, identity, finish_reason=reason,
                stop_reason=ids[-1] if reason == "stop" else None, execution={"backend": "transformers.generate"})
            record["attack_diagnostics"] = attack_stats
            record["record_sha256"] = digest({k: v for k, v in record.items() if k != "record_sha256"})
            _append_jsonl(journal, record)
            done[row["prompt_id"]] = record
            if len(done) % 20 == 0:
                print(f"delta={delta}: {len(done)}/{len(prompts)} responses", flush=True)
        if backend == "transformers" and len(done) != len(prompts):
            raise ValueError(f"delta={delta} generation journal incomplete")
        prompt_by_id = {row["prompt_id"]: row for row in prompts}
        normalized_diagnostics = []
        for record in done.values():
            attack_diagnostics = record.get("attack_diagnostics", {})
            if "expected_nonzero_adjustment_values" not in attack_diagnostics:
                prompt_row = prompt_by_id[record["prompt_id"]]
                attack_diagnostics = _expected_adjustment_stats(
                    _prompt_ids(tokenizer, prompt_row["prompt"]),
                    record.get("generated_token_ids", []), mapping_stats, delta)
            normalized_diagnostics.append(attack_diagnostics)
        step_count = sum(row.get("steps", 0) for row in normalized_diagnostics)
        covered_count = sum(row.get("covered_steps", 0) for row in normalized_diagnostics)
        expected_values = sum(row.get("expected_nonzero_adjustment_values", 0)
                              for row in normalized_diagnostics)
        adjustment_sum = sum(row.get("expected_absolute_adjustment_sum", 0.0)
                             for row in normalized_diagnostics)
        value_count = sum(row.get("expected_adjustment_values", 0)
                          for row in normalized_diagnostics)
        adjusted_steps = sum(row.get("expected_adjusted_steps", 0)
                             for row in normalized_diagnostics)
        if not step_count or not expected_values:
            raise RuntimeError(f"delta={delta} produced no measured logit adjustment; check mapping/processor")
        atomic_json(out / "diagnostics.json", {"delta": delta, "generation_steps": step_count,
            "covered_steps": covered_count,
            "covered_step_fraction": covered_count / max(step_count, 1),
            "adjusted_step_fraction": adjusted_steps / max(step_count, 1),
            "mean_expected_absolute_logit_adjustment": adjustment_sum / max(value_count, 1),
            "expected_nonzero_adjustment_value_fraction": expected_values / max(value_count, 1),
            "expected_nonzero_adjustment_values": expected_values,
            "observed_floating_point_changes_measured": False,
            "mapping_preceding_tokens": len(mapping_cpu), "backend": f"{backend}.generate",
            "e3_reference_backend": state["cfg"].generation_backend,
            "sampling_parity_claim": False})


def verify_attacks(cfg):
    from klrd_rs.e3full.evaluation import load_observations, generation_utility
    from klrd_rs.e3.evaluate import score_rows, tables, csv
    from klrd_rs.io import digest
    state = _e3_state(cfg)
    phase, baseline, groups, source = load_observations(state, 1024)
    root = cfg["paths"]["output"]
    generation_root = _generation_root(cfg)
    all_conditions = {"baseline": baseline["student"]}
    for delta in cfg["experiment"]["deltas"]:
        rows = _jsonl(generation_root / f"delta_{delta}/responses.jsonl")
        if [r["prompt_id"] for r in rows] != source["ids"]:
            raise ValueError(f"delta={delta} responses do not exactly match frozen E3 prompt ordering")
        condition = dict(phase, out=root / f"verification/delta_{delta}",
                         identity=digest({"e3": state["identity"], "delta": delta,
                           "rows": digest(rows), "kind": "E4-statistical-analysis"}))
        vectors, tokens = score_rows(condition, rows, f"student_delta_{delta}", source["environment"])
        all_conditions[f"delta_{delta}"] = (rows, vectors, tokens)
        summary = tables(condition, {f"delta_{delta}": all_conditions[f"delta_{delta}"]}, groups,
                         condition["out"] / "scores")
        atomic_json(condition["out"] / "summary.json", {"delta": delta, "results": summary,
            "nominal_random_key_null_alpha_not_empirical_fpr": True})
    baseline_condition = dict(phase, out=root / "verification/baseline",
                              identity=digest({"e3": state["identity"], "kind": "E4-baseline-reuse"}))
    baseline_summary = tables(baseline_condition, {"baseline": baseline["student"],
        "clean_control": baseline["clean"]}, groups,
                              baseline_condition["out"] / "scores")
    atomic_json(root / "verification/baseline/summary.json", {"results": baseline_summary,
        "reused_e3_full": True})
    utilities = generation_utility(all_conditions, 1024)
    for delta in cfg["experiment"]["deltas"]:
        atomic_json(generation_root / f"delta_{delta}/utility.json", utilities[f"delta_{delta}"])


def arc_utility(cfg):
    """Score ARC with E3's lm-eval protocol and the same position-wise logits attack."""
    import torch
    import lm_eval
    from importlib import metadata
    from lm_eval.models.huggingface import HFLM
    if not torch.cuda.is_available():
        raise RuntimeError("ARC utility requires the GPU server")
    state = _e3_state(cfg)
    checkpoint = _merged_checkpoint(state)
    root = cfg["paths"]["output"]
    mapping = torch.load(root / "mapping/suppression_mapping.pt", map_location="cpu", weights_only=True)
    mapping = {int(key): (token_ids.to("cuda"), weights.to("cuda"))
               for key, (token_ids, weights) in mapping.items()}
    e3_arc = __import__("klrd_rs.e3full.arc", fromlist=["SETTINGS", "TASK", "DATA_SEED"])
    if metadata.version("lm_eval") != e3_arc.HARNESS_VERSION:
        raise RuntimeError(f"ARC requires the E3-pinned lm_eval=={e3_arc.HARNESS_VERSION}")
    original_call = HFLM._model_call
    rows = []
    e3_cfg = state["cfg"]
    conditions = [("clean", f"pretrained={e3_cfg.model},revision={e3_cfg.revision},dtype=float32,attn_implementation=sdpa,trust_remote_code=False", None),
                  ("baseline", f"pretrained={checkpoint},dtype=float32,attn_implementation=sdpa,trust_remote_code=False", 0)]
    conditions.extend((f"delta_{delta}", f"pretrained={checkpoint},dtype=float32,attn_implementation=sdpa,trust_remote_code=False", delta)
                      for delta in cfg["experiment"]["deltas"])
    for condition_name, model_args, delta in conditions:
        def adjusted_call(self, *args, **kwargs):
            output = original_call(self, *args, **kwargs)
            logits = output[0] if isinstance(output, tuple) else output
            if delta is None or delta == 0:
                return output
            input_ids = args[0] if args else kwargs.get("inps", kwargs.get("input_ids"))
            if input_ids is None:
                raise ValueError("Could not recover ARC input token IDs for SaTML adjustment")
            logits = adjust_positionwise_logits(input_ids, logits, mapping, delta)
            return (logits,) + output[1:] if isinstance(output, tuple) else logits
        HFLM._model_call = adjusted_call
        try:
            cache = root / "private/arc" / condition_name
            cache.mkdir(parents=True, exist_ok=True)
            from klrd_rs.io import read_json, write_json
            binding = {"e3_identity": state["identity"],
                "checkpoint": model_args, "delta": delta,
                "mapping_sha256": hashlib.sha256((root / "mapping/suppression_mapping.pt").read_bytes()).hexdigest()}
            binding_path = cache / "binding.json"
            if binding_path.exists() and read_json(binding_path) != binding:
                raise ValueError(f"ARC cache identity mismatch: {condition_name}")
            if not binding_path.exists():
                write_json(binding_path, binding)
            result = lm_eval.simple_evaluate(model="hf", model_args=model_args,
                tasks=[e3_arc.TASK], num_fewshot=0, batch_size=1, limit=None,
                apply_chat_template=True, log_samples=True, bootstrap_iters=1000,
                random_seed=e3_arc.DATA_SEED, numpy_random_seed=e3_arc.DATA_SEED,
                torch_random_seed=e3_arc.DATA_SEED, fewshot_random_seed=e3_arc.DATA_SEED,
                use_cache=str(cache / "likelihood"))
            task = result["results"][e3_arc.TASK]
            metric = task.get("acc_norm,none")
            if metric is None:
                raise ValueError("ARC harness omitted acc_norm,none")
            sample_rows = result.get("samples", {}).get(e3_arc.TASK, [])
            if len(sample_rows) != cfg["experiment"]["arc_examples"]:
                raise ValueError("ARC sample results do not contain all 1,172 examples; refusing to infer correct count")
            sample_scores = [row.get("acc_norm,none") for row in sample_rows]
            if any(score not in (0, 1, 0.0, 1.0) for score in sample_scores):
                raise ValueError("ARC per-example acc_norm,none decisions are missing or not binary")
            correct_count = int(sum(sample_scores))
            rows.append({"condition": condition_name,
                "delta": delta, "accuracy": float(metric), "accuracy_percent": 100 * float(metric),
                "correct": correct_count, "total": len(sample_rows),
                "backend": "lm_eval HFLM; E3 ARC-Challenge acc_norm,none FP32 batch1"})
        finally:
            HFLM._model_call = original_call
    from klrd_rs.io import write_csv
    root.joinpath("arc").mkdir(parents=True, exist_ok=True)
    write_csv(root / "arc/results.csv", rows, list(rows[0]))
    atomic_json(root / "arc/results.json", {"settings": e3_arc.SETTINGS, "models": rows,
        "attacked_scoring": "HFLM logits adjusted at every predicted continuation token using preceding input token"})


def report(cfg):
    root = cfg["paths"]["output"]
    generation_root = _generation_root(cfg)
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((root / "verification/baseline/summary.json").read_text(encoding="utf-8"))
    deltas = cfg["experiment"]["deltas"]
    summaries = {d: json.loads((root / f"verification/delta_{d}/summary.json").read_text(encoding="utf-8"))["results"]
                 for d in deltas}
    metrics = []
    for q in (20, 50, 100):
        for alpha in cfg["experiment"]["verification_alphas"]:
            metrics.append((f"TPR Q{q} alpha={alpha}", lambda data, q=q, alpha=alpha:
                next((x["rate"] for x in data if x["model"] == "baseline" and x["Q"] == q and x["alpha"] == alpha), None),
                lambda data, q=q, alpha=alpha, d=None: next((x["rate"] for x in data if x["Q"] == q and x["alpha"] == alpha), None)))
    metrics.append(("Wrong-key acceptance mean", lambda data: next((x["wrong_mean"] for x in data if x["model"] == "baseline" and x["Q"] == 100 and x["alpha"] == .01), None),
                    lambda data, d=None: next((x["wrong_mean"] for x in data if x["Q"] == 100 and x["alpha"] == .01), None)))
    e3_phase = _e3_state(cfg)
    e3_summary = json.loads((e3_phase["out"] / "eval1024/utility/generation_summary.json").read_text(encoding="utf-8"))
    utilities = {d: json.loads((generation_root / f"delta_{d}/utility.json").read_text(encoding="utf-8")) for d in deltas}
    metrics.append(("ARC-C accuracy (%)", lambda data: None, lambda data: None))
    metrics.append(("Mean response length (words)", lambda data: e3_summary["student"]["mean_words"],
                    lambda data, d=None: utilities[d]["mean_words"]))
    metrics.append(("Truncation rate (%)", lambda data: 100 * e3_summary["student"]["max_length_truncation_rate"],
                    lambda data, d=None: 100 * utilities[d]["max_length_truncation_rate"]))
    metrics.append(("Severe repetition rate (%)", lambda data: 100 * e3_summary["student"]["severe_repetition_rate"],
                    lambda data, d=None: 100 * utilities[d]["severe_repetition_rate"]))
    metrics.append(("Suppression coverage (%)", lambda data: None, lambda data, d=None: 100 * json.loads(
        (generation_root / f"delta_{d}/diagnostics.json").read_text(encoding="utf-8"))["covered_step_fraction"]))
    arc = None
    try:
        arc = json.loads((root / "arc/results.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    metrics.extend((f"Clean acceptance Q100 alpha={alpha}",
        lambda data, alpha=alpha: next((x["rate"] for x in data if x["model"] == "clean_control" and x["Q"] == 100 and x["alpha"] == alpha), None),
        lambda data, d=None: None) for alpha in cfg["experiment"]["verification_alphas"])
    rows = []
    for label, base_fn, delta_fn in metrics:
        row = {"metric": label, "no_attack": base_fn(baseline["results"])}
        for delta in deltas:
            row[f"delta_{delta}"] = delta_fn(summaries[delta], d=delta)
        if label == "ARC-C accuracy (%)" and arc:
            by_name = {item["condition"]: item["accuracy_percent"] for item in arc["models"]}
            row["no_attack"] = by_name.get("baseline")
            for delta in deltas:
                row[f"delta_{delta}"] = by_name.get(f"delta_{delta}")
            row["clean_control"] = by_name.get("clean")
        rows.append(row)
    if arc:
        by_name = {item["condition"]: f"{item['correct']}/{item['total']}" for item in arc["models"]}
        rows.append({"metric": "ARC-C correct/total", "clean_control": by_name.get("clean"),
            "no_attack": by_name.get("baseline"),
            **{f"delta_{d}": by_name.get(f"delta_{d}") for d in deltas}})
    rows.append({"metric": "Mean generated tokens/response",
        "no_attack": e3_summary["student"]["mean_generated_tokens"],
        **{f"delta_{d}": utilities[d]["mean_generated_tokens"] for d in deltas}})
    q1000 = {}
    files = [("baseline", root / "verification/baseline/scores/per_key_group.csv")]
    files.extend((f"delta_{d}", root / f"verification/delta_{d}/scores/per_key_group.csv") for d in deltas)
    for condition, file in files:
        with file.open(encoding="utf-8", newline="") as stream:
            qrows = list(__import__("csv").DictReader(stream))
        owner = [r for r in qrows if int(r["Q"]) == 1000 and int(r["test_key"]) == 0]
        if len(owner) != 1:
            raise ValueError(f"Expected one exact owner Q=1000 p-value for {condition}")
        pvalue = float(owner[0]["p_value"])
        q1000[condition] = {"p_value": pvalue,
            "decisions": {str(alpha): pvalue <= alpha for alpha in cfg["experiment"]["verification_alphas"]},
            "groups": 1, "Q": 1000}
    from klrd_rs.io import write_csv
    headers = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(report_dir / "results.csv", rows, headers)
    columns = ["Metric", "Clean control", "No attack", *[f"δ={d}" for d in cfg["experiment"]["deltas"]]]
    result_columns = ["metric", "clean_control", "no_attack", *[f"delta_{d}" for d in deltas]]
    cells = []
    for q in cfg["experiment"]["group_sizes"]:
        for alpha in cfg["experiment"]["verification_alphas"]:
            key = f"TPR Q{q} α={alpha}"
            base_item = next((x for x in baseline["results"] if x["model"] == "baseline" and x["Q"] == q and x["alpha"] == alpha), None)
            row = [key, "n/a", f"{base_item['detections']}/{base_item['groups']} ({base_item['rate']:.3f})" if base_item else "n/a"]
            for delta in cfg["experiment"]["deltas"]:
                result = json.loads((root / f"verification/delta_{delta}/summary.json").read_text(encoding="utf-8"))
                item = next((x for x in result["results"] if x["Q"] == q and x["alpha"] == alpha), None)
                row.append(f"{item['detections']}/{item['groups']} ({item['rate']:.3f})" if item else "n/a")
            cells.append(row)
    diagnostics = {str(d): json.loads((generation_root / f"delta_{d}/diagnostics.json").read_text(encoding="utf-8"))
                   for d in cfg["experiment"]["deltas"]}
    lines = ["# Experiment 4 — SaTML statistical bigram suppression", "",
        "Method: **SaTML statistical bigram suppression with reconstructed public calibration assets.** This is not an exact-data reproduction.", "",
        "Calibration is fixed at 128 contexts. The 64-context view is diagnostic only; the final mapping uses all 128. No fingerprint outcome is used to tune mapping or parameters.", "",
        "## Main results", "", "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|"]
    lines.extend("| " + " | ".join(str(row.get(key, "n/a")) if row.get(key) is not None else "n/a"
                                   for key in result_columns) + " |" for row in rows)
    lines += ["", "## Fingerprint verification", "", "| Metric | No attack | " +
        " | ".join(f"δ={d}" for d in deltas) + " |",
        "|" + "|".join(["---"] * (2 + len(deltas))) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in cells)
    lines += ["", "See per-condition CSVs under `verification/` for group decisions, exact p-values, owner and 20 wrong keys.", "",
        "Alpha is the nominal spherical random-key null significance level, not empirical clean FPR. Q=1000 has one group and is descriptive only.", "",
        "## Q=1000 exact owner result", "", "| Condition | Exact p-value | Decision at α=.01 | Decision at α=.0001 |",
        "|---|---:|:---:|:---:|"]
    lines.extend(f"| {name} | {value['p_value']:.8g} | {value['decisions'].get('0.01')} | {value['decisions'].get('0.0001')} |"
                 for name, value in q1000.items())
    lines += ["",
        "## ARC-Challenge", "", ("ARC results are present in `arc/results.csv`." if arc else "ARC utility was not run."), "",
        "## Attack diagnostics", "", "| δ | Previous-token coverage | Expected adjusted steps | Expected nonzero mapped values | Mean expected absolute adjustment |",
        "|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {delta} | {value['covered_step_fraction']:.4f} | {value['adjusted_step_fraction']:.4f} | {value['expected_nonzero_adjustment_value_fraction']:.6f} | {value['mean_expected_absolute_logit_adjustment']:.8g} |"
                 for delta, value in diagnostics.items())
    lines += ["", "Interpret retained TPR together with coverage before claiming robustness.", ""]
    (report_dir / "experiment4_report.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(report_dir / "results.json", {"baseline": baseline, "conditions": rows, "q1000_exact": q1000, "arc": arc})


def run(stage: str, config_path: str):
    cfg = _config(Path(config_path))
    _record_run_config(cfg)
    stages = {"prepare": prepare, "calibration": generate_calibration,
              "mapping": build_mapping, "generation": generate_attacks,
              "verification": verify_attacks, "arc": arc_utility, "report": report}
    if stage == "all":
        for name in ("prepare", "calibration", "mapping", "generation", "verification", "arc", "report"):
            print(f"=== Experiment 4 stage: {name} ===", flush=True)
            stages[name](cfg)
    else:
        stages[stage](cfg)
