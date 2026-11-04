from __future__ import annotations

import hashlib
import ast
import math
import json
import os
import types
import urllib.request
from pathlib import Path

SEED_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"


def top_seed_words(source_text: str, count: int = 6000) -> list[str]:
    """Filter only the first ``count`` source rows; never backfill later words."""
    result, seen = [], set()
    for raw in source_text.splitlines()[:count]:
        word = raw.strip()
        if word and word not in seen:
            result.append(word)
            seen.add(word)
    if not result:
        raise ValueError("Seed source has no non-empty words in the retained rows")
    return result


def valid_seed_words(words: list[str], tokenizer) -> list[str]:
    return [word for word in words if len(tokenizer.encode(f" {word}", add_special_tokens=False)) == 1]


def calibration_prompt(bos: str | None, text: str, word: str) -> str:
    return f"{bos or ''}{text} {word}"


def choose_calibration_words(words: list[str], count: int = 128, seed: int = 42) -> list[str]:
    import random
    if len(set(words)) < count:
        raise ValueError(f"Need at least {count} distinct valid seed words; found {len(set(words))}")
    return random.Random(seed).sample(list(dict.fromkeys(words)), count)


def assert_tokenizer_compatible(attacked, calibration) -> dict:
    """Require identical token IDs; record special-token annotations for audit.

    Experiment 4 encodes prompts with the attacked tokenizer for both models.
    The independent calibration model's special-token annotations therefore do
    not affect the input IDs or output-logit alignment; its vocabulary mapping
    must still match exactly.
    """
    a_vocab, c_vocab = attacked.get_vocab(), calibration.get_vocab()
    if len(a_vocab) != len(c_vocab):
        raise ValueError(f"Tokenizer vocabulary sizes differ: attacked={len(a_vocab)}, calibration={len(c_vocab)}")
    if a_vocab != c_vocab:
        differing = next((token for token in a_vocab.keys() | c_vocab.keys()
                          if a_vocab.get(token) != c_vocab.get(token)), "<unknown>")
        raise ValueError(f"Tokenizer token-to-ID mappings differ (first token: {differing!r})")
    a_special = dict(attacked.special_tokens_map)
    c_special = dict(calibration.special_tokens_map)
    a_ids = sorted(int(token_id) for token_id in attacked.all_special_ids)
    c_ids = sorted(int(token_id) for token_id in calibration.all_special_ids)
    return {"vocab_size": len(a_vocab), "vocabulary_sha256": hashlib.sha256(
        json.dumps(sorted(a_vocab.items()), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "special_token_metadata_match": a_special == c_special and a_ids == c_ids,
        "special_tokens": a_special, "all_special_ids": a_ids,
        "attacked_special_tokens": a_special, "calibration_special_tokens": c_special,
        "attacked_all_special_ids": a_ids, "calibration_all_special_ids": c_ids,
        "attacked_tokenizer_revision": getattr(attacked, "init_kwargs", {}).get("_commit_hash"),
        "calibration_tokenizer_revision": getattr(calibration, "init_kwargs", {}).get("_commit_hash")}


def validate_output_vocab_sizes(attacked_vocab_size: int, calibration_vocab_size: int,
                                tokenizer) -> dict:
    """Validate aligned model logits while allowing padded output vocabulary rows."""
    attacked_vocab_size = int(attacked_vocab_size)
    calibration_vocab_size = int(calibration_vocab_size)
    if attacked_vocab_size != calibration_vocab_size:
        raise ValueError(
            "Attacked and calibration model output vocabulary sizes differ: "
            f"attacked={attacked_vocab_size}, calibration={calibration_vocab_size}")
    token_ids = [int(token_id) for token_id in tokenizer.get_vocab().values()]
    if not token_ids:
        raise ValueError("Tokenizer vocabulary is empty")
    max_token_id = max(token_ids)
    if min(token_ids) < 0 or max_token_id >= attacked_vocab_size:
        raise ValueError(
            "Tokenizer token ID exceeds model output dimension: "
            f"min_id={min(token_ids)}, max_id={max_token_id}, "
            f"model_output_vocab_size={attacked_vocab_size}")
    return {"model_output_vocab_size": attacked_vocab_size,
            "tokenizer_token_count": len(token_ids), "max_token_id": max_token_id}


def prefill_prefix_cache(model, prefix_ids: list[list[int]], device):
    """Run equal-length prompt prefixes once and return reusable past-key-values."""
    import torch
    if not prefix_ids or not prefix_ids[0]:
        raise ValueError("Prefix cache requires non-empty token prefixes")
    prefix_length = len(prefix_ids[0])
    if any(len(row) != prefix_length for row in prefix_ids):
        raise ValueError("Prefix cache rows must have equal token lengths")
    input_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True)
    cache = getattr(outputs, "past_key_values", None)
    if cache is None:
        raise RuntimeError("Model did not return past_key_values for prefix cache")
    # Store immutable tensors rather than the mutable Cache object returned by
    # Transformers. A fresh Cache wrapper is reconstructed for each suffix call.
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    elif hasattr(cache, "layers"):
        cache = tuple((layer.keys, layer.values) for layer in cache.layers)
    if not isinstance(cache, (tuple, list)) or not cache:
        raise TypeError(f"Unsupported past_key_values type: {type(cache).__name__}")
    return tuple(tuple(layer) for layer in cache), prefix_length


def run_cached_next_token_logits(model, cache, prefix_length: int,
                                 suffix_ids: list[list[int]], pad_token_id: int,
                                 device):
    """Return logits after each exact suffix using a reusable shared prefix cache."""
    import torch
    if not suffix_ids or any(not row for row in suffix_ids):
        raise ValueError("Cached inference requires a non-empty suffix for every row")
    if prefix_length <= 0:
        raise ValueError("Cached inference requires a positive prefix length")
    batch_size = len(suffix_ids)
    lengths = torch.tensor([len(row) for row in suffix_ids], dtype=torch.long, device=device)
    suffix_width = int(lengths.max().item())
    input_ids = torch.full((batch_size, suffix_width), int(pad_token_id),
                           dtype=torch.long, device=device)
    suffix_mask = torch.zeros((batch_size, suffix_width), dtype=torch.long, device=device)
    for row_index, row in enumerate(suffix_ids):
        input_ids[row_index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        suffix_mask[row_index, :len(row)] = 1
    prefix_mask = torch.ones((batch_size, int(prefix_length)), dtype=torch.long, device=device)
    attention_mask = torch.cat((prefix_mask, suffix_mask), dim=1)
    position_ids = torch.arange(int(prefix_length), int(prefix_length) + suffix_width,
                                dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    # Transformers 5 removed legacy tuple cache inputs. Wrap the stored tensors
    # in the current Cache API, while retaining compatibility with v4 models.
    model_cache = cache
    if isinstance(cache, (tuple, list)):
        try:
            from transformers import DynamicCache
            import inspect
            parameters = inspect.signature(DynamicCache).parameters
            if "ddp_cache_data" in parameters:
                model_cache = DynamicCache(ddp_cache_data=cache)
            elif hasattr(DynamicCache, "from_legacy_cache"):
                model_cache = DynamicCache.from_legacy_cache(cache)
            else:
                model_cache = DynamicCache(cache)
        except (ImportError, TypeError, ValueError):
            # Older model families accept the legacy tuple directly.
            model_cache = cache
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=model_cache, use_cache=False)
    last_positions = lengths - 1
    return outputs.logits[torch.arange(batch_size, device=device), last_positions, :]


def split_cached_prefix(full_ids: list[int], prefix_ids: list[int]) -> list[int] | None:
    """Return the exact non-empty suffix, or None when a prompt cannot reuse cache."""
    if len(full_ids) <= len(prefix_ids) or full_ids[:len(prefix_ids)] != prefix_ids:
        return None
    return list(full_ids[len(prefix_ids):])


def _encode_text_batch(tokenizer, prompts: list[str]) -> list[list[int]]:
    if callable(tokenizer):
        encoded = tokenizer(prompts, add_special_tokens=False)
        return [[int(token_id) for token_id in row] for row in encoded["input_ids"]]
    return [[int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False)]
            for prompt in prompts]


def _longest_common_token_prefix(sequences: list[list[int]]) -> list[int]:
    if not sequences or not sequences[0]:
        raise ValueError("Cannot find a shared prefix in empty token sequences")
    limit = min(len(sequence) for sequence in sequences)
    length = 0
    while length < limit and all(sequence[length] == sequences[0][length]
                                  for sequence in sequences[1:]):
        length += 1
    return sequences[0][:length]


def build_prefix_suffix_plan(tokenizer, contexts: list[str], words: list[str],
                             bos_token: str | None = None) -> dict:
    """Find an exact shared prefix and retain each full prompt's suffix token IDs."""
    if not contexts or not words or len(set(words)) != len(words):
        raise ValueError("Prefix plan requires contexts and distinct seed words")
    bos = bos_token or ""
    context_prefix_candidates = []
    for context_index, context in enumerate(contexts):
        prompts = [calibration_prompt(bos, context, word) for word in words]
        token_rows = _encode_text_batch(tokenizer, prompts)
        if len(token_rows) != len(words):
            raise ValueError("Tokenizer returned a different number of prompt rows")
        context_prefix_candidates.append(_longest_common_token_prefix(token_rows))
        if (context_index + 1) % 8 == 0 or context_index + 1 == len(contexts):
            print(f"Cache prefix scan: contexts {context_index + 1}/{len(contexts)}", flush=True)
    prefix_length = min(map(len, context_prefix_candidates))
    prefix_ids = [candidate[:prefix_length] for candidate in context_prefix_candidates]
    suffix_ids_by_word = {word: [] for word in words}
    for context_index, context in enumerate(contexts):
        prompts = [calibration_prompt(bos, context, word) for word in words]
        token_rows = _encode_text_batch(tokenizer, prompts)
        prefix = prefix_ids[context_index]
        for word, full_ids in zip(words, token_rows):
            suffix = split_cached_prefix(full_ids, prefix)
            if suffix is None:
                raise ValueError(f"Prompt tokens do not share the scanned prefix at context {context_index}, word {word!r}")
            suffix_ids_by_word[word].append(suffix)
    return {"prefix_ids": prefix_ids, "prefix_length": prefix_length,
            "suffix_ids_by_word": suffix_ids_by_word}


def select_prefix_cache_rows(cache, row_indices: list[int], device):
    """Select a small non-mutating batch from a legacy per-layer KV tuple."""
    import torch
    indices = torch.tensor(row_indices, dtype=torch.long, device=device)
    return tuple(tuple(tensor.index_select(0, indices) if torch.is_tensor(tensor) else tensor
                       for tensor in layer) for layer in cache)


class PrefixCacheInferenceAdapter:
    """Provide the upstream attack's run_inference contract using prefix KV caches."""

    def __init__(self, attacked_model, calibration_model, tokenizer, contexts, words,
                 plan, caches, batch_size, fallback_inference, pad_token_id, device,
                 bos_token=None):
        self.models = {"attacked": attacked_model, "calibration": calibration_model}
        self.tokenizer = tokenizer
        self.contexts = contexts
        self.words = set(words)
        self.plan = plan
        self.caches = caches
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("Cache adapter batch_size must be positive")
        self.fallback_inference = fallback_inference
        self.pad_token_id = int(pad_token_id)
        self.device = device
        self.bos = bos_token or ""
        self.active_word = None
        self.call_index = 0

    def begin_word(self, word: str):
        if word not in self.words:
            raise ValueError(f"No prefix cache plan for seed word {word!r}")
        self.active_word = word
        self.call_index = 0

    def __call__(self, model, tokenizer, prompts, **kwargs):
        if self.active_word is None:
            raise RuntimeError("Call begin_word before cached inference")
        side = "attacked" if self.call_index % 2 == 0 else "calibration"
        expected_model = self.models[side]
        if model is not expected_model:
            raise RuntimeError(f"Upstream scorer call order mismatch: expected {side} model")
        batch_index = self.call_index // 2
        start = batch_index * self.batch_size
        end = start + len(prompts)
        self.call_index += 1
        if end > len(self.contexts):
            print(f"Prefix cache context range {start}:{end} is out of bounds; using full inference.",
                  flush=True)
            return self.fallback_inference(model, tokenizer, prompts, **kwargs)
        expected = [calibration_prompt(self.bos, self.contexts[index], self.active_word)
                    for index in range(start, end)]
        if list(prompts) != expected:
            print(f"Prefix cache prompt order mismatch for {self.active_word!r}; using full inference.",
                  flush=True)
            return self.fallback_inference(model, tokenizer, prompts, **kwargs)
        if batch_index >= len(self.caches[side]):
            print(f"Prefix cache chunk {batch_index} missing for {side}; using full inference.",
                  flush=True)
            return self.fallback_inference(model, tokenizer, prompts, **kwargs)
        prefix_rows = self.plan["prefix_ids"][start:end]
        token_rows = _encode_text_batch(tokenizer, list(prompts))
        suffixes = [split_cached_prefix(full_ids, prefix)
                    for full_ids, prefix in zip(token_rows, prefix_rows)]
        if len(suffixes) != len(prompts) or any(suffix is None for suffix in suffixes):
            print(f"Prefix token IDs changed for {self.active_word!r}; using full inference.",
                  flush=True)
            return self.fallback_inference(model, tokenizer, prompts, **kwargs)
        logits = run_cached_next_token_logits(model, self.caches[side][batch_index],
            self.plan["prefix_length"], suffixes, self.pad_token_id, self.device)
        return {"scores": [logits]}

    def verify_parity(self, words: list[str], atol: float = 1e-4,
                      fp16_atol: float = 5e-2) -> dict:
        """Compare cached and full-prompt logits before mapping can write shards."""
        import torch
        sample_count = min(8, len(self.contexts))
        if sample_count <= 0:
            raise ValueError("Cannot verify cache parity without contexts")
        indices = sorted({round(index * (len(self.contexts) - 1) / max(1, sample_count - 1))
                          for index in range(sample_count)})
        grouped_indices = {}
        for context_index in indices:
            chunk_index = context_index // self.batch_size
            grouped_indices.setdefault(chunk_index, []).append(context_index)
        results = {}
        for side, model in self.models.items():
            model_errors = []
            selected_caches = {}
            local_rows = {}
            for chunk_index, context_indices in grouped_indices.items():
                chunk_start = chunk_index * self.batch_size
                rows = [index - chunk_start for index in context_indices]
                selected_caches[chunk_index] = select_prefix_cache_rows(
                    self.caches[side][chunk_index], rows, self.device)
                local_rows[chunk_index] = context_indices
            params = next(model.parameters(), None)
            low_precision = params is not None and params.dtype in (
                torch.float16, torch.bfloat16)
            tolerance = fp16_atol if low_precision else atol
            relative_tolerance = 5e-3 if low_precision else atol
            for word in words:
                if word not in self.words:
                    raise ValueError(f"Parity word absent from cache plan: {word!r}")
                for chunk_index, context_indices in local_rows.items():
                    suffixes = [self.plan["suffix_ids_by_word"][word][index]
                                for index in context_indices]
                    cached_logits = run_cached_next_token_logits(model,
                        selected_caches[chunk_index], self.plan["prefix_length"],
                        suffixes, self.pad_token_id, self.device)
                    prompts = [calibration_prompt(self.bos, self.contexts[index], word)
                               for index in context_indices]
                    full_result = self.fallback_inference(model, self.tokenizer, prompts,
                        apply_chat_template=False, max_new_tokens=1, do_sample=False,
                        output_scores=True)
                    full_logits = full_result["scores"][0]
                    difference = (cached_logits.float() - full_logits.float()).abs()
                    if not torch.allclose(cached_logits.float(), full_logits.float(),
                                          atol=tolerance, rtol=relative_tolerance):
                        max_error = float(difference.max().item())
                        raise RuntimeError(
                            f"Prefix-cache logits failed parity for {side} model, word={word!r}: "
                            f"max_abs_error={max_error:.6g}, atol={tolerance:.6g}, "
                            f"rtol={relative_tolerance:.6g}")
                    model_errors.append(float(difference.max().item()))
            results[side] = {"words_checked": len(words), "contexts_per_word": sample_count,
                             "max_abs_error": max(model_errors, default=0.0),
                             "atol_rtol": {"atol": tolerance, "rtol": relative_tolerance}}
        return results


def load_official_mlprints(config: dict):
    """Load vendored upstream attack classes without importing MLPrints modules.

    The original source is copied into this project under vendor/mlprints.
    Current MLPrints package initialization imports its entire inference stack
    and requires Transformers 5. The E3 server environment may use an older
    compatible release, so we execute the vendored attack definitions' AST and
    supply only a small HF next-token-logit adapter.
    """
    attack = Path(__file__).resolve().parent / "vendor" / "mlprints" / "bigram_suppression.py"
    if not attack.is_file():
        raise FileNotFoundError(f"Official SaTML source not found: {attack}")
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class AttackModel:
        def __init__(self, model, tokenizer):
            self.model, self.tokenizer = model, tokenizer

    def run_inference(model, tokenizer, prompts, **kwargs):
        if kwargs.get("apply_chat_template") is not False or kwargs.get("max_new_tokens") != 1:
            raise ValueError("SaTML calibration adapter only supports raw one-step prompts")
        rows = [tokenizer.encode(text, add_special_tokens=False) for text in prompts]
        if not rows or any(not row for row in rows):
            raise ValueError("SaTML calibration received an empty prompt/token sequence")
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, :len(row)] = torch.tensor(row, dtype=torch.long)
            mask[index, :len(row)] = 1
        device = model.get_input_embeddings().weight.device
        ids, mask = ids.to(device), mask.to(device)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        last = mask.sum(dim=1) - 1
        scores = logits[torch.arange(len(prompts), device=device), last, :]
        return {"scores": [scores]}

    namespace = {"math": math, "torch": torch, "LogitsProcessor": LogitsProcessor,
        "LogitsProcessorList": LogitsProcessorList, "AttackModel": AttackModel,
        "run_inference": run_inference, "load_model": None, "load_tokenizer": None,
        "resolve_cached_attack_asset": None}
    tree = ast.parse(attack.read_text(encoding="utf-8"), filename=str(attack))
    wanted = {"BigramSuppressionProcessor", "BigramSuppressionAttackModel", "bigram_suppression"}
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in wanted]
    if {node.name for node in selected} != wanted:
        raise ValueError(f"Official source definitions missing in {attack}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(attack), "exec"), namespace)

    class SourceModule:
        def __init__(self):
            object.__setattr__(self, "_namespace", namespace)
            for name in wanted:
                setattr(self, name, namespace[name])
            self.source_path = str(attack)
            self.inference_adapter = "HF forward next-token logits; no chat template"
            self.run_inference = namespace["run_inference"]

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name in self._namespace:
                self._namespace[name] = value

    return SourceModule()


def adjust_positionwise_logits(input_ids, logits, mapping, delta):
    """Apply the SaTML signed bigram adjustment to each scored next-token position."""
    for position in range(max(0, input_ids.shape[1] - 1)):
        for row, previous_id in enumerate(input_ids[:, position].tolist()):
            if previous_id not in mapping:
                continue
            token_ids, weights = mapping[previous_id]
            token_ids = token_ids.to(logits.device)
            logits[row, position, token_ids] -= float(delta) * weights.to(
                logits.device, dtype=logits.dtype)
    return logits


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def download_seed_source(path: Path) -> tuple[str, str]:
    """Fetch only the small public word list, only when the configured file is absent."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        urllib.request.urlretrieve(SEED_URL, tmp)
        os.replace(tmp, path)
    raw = path.read_bytes()
    return raw.decode("utf-8-sig"), hashlib.sha256(raw).hexdigest()
