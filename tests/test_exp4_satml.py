import ast
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
torch = pytest.importorskip("torch")
from experiment4_satml.core import (assert_tokenizer_compatible, calibration_prompt,
    adjust_positionwise_logits, build_prefix_suffix_plan, choose_calibration_words,
    load_official_mlprints, PrefixCacheInferenceAdapter, prefill_prefix_cache,
    run_cached_next_token_logits, split_cached_prefix, top_seed_words,
    valid_seed_words, validate_output_vocab_sizes)
from experiment4_satml.pipeline import (_cached_hub_revision, _estimate_mapping_eta_seconds,
    _jsonl, _unique_index)

source_path = ROOT / "experiment4_satml/vendor/mlprints/bigram_suppression.py"
tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
processor_node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "BigramSuppressionProcessor")
class _LogitsProcessor:
    pass
namespace = {"torch": torch, "LogitsProcessor": _LogitsProcessor}
exec(compile(ast.Module(body=[processor_node], type_ignores=[]), str(source_path), "exec"), namespace)
BigramSuppressionProcessor = namespace["BigramSuppressionProcessor"]


class FakeTokenizer:
    def __init__(self, vocab, eos_token="<eos>", special_ids=None):
        self.vocab = vocab
        self.special_tokens_map = {"eos_token": eos_token}
        self.all_special_ids = [2] if special_ids is None else list(special_ids)
    def get_vocab(self):
        return dict(self.vocab)
    def encode(self, text, add_special_tokens=False):
        word = text.strip()
        return [self.vocab[word]] if word in self.vocab else [0, 1]


def test_seed_list_keeps_frequency_order_and_removes_duplicates():
    assert top_seed_words("the\n of \n\nthe\nand\n", 5) == ["the", "of", "and"]
    assert top_seed_words("the\n of \n\nthe\nand\n", 3) == ["the", "of"]


def test_seed_token_filter_matches_saTML_leading_space_encoding():
    tokenizer = FakeTokenizer({"one": 4})
    assert valid_seed_words(["one", "two"], tokenizer) == ["one"]


def test_calibration_prompt_has_only_bos_text_and_preceding_word():
    assert calibration_prompt("<bos>", "continuation", "word") == "<bos>continuation word"
    assert calibration_prompt(None, "continuation", "word") == "continuation word"


def test_calibration_word_sampling_is_fixed_and_distinct():
    words = [f"w{i}" for i in range(160)]
    assert choose_calibration_words(words, 128, 42) == choose_calibration_words(words, 128, 42)
    assert len(set(choose_calibration_words(words, 128, 42))) == 128


def test_resume_index_rejects_duplicate_work_items():
    with pytest.raises(ValueError, match="Duplicate index"):
        _unique_index([{"index": 3}, {"index": 3}], "index", "calibration journal")


def test_resume_preserves_and_repairs_only_an_incomplete_final_journal_row(tmp_path):
    path = tmp_path / "rows.jsonl"
    first = b'{"index":0}\n'
    tail = b'{"index":1'
    path.write_bytes(first + tail)
    assert _jsonl(path) == [{"index": 0}]
    assert path.read_bytes() == first
    assert path.with_suffix(".jsonl.incomplete-tail.bak").read_bytes() == tail


def test_tokenizer_compatibility_checks_token_ids_and_special_tokens():
    a = FakeTokenizer({"a": 0, "b": 1})
    b = FakeTokenizer({"a": 0, "b": 1})
    assert assert_tokenizer_compatible(a, b)["vocab_size"] == 2
    with pytest.raises(ValueError, match="token-to-ID"):
        assert_tokenizer_compatible(a, FakeTokenizer({"a": 1, "b": 0}))


def test_tokenizer_compatibility_allows_special_metadata_difference_with_same_vocab():
    attacked = FakeTokenizer({"a": 0, "b": 1}, eos_token="<|im_end|>", special_ids=[2])
    calibration = FakeTokenizer({"a": 0, "b": 1}, eos_token="<|endoftext|>", special_ids=[2, 3])
    result = assert_tokenizer_compatible(attacked, calibration)
    assert result["vocab_size"] == 2
    assert result["special_token_metadata_match"] is False
    assert result["attacked_special_tokens"] != result["calibration_special_tokens"]


def test_cached_hub_revision_comes_from_snapshot_path_without_model_config(monkeypatch, tmp_path):
    import huggingface_hub

    commit = "a" * 40
    snapshot = tmp_path / "snapshots" / commit
    snapshot.mkdir(parents=True)
    calls = {}
    def fake_snapshot_download(repo_id, **kwargs):
        calls["repo_id"] = repo_id
        calls.update(kwargs)
        return str(snapshot)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    assert _cached_hub_revision("Qwen/Qwen2.5-1.5B", "main") == commit
    assert calls["local_files_only"] is True
    assert calls["allow_patterns"] == ["config.json"]


def test_output_vocab_validation_allows_padded_rows_but_requires_all_token_ids_in_bounds():
    tokenizer = FakeTokenizer({"first": 0, "last": 151664})
    info = validate_output_vocab_sizes(151936, 151936, tokenizer)
    assert info["model_output_vocab_size"] == 151936
    assert info["tokenizer_token_count"] == 2
    assert info["max_token_id"] == 151664
    with pytest.raises(ValueError, match="output vocabulary sizes differ"):
        validate_output_vocab_sizes(151936, 151935, tokenizer)
    with pytest.raises(ValueError, match="token ID exceeds model output dimension"):
        validate_output_vocab_sizes(151664, 151664, tokenizer)


def test_mapping_eta_uses_only_newly_completed_shards():
    assert _estimate_mapping_eta_seconds(8.0, 1, 5438) == 43504.0
    assert _estimate_mapping_eta_seconds(80.0, 10, 5400) == 43200.0
    assert _estimate_mapping_eta_seconds(0.0, 0, 5400) is None


def test_cached_next_token_logits_match_full_prompt_with_variable_suffix_lengths():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(17)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=32,
        n_embd=16, n_layer=2, n_head=2, resid_pdrop=0.0,
        embd_pdrop=0.0, attn_pdrop=0.0)).eval()
    prefixes = [[1, 2, 3], [4, 5, 6]]
    suffixes = [[7, 8], [9]]
    cache, prefix_length = prefill_prefix_cache(model, prefixes, "cpu")
    cached = run_cached_next_token_logits(model, cache, prefix_length,
        suffixes, pad_token_id=0, device="cpu")
    with torch.inference_mode():
        full = torch.stack([
            model(input_ids=torch.tensor([prefix + suffix]))
                .logits[0, -1, :]
            for prefix, suffix in zip(prefixes, suffixes)
        ])
    assert torch.allclose(cached, full, atol=1e-5, rtol=1e-5)


def test_cached_logits_use_transformers5_dynamic_cache_constructor(monkeypatch):
    import types
    import transformers

    class DynamicCacheV5:
        def __init__(self, ddp_cache_data):
            self.ddp_cache_data = ddp_cache_data

    monkeypatch.setattr(transformers, "DynamicCache", DynamicCacheV5)
    key = torch.zeros((1, 1, 2, 3))
    value = torch.ones_like(key)
    legacy_cache = ((key, value),)
    calls = []

    class Model:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            batch, width = kwargs["input_ids"].shape
            return types.SimpleNamespace(logits=torch.zeros((batch, width, 5)))

    logits = run_cached_next_token_logits(
        Model(), legacy_cache, prefix_length=2, suffix_ids=[[4], [3, 4]],
        pad_token_id=0, device="cpu")

    assert logits.shape == (2, 5)
    assert isinstance(calls[0]["past_key_values"], DynamicCacheV5)
    assert calls[0]["past_key_values"].ddp_cache_data is legacy_cache


def test_cached_next_token_logits_reject_empty_suffix():
    with pytest.raises(ValueError, match="non-empty suffix"):
        run_cached_next_token_logits(object(), (), 3, [[], []], 0, "cpu")


def test_prefix_suffix_plan_reconstructs_exact_prompt_tokens_for_every_word_and_context():
    class CharacterTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(char) for char in text]

    contexts = ["context one", "another context"]
    words = ["alpha", "beta", "gamma"]
    plan = build_prefix_suffix_plan(CharacterTokenizer(), contexts, words, bos_token="<B>")
    assert plan["prefix_length"] > 0
    assert all(len(prefix) == plan["prefix_length"] for prefix in plan["prefix_ids"])
    for context_index, context in enumerate(contexts):
        for word in words:
            full_ids = [ord(char) for char in f"<B>{context} {word}"]
            suffix = plan["suffix_ids_by_word"][word][context_index]
            assert plan["prefix_ids"][context_index] + suffix == full_ids


def test_split_cached_prefix_returns_none_for_mismatch_or_empty_suffix():
    assert split_cached_prefix([1, 2, 3, 4], [1, 2]) == [3, 4]
    assert split_cached_prefix([1, 9, 3], [1, 2]) is None
    assert split_cached_prefix([1, 2], [1, 2]) is None


def test_cached_inference_adapter_routes_model_order_and_falls_back_on_prompt_mismatch(monkeypatch):
    import experiment4_satml.core as core

    class CharacterTokenizer:
        offset = 0
        def encode(self, text, add_special_tokens=False):
            return [ord(char) + self.offset for char in text]

    tokenizer = CharacterTokenizer()
    contexts, words = ["ctx zero", "ctx one"], ["alpha", "beta"]
    plan = build_prefix_suffix_plan(tokenizer, contexts, words)
    attacked, calibration = object(), object()
    fallback_calls = []
    def fallback(model, tokenizer, prompts, **kwargs):
        fallback_calls.append((model, prompts))
        return {"scores": [torch.full((len(prompts), 1), -3.0)]}
    def cached(model, cache, prefix_length, suffixes, pad_token_id, device):
        fill = 11.0 if model is attacked else 22.0
        return torch.full((len(suffixes), 1), fill)
    monkeypatch.setattr(core, "run_cached_next_token_logits", cached)
    caches = {"attacked": [object()], "calibration": [object()]}
    adapter = PrefixCacheInferenceAdapter(attacked, calibration, tokenizer,
        contexts, words, plan, caches, batch_size=2,
        fallback_inference=fallback, pad_token_id=0, device="cpu")
    adapter.begin_word("alpha")
    prompts = [f"{context} alpha" for context in contexts]
    attacked_scores = adapter(attacked, tokenizer, prompts)
    calibration_scores = adapter(calibration, tokenizer, prompts)
    assert attacked_scores["scores"][0][:, 0].tolist() == [11.0, 11.0]
    assert calibration_scores["scores"][0][:, 0].tolist() == [22.0, 22.0]
    mismatch = adapter(attacked, tokenizer, ["different prompt"])
    assert mismatch["scores"][0][0, 0].item() == -3.0
    assert fallback_calls == [(attacked, ["different prompt"])]
    tokenizer.offset = 1
    adapter.begin_word("alpha")
    exact_prompts = [f"{context} alpha" for context in contexts]
    changed_tokenization = adapter(attacked, tokenizer, exact_prompts)
    assert changed_tokenization["scores"][0][0, 0].item() == -3.0
    assert fallback_calls[-1] == (attacked, exact_prompts)


def test_cached_inference_adapter_runtime_parity_gate_matches_full_forward():
    from transformers import GPT2Config, GPT2LMHeadModel

    class CharacterTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [(ord(char) % 31) + 1 for char in text]

    torch.manual_seed(23)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=64,
        n_embd=16, n_layer=2, n_head=2, resid_pdrop=0.0,
        embd_pdrop=0.0, attn_pdrop=0.0)).eval()
    tokenizer = CharacterTokenizer()
    contexts, words = ["ctx one", "a longer ctx"], ["alpha", "beta"]
    plan = build_prefix_suffix_plan(tokenizer, contexts, words)
    cache, _ = prefill_prefix_cache(model, plan["prefix_ids"], "cpu")
    def full_inference(model, tokenizer, prompts, **kwargs):
        rows = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
        width = max(map(len, rows))
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, :len(row)] = torch.tensor(row)
            mask[index, :len(row)] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=mask).logits
        last = mask.sum(1) - 1
        return {"scores": [logits[torch.arange(len(rows)), last, :]]}
    adapter = PrefixCacheInferenceAdapter(model, model, tokenizer, contexts, words,
        plan, {"attacked": [cache], "calibration": [cache]}, batch_size=2,
        fallback_inference=full_inference, pad_token_id=0, device="cpu")
    result = adapter.verify_parity(words)
    assert result["attacked"]["max_abs_error"] < 1e-5
    assert result["calibration"]["max_abs_error"] < 1e-5


@pytest.mark.parametrize("error, passes", [(0.0390625, True), (0.051, False)])
def test_fp16_parity_tolerance_accepts_roundoff_but_still_fails_closed(
        monkeypatch, error, passes):
    import experiment4_satml.core as core

    class HalfModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))

    model_a, model_c = HalfModel(), HalfModel()
    cache = ((torch.zeros((1, 1)), torch.zeros((1, 1))),)
    monkeypatch.setattr(core, "run_cached_next_token_logits",
        lambda *args, **kwargs: torch.tensor([[error, 0., 0., 0.]]))
    def full_inference(model, tokenizer, prompts, **kwargs):
        return {"scores": [torch.zeros((len(prompts), 4))]}
    adapter = PrefixCacheInferenceAdapter(
        model_a, model_c, FakeTokenizer({"the": 1}), ["context"], ["the"],
        {"prefix_ids": [[1]], "prefix_length": 1,
         "suffix_ids_by_word": {"the": [[2]]}},
        {"attacked": [cache], "calibration": [cache]}, batch_size=1,
        fallback_inference=full_inference, pad_token_id=0, device="cpu")

    if passes:
        assert adapter.verify_parity(["the"])["calibration"]["max_abs_error"] == error
    else:
        with pytest.raises(RuntimeError, match="failed parity"):
            adapter.verify_parity(["the"])


def test_official_processor_keeps_signed_adjustment_and_delta_zero():
    mapping = {9: (torch.tensor([1, 2]), torch.tensor([0.5, -0.25]))}
    input_ids = torch.tensor([[7, 9]])
    original = torch.tensor([[0.0, 10.0, 10.0, 0.0]])
    adjusted = BigramSuppressionProcessor(mapping, 4.0)(input_ids, original.clone())
    assert torch.equal(adjusted, torch.tensor([[0.0, 8.0, 11.0, 0.0]]))
    zero = BigramSuppressionProcessor(mapping, 0.0)(input_ids, original.clone())
    assert torch.equal(zero, original)


def test_official_processor_leaves_unmapped_previous_tokens_unchanged():
    mapping = {9: (torch.tensor([1]), torch.tensor([0.5]))}
    scores = torch.tensor([[0.0, 10.0, 0.0]])
    actual = BigramSuppressionProcessor(mapping, 4.0)(torch.tensor([[8]]), scores.clone())
    assert torch.equal(actual, scores)


def test_positionwise_arc_adjustment_scores_candidate_tokens_using_previous_token():
    mapping = {8: (torch.tensor([1, 2]), torch.tensor([0.5, -0.25]))}
    inputs = torch.tensor([[8, 1, 5]])
    logits = torch.zeros((1, 3, 4))
    logits[0, :, 1:3] = 10.0
    adjusted = adjust_positionwise_logits(inputs, logits, mapping, 2)
    assert torch.equal(adjusted[0, 0], torch.tensor([0.0, 9.0, 10.5, 0.0]))
    assert torch.equal(adjusted[0, 1], torch.tensor([0.0, 10.0, 10.0, 0.0]))


def test_official_source_loader_uses_checked_in_file_without_package_wide_import():
    module = load_official_mlprints({})
    assert Path(module.source_path).resolve() == source_path.resolve()
    mapping = {9: (torch.tensor([1]), torch.tensor([0.5]))}
    assert module.BigramSuppressionProcessor(mapping, 2)(torch.tensor([[9]]),
        torch.tensor([[0.0, 10.0]]))[0, 1].item() == 9.0


def test_official_statistical_mapping_matches_reference_on_synthetic_logits():
    module = load_official_mlprints({})
    source_model = object()
    calibration_model = object()
    model_logits = torch.tensor([
        [2., 1., 0., 3., -1.], [1., 2., 0., -2., 2.],
        [3., -1., 1., 1., 0.], [-1., 3., 2., 0., 1.],
    ])
    calibration_logits = torch.tensor([
        [1., 2., 1., 0., 0.], [0., 1., 2., 1., 0.],
        [2., 0., 0., 1., 2.], [1., 0., 1., 2., 0.],
    ])
    calls = {"index": 0}
    globals_ = module._namespace
    original = globals_["run_inference"]
    def synthetic_inference(model, tokenizer, prompts, **kwargs):
        batch_index = calls["index"] // 2
        is_model = calls["index"] % 2 == 0
        calls["index"] += 1
        values = model_logits if is_model else calibration_logits
        start = batch_index * 2
        return {"scores": [values[start:start + len(prompts)].clone()]}
    globals_["run_inference"] = synthetic_inference
    class Tokenizer:
        all_special_ids = [0]
        bos_token = ""
        def encode(self, text, add_special_tokens=False):
            return [9]
    try:
        attack = module.BigramSuppressionAttackModel(source_model, Tokenizer(), calibration_model,
            seed_words=["word"], calibration_texts=["ctx0", "ctx1", "ctx2", "ctx3"],
            probability_threshold=1e-10, clip_quantile=.999, suppression_delta=1., batch_size=2)
    finally:
        globals_["run_inference"] = original
    actual_ids, actual = attack.processor.mapping[9]
    model_logprobs = model_logits.log_softmax(-1)
    ratios = model_logprobs - calibration_logits.log_softmax(-1)
    ratios[model_logprobs <= math.log(1e-10)] = torch.nan
    counts = ratios.isfinite().sum(0)
    maximum = ratios.nan_to_num(nan=-torch.inf).amax(0)
    mean = ratios.nan_to_num(nan=0.).sum(0) / counts
    centered = (ratios - mean).nan_to_num(nan=0.)
    standard_error = (centered.square().sum(0) / (counts - 1)).sqrt() / counts.sqrt()
    zscore = mean / standard_error
    maximum_bound = maximum[maximum.isfinite()].abs().quantile(.999)
    maximum = maximum.clamp(-maximum_bound, maximum_bound) / maximum_bound
    zscore_bound = zscore[zscore.isfinite()].abs().quantile(.999)
    zscore = zscore.clamp(-zscore_bound, zscore_bound) / zscore_bound
    scores = torch.hypot(maximum, zscore)
    scores = torch.where((maximum > 0) & (zscore > 0), scores, -scores)
    scores[0] = torch.nan
    expected_ids = scores.isfinite().nonzero().flatten()
    assert torch.equal(actual_ids, expected_ids)
    assert torch.allclose(actual, scores[expected_ids], rtol=1e-6, atol=1e-6)
