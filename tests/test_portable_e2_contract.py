import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from klrd_rs.portable.profile import ModelProfile
from klrd_rs.portable.e2 import (
    LlamaTextVerifier,
    build_e2_groups,
    encode_visible_response,
    prepare_e2_cohort,
    wrong_key_indices,
    run_e2,
)
from klrd_rs.portable.vllm_adapter import (
    engine_options,
    exact_final_norm,
    register_profile_architecture,
    runtime_identity,
)


def test_registration_is_lazy_and_profile_specific(monkeypatch):
    calls = []
    fake_vllm = ModuleType("vllm")
    fake_vllm.__version__ = "0.32.0"
    fake_vllm.ModelRegistry = SimpleNamespace(register_model=lambda *args: calls.append(args))
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    profile = ModelProfile.load("llama32_3b")
    register_profile_architecture(profile)
    assert calls == [("KLRDRSLlamaForCausalLM", "klrd_rs.portable.vllm_llama:KLRDRSLlamaForCausalLM")]
    assert all("Qwen" not in value for pair in calls for value in pair)


def test_runtime_metadata_and_engine_options_need_no_vllm_import(monkeypatch):
    profile = ModelProfile.load("llama32_3b")
    identity = runtime_identity(profile)
    assert identity["vllm_version"] == "0.32.0"
    options = engine_options(profile, {"model_revision": "a" * 40, "max_model_len": 4096}, eager=True)
    assert options["hf_overrides"]["architectures"] == ["KLRDRSLlamaForCausalLM"]
    assert options["revision"] == options["tokenizer_revision"] == "a" * 40
    assert options["max_model_len"] == 4096
    assert options["dtype"] == "float32" and options["enforce_eager"] is True


def test_final_norm_rotates_combined_residual_exactly_once_before_native_norm():
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    residual = torch.tensor([[0.5, -0.25, 0.75, -0.5]])
    p, q = torch.eye(4)[:, :2:1], torch.eye(4)[:, 2:4]
    angles = torch.tensor([0.2, -0.3])

    class TraceNorm:
        def __init__(self):
            self.seen = None

        def __call__(self, value):
            self.seen = value.clone()
            return value * 2

    native = TraceNorm()
    output, combined = exact_final_norm(hidden, residual, p, q, angles, native)
    expected_combined = hidden + residual
    assert torch.equal(combined, expected_combined)
    assert torch.allclose(native.seen.norm(dim=-1), expected_combined.norm(dim=-1), atol=1e-6)
    assert torch.allclose(output, native.seen * 2)


def test_rotation_helper_preserves_native_residual_contract_when_unfused():
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    p, q = torch.eye(4)[:, :2], torch.eye(4)[:, 2:4]
    angles = torch.zeros(2)

    class IdentityNorm:
        def __call__(self, value):
            return value

    output = exact_final_norm(hidden, None, p, q, angles, IdentityNorm())
    assert torch.equal(output, hidden)


def test_engine_requires_frozen_revision_and_does_not_add_temperature_to_model():
    profile = ModelProfile.load("llama32_3b")
    with pytest.raises(ValueError, match="immutable"):
        engine_options(profile, {"max_model_len": 512})
    source = open("klrd_rs/portable/vllm_llama.py", encoding="utf-8").read()
    assert "class KLRDRSLlamaForCausalLM(LlamaForCausalLM)" in source
    assert "logits /" not in source and " / tau" not in source
    assert "super().compute_logits(hidden_states)" in source
    assert "vocab_size" not in source and "logits[:," not in source
    assert "model.lm_head" not in source and "self.lm_head" not in source
    e2_source = open("klrd_rs/portable/e2.py", encoding="utf-8").read()
    assert "eos_token_id" in e2_source and "stop_token_ids=eos_ids" in e2_source
    assert "HF generation" not in e2_source


class FakeTokenizer:
    all_special_ids = [0, 2, 9]

    def encode(self, text, add_special_tokens=False):
        if text.startswith("a sufficiently long"):
            return list(range(20))
        return [4, 5, 2, 9]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [10, 11, 12]


def test_visible_text_encoding_uses_prefix_len_minus_one_and_excludes_special_eos():
    encoded = encode_visible_response(FakeTokenizer(), "prompt", "visible answer")
    assert encoded["input_ids"] == [10, 11, 12, 4, 5, 2, 9]
    assert encoded["positions"] == [2, 3]
    assert encoded["targets"] == [4, 5]


def test_cohort_and_same_budget_groups_are_unique_and_disjoint():
    tokenizer = FakeTokenizer()
    profile = ModelProfile.load("llama32_3b")
    rows = {
        "test_gen": [{"id": f"t{i}", "prompt": f"a sufficiently long test prompt number {i}"} for i in range(24)],
        "train_gen": [{"id": f"r{i}", "prompt": f"a sufficiently long train prompt number {i}"} for i in range(24)],
    }
    selected = prepare_e2_cohort(rows, tokenizer, profile, excluded={"ids": set(), "exact": set(), "normalized": set(), "legacy_normalized": set()}, count=20)
    assert len(selected) == len({row["prompt_id"] for row in selected}) == 20
    assert {row["source_split"] for row in selected} == {"test_gen"}
    grouped = build_e2_groups([{"prompt_id": str(i)} for i in range(200)], seed=7)
    q100 = [group for group in grouped if group["Q"] == 100]
    assert len(q100) == 2
    assert len(set(q100[0]["prompt_ids"]) & set(q100[1]["prompt_ids"])) == 0


def test_wrong_key_mapping_is_deterministic_unique_and_excludes_owner():
    profile = ModelProfile.load("llama32_3b")
    first = wrong_key_indices(profile, owner_index=3)
    second = wrong_key_indices(profile, owner_index=3)
    assert first == second
    assert len(first) == 20 and len(set(first)) == 20 and 3 not in first


def test_one_score_vector_drives_all_key_scores():
    verifier = object.__new__(LlamaTextVerifier)
    vector = np.arange(1, 33, dtype=np.float64)
    bank = np.random.default_rng(9).normal(size=(24, 32))
    bank /= np.linalg.norm(bank, axis=1, keepdims=True)
    result = verifier.score_keys(vector, bank)
    from klrd_rs.math import sphere_p
    np.testing.assert_allclose(result["p_values"], sphere_p(vector, bank))
    np.testing.assert_allclose(result["cosines"], bank @ vector / np.linalg.norm(vector))


def test_missing_frozen_e1_stops_before_model_or_dataset_load(tmp_path, monkeypatch):
    import klrd_rs.portable.e2 as e2

    calls = []
    def reject(*_args, **_kwargs):
        calls.append("frozen")
        raise ValueError("MISSING_FROZEN_E1")
    monkeypatch.setattr(e2, "load_frozen_e1", reject)
    monkeypatch.setattr(e2, "load_backbone", lambda *_args, **_kwargs: calls.append("model"))
    monkeypatch.setattr(e2, "_dataset_rows", lambda *_args, **_kwargs: calls.append("dataset"))
    with pytest.raises(ValueError, match="MISSING_FROZEN_E1"):
        run_e2(ModelProfile.load("llama32_3b"), tmp_path / "missing-e1", tmp_path / "e2", "smoke")
    assert calls == ["frozen"]

    calls.clear()
    def corrupt(*_args, **_kwargs):
        calls.append("frozen")
        raise ValueError("FROZEN_E1_ARTIFACT_CHECKSUM_MISMATCH")
    monkeypatch.setattr(e2, "load_frozen_e1", corrupt)
    with pytest.raises(ValueError, match="CHECKSUM"):
        run_e2(ModelProfile.load("llama32_3b"), tmp_path / "corrupt-e1", tmp_path / "e2", "smoke")
    assert calls == ["frozen"]
