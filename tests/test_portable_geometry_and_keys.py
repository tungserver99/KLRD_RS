import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from klrd_rs.portable.calibration import (
    fit_public_basis,
    partition_calibration_rows,
    resolve_model_identity,
    select_calibration_prompts,
    serialize_clean_generations,
    build_calibration_activation_rows,
)
from klrd_rs.portable.e1 import freeze_e1_geometry
from klrd_rs.portable.frozen import load_frozen_e1
from klrd_rs.portable.profile import ModelProfile
from klrd_rs.math import keys, sphere_p
from klrd_rs.io import digest, write_json


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    chat_template = "{{ messages }}"
    init_kwargs = {"_commit_hash": "b" * 40}

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 251 + 3 for char in text]

    def get_vocab(self):
        return {"a": 3, "b": 4, "c": 5}

    def decode(self, token_ids, skip_special_tokens=False):
        return "clean response"

    @property
    def all_special_ids(self):
        return [0, 2]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [11, 12] if tokenize else "<|user|>prompt<|assistant|>"


def _source_rows(count=30):
    return [
        {"messages": [{"role": "user", "content": f"A distinct calibration prompt number {i:03d}."}], "id": f"u{i:03d}"}
        for i in range(count)
    ]


def _test_identity(profile):
    identity = {
        "profile_id": profile.profile_id, "model_id": profile.model_id,
        "model_type": profile.model_type, "hf_architecture": profile.hf_architecture,
        "model_revision": "b" * 40, "tokenizer_revision": "b" * 40,
        "shapes": {"hidden_size": 256, "vocab_size": 64, "max_position_embeddings": 512, "num_hidden_layers": 2},
    }
    identity["identity_sha256"] = digest(identity)
    return identity


def test_prompt_selection_is_deterministic_eligible_and_source_hashed():
    profile = ModelProfile.load("llama32_3b")
    first = select_calibration_prompts(_source_rows(), TinyTokenizer(), profile, seed=13, count=10)
    second = select_calibration_prompts(_source_rows(), TinyTokenizer(), profile, seed=13, count=10)
    assert [row["prompt_id"] for row in first] == [row["prompt_id"] for row in second]
    assert len({row["prompt_text_sha256"] for row in first}) == 10
    assert all(row["source_split"] == "train_gen" and row["prompt_length"] >= 16 for row in first)
    assert all(len(row["source_row_sha256"]) == 64 for row in first)


def test_calibration_partitions_are_disjoint_and_cover_exactly_selected_rows():
    profile = ModelProfile.load("llama32_3b")
    rows = select_calibration_prompts(_source_rows(40), TinyTokenizer(), profile, seed=5, count=20)
    partitions = partition_calibration_rows(rows, seed=19)
    flattened = [row["prompt_id"] for values in partitions.values() for row in values]
    assert len(flattened) == len(set(flattened)) == len(rows)
    assert set(flattened) == {row["prompt_id"] for row in rows}
    assert set(partitions) == {"pca_covariance", "fisher", "strength", "local"}


def test_identity_receipt_hashes_vocab_template_and_records_shapes(tmp_path):
    profile = ModelProfile.load("llama32_3b")
    config = SimpleNamespace(
        _commit_hash="b" * 40, model_type="llama", architectures=["LlamaForCausalLM"],
        hidden_size=8, vocab_size=3,
        max_position_embeddings=32, num_hidden_layers=2,
        generation_config=SimpleNamespace(eos_token_id=[2, 9], bos_token_id=1),
    )
    receipt = resolve_model_identity(profile, TinyTokenizer(), config, tmp_path)
    assert receipt["model_revision"] == "b" * 40
    assert receipt["tokenizer_revision"] == "b" * 40
    assert receipt["vocab_sha256"] == hashlib.sha256(b'{"a":3,"b":4,"c":5}').hexdigest()
    assert receipt["chat_template_sha256"] == hashlib.sha256(b"{{ messages }}").hexdigest()
    assert receipt["shapes"] == {"hidden_size": 8, "vocab_size": 3, "max_position_embeddings": 32, "num_hidden_layers": 2}
    assert receipt["generation_config"]["eos_token_id"] == [2, 9]
    assert (tmp_path / "model_tokenizer_identity.json").is_file()


def test_generation_serialization_records_roundtrip_and_identity_hashes():
    profile = ModelProfile.load("llama32_3b")
    prompt = select_calibration_prompts(_source_rows(1), TinyTokenizer(), profile, seed=1, count=1)[0]
    generated = [{"token_ids": [7, 8, 2], "text": "clean response", "finish_reason": "stop", "seed": 101}]
    rows = serialize_clean_generations([prompt], generated, TinyTokenizer(), model_identity={"model_revision": "b" * 40, "tokenizer_revision": "b" * 40})
    row = rows[0]
    assert row["generated_token_ids"] == [7, 8, 2]
    assert row["eos_seen"] is True and row["termination"] == "eos"
    assert row["roundtrip_exact"] is True
    assert row["response_sha256"] == hashlib.sha256(b"clean response").hexdigest()
    assert row["model_revision"] == "b" * 40


def test_activation_serialization_uses_visible_text_and_refuses_truncation():
    record = {
        "prompt_id": "u1", "source_split": "train_gen", "prompt": "source prompt",
        "response": "visible clean response", "response_sha256": "c" * 64,
        "model_revision": "b" * 40, "tokenizer_revision": "b" * 40,
    }
    encoded = build_calibration_activation_rows([record], TinyTokenizer(), max_position_embeddings=100)
    assert encoded[0]["input_ids"][:2] == [11, 12]
    assert encoded[0]["positions"][0] == 1
    assert encoded[0]["model_revision"] == "b" * 40
    with pytest.raises(ValueError, match="truncation is forbidden"):
        build_calibration_activation_rows([record], TinyTokenizer(), max_position_embeddings=3)


def test_public_basis_is_orthonormal_and_mean_orthogonal():
    rng = np.random.default_rng(21)
    residuals = rng.normal(size=(80, 12)) + np.arange(12)[None, :]
    mean = residuals.mean(axis=0)
    basis, returned_mean = fit_public_basis(residuals, mean, pca_dim=4, seed=7)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-6)
    np.testing.assert_allclose(basis.T @ returned_mean, np.zeros(4), atol=1e-6)
    np.testing.assert_allclose(returned_mean, mean, atol=0, rtol=0)


def test_public_basis_assertions_fail_closed(monkeypatch):
    import klrd_rs.portable.calibration as calibration

    monkeypatch.setattr(calibration, "_basis_sanity", lambda *_args, **_kwargs: False)
    residuals = np.arange(60, dtype=float).reshape(20, 3)
    with pytest.raises(ValueError, match="basis sanity"):
        calibration.fit_public_basis(residuals, residuals.mean(axis=0), pca_dim=2, seed=2)


def test_e1_freezes_geometry_lambda_before_independent_owner_keys_and_loads_receipt(tmp_path, monkeypatch):
    import klrd_rs.portable.e1 as e1

    profile = ModelProfile.load("llama32_3b")
    identity = _test_identity(profile)
    write_json(tmp_path / "model_tokenizer_identity.json", identity)
    basis = np.eye(256, dtype=np.float64)[:, :128]
    mean = np.zeros(256)
    fisher_scores = np.random.default_rng(88).normal(size=(160, 32))
    original_keys = e1.keys
    owner_key_generation_saw_frozen_lambda = []

    def observed_keys(n, m, seed):
        if seed == profile.seeds["final_keys"]:
            lambda_path = tmp_path / "teacher" / "global_lambda.json"
            owner_key_generation_saw_frozen_lambda.append(lambda_path.is_file() and __import__("json").loads(lambda_path.read_text())["status"] == "FROZEN")
        return original_keys(n, m, seed)

    monkeypatch.setattr(e1, "keys", observed_keys)
    result = freeze_e1_geometry(
        profile, tmp_path, basis, mean, fisher_scores,
        lambda_evaluator=lambda strength, _bank, _whitener: np.full(128, 0.005 * strength**2),
        model_identity=identity,
    )
    assert owner_key_generation_saw_frozen_lambda == [True]
    assert result["lambda"] > 0
    assert result["key_ids"][0]["index"] == 0
    assert result["key_ids"] == __import__("klrd_rs.math", fromlist=["key_ids"]).key_ids(keys(64, 32, profile.seeds["final_keys"]))
    loaded = load_frozen_e1(tmp_path, profile)
    assert loaded.lambda_value == result["lambda"]
    assert loaded.P.shape == loaded.Q.shape == (256, 32)
    assert loaded.key_bank.shape == (64, 32)


def test_frozen_e1_rejects_cross_profile_missing_and_corrupt_artifacts(tmp_path):
    profile = ModelProfile.load("llama32_3b")
    with pytest.raises(ValueError, match="MISSING_FROZEN_E1"):
        load_frozen_e1(tmp_path, profile)

    identity = _test_identity(profile)
    write_json(tmp_path / "model_tokenizer_identity.json", identity)
    freeze_e1_geometry(
        profile, tmp_path, np.eye(256)[:, :128], np.zeros(256),
        np.random.default_rng(89).normal(size=(160, 32)),
        lambda_evaluator=lambda strength, _bank, _whitener: np.full(128, 0.005 * strength**2),
        model_identity=identity,
    )
    with pytest.raises(ValueError, match="(?i)profile"):
        load_frozen_e1(tmp_path, replace(profile, profile_id="other_profile"))
    with pytest.raises(ValueError, match="(?i)hash|checksum"):
        with (tmp_path / "geometry" / "basis.npz").open("ab") as stream:
            stream.write(b"corrupt")
        load_frozen_e1(tmp_path, profile)


def test_zero_score_vector_keeps_exact_spherical_p_value_one():
    assert sphere_p(np.zeros(32), keys(4, 32, 23)) == pytest.approx(1.0)
