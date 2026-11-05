from pathlib import Path
import subprocess
import sys

import pytest

from klrd_rs.portable.profile import ModelProfile


def test_llama_profile_keeps_revision_unresolved_until_prepare():
    profile = ModelProfile.load("llama32_3b")

    assert profile.profile_id == "llama32_3b"
    assert profile.model_id == "meta-llama/Llama-3.2-3B-Instruct"
    assert profile.revision is None
    assert profile.model_type == "llama"
    assert profile.hf_architecture == "LlamaForCausalLM"
    assert profile.vllm_architecture == "KLRDRSLlamaForCausalLM"
    assert profile.scientific["tau"] == 0.8
    assert profile.scientific["pca_dim"] == 128
    assert profile.scientific["m"] == 32
    assert profile.scientific["alphas"] == (0.01, 0.0001)

    with pytest.raises(ValueError, match="immutable.*revision"):
        profile.validate_for_prepare()


def test_profile_requires_a_commit_sha_and_supported_architecture(tmp_path):
    profile_file = tmp_path / "bad.yaml"
    profile_file.write_text(
        """profile_id: bad
model_id: meta-llama/Llama-3.2-3B-Instruct
revision: main
model_type: llama
hf_architecture: Qwen2ForCausalLM
vllm_architecture: UnknownForCausalLM
trust_remote_code: false
scientific: {tau: 0.8, pca_dim: 128, m: 32, alphas: [0.01, 0.0001]}
data: {dataset: HuggingFaceH4/ultrachat_200k, dataset_revision: 8049631c405ae6576f93f445c6b8166f76f5505a}
runtime: {}
seeds: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="revision.*40-character commit SHA"):
        ModelProfile.load("bad", profile_dir=tmp_path)


def test_profile_rejects_invalid_geometry_alpha_and_unknown_fields(tmp_path):
    profile_file = tmp_path / "bad.yaml"
    profile_file.write_text(
        """profile_id: llama32_3b
model_id: meta-llama/Llama-3.2-3B-Instruct
revision: null
model_type: llama
hf_architecture: LlamaForCausalLM
vllm_architecture: KLRDRSLlamaForCausalLM
trust_remote_code: false
unexpected: true
scientific: {tau: 0.8, pca_dim: 32, m: 32, alphas: [1.2, 0.0001]}
data: {dataset: HuggingFaceH4/ultrachat_200k, dataset_revision: 8049631c405ae6576f93f445c6b8166f76f5505a}
runtime: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown profile field.*unexpected"):
        ModelProfile.load("bad", profile_dir=tmp_path)


def test_profile_output_is_confined_to_its_run_namespace(tmp_path):
    profile = ModelProfile.load("llama32_3b")

    output = profile.output_path(tmp_path, "e1", "run_001")
    assert output == (tmp_path / "runs/llama32_3b/e1/run_001").resolve()
    assert output != (tmp_path / "experiment1_rs").resolve()
    assert output != (tmp_path / "experiment1_rs_continuation").resolve()

    with pytest.raises(ValueError, match="unsafe run identifier"):
        profile.output_path(tmp_path, "e1", "../../experiment1_rs")
    with pytest.raises(ValueError, match="unsafe E1 output path"):
        profile.validate_run_path(tmp_path, "e1", tmp_path / "experiment1_rs")


def test_profile_is_immutable_and_accepts_only_known_e1_e2_stages():
    profile = ModelProfile.load("llama32_3b")

    with pytest.raises((AttributeError, TypeError)):
        profile.model_id = "Qwen/Qwen2.5-3B-Instruct"
    with pytest.raises(ValueError, match="stage must be e1 or e2"):
        profile.output_path(Path.cwd(), "e3", "run_001")


def test_portable_help_and_inspect_are_offline():
    root = Path(__file__).resolve().parents[1]
    for entrypoint in ("main_portable_e1.py", "main_portable_e2.py"):
        help_result = subprocess.run(
            [sys.executable, str(root / entrypoint), "--help"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert help_result.returncode == 0, help_result.stderr
        assert "--mode" in help_result.stdout

        inspect_result = subprocess.run(
            [sys.executable, str(root / entrypoint), "--profile", "llama32_3b", "--mode", "inspect"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert inspect_result.returncode == 0, inspect_result.stderr
        assert "llama32_3b" in inspect_result.stdout
        assert "meta-llama/Llama-3.2-3B-Instruct" in inspect_result.stdout
