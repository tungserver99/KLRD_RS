"""Strict, import-safe model profiles for the portable E1/E2 path."""

from dataclasses import dataclass
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


_TOP_LEVEL_FIELDS = {
    "profile_id",
    "model_id",
    "revision",
    "model_type",
    "hf_architecture",
    "vllm_architecture",
    "trust_remote_code",
    "scientific",
    "seeds",
    "data",
    "runtime",
}
_ARCHITECTURES = {
    "llama": ("LlamaForCausalLM", "KLRDRSLlamaForCausalLM"),
    "qwen2": ("Qwen2ForCausalLM", "KLRDRSQwen2ForCausalLM"),
}
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_PROFILE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_RUN_ID = re.compile(r"^run_[0-9]{3,}$")
_ALPHAS = (0.01, 0.0001)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    model_id: str
    revision: str | None
    model_type: str
    hf_architecture: str
    vllm_architecture: str
    trust_remote_code: bool
    scientific: Mapping[str, Any]
    seeds: Mapping[str, Any]
    data: Mapping[str, Any]
    runtime: Mapping[str, Any]

    @classmethod
    def load(cls, profile_id: str, profile_dir: Path | None = None) -> "ModelProfile":
        if not isinstance(profile_id, str) or not _PROFILE_ID.fullmatch(profile_id):
            raise ValueError("unsafe profile identifier")
        directory = Path(profile_dir) if profile_dir is not None else (
            Path(__file__).resolve().parents[2] / "configs" / "model_profiles"
        )
        path = directory / f"{profile_id}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"model profile not found: {path}")
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("model profile must be a YAML mapping")
        unknown = set(raw) - _TOP_LEVEL_FIELDS
        if unknown:
            raise ValueError("unknown profile field(s): " + ", ".join(sorted(unknown)))
        missing = _TOP_LEVEL_FIELDS - set(raw)
        if missing:
            raise ValueError("missing profile field(s): " + ", ".join(sorted(missing)))
        if raw["profile_id"] != profile_id:
            raise ValueError("profile_id must match the requested profile filename")
        profile = cls(
            profile_id=raw["profile_id"],
            model_id=raw["model_id"],
            revision=raw["revision"],
            model_type=raw["model_type"],
            hf_architecture=raw["hf_architecture"],
            vllm_architecture=raw["vllm_architecture"],
            trust_remote_code=raw["trust_remote_code"],
            scientific=_freeze(raw["scientific"]),
            seeds=_freeze(raw["seeds"]),
            data=_freeze(raw["data"]),
            runtime=_freeze(raw["runtime"]),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if not _PROFILE_ID.fullmatch(self.profile_id):
            raise ValueError("unsafe profile identifier")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be nonempty")
        if self.revision is not None and (
            not isinstance(self.revision, str) or not _SHA40.fullmatch(self.revision)
        ):
            raise ValueError("revision must be null or a 40-character commit SHA")
        if self.model_type not in _ARCHITECTURES:
            raise ValueError(f"unsupported model_type: {self.model_type}")
        expected_hf, expected_vllm = _ARCHITECTURES[self.model_type]
        if self.hf_architecture != expected_hf or self.vllm_architecture != expected_vllm:
            raise ValueError("unknown or mismatched model architecture")
        if self.trust_remote_code is not False:
            raise ValueError("trust_remote_code must be false")
        if not all(isinstance(section, Mapping) for section in (self.scientific, self.seeds, self.data, self.runtime)):
            raise ValueError("scientific, seeds, data, and runtime must be mappings")
        required_science = {
            "tau", "pca_dim", "m", "fisher_ridge_fraction", "target_mean_kl_nats",
            "safeguard_p99_kl_nats", "alphas", "calibration_keys", "final_keys",
            "owner_key_rule", "calibration_responses", "clean_original_diagnostic_responses",
            "e2_prompt_count", "e2_owner_indices", "wrong_keys_per_owner", "primary_query_budgets",
            "sampling_max_new_tokens", "sampling_top_p", "sampling_top_k_unrestricted",
            "calibration_context_caps",
        }
        missing = required_science - set(self.scientific)
        if missing:
            raise ValueError("missing scientific field(s): " + ", ".join(sorted(missing)))
        exact_science = {
            "tau": 0.8, "pca_dim": 128, "m": 32, "fisher_ridge_fraction": 0.001,
            "target_mean_kl_nats": 0.005, "safeguard_p99_kl_nats": 0.05,
            "calibration_responses": 2000, "clean_original_diagnostic_responses": 2500,
            "e2_prompt_count": 5000, "wrong_keys_per_owner": 20,
            "sampling_max_new_tokens": 512, "sampling_top_p": 1.0,
        }
        for key, expected in exact_science.items():
            if self.scientific[key] != expected:
                raise ValueError(f"portable KLRD-RS {key} must remain {expected}")
        if self.scientific["sampling_top_k_unrestricted"] is not True:
            raise ValueError("portable full-vocabulary sampling must leave top-k unrestricted")
        m, pca_dim = self.scientific["m"], self.scientific["pca_dim"]
        if not isinstance(m, int) or not isinstance(pca_dim, int) or m < 2 or pca_dim < 2 * m:
            raise ValueError("pca_dim must be at least 2*m and m must be >=2")
        if tuple(self.scientific["alphas"]) != _ALPHAS:
            raise ValueError("portable KLRD-RS alphas must remain [0.01, 0.0001]")
        for key in ("calibration_keys", "final_keys", "calibration_responses", "clean_original_diagnostic_responses",
                    "e2_prompt_count", "wrong_keys_per_owner"):
            if not isinstance(self.scientific[key], int) or self.scientific[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.scientific["owner_key_rule"] != "predeclared_first_key":
            raise ValueError("owner key rule must be predeclared_first_key")
        if tuple(self.scientific["e2_owner_indices"]) != tuple(range(8)):
            raise ValueError("portable E2 owners must remain predeclared indices 0..7")
        if tuple(self.scientific["primary_query_budgets"]) != (100, 1000):
            raise ValueError("portable E2 primary query budgets must remain Q=100 and Q=1000")
        if self.scientific["wrong_keys_per_owner"] >= self.scientific["final_keys"]:
            raise ValueError("wrong-key count must be smaller than the frozen owner key bank")
        expected_caps = {"pca_covariance": 4096, "fisher": 100000, "strength": 128, "local": 512}
        if dict(self.scientific["calibration_context_caps"]) != expected_caps:
            raise ValueError("portable E1 context caps must follow the predeclared matched protocol")
        dataset_revision = self.data.get("dataset_revision")
        if self.data.get("dataset") != "HuggingFaceH4/ultrachat_200k":
            raise ValueError("portable calibration/evaluation data source must remain UltraChat 200k")
        if not isinstance(dataset_revision, str) or not _SHA40.fullmatch(dataset_revision):
            raise ValueError("dataset_revision must be a 40-character commit SHA")
        if self.data.get("prompt_template_policy") != "native_tokenizer_apply_chat_template":
            raise ValueError("portable prompt serialization must use the native tokenizer chat template")
        if self.data.get("calibration_split") != "train_gen" or self.data.get("diagnostic_split") != "test_gen":
            raise ValueError("portable E1 calibration/diagnostic splits must remain train_gen/test_gen")
        if tuple(self.data.get("e2_split_order", ())) != ("test_gen", "train_gen"):
            raise ValueError("portable E2 split order must remain test_gen then train_gen")
        required_seeds = {
            "geometry", "calibration_keys", "final_keys", "calibration_partition",
            "clean_generation", "diagnostic_generation", "e2_prompts", "e2_generation",
            "e2_groups", "context_sampling", "token_sampling",
        }
        if required_seeds - set(self.seeds):
            raise ValueError("missing portable seed(s): " + ", ".join(sorted(required_seeds - set(self.seeds))))
        if any(not isinstance(self.seeds[key], int) or self.seeds[key] < 0 for key in required_seeds):
            raise ValueError("portable experiment seeds must be nonnegative integers")
        if self.runtime.get("use_tf32") is not False:
            raise ValueError("TF32 must be disabled for the portable exact teacher")
        if self.runtime.get("teacher_dtype") != "float32":
            raise ValueError("teacher_dtype must be float32")

    def validate_for_prepare(self) -> None:
        self.validate()
        if self.revision is None:
            raise ValueError("an immutable model revision is required before prepare")

    def output_path(self, repo_root: Path, stage: str, run_id: str) -> Path:
        if stage not in ("e1", "e2"):
            raise ValueError("stage must be e1 or e2")
        if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
            raise ValueError("unsafe run identifier")
        return (Path(repo_root).resolve() / "runs" / self.profile_id / stage / run_id).resolve()

    def validate_run_path(self, repo_root: Path, stage: str, path: Path) -> Path:
        """Require portable writes to stay in this profile's isolated run namespace."""
        if stage not in ("e1", "e2"):
            raise ValueError("stage must be e1 or e2")
        resolved = Path(path).resolve()
        allowed_parent = (Path(repo_root).resolve() / "runs" / self.profile_id / stage).resolve()
        if resolved.parent != allowed_parent or not _RUN_ID.fullmatch(resolved.name):
            raise ValueError(f"unsafe {stage.upper()} output path; expected runs/{self.profile_id}/{stage}/run_NNN")
        return resolved
