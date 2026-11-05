"""Lazy profile-specific vLLM registration and exact Llama runtime options."""

from importlib.metadata import PackageNotFoundError, version
import re
from typing import Any, Mapping

import torch

from ..math import rotate
from .profile import ModelProfile


_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_LLAMA_IMPORT = "klrd_rs.portable.vllm_llama:KLRDRSLlamaForCausalLM"


def rotate_final_state(hidden: torch.Tensor, residual: torch.Tensor | None, P, Q, angles):
    """Return one combined pre-norm residual rotated in FP32 exactly once."""
    if hidden.dtype != torch.float32:
        raise RuntimeError("Exact portable vLLM teacher requires FP32 residuals")
    combined = hidden if residual is None else hidden + residual
    return rotate(combined, P, Q, angles), combined


def exact_final_norm(hidden, residual, P, Q, angles, native_norm):
    """Small callable contract shared by the custom norm and CPU synthetic tests."""
    rotated, combined = rotate_final_state(hidden, residual, P, Q, angles)
    normalized = native_norm(rotated)
    return normalized if residual is None else (normalized, combined)


def register_profile_architecture(profile: ModelProfile) -> None:
    profile.validate()
    if profile.model_type != "llama" or profile.vllm_architecture != "KLRDRSLlamaForCausalLM":
        raise ValueError("PORTABLE_VLLM_FAMILY_UNSUPPORTED: only the explicit Llama architecture is registered")
    # Resolve vLLM and the custom class only in the server-side engine process.
    import vllm
    installed = getattr(vllm, "__version__", None)
    if installed is None:
        try:
            installed = version("vllm")
        except PackageNotFoundError:
            installed = None
    ModelRegistry = vllm.ModelRegistry

    ModelRegistry.register_model(profile.vllm_architecture, _LLAMA_IMPORT)


def runtime_identity(profile: ModelProfile) -> dict[str, Any]:
    profile.validate()
    try:
        installed = version("vllm")
    except PackageNotFoundError:
        installed = None
    return {
        "reference_vllm_version": profile.runtime.get("vllm_version"),
        "installed_vllm_version": installed,
        "architecture": profile.vllm_architecture,
        "model_type": profile.model_type,
        "dtype": profile.runtime["teacher_dtype"],
        "use_tf32": profile.runtime["use_tf32"],
        "attention_backend": "TRITON_ATTN",
        "temperature_location": "SamplingParams only",
        "logits_temperature_divisions": 0,
        "generation_backend": profile.runtime["gpu_generation_backend"],
    }


def engine_options(profile: ModelProfile, settings: Mapping[str, Any], eager: bool = False) -> dict[str, Any]:
    profile.validate()
    if profile.model_type != "llama":
        raise ValueError("PORTABLE_VLLM_FAMILY_UNSUPPORTED: no Qwen registration or teacher fallback")
    revision = settings.get("model_revision", profile.revision)
    if not isinstance(revision, str) or not _SHA40.fullmatch(revision):
        raise ValueError("an immutable model revision SHA is required for the vLLM teacher")
    max_model_len = settings.get("max_model_len", settings.get("max_position_embeddings"))
    if not isinstance(max_model_len, int) or max_model_len < 1:
        raise ValueError("max_model_len must come from the frozen native model identity")
    return {
        "model": profile.model_id,
        "revision": revision,
        "tokenizer": profile.model_id,
        "tokenizer_revision": revision,
        "dtype": "float32",
        "kv_cache_dtype": "auto",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "quantization": None,
        "hf_overrides": {"architectures": [profile.vllm_architecture]},
        "trust_remote_code": False,
        "attention_config": {"backend": "TRITON_ATTN"},
        "logprobs_mode": "processed_logprobs",
        "gpu_memory_utilization": float(settings.get("gpu_memory_utilization", 0.4)),
        "max_model_len": int(max_model_len),
        "max_num_seqs": int(settings.get("max_num_seqs", 8)),
        "max_num_batched_tokens": int(settings.get("max_num_batched_tokens", 8192)),
        "enforce_eager": bool(eager),
        "compilation_config": {
            "mode": 0,
            "cudagraph_mode": "NONE" if eager else "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [] if eager else list(range(1, int(settings.get("max_num_seqs", 8)) + 1)),
        },
        "enable_prefix_caching": False,
        "enable_chunked_prefill": not eager,
        "async_scheduling": False,
        "generation_config": "vllm",
        "seed": int(settings.get("generation_seed", profile.seeds["e2_generation"])),
        "distributed_executor_backend": "uni",
    }
