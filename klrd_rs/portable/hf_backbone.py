"""Supported Hugging Face backbone contracts for portable E1/E2."""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from ..readout import RotationReadout
from .profile import ModelProfile
from .auth import get_hf_token, resolve_hf_revision


@dataclass(frozen=True)
class _FamilyContract:
    model_type: str
    norm_class_names: tuple[str, ...]


_FAMILIES = {
    "llama": _FamilyContract("llama", ("LlamaRMSNorm",)),
    "qwen2": _FamilyContract("qwen2", ("Qwen2RMSNorm",)),
}


class HFBackbone:
    """Validated adapter over an already loaded supported HF causal LM."""

    def __init__(self, model: Any, tokenizer: Any, profile: ModelProfile):
        profile.validate()
        if profile.model_type not in _FAMILIES:
            raise ValueError(f"unsupported model family: {profile.model_type}")
        self.model = model
        self.tokenizer = tokenizer
        self.profile = profile
        self.contract = _FAMILIES[profile.model_type]

        config = getattr(model, "config", None)
        if config is None or getattr(config, "model_type", None) != self.contract.model_type:
            raise ValueError(f"unsupported model family: expected {self.contract.model_type}")
        architectures = tuple(getattr(config, "architectures", None) or ())
        if profile.hf_architecture not in architectures and type(model).__name__ != profile.hf_architecture:
            raise ValueError("loaded Hugging Face architecture does not match profile")

        core = getattr(model, "model", None)
        self.norm = getattr(core, "norm", None)
        if self.norm is None or type(self.norm).__name__ not in self.contract.norm_class_names:
            raise ValueError(f"unsupported {self.contract.model_type} final RMSNorm structure")
        self.head = model.get_output_embeddings()
        if self.head is None or not hasattr(self.head, "weight"):
            raise ValueError("native output head is missing")

        self.hidden_size = int(getattr(config, "hidden_size", 0))
        self.vocab_size = int(getattr(config, "vocab_size", 0))
        self.max_position_embeddings = int(getattr(config, "max_position_embeddings", 0))
        if min(self.hidden_size, self.vocab_size, self.max_position_embeddings) < 1:
            raise ValueError("loaded config must declare hidden, vocabulary, and context dimensions")
        if tuple(self.head.weight.shape) != (self.vocab_size, self.hidden_size):
            raise ValueError("output head shape does not match config vocabulary and hidden size")
        if tuple(self.norm.weight.shape) != (self.hidden_size,):
            raise ValueError("final RMSNorm gain shape does not match config hidden size")
        if self.head.weight.dtype != torch.float32 or self.norm.weight.dtype != torch.float32:
            raise ValueError("portable reference model and output head must remain float32")
        bias = getattr(self.head, "bias", None)
        if bias is not None and tuple(bias.shape) != (self.vocab_size,):
            raise ValueError("output head bias shape does not match config vocabulary")
        epsilon = getattr(self.norm, "variance_epsilon", getattr(self.norm, "eps", None))
        if epsilon is None:
            epsilon = getattr(config, "rms_norm_eps", None)
        if epsilon is None or not np.isfinite(float(epsilon)) or float(epsilon) <= 0:
            raise ValueError("native RMSNorm epsilon is missing or invalid")
        self.epsilon = float(epsilon)
        try:
            self.device = self.head.weight.device
        except AttributeError:
            self.device = torch.device("cpu")

    @classmethod
    def from_loaded(cls, model: Any, tokenizer: Any, profile: ModelProfile) -> "HFBackbone":
        """Validate an in-memory model; supports tiny synthetic contract fixtures."""
        return cls(model, tokenizer, profile)

    def readout(self, P, Q, tau: float) -> RotationReadout:
        return RotationReadout(
            self.head.weight,
            self.norm.weight,
            self.epsilon,
            torch.as_tensor(P),
            torch.as_tensor(Q),
            tau,
            getattr(self.head, "bias", None),
        )

    @torch.inference_mode()
    def native_tail_equivalence_check(self, residuals) -> float:
        values = torch.as_tensor(residuals, device=self.device, dtype=self.head.weight.dtype)
        expected = self.head(self.norm(values))
        observed = self.readout(
            torch.eye(self.hidden_size, dtype=self.head.weight.dtype, device=self.device)[:, :2],
            torch.eye(self.hidden_size, dtype=self.head.weight.dtype, device=self.device)[:, 2:4],
            self.profile.scientific["tau"],
        ).tail(values)
        return float((expected - observed).abs().max().cpu())

    @torch.inference_mode()
    def capture_pre_norm(self, encoded_rows: list[dict[str, Any]]) -> list[np.ndarray]:
        """Capture selected residual rows before the native final normalization."""
        if not encoded_rows:
            return []
        if self.tokenizer is None:
            raise ValueError("tokenizer is required for residual capture")
        input_rows = []
        position_rows = []
        for row in encoded_rows:
            ids = [int(token) for token in row.get("input_ids", [])]
            positions = [int(position) for position in row.get("positions", [])]
            if not ids:
                raise ValueError("encoded row has empty input_ids")
            if len(ids) > self.max_position_embeddings:
                raise ValueError("encoded row exceeds model context window; truncation is forbidden")
            if any(position < 0 or position >= len(ids) for position in positions):
                raise ValueError("selected residual position is outside input_ids")
            input_rows.append(ids)
            position_rows.append(positions)

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is None:
            raise ValueError("tokenizer has neither a pad nor EOS token for masked replay")
        width = max(map(len, input_rows))
        token_ids = torch.full((len(input_rows), width), int(pad_id), dtype=torch.long, device=self.device)
        attention_mask = torch.zeros_like(token_ids)
        for index, ids in enumerate(input_rows):
            token_ids[index, :len(ids)] = torch.as_tensor(ids, dtype=torch.long, device=self.device)
            attention_mask[index, :len(ids)] = 1

        captured = []

        def capture(_module, args):
            if not args:
                raise RuntimeError("final RMSNorm received no residual input")
            residual = args[0]
            if len(args) > 1 and args[1] is not None:
                residual = residual + args[1]
            captured.append(residual.detach())

        hook = self.norm.register_forward_pre_hook(capture)
        try:
            self.model.model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("expected exactly one final RMSNorm input capture")
        full = captured[0]
        return [
            full[index, positions].float().cpu().numpy()
            for index, positions in enumerate(position_rows)
        ]


def load_backbone(profile: ModelProfile, device: str = "cuda") -> HFBackbone:
    """Load one explicitly supported, pinned HF reference model."""
    profile.validate()
    if profile.model_type not in _FAMILIES:
        raise ValueError(f"unsupported model family: {profile.model_type}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU_UNAVAILABLE: portable model loading requires the requested CUDA device")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = get_hf_token()
    revision = profile.revision
    if revision is None:
        revision = resolve_hf_revision(profile.model_id, token)

    torch.set_float32_matmul_precision("highest")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        profile.model_id,
        revision=revision,
        trust_remote_code=False,
        token=token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        profile.model_id,
        revision=revision,
        torch_dtype=torch.float32,
        trust_remote_code=False,
        attn_implementation="eager",
        token=token,
    ).to(device).eval()
    model.requires_grad_(False)
    pinned_profile = profile if profile.revision == revision else replace(profile, revision=revision)
    return HFBackbone(model, tokenizer, pinned_profile)
