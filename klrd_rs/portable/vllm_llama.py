"""Llama-derived exact-rotation model, imported in vLLM GPU workers only."""

import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.llama import LlamaForCausalLM

from .vllm_adapter import exact_final_norm


class RotationFinalRMSNorm(RMSNorm):
    @classmethod
    def from_native(cls, native, m: int):
        if getattr(native, "variance_size_override", None) is not None:
            raise RuntimeError("Unsupported native Llama final RMSNorm variance override")
        result = cls(native.hidden_size, eps=native.variance_epsilon, dtype=native.weight.dtype)
        # Reuse the original Parameter so native checkpoint weight loading remains authoritative.
        result.weight = native.weight
        result.register_buffer("P", torch.zeros(native.hidden_size, m, device=native.weight.device, dtype=torch.float32), persistent=False)
        result.register_buffer("Q", torch.zeros_like(result.P), persistent=False)
        result.register_buffer("angles", torch.zeros(m, device=native.weight.device, dtype=torch.float32), persistent=False)
        return result

    @torch.inference_mode()
    def configure(self, P, Q, angles):
        if self.weight.dtype != torch.float32:
            raise RuntimeError("Portable Llama vLLM weights must remain float32")
        for target, value in ((self.P, P), (self.Q, Q), (self.angles, angles)):
            value = torch.as_tensor(value, device=target.device, dtype=target.dtype)
            if value.shape != target.shape or not torch.isfinite(value).all():
                raise RuntimeError("Invalid frozen Llama rotation geometry or owner key")
            target.copy_(value)

    def forward(self, hidden_states, residual=None):
        return exact_final_norm(hidden_states, residual, self.P, self.Q, self.angles, super().forward)


class KLRDRSLlamaForCausalLM(LlamaForCausalLM):
    """Native Llama weights/forward/head with only the final norm path overridden."""

    def __init__(self, *, vllm_config, prefix=""):
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise RuntimeError("Portable Llama exact teacher requires tensor/pipeline parallel size 1")
        if vllm_config.quant_config is not None:
            raise RuntimeError("Portable Llama exact teacher does not support quantized weights")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hidden_size = int(self.model.norm.hidden_size)
        self.model.norm = RotationFinalRMSNorm.from_native(self.model.norm, m=32)
        self.register_buffer("_klrd_logits_finite", torch.ones((), dtype=torch.bool, device=self.model.norm.weight.device), persistent=False)
        self._klrd_hidden_size = hidden_size

    @torch.inference_mode()
    def configure_rotation(self, P, Q, angles):
        self.model.norm.configure(P, Q, angles)
        self._klrd_logits_finite.fill_(True)

    def compute_logits(self, hidden_states):
        # The exact native full-vocabulary head is preserved. SamplingParams applies tau once.
        logits = super().compute_logits(hidden_states)
        if logits is not None:
            self._klrd_logits_finite.logical_and_(torch.isfinite(logits).all())
        return logits


def check_logits_finite(model):
    if not bool(model._klrd_logits_finite.item()):
        raise RuntimeError("Nonfinite Llama vLLM logits; reject the generation batch")
    return {"logits_finite": True, "temperature_applications_in_model": 0}
