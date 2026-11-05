from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from klrd_rs.portable.hf_backbone import HFBackbone
from klrd_rs.portable.profile import ModelProfile
from klrd_rs.math import rotate


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.7, 1.3, hidden_size))
        self.variance_epsilon = eps

    def forward(self, value, residual=None):
        if residual is not None:
            value = value + residual
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon) * self.weight


class TinyCore(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embed_tokens = embedding
        self.norm = LlamaRMSNorm(8)
        self.fused_residual = 0.0

    def forward(self, input_ids, **_kwargs):
        hidden = self.embed_tokens(input_ids)
        residual = torch.full_like(hidden, self.fused_residual) if self.fused_residual else None
        return self.norm(hidden, residual)


class TinyCausalLM(nn.Module):
    def __init__(self, tied=False, bias=False):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="llama",
            architectures=["LlamaForCausalLM"],
            hidden_size=8,
            vocab_size=13,
            max_position_embeddings=32,
            rms_norm_eps=1e-5,
        )
        self.embeddings = nn.Embedding(13, 8)
        self.model = TinyCore(self.embeddings)
        self.head = nn.Linear(8, 13, bias=bias)
        if tied:
            self.head.weight = self.embeddings.weight

    def get_output_embeddings(self):
        return self.head


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2


def test_capture_pre_norm_combines_fused_residual_once_with_variable_rows():
    model = TinyCausalLM()
    model.model.fused_residual = 0.25
    tokenizer = TinyTokenizer()
    backbone = HFBackbone.from_loaded(model, tokenizer=tokenizer, profile=ModelProfile.load("llama32_3b"))
    rows = [
        {"input_ids": [1, 2, 3], "positions": [0, 2]},
        {"input_ids": [4, 5], "positions": [1]},
    ]
    actual = backbone.capture_pre_norm(rows)
    expected = [
        (model.embeddings(torch.tensor([1, 2, 3])) + 0.25)[[0, 2]].detach().numpy(),
        (model.embeddings(torch.tensor([4, 5])) + 0.25)[[1]].detach().numpy(),
    ]
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=0, rtol=0)


def _planes():
    frame = torch.eye(8)[:, :4]
    return frame[:, ::2].contiguous(), frame[:, 1::2].contiguous()


@pytest.mark.parametrize("tied,bias", [(False, False), (False, True), (True, False)])
def test_rotation_readout_matches_native_rmsnorm_and_head_for_tiny_llama(tied, bias):
    torch.manual_seed(41)
    model = TinyCausalLM(tied=tied, bias=bias)
    profile = ModelProfile.load("llama32_3b")
    backbone = HFBackbone.from_loaded(model, tokenizer=None, profile=profile)
    p, q = _planes()
    readout = backbone.readout(p, q, tau=0.8)
    residuals = torch.randn(5, 8)
    angles = torch.tensor([0.17, -0.23])

    native_tail = model.get_output_embeddings()(model.model.norm(residuals))
    torch.testing.assert_close(readout.tail(residuals), native_tail, rtol=1e-6, atol=1e-6)

    for values in (torch.zeros_like(angles), angles):
        direct = torch.log_softmax(
            model.get_output_embeddings()(model.model.norm(rotate(residuals, p, q, values))) / 0.8,
            dim=-1,
        )
        torch.testing.assert_close(readout.log_probs(residuals, values), direct, rtol=2e-5, atol=2e-5)


def test_backbone_rejects_family_and_shape_mismatch():
    model = TinyCausalLM()
    model.config.model_type = "mystery"
    with pytest.raises(ValueError, match="unsupported model family"):
        HFBackbone.from_loaded(model, tokenizer=None, profile=ModelProfile.load("llama32_3b"))

    model = TinyCausalLM()
    model.config.vocab_size = 12
    with pytest.raises(ValueError, match="output head.*vocabulary"):
        HFBackbone.from_loaded(model, tokenizer=None, profile=ModelProfile.load("llama32_3b"))
