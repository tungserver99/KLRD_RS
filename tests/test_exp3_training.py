"""SERVER ONLY: delivered test source; do not execute on the local workstation.

Toy contracts check exact derivatives without any model downloads. The real
adapter/base/checkpoint integration is the opt-in GPU server_smoke helper.
"""
from contextlib import nullcontext
import inspect
from types import SimpleNamespace
import pytest
import torch
from klrd_rs.e3.teacher import TeacherStudent, distill_backward, forward_kl
from klrd_rs.e3 import train


class ToyTail:
    """A full-vocabulary frozen tail; categories are never selected/truncated."""
    def __init__(self, chunk):
        self.cfg = SimpleNamespace(full_vocab_logit_chunk=chunk)
        self.weight = torch.tensor([[.2, -.3], [-.4, .5], [.1, .7],
            [.8, -.2], [-.6, -.1], [.3, .4], [-.7, .9]], dtype=torch.float32)
        self.angles = torch.tensor([.13, -.08], dtype=torch.float32)
        self.teacher_calls = []
        self.student_calls = []

    def teacher_log_probs(self, residual):
        with torch.no_grad():
            value = torch.log_softmax((residual.float() + self.angles) @ self.weight.T / .8, -1)
        self.teacher_calls.append((residual.shape[0], value.requires_grad, value.shape[-1]))
        return value

    def student_log_probs(self, residual):
        value = torch.log_softmax(residual.float() @ self.weight.T / .8, -1)
        self.student_calls.append((residual.shape[0], value.requires_grad, value.shape[-1]))
        return value


@pytest.mark.parametrize('lengths', [(2, 5), (3,)])
@pytest.mark.parametrize('chunk', [1, 2, 32])
def test_streamed_exact_full_vocab_gradient_and_whole_window_token_weighting(lengths, chunk):
    # (3,) is the final incomplete accumulation window: normalization is by its
    # three positions, never by the nominal accumulation16 or a sample mean.
    weight = torch.tensor([[.6, -.1], [.2, .7]], requires_grad=True)
    expected_weight = weight.detach().clone().requires_grad_(True)
    engine = ToyTail(chunk)
    reference_engine = ToyTail(chunk)
    inputs = [torch.arange(1, n * 2 + 1, dtype=torch.float32).reshape(n, 2) / 11 for n in lengths]
    cleans = [x + .07 for x in inputs]
    total = sum(lengths)
    expected_losses = []
    for x, clean in zip(inputs, cleans):
        q = reference_engine.teacher_log_probs(clean)
        p = reference_engine.student_log_probs(x @ expected_weight)
        expected_losses.append(forward_kl(q, p))
    expected = torch.cat(expected_losses).sum() / total
    expected.backward()
    observed_values = []
    backbone_backward_calls = []
    for x, clean in zip(inputs, cleans):
        student = x @ weight
        student.register_hook(lambda gradient: backbone_backward_calls.append(gradient.detach().clone()))
        observed_values.extend(distill_backward(engine, student, clean, total))
    assert len(backbone_backward_calls) == len(lengths)
    torch.testing.assert_close(weight.grad, expected_weight.grad, rtol=2e-5, atol=2e-7)
    assert sum(observed_values) / total == pytest.approx(float(expected.detach()), rel=2e-5, abs=2e-7)
    assert all(n <= chunk and detached is False and vocab == 7
        for n, detached, vocab in engine.teacher_calls)
    assert all(n <= chunk and attached is True and vocab == 7
        for n, attached, vocab in engine.student_calls)


def test_forward_kl_keeps_last_vocabulary_candidate_and_no_temperature_squared():
    q = torch.tensor([[.1, .2, .7]])
    logits = torch.tensor([[.2, -.1, .4]], requires_grad=True)
    log_p = torch.log_softmax(logits / .8, -1)
    value = forward_kl(q.log(), log_p)
    expected = (q * (q.log() - log_p)).sum(-1)
    torch.testing.assert_close(value, expected)
    value.sum().backward()
    assert logits.grad[0, -1] != 0  # Last category may be EOS; it stays in KL.


def test_authoritative_teacher_is_detached_but_native_student_tail_is_attached():
    engine = TeacherStudent.__new__(TeacherStudent)
    engine.cfg = SimpleNamespace(tau=.8)
    engine.angles = torch.tensor([.2, -.1])
    engine.norm = torch.nn.Identity()
    engine.head = torch.nn.Linear(2, 3, bias=False)
    engine.head.requires_grad_(False)
    received = []
    class Readout:
        def log_probs(self, residual, angles):
            received.append(angles)
            return torch.log_softmax(torch.cat([residual, residual[:, :1]], -1), -1)
    engine.readout = Readout()
    residual = torch.tensor([[.3, -.2]], requires_grad=True)
    log_q = engine.teacher_log_probs(residual)
    log_p = engine.student_log_probs(residual)
    assert received[0] is engine.angles
    assert not log_q.requires_grad and log_p.requires_grad
    forward_kl(log_q, log_p).sum().backward()
    assert residual.grad is not None and torch.isfinite(residual.grad).all()
    assert engine.head.weight.grad is None


def test_public_training_helpers_and_canonical_position_policy():
    assert tuple(inspect.signature(train.run).parameters) == ('state', 'dataset')
    assert tuple(inspect.signature(train.merge).parameters) == ('state',)
    assert tuple(inspect.signature(train.server_smoke).parameters) == ('state', 'dataset')
    source = inspect.getsource(train._encode)
    assert 'encode_response(tokenizer' in source
    assert 'generated_ids' not in source and 'generated_token_ids' not in source
    # Encoding, not sampled native EOS metadata, owns response prediction steps.
    selected = inspect.getsource(TeacherStudent.selected)
    assert 'self.backbone(' in selected and 'self.model(' not in selected
    assert 'register_forward_pre_hook' in selected and 'hook.remove()' in selected
    assert 'self.model.disable_adapter() if clean' in selected
    assert 'torch.no_grad() if clean' in selected


def test_first_effective_lora_update_check_skips_zero_lr_warmup_and_survives_resume():
    from klrd_rs.e3.train import needs_first_update_check
    progress={'global_step':0}
    assert not needs_first_update_check(progress,0.)
    progress['global_step']=1
    assert needs_first_update_check(progress,1e-6)
    progress['first_real_step_verified']=True
    assert not needs_first_update_check(progress,2e-6)
