"""Exact E1 teacher and native student tails; no generation or alternate target."""
from contextlib import nullcontext
import torch
from ..readout import RotationReadout


class TeacherStudent:
    """One FP32-master PEFT model, clean adapter-disabled teacher forwards."""
    def __init__(self, model, frozen, cfg):
        self.model, self.cfg = model, cfg
        self.base = model.get_base_model()
        if self.base.config.model_type != 'qwen2':
            raise ValueError('E3 requires the pinned ordinary Qwen2 architecture')
        self.backbone = self.base.model
        self.norm, self.head = self.backbone.norm, self.base.get_output_embeddings()
        self.device = self.head.weight.device
        for module in (self.norm, self.head):
            if any(p.dtype != torch.float32 or p.requires_grad for p in module.parameters()):
                raise ValueError('Original final norm/head must remain frozen FP32')
        self.readout = RotationReadout(self.head.weight, self.norm.weight,
            self.norm.variance_epsilon, torch.as_tensor(frozen['P']),
            torch.as_tensor(frozen['Q']), cfg.tau, self.head.bias)
        self.angles = (float(frozen['strength']) * (
            torch.as_tensor(frozen['L'], dtype=torch.float32, device=self.device) @
            torch.as_tensor(frozen['bank'][cfg.owner_key_id], dtype=torch.float32, device=self.device)))
        if self.angles.shape != (cfg.m,) or not torch.isfinite(self.angles).all():
            raise ValueError('Invalid frozen teacher angles')

    def selected(self, example, *, clean=False, training=True, force_fp32=False):
        """Hook args[0] already includes HF's residual; never add it again.

        Invoke the backbone directly, so a full-sequence vocabulary tensor is
        never constructed. The temporary hook is gone before backward/checkpoint
        recomputation, and the selected student tensor retains its LoRA graph.
        """
        ids = torch.tensor([example['input_ids']], dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        positions = example['positions']
        if not positions or min(positions) < 0 or max(positions) >= ids.shape[1] - 1:
            raise ValueError('Invalid/noncausal response prediction positions')
        captured = []
        def capture(module, args):
            captured.append(args[0][0, positions])
        self.model.train(bool(training and not clean))
        adapter_context = self.model.disable_adapter() if clean else nullcontext()
        grad_context = torch.no_grad() if clean else nullcontext()
        cast = torch.autocast('cuda', dtype=torch.bfloat16,
            enabled=bool(self.cfg.student_bf16_autocast and not clean and not force_fp32))
        hook = self.norm.register_forward_pre_hook(capture)
        try:
            with adapter_context, grad_context, cast:
                output = self.backbone(input_ids=ids, attention_mask=mask,
                    use_cache=False, return_dict=True)
                del output
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected exactly one HF pre-final-RMSNorm capture')
        return captured[0].detach() if clean else captured[0]

    def student_log_probs(self, residual):
        # FP32 native unrotated norm/head; frozen weights still propagate dr.
        with torch.autocast('cuda', enabled=False):
            return torch.log_softmax(self.head(self.norm(residual.float())) / self.cfg.tau, dim=-1)

    def teacher_log_probs(self, residual, *, rotated=True):
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            return self.readout.log_probs(residual.float(), self.angles if rotated else None)


def forward_kl(log_q, log_p):
    """Full vocabulary forward KL, including EOS, with no tau-squared factor."""
    return (log_q.exp() * (log_q - log_p)).sum(dim=-1)


def distill_backward(engine, student, clean, window_tokens):
    """Exact chain rule with bounded vocabulary graphs and ONE backbone backward.

    Each temporary leaf receives the exact tail gradient for its positions. We
    concatenate these gradients and inject them into the original, attached
    student residual. Only vocabulary graphs are detached/recreated; the LoRA
    backbone graph is traversed exactly once. Normalization covers the WHOLE
    accumulation window, including a short final window.
    """
    if student.shape != clean.shape or not student.requires_grad or window_tokens <= 0:
        raise ValueError('Attached matching student/clean residuals and token count required')
    gradient = torch.empty_like(student)
    values = []
    chunk = engine.cfg.full_vocab_logit_chunk
    for start in range(0, student.shape[0], chunk):
        end = min(start + chunk, student.shape[0])
        leaf = student[start:end].detach().float().requires_grad_(True)
        log_q = engine.teacher_log_probs(clean[start:end])
        log_p = engine.student_log_probs(leaf)
        if log_q.requires_grad or not log_p.requires_grad:
            raise RuntimeError('Teacher/student gradient isolation failed')
        losses = forward_kl(log_q, log_p)
        if not torch.isfinite(losses).all():
            raise RuntimeError('Nonfinite full-vocabulary KL')
        gradient[start:end] = torch.autograd.grad(losses.sum() / window_tokens, leaf)[0].to(student)
        values.extend(losses.detach().double().cpu().tolist())
        del leaf, log_q, log_p, losses
    student.backward(gradient)
    return values
