"""Server-only exact forward-KL LoRA training, stateful resume and ordinary export.

No model is loaded at import time. All generated checkpoints remain private;
public export directories contain only standard PEFT/HF assets and tokenizer.
"""
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import torch
from ..data import encode_response, prompt_ids
from .teacher import TeacherStudent, distill_backward, forward_kl
from .provenance import atomic_directory, file_manifest, append_event, get_frozen


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')).hexdigest()


def _seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _runtime(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError('E3 training/smoke/merge requires the GPU server; no CPU fallback')
    if not cfg.teacher_fp32 or cfg.per_device_train_batch_size != 1:
        raise ValueError('E3 requires FP32 teacher and batch1 causal encoding')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def assert_only_lora(model):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named or any('lora_' not in n or any(s in n for s in
            ('lm_head', 'embed_tokens', '.norm.')) for n, _ in named):
        raise RuntimeError('Only backbone LoRA matrices may be trainable')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError('All base and adapter master weights must stay FP32')
    return [p for _, p in named]


def _tokenizer(cfg, frozen=None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision, trust_remote_code=False)
    if frozen is not None and tok.chat_template != frozen['chat_template']:
        raise ValueError('Pinned tokenizer chat template differs from frozen E1')
    return tok


def _model(cfg, adapter=None):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model
    base = AutoModelForCausalLM.from_pretrained(cfg.model, revision=cfg.revision,
        torch_dtype=torch.float32, attn_implementation='sdpa', trust_remote_code=False).to('cuda')
    if base.config.model_type != 'qwen2':
        raise ValueError('Expected ordinary pinned Qwen2 base')
    base.requires_grad_(False)
    if adapter is None:
        model = get_peft_model(base, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout, bias='none', task_type='CAUSAL_LM',
            target_modules=list(cfg.target_modules)))
    else:
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    assert_only_lora(model)
    return model


def _encode(tokenizer, dataset):
    encoded, excluded = {}, []
    for split in ('train', 'validation'):
        encoded[split] = []
        for row in dataset[split]:
            item = encode_response(tokenizer, row['prompt'], row['response'])
            item['prompt_id'] = row['prompt_id']
            if item['positions']:
                encoded[split].append(item)
            else:
                excluded.append({'split': split, 'prompt_id': row['prompt_id'],
                    'reason': 'NO_VISIBLE_RESPONSE_PREDICTION_POSITIONS'})
        if not encoded[split]:
            raise ValueError('No eligible response positions in ' + split)
    return encoded, excluded


def _statistics(values):
    if not values:
        raise RuntimeError('Empty KL evidence')
    x = torch.tensor(values, dtype=torch.float64)
    if not torch.isfinite(x).all():
        raise RuntimeError('Nonfinite KL evidence')
    return {'mean': float(x.mean()), 'minimum': float(x.min()), 'maximum': float(x.max()),
        'p95': float(torch.quantile(x, .95)),
        'p99': float(torch.quantile(x, .99)), 'tokens': len(values)}


@torch.no_grad()
def validate(engine, examples):
    teacher, drift, true_clean = [], [], []
    start = time.monotonic()
    for number,example in enumerate(examples,1):
        clean = engine.selected(example, clean=True)
        student = engine.selected(example, training=False)
        for r0, rs in zip(clean.split(engine.cfg.full_vocab_logit_chunk),
                          student.split(engine.cfg.full_vocab_logit_chunk)):
            log_p = engine.student_log_probs(rs)
            log_q = engine.teacher_log_probs(r0)
            log_p0 = engine.teacher_log_probs(r0, rotated=False)
            teacher.extend(forward_kl(log_q, log_p).double().cpu().tolist())
            drift.extend(forward_kl(log_p0, log_p).double().cpu().tolist())
            true_clean.extend(forward_kl(log_q, log_p0).double().cpu().tolist())
        del clean, student
        if number==1 or number%10==0 or number==len(examples):
            print(f'E3 validation: {number}/{len(examples)} sequences; {len(teacher)} response positions scored.',flush=True)
    seconds = time.monotonic() - start
    return {'teacher_kl': _statistics(teacher), 'clean_kl': _statistics(drift),
        'teacher_vs_clean_fp32_kl': _statistics(true_clean),
        'seconds': seconds, 'tokens_per_second': len(teacher) / max(seconds, 1e-9),
        'units': 'nats/token'}


def _rng_state():
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all()}


def _restore_rng(saved):
    random.setstate(saved['python'])
    torch.set_rng_state(saved['torch'])
    torch.cuda.set_rng_state_all(saved['cuda'])


def _check_checkpoint(path, identity):
    meta = _read(path / 'complete.json')
    if meta['identity'] != identity:
        raise ValueError('Checkpoint run identity mismatch: ' + str(path))
    actual = file_manifest(path)
    actual.pop('complete.json', None)
    if actual != meta['files']:
        raise ValueError('Checkpoint integrity mismatch: ' + str(path))
    return meta


def _latest(checkpoints, identity):
    candidates = []
    if checkpoints.exists():
        for path in checkpoints.glob('resume_*'):
            if not (path / 'complete.json').is_file():
                raise ValueError('Incomplete checkpoint requires inspection: ' + str(path))
            meta = _check_checkpoint(path, identity)
            candidates.append((meta['serial'], path, meta))
    return max(candidates, default=None, key=lambda item: item[0])


def _save(checkpoints, model, optimizer, scheduler, progress, state):
    progress['serial'] += 1
    destination = checkpoints / ('resume_%08d' % progress['serial'])
    def writer(temp):
        temp = Path(temp)
        model.save_pretrained(temp / 'adapter', safe_serialization=True)
        torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'rng': _rng_state()}, temp / 'runtime.pt')
        metadata = dict(progress, identity=state['identity'], files=file_manifest(temp))
        _json(temp / 'complete.json', metadata)
    atomic_directory(destination, writer)
    return destination


def _load_runtime(path, optimizer, scheduler):
    # Locally produced private state contains Python RNG tuples; never load an
    # untrusted downloaded pickle here. Integrity was checked before this call.
    saved = torch.load(path / 'runtime.pt', map_location='cpu', weights_only=False)
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    _restore_rng(saved['rng'])


def _publish_adapter(source, destination):
    def writer(temp):
        shutil.copytree(source, temp, dirs_exist_ok=True)
    atomic_directory(destination, writer)


@torch.no_grad()
def _baseline(engine, validation):
    b = [p for name, p in engine.model.named_parameters() if 'lora_B' in name]
    zero_b = bool(b) and all(not bool(torch.count_nonzero(p)) for p in b)
    if not zero_b:
        raise RuntimeError('Fresh PEFT adapter must have zero initial LoRA B matrices')
    # Exact FP32 backbone check on one held-out clean prefix, not a calibration gate.
    example = validation[0]
    clean = engine.selected(example, clean=True)
    old_precision = engine.cfg.student_bf16_autocast
    student = engine.selected(example, training=False, force_fp32=True)
    residual_difference = float((student.float() - clean).abs().max())
    result = validate(engine, validation)
    result.update(zero_initial_lora_B=zero_b,
        untrained_zero_lora_is_clean_model=True,
        fp32_adapter_on_vs_disabled_residual_max_abs=residual_difference,
        student_backbone_bf16_autocast=old_precision,
        note='BF16 backbone numerical drift, if present, is reported in clean_kl; zero LoRA is the clean architecture.')
    return result


def needs_first_update_check(progress,learning_rate):
    """Warmup LR0 is an intended optimizer no-op, not evidence of no learning."""
    return learning_rate>0 and not progress.get('first_real_step_verified',False)

def run(state, dataset):
    """Train only; automatically export validation-best adapter + merged weights."""
    cfg, out = state['cfg'], Path(state['out'])
    _runtime(cfg)
    selection_path = out / 'training' / 'selection.json'
    if selection_path.is_file():
        if _read(selection_path)['identity'] != state['identity']:
            raise ValueError('Training selection identity mismatch')
        return merge(state)
    frozen = get_frozen(state)
    tokenizer = _tokenizer(cfg, frozen)
    encoded, excluded = _encode(tokenizer, dataset)
    data_digest = _digest(encoded)
    runtime_signature = {name: version(name) for name in ('torch', 'transformers', 'peft')}
    runtime_signature['cuda'] = torch.version.cuda
    numerical_source_files = ('klrd_rs/e3/train.py', 'klrd_rs/e3/teacher.py',
        'klrd_rs/readout.py', 'klrd_rs/data.py', 'klrd_rs/e3/config.py')
    numerical_source = {name: hashlib.sha256((Path(state['root']) / name).read_bytes()).hexdigest()
        for name in numerical_source_files}
    training_dir, checkpoints = out / 'training', out / 'checkpoints'
    training_dir.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    _json(training_dir / 'eligibility.json', {'excluded': excluded,
        'eligible_sequences': {s: len(x) for s, x in encoded.items()},
        'eligible_tokens': {s: sum(len(e['positions']) for e in x) for s, x in encoded.items()}})
    previous = _latest(checkpoints, state['identity'])
    _seed(cfg.seed)
    model = _model(cfg, previous[1] / 'adapter' if previous else None)
    engine = TeacherStudent(model, frozen, cfg)
    params = assert_only_lora(model)
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    from transformers import get_linear_schedule_with_warmup
    updates = math.ceil(len(encoded['train']) / cfg.gradient_accumulation_steps) * cfg.max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=math.ceil(cfg.warmup_ratio * updates), num_training_steps=updates)
    if previous:
        progress = {k: v for k, v in previous[2].items() if k not in ('identity', 'files')}
        if progress.get('runtime_signature') != runtime_signature:
            raise ValueError('Numerical runtime versions changed; use an independent run')
        if progress.get('numerical_source') != numerical_source:
            raise ValueError('Numerical training source changed; use an independent run')
        if progress['data_digest'] != data_digest:
            raise ValueError('Encoded training/validation inputs changed on resume')
        _load_runtime(previous[1], optimizer, scheduler)
        _json(training_dir / 'baseline.json', progress['baseline'])
    else:
        baseline = _baseline(engine, encoded['validation'])
        _json(training_dir / 'baseline.json', baseline)
        progress = {'serial': 0, 'epoch': 0, 'cursor': 0, 'order': [], 'global_step': 0,
            'data_digest': data_digest, 'numerical_source': numerical_source,
            'runtime_signature': runtime_signature, 'baseline': baseline, 'history': [],
            'epoch_losses': [], 'epoch_tokens': 0, 'elapsed_training_seconds': 0.0}
        _save(checkpoints, model, optimizer, scheduler, progress, state)
    while progress['epoch'] < cfg.max_epochs:
        epoch = progress['epoch']
        if not progress['order']:
            order = list(range(len(encoded['train'])))
            random.Random(cfg.seed + epoch).shuffle(order)
            progress['order'] = order
        order = progress['order']
        torch.cuda.reset_peak_memory_stats()
        while progress['cursor'] < len(order):
            start = time.monotonic()
            cursor = progress['cursor']
            indexes = order[cursor: cursor + cfg.gradient_accumulation_steps]
            group = [encoded['train'][i] for i in indexes]
            window_tokens = sum(len(e['positions']) for e in group)
            optimizer.zero_grad(set_to_none=True)
            update_lr = float(optimizer.param_groups[0]['lr'])
            # Linear warmup intentionally starts at LR0; check the first
            # effective update rather than treating that no-op as broken LoRA.
            first_step = needs_first_update_check(progress, update_lr)
            before_lora = [p.detach().clone() for p in params] if first_step else None
            before_base = {n: p._version for n, p in model.named_parameters()
                if not p.requires_grad} if first_step else None
            window_losses = []
            for example in group:
                clean = engine.selected(example, clean=True)
                student = engine.selected(example)
                window_losses.extend(distill_backward(engine, student, clean, window_tokens))
                del student, clean
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm, error_if_nonfinite=True)
            if not torch.isfinite(grad_norm) or float(grad_norm) == 0:
                raise RuntimeError('LoRA gradients are zero/nonfinite; training incomplete')
            optimizer.step()
            first_step_diagnostics = {}
            if first_step:
                lora_changed = any(not torch.equal(p.detach(), b) for p, b in zip(params, before_lora))
                base_unchanged = all(p._version == before_base[n] for n, p in model.named_parameters()
                    if n in before_base)
                if not lora_changed or not base_unchanged:
                    raise RuntimeError('First training update changed frozen weights or failed to update LoRA')
                progress['first_real_step_verified'] = True
                first_step_diagnostics = {'first_real_step_lora_changed': lora_changed,
                    'first_real_step_all_base_parameter_versions_unchanged': base_unchanged}
                del before_lora, before_base
            scheduler.step()
            progress['global_step'] += 1
            progress['cursor'] += len(indexes)
            progress['epoch_losses'].extend(window_losses)
            progress['epoch_tokens'] += window_tokens
            seconds = time.monotonic() - start
            progress['elapsed_training_seconds'] += seconds
            event = {'identity': state['identity'], 'global_step': progress['global_step'],
                'epoch': epoch + 1, 'cursor': progress['cursor'], 'teacher_kl': _statistics(window_losses),
                'window_tokens': window_tokens, 'seconds': seconds,
                'tokens_per_second': window_tokens / max(seconds, 1e-9),
                'learning_rate': update_lr, 'next_learning_rate': scheduler.get_last_lr()[0],
                'zero_lr_warmup': update_lr == 0.0, 'lora_grad_norm': float(grad_norm),
                'gpu_device': torch.cuda.get_device_name(),
                'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(), **first_step_diagnostics}
            # A crash can leave log rows beyond the last committed checkpoint.
            # Checkpoint cursor/global_step is authoritative; event IDs permit dedup.
            append_event(training_dir / 'steps.jsonl', event)
            print(json.dumps(event), flush=True)
            if progress['global_step'] % cfg.checkpoint_steps == 0:
                _save(checkpoints, model, optimizer, scheduler, progress, state)
        validation = validate(engine, encoded['validation'])
        epoch_metrics = {'epoch': epoch + 1, 'global_step': progress['global_step'],
            'training_teacher_kl': _statistics(progress['epoch_losses']), **validation,
            'elapsed_training_seconds': progress['elapsed_training_seconds']}
        progress['history'].append(epoch_metrics)
        epoch_adapter = checkpoints / ('epoch_%03d' % (epoch + 1))
        if not epoch_adapter.exists():
            def writer(temp):
                model.save_pretrained(Path(temp) / 'adapter', safe_serialization=True)
                _json(Path(temp) / 'metrics.json', dict(epoch_metrics, identity=state['identity']))
                _json(Path(temp) / 'complete.json', {'identity': state['identity'], 'files': file_manifest(temp)})
            atomic_directory(epoch_adapter, writer)
        else:
            _check_checkpoint(epoch_adapter, state['identity'])
        progress.update(epoch=epoch + 1, cursor=0, order=[], epoch_losses=[], epoch_tokens=0)
        _save(checkpoints, model, optimizer, scheduler, progress, state)
        _json(training_dir / 'epochs.json', progress['history'])
        append_event(training_dir / 'validation.jsonl', dict(epoch_metrics, identity=state['identity']))
    best = min(progress['history'], key=lambda row: row['teacher_kl']['mean'])
    best_path = checkpoints / 'best_adapter'
    best_epoch_path = checkpoints / ('epoch_%03d' % best['epoch'])
    _check_checkpoint(best_epoch_path, state['identity'])
    source = best_epoch_path / 'adapter'
    if not best_path.exists():
        _publish_adapter(source, best_path)
    elif file_manifest(best_path) != file_manifest(source):
        raise ValueError('Existing best adapter differs from selected validation epoch')
    selection = {'identity': state['identity'], 'selected': best, 'history': progress['history'],
        'baseline': progress['baseline'], 'selection_rule': 'minimum_validation_teacher_forward_KL',
        'adapter_checkpoint': 'checkpoints/best_adapter', 'adapter_files': file_manifest(best_path),
        'status': ('TRAINING_ONLY' if best['teacher_kl']['mean'] < progress['baseline']['teacher_kl']['mean']
                   else 'NO_TRANSFER_EVIDENCE'), 'black_box_transfer_established': False}
    _json(selection_path, selection)
    del engine, model, params, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return merge(state)


def merge(state):
    """Export selected adapter into ordinary Qwen2; export reads no frozen keys."""
    cfg, out = state['cfg'], Path(state['out'])
    selection = _read(out / 'training' / 'selection.json')
    adapter = out / 'checkpoints' / 'best_adapter'
    if selection['identity'] != state['identity'] or file_manifest(adapter) != selection['adapter_files']:
        raise ValueError('Missing/changed selected adapter identity or hashes')
    manifest_path = out / 'provenance' / 'merged_manifest.json'
    destination = out / 'merged_student'
    if manifest_path.is_file():
        manifest = _read(manifest_path)
        if manifest['identity'] != state['identity'] or file_manifest(destination) != manifest['files']:
            raise ValueError('Merged model identity/hash mismatch')
        return manifest
    _runtime(cfg)
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    # Training model was released before loading the clean export backbone.
    base = AutoModelForCausalLM.from_pretrained(cfg.model, revision=cfg.revision,
        torch_dtype=torch.float32, trust_remote_code=False, attn_implementation='sdpa').to('cuda').eval()
    model = PeftModel.from_pretrained(base, str(adapter)).eval()
    tokenizer = _tokenizer(cfg)
    prompt = torch.tensor([prompt_ids(tokenizer, 'Explain what a prime number is.')],
        dtype=torch.long, device='cuda')
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        before = model(input_ids=prompt, use_cache=False).logits[:, -1].float().cpu()
    ordinary = model.merge_and_unload(safe_merge=True).eval()
    if ordinary.__class__.__name__ != 'Qwen2ForCausalLM':
        raise RuntimeError('Merged export must be ordinary Qwen2ForCausalLM')
    export_receipt = {'identity': state['identity'], 'adapter_files': selection['adapter_files'],
        'base_revision': cfg.revision, 'model_architecture': 'Qwen2ForCausalLM'}
    if destination.exists():
        receipt_path = destination / 'export_receipt.json'
        receipt = _read(receipt_path) if receipt_path.is_file() else {}
        expected_files = receipt.pop('files', None)
        actual_files = file_manifest(destination)
        actual_files.pop('export_receipt.json', None)
        if receipt != export_receipt or actual_files != expected_files:
            raise ValueError('Unreceipted/changed merged_student directory; inspect before resuming merge')
    def writer(temp):
        ordinary.save_pretrained(temp, safe_serialization=True)
        tokenizer.save_pretrained(temp)
        _json(Path(temp) / 'export_receipt.json', dict(export_receipt, files=file_manifest(temp)))
    if not destination.exists():
        atomic_directory(destination, writer)
    del model, base, ordinary
    gc.collect()
    torch.cuda.empty_cache()
    # Reload the actual on-disk ordinary checkpoint without owner material.
    reloaded = AutoModelForCausalLM.from_pretrained(str(destination), torch_dtype=torch.float32,
        trust_remote_code=False, attn_implementation='sdpa').to('cuda').eval()
    if reloaded.__class__.__name__ != 'Qwen2ForCausalLM':
        raise RuntimeError('Reloaded merged student is not an ordinary Qwen2 model')
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        after = reloaded(input_ids=prompt, use_cache=False).logits[:, -1].float().cpu()
        generated = reloaded.generate(input_ids=prompt, max_new_tokens=16, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise RuntimeError('Nonfinite adapter/merged smoke logits')
    smoke = {'fixed_public_prompt': 'Explain what a prime number is.',
        'last_position_logits_max_abs': float((before - after).abs().max()),
        'generated_new_tokens': int(generated.shape[-1] - prompt.shape[-1]),
        'ordinary_reload': True, 'owner_material_loaded_for_merge': False,
        'note': 'Diagnostic only; no strict numerical parity threshold.'}
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    manifest = {'identity': state['identity'], 'files': file_manifest(destination),
        'adapter_checkpoint': 'checkpoints/best_adapter', 'base_revision': cfg.revision,
        'model_architecture': 'Qwen2ForCausalLM', 'precision': 'float32',
        'optimizer': {'name': 'AdamW', 'learning_rate': cfg.learning_rate,
            'weight_decay': cfg.weight_decay, 'max_grad_norm': cfg.max_grad_norm},
        'checkpoint_reason': 'minimum_validation_teacher_forward_KL',
        'selected_epoch': selection['selected']['epoch'], 'smoke': smoke,
        'merged_model_path': str(destination), 'status': selection['status']}
    _json(manifest_path, manifest)
    return manifest


def _frozen_hashes(model):
    """Server-smoke-only full weight hashes, bounded host copy per tensor chunk."""
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        digest = hashlib.sha256()
        for chunk in parameter.detach().reshape(-1).split(4 * 1024 * 1024):
            digest.update(chunk.cpu().contiguous().numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def server_smoke(state, dataset):
    """Opt-in one-update GPU bug catch; never publishes a trained checkpoint."""
    cfg, out = state['cfg'], Path(state['out'])
    _runtime(cfg)
    frozen = get_frozen(state)
    tokenizer = _tokenizer(cfg, frozen)
    encoded, _ = _encode(tokenizer, dataset)
    _seed(cfg.seed)
    model = _model(cfg)
    engine = TeacherStudent(model, frozen, cfg)
    params = assert_only_lora(model)
    # Version counters cover every frozen parameter without cloning 12 GB.
    frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    frozen_hashes_before = _frozen_hashes(model)
    originals = [p.detach().clone() for p in params]
    examples = encoded['train'][:2]
    count = sum(len(e['positions']) for e in examples)
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    optimizer.zero_grad(set_to_none=True)
    initial_clean = engine.selected(examples[0], clean=True).clone()
    initial_student = engine.selected(examples[0], training=False, force_fp32=True).detach()
    initial_difference = float((initial_clean - initial_student).abs().max())
    losses = []
    for number,example in enumerate(examples,1):
        clean = engine.selected(example, clean=True)
        student = engine.selected(example)
        losses.extend(distill_backward(engine, student, clean, count))
        del clean, student
    grad = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm, error_if_nonfinite=True)
    if not float(grad) > 0:
        raise RuntimeError('Smoke has zero LoRA gradients')
    optimizer.step()
    changed = any(not torch.equal(p.detach(), old) for p, old in zip(params, originals))
    frozen_unchanged = all(p._version == frozen_versions[n]
        for n, p in model.named_parameters() if n in frozen_versions)
    frozen_hashes_after = _frozen_hashes(model)
    frozen_hashes_unchanged = frozen_hashes_before == frozen_hashes_after
    after_clean = engine.selected(examples[0], clean=True)
    clean_difference = float((after_clean - initial_clean).abs().max())
    if not changed or not frozen_unchanged or not frozen_hashes_unchanged:
        raise RuntimeError('Smoke failed: LoRA did not change or frozen weights were mutated')
    result = {'identity': state['identity'], 'status': 'SERVER_SMOKE_COMPLETE',
        'teacher_kl': _statistics(losses), 'lora_grad_norm': float(grad),
        'lora_weights_changed': changed, 'all_frozen_parameter_versions_unchanged': frozen_unchanged,
        'all_frozen_parameter_hashes_unchanged': frozen_hashes_unchanged,
        'frozen_parameter_hashes': frozen_hashes_before,
        'fp32_untrained_adapter_vs_clean_residual_max_abs': initial_difference,
        'disabled_adapter_clean_after_update_residual_max_abs': clean_difference,
        'examples': [{'prompt_id': e['prompt_id'], 'positions': e['positions'],
            'count': len(e['positions'])} for e in examples],
        'teacher_detached_student_attached': True, 'optimizer_steps': 1,
        'note': 'One-update bug catch, not an experiment result or numerical parity campaign.'}
    _json(out / 'training' / 'server_smoke.json', result)
    del engine, model, optimizer, params, originals
    gc.collect()
    torch.cuda.empty_cache()
    return result
