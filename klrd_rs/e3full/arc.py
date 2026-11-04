"""Reuse the byte-identical KLRD ARC driver with explicit E3 model adapters.

The original driver owns batching, scoring, seeds and serialization. Temporary
injections supply ordinary FP32 models, the exact teacher hook, and the native
harness response cache. No generated-answer scorer or free-generation tau is
introduced. Run only as the explicit server ARC phase under the full-run lock.
"""
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
import gc
import hashlib
import json
import math
import os
import shutil
import time

from ..io import digest, read_json, sha256, write_csv, write_json
from ..e3.evaluate import verify_merged
from ..e3.provenance import environment, get_frozen
from .common import checked, complete


TASK = 'arc_challenge'
METRIC = 'acc_norm,none'
HARNESS_VERSION = '0.4.13'
DATA_SEED = 20261021
SOURCE_HASHES = {
    'evaluate.py': 'f5662e30593fbfaf9e7f4a2aba549d4afe49851540478aafe71652609f98c46a',
    'model.py': '786dbb44f2f3aceeae9e2208ea0a3d7cc99ae159a3a25533bafa8d7ea8ba3aea',
    'data.py': '24ab4603b1c8200a5bd419eb08079965093571ca5450d73279edd8425de9527b',
    'artifacts.py': '31f4a3f294e31d13685718b59884d7d2cd3b7d7ccfe258a933f5505afef0f583',
}
SETTINGS = {
    'tasks': [TASK], 'metric': METRIC, 'num_fewshot': 0,
    'batch_size': 1, 'max_length': 4096, 'limit': None,
    'apply_chat_template': True, 'log_samples': False,
    'bootstrap_iters': 1000, 'data_seed': DATA_SEED,
    'scoring': 'native HFLM answer-choice loglikelihood; character-normalized acc_norm',
    'likelihood_temperature': 1.0,
    'free_generation_tau_applied': False,
    'attention_implementation': 'sdpa', 'autocast': False,
    'historical_model_dtype': 'bfloat16', 'model_dtype': 'float32',
    'numerically_equivalent_to_historical_bf16': False,
    'precision_reason': 'Match the frozen exact teacher FP32 geometry/native readout across all four models',
    'checkpoint_selection': 'existing validation-KL selection only; no ARC-based selection',
}


def _canonical(value):
    """JSON values for provenance only; never store raw benchmark documents."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False,
        default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))


def _immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('ARC protocol/model/cache binding changed: ' + str(path))
    else:
        write_json(path, value)


def _verify_vendor(root):
    actual = {name: sha256(Path(root) / 'klrd_arc' / name) for name in SOURCE_HASHES}
    if actual != SOURCE_HASHES:
        raise ValueError('Original vendored ARC source bytes changed')
    receipt = read_json(Path(root) / 'klrd_arc/SOURCE_RECEIPT.json')
    if receipt['files'] != SOURCE_HASHES:
        raise ValueError('ARC vendoring receipt changed')
    return actual


def _load_task():
    from lm_eval.tasks import TaskManager
    manager = TaskManager()
    loaded = manager.load([TASK])
    if set(loaded['tasks']) != {TASK}:
        raise ValueError('Expected exactly the original ARC-Challenge task')
    task = loaded['tasks'][TASK]
    # This is the very same override made by the original simple_evaluate call.
    task.set_config(key='num_fewshot', value=0)
    return manager, task


def _task_binding(task):
    config = _canonical(task.dump_config())
    if (config.get('dataset_path') != 'allenai/ai2_arc'
            or config.get('dataset_name') != 'ARC-Challenge'
            or config.get('test_split') != 'test'
            or config.get('output_type') != 'multiple_choice'
            or config.get('num_fewshot') != 0):
        raise ValueError('Original harness ARC-Challenge task semantics changed')
    docs = task.eval_docs
    ids = [str(row['id']) for row in docs]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('ARC test examples must have nonempty unique IDs')
    # Hash complete ordered document contents as well as IDs. Matching row IDs
    # alone would not detect changed questions/choices/answer keys.
    rows_hash = hashlib.sha256()
    for row in docs:
        rows_hash.update(json.dumps(_canonical(row), sort_keys=True,
            ensure_ascii=False, separators=(',', ':')).encode('utf-8') + b'\n')
    fingerprint = getattr(docs, '_fingerprint', None)
    if not fingerprint:
        raise ValueError('ARC test dataset has no Hugging Face fingerprint')
    info = getattr(docs, 'info', None)
    return {
        'task_config': config, 'task_config_sha256': digest(config),
        'task_version': _canonical(task.VERSION),
        'test_split': 'test', 'test_count': len(ids),
        'ordered_test_ids_sha256': digest(ids),
        'ordered_test_documents_sha256': rows_hash.hexdigest(),
        'dataset_fingerprint': fingerprint,
        'dataset_version': str(getattr(info, 'version', None)),
        'original_dataset_revision': None,
        'original_dataset_revision_limitation': 'Original KLRD did not explicitly pin the dataset commit; this run freezes actual loaded data/config and checks every model against it',
    }


def _harness_sources():
    import lm_eval
    root = Path(lm_eval.__file__).parent
    names = ['evaluator.py', 'evaluator_utils.py', 'api/task.py', 'api/model.py',
        'models/huggingface.py', 'tasks/arc/arc_challenge.yaml', 'tasks/arc/arc_easy.yaml']
    return {name: sha256(root / name) for name in names}


def _counts(result, expected_count):
    """Recover the integer normalized-correct count, rejecting partial results."""
    samples = result['n-samples'][TASK]
    original, effective = samples['original'], samples['effective']
    if (isinstance(original, bool) or isinstance(effective, bool)
            or int(original) != original or int(effective) != effective
            or int(original) != expected_count or int(effective) != expected_count):
        raise ValueError('ARC must evaluate every original test example')
    accuracy = float(result['results'][TASK][METRIC])
    count = accuracy * int(effective)
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1 or abs(count - round(count)) > 1e-6:
        raise ValueError('ARC normalized accuracy does not recover an integer correct count')
    if result['n-shot'][TASK] != 0:
        raise ValueError('ARC result is not zero-shot')
    return {'number_evaluated': int(effective), 'correct_count': int(round(count)),
        'total_count': int(effective), 'accuracy': accuracy, 'accuracy_percent': 100 * accuracy,
        'metric': METRIC,
        'correct_count_definition': 'normalized-choice correct count, recovered from unrounded native acc_norm times effective count and checked for integrality'}


def _request_binding(task):
    """Hash the exact native chat-templated requests after harness construction."""
    rows = []
    for request in task.instances:
        if request.request_type != 'loglikelihood':
            raise ValueError('ARC unexpectedly requested generation or another scorer')
        rows.append({'doc_id': request.doc_id, 'idx': request.idx,
            'request_type': request.request_type, 'args': _canonical(request.args)})
    if not rows:
        raise ValueError('ARC produced no likelihood requests')
    return {'request_count': len(rows), 'ordered_requests_sha256': digest(rows)}


def _teacher_hook(model, frozen, cfg):
    import torch
    from ..math import rotate
    if model.config.model_type != 'qwen2':
        raise ValueError('Exact ARC teacher requires the frozen Qwen2 architecture')
    norm, head = model.model.norm, model.get_output_embeddings()
    if any(p.dtype != torch.float32 for module in (norm, head) for p in module.parameters()):
        raise ValueError('Exact ARC teacher requires native FP32 RMSNorm/head')
    device = head.weight.device
    p = torch.as_tensor(frozen['P'], device=device, dtype=torch.float32)
    q = torch.as_tensor(frozen['Q'], device=device, dtype=torch.float32)
    angles = float(frozen['strength']) * (
        torch.as_tensor(frozen['L'], device=device, dtype=torch.float32) @
        torch.as_tensor(frozen['bank'][cfg.owner_key_id], device=device, dtype=torch.float32))
    if angles.shape != (cfg.m,) or not torch.isfinite(angles).all():
        raise ValueError('Invalid frozen exact ARC teacher angles')

    def pre_final_norm(module, args):
        # args[0] already contains the final residual sum. Native norm/head run
        # afterwards; exact finite rotation only, without tau or logits bias.
        with torch.autocast('cuda', enabled=False):
            return (rotate(args[0].float(), p, q, angles),) + args[1:]

    return norm.register_forward_pre_hook(pre_final_norm)


@contextmanager
def _driver_injections(state, label, descriptor, protocol, tokenizer):
    """Restore all process globals and remove the teacher hook on any failure."""
    import torch
    import lm_eval
    from transformers import AutoModelForCausalLM
    from klrd_arc import evaluate as original
    previous_loader, previous_evaluate = original.load_model, lm_eval.simple_evaluate
    previous_dump = original.dump_json
    active = {'label': None, 'hooks': []}
    protocol_hash = digest(protocol)

    def load_model(model_id, revision=None):
        is_clean = model_id == state['cfg'].model and revision == state['cfg'].revision
        current = 'clean' if is_clean else label
        if not is_clean and model_id != 'klrd-rs-arc:' + label:
            raise ValueError('Unexpected original ARC model request')
        if active['hooks']:
            raise RuntimeError('Previous ARC model hook was not released')
        checkpoint = state['cfg'].model if current in ('clean', 'teacher') else descriptor['path']
        model_revision = state['cfg'].revision if current in ('clean', 'teacher') else None
        if not torch.cuda.is_available():
            raise RuntimeError('ARC requires the server CUDA GPU')
        model = AutoModelForCausalLM.from_pretrained(checkpoint, revision=model_revision,
            torch_dtype=torch.float32, attn_implementation='sdpa',
            local_files_only=True, trust_remote_code=False).to('cuda').eval()
        model.requires_grad_(False)
        if any(parameter.dtype != torch.float32 for parameter in model.parameters() if parameter.is_floating_point()):
            raise ValueError('All ARC models must use matched FP32 parameters')
        if current == 'teacher':
            active['hooks'].append(_teacher_hook(model, get_frozen(state), state['cfg']))
        active['label'] = current
        return model

    def simple_evaluate(**kwargs):
        current = active['label']
        if current not in ('clean', label):
            raise RuntimeError('ARC model adapter was not initialized')
        required = {'tasks': [TASK], 'num_fewshot': 0, 'batch_size': 1, 'limit': None,
            'apply_chat_template': True, 'log_samples': False, 'bootstrap_iters': 1000,
            'random_seed': DATA_SEED, 'numpy_random_seed': DATA_SEED,
            'torch_random_seed': DATA_SEED, 'fewshot_random_seed': DATA_SEED}
        if any(kwargs.get(key) != value for key, value in required.items()):
            raise ValueError('Original ARC evaluator call settings changed')
        manager, task = _load_task()
        if _task_binding(task) != protocol['benchmark']:
            raise ValueError('ARC test dataset/config changed between models or resume')
        current_descriptor = protocol['models'][current]
        binding = {'identity': state['identity'], 'model': current_descriptor,
            'protocol_sha256': protocol_hash}
        cache = state['out'] / 'private/arc_c' / current
        _immutable_json(cache / 'binding.json', binding)
        # Use the harness's own CachingLM via its supported use_cache option.
        # It installs native per-request partial hooks and an autocommit SQLite
        # database; no benchmark examples/requests are copied into public output.
        kwargs.update(tasks=[task], task_manager=manager, use_cache=str(cache / 'likelihood'))
        started = time.monotonic()
        try:
            with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                result = previous_evaluate(**kwargs)
            _counts(result, protocol['benchmark']['test_count'])
            requests = _request_binding(task)
            _immutable_json(state['out'] / 'arc_c/request_binding.json', requests)
            result['klrd_rs_arc_provenance'] = dict(binding,
                request_binding=requests, runtime_seconds=time.monotonic() - started,
                runtime_scope='this invocation including cache hits; interrupted invocations excluded',
                matched_dtype='float32', numerical_equivalence_to_historical_bf16=False)
            return result
        finally:
            for hook in active['hooks']:
                hook.remove()
            active['hooks'].clear()
            # CachingLM installs its hook on the HFLM instance. Explicitly flush
            # and close its SQLitedict thread so it cannot retain a GPU model.
            wrapper = kwargs['model']
            cache_hook = getattr(wrapper, 'cache_hook', None)
            database = getattr(cache_hook, 'dbdict', None)
            if database is not None:
                database.commit()
                database.close()
                from lm_eval.api.model import CacheHook
                wrapper.set_cache_hook(CacheHook(None))

    def dump_json(path, value):
        # Preserve the original byte serializer, then publish each completed
        # model immediately. Clean remains reviewable if the teacher is killed.
        previous_dump(path, value)
        name = Path(path).name
        if name == 'utility_arc_challenge_clean.json':
            _publish_result(state, Path(path), 'clean', protocol)
        elif name == 'utility_arc_challenge_fingerprint.json':
            _publish_result(state, Path(path), label, protocol)

    original.load_model, lm_eval.simple_evaluate = load_model, simple_evaluate
    original.dump_json = dump_json
    try:
        yield original
    finally:
        original.load_model, lm_eval.simple_evaluate = previous_loader, previous_evaluate
        original.dump_json = previous_dump
        for hook in active['hooks']:
            hook.remove()
        active['hooks'].clear()
        gc.collect()
        torch.cuda.empty_cache()


def _validated_result(state, path, label, protocol):
    result = read_json(path)
    provenance = result.get('klrd_rs_arc_provenance', {})
    if (provenance.get('identity') != state['identity']
            or provenance.get('model') != protocol['models'][label]
            or provenance.get('protocol_sha256') != digest(protocol)
            or provenance.get('request_binding') != read_json(state['out'] / 'arc_c/request_binding.json')
            or result.get('configs', {}).get(TASK) != protocol['benchmark']['task_config']):
        raise ValueError('Unbound/changed ARC result cannot be reused: ' + str(path))
    _counts(result, protocol['benchmark']['test_count'])
    return result


def _publish_result(state, source, label, protocol):
    committed = checked(state, 'arc_c/' + label + '/complete.json')
    result = _validated_result(state, source, label, protocol)
    destination = state['out'] / 'arc_c' / label / 'result.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise ValueError('Committed ARC result changed: ' + label)
    elif source != destination:
        temporary = destination.with_name(destination.name + '.tmp')
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    summary = dict(_counts(result, protocol['benchmark']['test_count']), model=label,
        checkpoint=protocol['models'][label], identity=state['identity'],
        protocol_sha256=digest(protocol), result_sha256=sha256(destination), dtype='float32',
        runtime_seconds=result['klrd_rs_arc_provenance']['runtime_seconds'])
    summary.update(correct=summary['correct_count'], total=summary['total_count'])
    _immutable_json(destination.with_name('summary.json'), summary)
    if not committed:
        complete(state, 'arc_c/' + label + '/complete.json',
            ['arc_c/' + label + '/result.json', 'arc_c/' + label + '/summary.json'],
            model=label, status='complete', protocol_sha256=digest(protocol),
            correct=summary['correct_count'], total=summary['total_count'],
            accuracy=summary['accuracy'], dtype='float32')
    return summary


def run(state):
    """Evaluate clean, exact key0 teacher, pilot and full exports sequentially."""
    if state['cfg'].profile != 'full' or state['cfg'].owner_key_id != 0:
        raise ValueError('ARC phase requires the full run and predeclared owner key0')
    pilot = state['pilot']
    if (pilot['cfg'].profile != 'pilot' or pilot['cfg'].model != state['cfg'].model
            or pilot['cfg'].revision != state['cfg'].revision):
        raise ValueError('ARC pilot must share the pinned clean base model/revision')
    vendor = _verify_vendor(state['root'])
    # Shallow copies prevent the legacy read-only verifier adding cache fields
    # to the shared pilot/root state. It never writes checkpoint artifacts.
    full_merge, pilot_merge = verify_merged(dict(state)), verify_merged(dict(pilot))
    done = checked(state, 'arc_c/complete.json')
    if done:
        return done
    if metadata.version('lm_eval') != HARNESS_VERSION:
        raise RuntimeError('Install the separate ARC server requirements: lm_eval==' + HARNESS_VERSION)
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or int(os.environ.get('LOCAL_RANK', '0')) != 0:
        raise ValueError('Original ARC protocol requires one sequential single-GPU evaluator')
    from transformers import AutoTokenizer
    cfg = state['cfg']
    frozen = get_frozen(state)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision,
        local_files_only=True, trust_remote_code=False)
    _, task = _load_task()
    geometry = digest(_canonical({name: frozen[name] for name in ('P', 'Q', 'L', 'strength')}))
    owner = digest(_canonical(frozen['bank'][cfg.owner_key_id]))
    models = {
        'clean': {'model': cfg.model, 'revision': cfg.revision, 'rotation': False},
        'teacher': {'model': cfg.model, 'revision': cfg.revision, 'rotation': 'exact pre-final-RMSNorm',
            'owner_key_id': 0, 'geometry_sha256': geometry, 'owner_key_sha256': owner},
        'pilot_student': {'path': str((pilot['out'] / 'merged_student').resolve()),
            'run_identity': pilot['identity'], 'merged_receipt_sha256': digest(pilot_merge), 'rotation': False},
        'full_student': {'path': str((state['out'] / 'merged_student').resolve()),
            'run_identity': state['identity'], 'merged_receipt_sha256': digest(full_merge), 'rotation': False},
    }
    protocol = {'identity': state['identity'], 'settings': SETTINGS,
        'adapter_source_sha256': sha256(__file__),
        'vendored_source_sha256': vendor, 'harness_version': HARNESS_VERSION,
        'harness_source_sha256': _harness_sources(), 'benchmark': _task_binding(task),
        'tokenizer': {'model': cfg.model, 'revision': cfg.revision,
            'chat_template_sha256': digest(tokenizer.chat_template),
            'vocabulary_sha256': digest(tokenizer.get_vocab()),
            'special_tokens_sha256': digest(_canonical(tokenizer.special_tokens_map))},
        'models': models, 'runtime': dict(environment(), lm_eval=HARNESS_VERSION),
        'historical_reuse': 'code/protocol reuse; no numeric equivalence claim to original BF16 scores'}
    _immutable_json(state['out'] / 'arc_c/protocol.json', protocol)
    del task
    summaries = {}
    clean_source = state['out'] / 'arc_c/clean/result.json'
    for label in ('teacher', 'pilot_student', 'full_student'):
        pair = state['out'] / 'arc_c' / label
        pair_clean = pair / 'reports/utility_arc_challenge_clean.json'
        if clean_source.exists():
            _validated_result(state, clean_source, 'clean', protocol)
            _publish_result(state, clean_source, 'clean', protocol)
            pair_clean.parent.mkdir(parents=True, exist_ok=True)
            if pair_clean.exists():
                if sha256(pair_clean) != sha256(clean_source):
                    raise ValueError('ARC pair clean result differs from authoritative clean')
            else:
                temporary = pair_clean.with_name(pair_clean.name + '.tmp')
                shutil.copyfile(clean_source, temporary)
                os.replace(temporary, pair_clean)
        for name in ('clean', 'fingerprint'):
            existing = pair / 'reports' / ('utility_arc_challenge_' + name + '.json')
            if existing.exists():
                _validated_result(state, existing, 'clean' if name == 'clean' else label, protocol)
        with _driver_injections(state, label, models[label], protocol, tokenizer) as original:
            original.evaluate_utility(SimpleNamespace(output=str(pair), model=cfg.model, data_seed=DATA_SEED),
                {'model_revision': cfg.revision}, tokenizer, 'klrd-rs-arc:' + label, tasks=[TASK])
        summaries['clean'] = _publish_result(state, pair_clean, 'clean', protocol)
        summaries[label] = _publish_result(state,
            pair / 'reports/utility_arc_challenge_fingerprint.json', label, protocol)
    clean_accuracy = summaries['clean']['accuracy']
    rows = []
    for label in ('clean', 'teacher', 'pilot_student', 'full_student'):
        value = summaries[label]
        rows.append({key: value[key] for key in ('model', 'accuracy', 'accuracy_percent',
            'correct_count', 'total_count', 'number_evaluated', 'metric', 'runtime_seconds')})
        rows[-1]['delta_vs_clean_percentage_points'] = 100 * (value['accuracy'] - clean_accuracy)
        rows[-1].update(correct=value['correct_count'], total=value['total_count'],
            delta_vs_clean_pp=rows[-1]['delta_vs_clean_percentage_points'], dtype='float32')
    write_csv(state['out'] / 'arc_c/comparison.csv', rows, list(rows[0]))
    _immutable_json(state['out'] / 'arc_c/summary.json', {'models': summaries,
        'protocol_sha256': digest(protocol), 'comparison': rows,
        'historical_bf16_numeric_equivalence': False, 'checkpoint_selection_uses_arc': False})
    products = ['arc_c/protocol.json', 'arc_c/request_binding.json', 'arc_c/comparison.csv', 'arc_c/summary.json']
    for label in ('clean', 'teacher', 'pilot_student', 'full_student'):
        products.extend(['arc_c/' + label + '/result.json', 'arc_c/' + label + '/summary.json',
            'arc_c/' + label + '/complete.json'])
    return complete(state, 'arc_c/complete.json', products,
        status='ARC_C_COMPLETE', protocol_sha256=digest(protocol), models=list(summaries), metric=METRIC,
        total_count=protocol['benchmark']['test_count'], checkpoint_selection_uses_arc=False)
