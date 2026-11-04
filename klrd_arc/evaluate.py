"""Identical standard utility evaluation and explicit metric deltas."""
import gc
from pathlib import Path
import torch
from .artifacts import dump_json, load_json
from .model import load_model

METRICS = {'mmlu': 'acc,none', 'hellaswag': 'acc_norm,none',
           'arc_challenge': 'acc_norm,none', 'winogrande': 'acc,none',
           'truthfulqa_mc2': 'acc,none'}


def utility_delta(clean, fingerprint, tasks=None):
    chosen = list(METRICS) if tasks is None else tasks
    tasks = {}
    for task in chosen:
        metric = METRICS[task]
        c = float(clean['results'][task][metric])
        f = float(fingerprint['results'][task][metric])
        tasks[task] = {'metric': metric, 'clean': c, 'fingerprint': f,
                       'delta_percentage_points': 100 * (f - c),
                       'absolute_delta_percentage_points': 100 * abs(f - c)}
    return {'tasks': tasks, 'average_delta_percentage_points':
        sum(r['delta_percentage_points'] for r in tasks.values()) / len(tasks),
        'average_absolute_delta_percentage_points':
        sum(r['absolute_delta_percentage_points'] for r in tasks.values()) / len(tasks)}


def evaluate_utility(config, metadata, tokenizer, export, tasks=None):
    tasks = ['arc_challenge'] if tasks is None else list(dict.fromkeys(tasks))
    if not tasks or any(task not in METRICS for task in tasks):
        raise ValueError('Unknown or empty utility task selection')
    profile = '__'.join(sorted(tasks))
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    root = Path(config.output) / 'reports'
    root.mkdir(parents=True, exist_ok=True)
    combined = {}
    for name, model_id, revision in [('clean', config.model, metadata['model_revision']),
                                    ('fingerprint', export, None)]:
        pending = [task for task in tasks if not (root / f'utility_{task}_{name}.json').exists()]
        if pending:
            model = load_model(model_id, revision)
            wrapper = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1, max_length=4096)
            try:
                for task in pending:
                    print(f'Utility START: model={name}, task={task}', flush=True)
                    result = simple_evaluate(model=wrapper, tasks=[task], num_fewshot=0,
                        batch_size=1, limit=None, apply_chat_template=True, log_samples=False,
                        bootstrap_iters=1000, random_seed=config.data_seed,
                        numpy_random_seed=config.data_seed, torch_random_seed=config.data_seed,
                        fewshot_random_seed=config.data_seed)
                    if result is None or task not in result.get('results', {}):
                        raise RuntimeError(f'Utility evaluation returned no results for {task}')
                    path = root / f'utility_{task}_{name}.json'
                    dump_json(path, result)
                    print(f"Utility SAVED: model={name}, task={task}, "
                          f"{METRICS[task]}={result['results'][task][METRICS[task]]}; {path}", flush=True)
                    del result
            finally:
                del wrapper, model
                gc.collect()
                torch.cuda.empty_cache()
        combined[name] = {'results': {}}
        for task in tasks:
            result = load_json(root / f'utility_{task}_{name}.json')
            combined[name]['results'][task] = result['results'][task]
    report = utility_delta(combined['clean'], combined['fingerprint'], tasks=tasks)
    report['settings'] = {'tasks': tasks, 'num_fewshot': 0, 'batch_size': 1, 'max_length': 4096,
                          'apply_chat_template': True, 'limit': None,
                          'seed': config.data_seed, 'bootstrap_iters': 1000,
                          'persistence': 'One raw result per task and model, saved immediately after that task returns'}
    dump_json(root / f'utility_comparison_{profile}.json', report)
    dump_json(root / 'utility_comparison.json', report)
    return report


def benchmark_saved_run(config, tasks=None):
    """Opt-in utility only, without repeating fingerprint stages."""
    from transformers import AutoTokenizer
    from .config import Config
    from .pipeline import check_environment, report_saved_run
    from dataclasses import replace
    root = Path(config.output)
    metadata = load_json(root / 'private' / 'manifest.json')
    saved = replace(Config(**metadata['signature']['config']), output=config.output)
    if not (root / 'reports' / 'detection.json').exists():
        raise RuntimeError('Run fingerprint detection before optional utility evaluation')
    merge = load_json(root / 'private' / 'checkpoints' / 'merge.json')
    if not merge.get('fidelity_gate_passed') or not merge.get('reload_verified'):
        raise RuntimeError('Utility evaluation requires a validated ordinary export')
    check_environment()
    tokenizer = AutoTokenizer.from_pretrained(root / 'private' / 'reference_tokenizer',
                                             local_files_only=True, trust_remote_code=False)
    evaluate_utility(saved, metadata, tokenizer, str(root / 'released_model'), tasks=tasks)
    return report_saved_run(saved)
