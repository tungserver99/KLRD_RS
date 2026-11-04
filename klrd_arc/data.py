"""Deterministic prompt selection and one shared text-only causal encoding."""
import random
from collections.abc import Mapping
from numbers import Integral

SIZES = {'calibration': 2000, 'train': 10000, 'validation': 1000,
         'null': 2500, 'final': 2500}


def prompt_ids(tokenizer, prompt):
    encoded = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=True,
        add_generation_prompt=True)
    # v4 may return IDs directly; v5 returns a BatchEncoding mapping.
    # Iterating a mapping yields field names, never token IDs.
    if isinstance(encoded, Mapping):
        if 'input_ids' not in encoded:
            raise TypeError('Chat template returned a mapping without input_ids')
        encoded = encoded['input_ids']
    if hasattr(encoded, 'tolist'):
        encoded = encoded.tolist()
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    if not isinstance(encoded, (list, tuple)) or not encoded or any(
            not isinstance(token, Integral) or isinstance(token, bool) or token < 0
            for token in encoded):
        raise TypeError('Chat template must return one nonempty sequence of integer input_ids')
    return [int(token) for token in encoded]


def encode_response(tokenizer, prompt, response):
    prefix = prompt_ids(tokenizer, prompt)
    if not prefix:
        raise ValueError('Empty chat prefix')
    response_ids = list(tokenizer.encode(response, add_special_tokens=False))
    special = set(tokenizer.all_special_ids)
    pairs = [(len(prefix) + i - 1, token) for i, token in enumerate(response_ids)
             if token not in special]
    return {'input_ids': prefix + response_ids,
            'positions': [p for p, _ in pairs],
            'targets': [t for _, t in pairs]}


def select_prompts(dataset, tokenizer, sizes=None, seed=20261021):
    sizes = SIZES if sizes is None else sizes
    rng = random.Random(seed)
    result, used, global_index = {}, set(), 0
    for source, partitions in [('train_gen', ['calibration', 'train', 'validation']),
                               ('test_gen', ['null', 'final'])]:
        rows = dataset[source]
        order = list(range(len(rows)))
        rng.shuffle(order)
        required = sum(sizes[p] for p in partitions)
        selected = []
        for i in order:
            row = rows[i]
            pid = str(row['prompt_id'])
            prompt = row['prompt']
            if pid in used or not isinstance(prompt, str):
                continue
            n = len(tokenizer.encode(prompt, add_special_tokens=False))
            if not 16 <= n <= 512:
                continue
            used.add(pid)
            selected.append({'prompt_id': pid, 'prompt': prompt,
                             'index': global_index, 'source': source})
            global_index += 1
            if len(selected) == required:
                break
        if len(selected) != required:
            raise ValueError(f'Insufficient unique filtered prompts in {source}: '
                             f'{len(selected)} < {required}')
        offset = 0
        for name in partitions:
            result[name] = [dict(x, split=name)
                            for x in selected[offset:offset + sizes[name]]]
            offset += sizes[name]
    return result

