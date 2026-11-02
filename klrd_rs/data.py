"""Historical cohort validation and text-only causal encoding.

prompt_ids/encode_response adapted from KLRD klrd/data.py; see code_reuse.json.
"""
from collections.abc import Mapping
from numbers import Integral
import numpy as np


def prompt_ids(tokenizer, prompt):
    encoded = tokenizer.apply_chat_template([{'role':'user', 'content':prompt}], tokenize=True, add_generation_prompt=True)
    if isinstance(encoded, Mapping): encoded = encoded['input_ids']
    if hasattr(encoded, 'tolist'): encoded = encoded.tolist()
    if len(encoded) == 1 and isinstance(encoded[0], (tuple,list)): encoded = encoded[0]
    if not encoded or any(not isinstance(x, Integral) or isinstance(x, bool) or x < 0 for x in encoded):
        raise ValueError('Chat template must return nonempty integer IDs')
    return list(encoded)


def encode_response(tokenizer, prompt, response):
    prefix = prompt_ids(tokenizer, prompt)
    ids = list(tokenizer.encode(response, add_special_tokens=False))
    special = set(tokenizer.all_special_ids)
    pairs = [(len(prefix)+i-1, t) for i,t in enumerate(ids) if t not in special]
    return {'input_ids':prefix+ids, 'positions':[i for i,t in pairs], 'targets':[t for i,t in pairs]}


def unique(rows):
    ids = [r['prompt_id'] for r in rows]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate prompt IDs')


def validate_rows(rows, selected, split):
    unique(rows)
    seen_ids, seen_texts = set(), set()
    for name, values in selected.items():
        unique(values)
        for row in values:
            if row['prompt_id'] in seen_ids or row['prompt'] in seen_texts:
                raise ValueError('Selection contains overlapping prompt IDs/text across splits')
            seen_ids.add(row['prompt_id']); seen_texts.add(row['prompt'])
            expected = 'test_gen' if name in ('null','final') else 'train_gen'
            if row['source'] != expected: raise ValueError('Wrong source cohort')
    expected = {r['prompt_id']:r for r in selected[split]}
    if set(expected) != {r['prompt_id'] for r in rows}: raise ValueError('Incomplete/mismatched historical cohort')
    for row in rows:
        old = expected[row['prompt_id']]
        for field in ('prompt','source'):
            if row[field] != old[field]: raise ValueError('Historical prompt identity mismatch')
        if row.get('split',split) != split or not isinstance(row.get('response'), str):
            raise ValueError('Malformed historical response')
        if row.get('generation_model_kind','clean') != 'clean': raise ValueError('Fingerprint data in clean cohort')
    return sorted(rows, key=lambda r: expected[r['prompt_id']].get('index',0))


def match_paraphrases(original, raw):
    if raw is None: return None
    clean = [r for r in raw if r['split']=='clean']
    unique(clean)
    by_id = {r['prompt_id']:r for r in original}
    if set(by_id) != {r['prompt_id'] for r in clean}: raise ValueError('Missing/unmatched clean paraphrase IDs')
    matched = []
    by_para = {r['prompt_id']:r for r in clean}
    for old in original:
        r = by_para[old['prompt_id']]
        if r['source_text'] != old['response'] or r.get('status') not in ('success','ok','completed'):
            raise ValueError('Paraphrase source/status mismatch')
        if not isinstance(r.get('paraphrased_text'),str): raise ValueError('Missing paraphrase text')
        matched.append(dict(old, response=r['paraphrased_text'], response_id=r.get('response_id',old['prompt_id'])))
    return matched


def groups(rows, budget):
    unique(rows)
    n = len(rows)//budget
    return [rows[i*budget:(i+1)*budget] for i in range(n)], len(rows)-n*budget


def partition(rows, seed):
    order = np.random.default_rng(seed).permutation(len(rows))
    cuts = [0, int(.5*len(rows)), int(.65*len(rows)), int(.8*len(rows)), len(rows)]
    if min(np.diff(cuts)) < 2: raise ValueError('Insufficient calibration prompts for independent partitions')
    return {name:[rows[int(i)] for i in order[lo:hi]] for name,lo,hi in zip(
        ['fisher','covariance','strength','local'], cuts[:-1],cuts[1:])}
