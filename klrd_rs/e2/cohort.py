"""Key-independent fresh cohort, source exclusion and nested trial groups."""
import hashlib
import unicodedata
import numpy as np
from ..io import digest


def text_hash(text,normalized=False):
    if normalized:text=' '.join(unicodedata.normalize('NFKC',text).casefold().split())
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def exclusion_sets(rows):
    return {'ids':{str(r['prompt_id']) for r in rows},
        'exact':{text_hash(r['prompt']) for r in rows},'normalized':{text_hash(r['prompt'],True) for r in rows}}


def select_cohort(datasets,tokenizer,excluded,count,seed):
    from .exclusions import legacy_text_hash,source_aliases
    legacy=excluded.get('legacy_normalized',set())
    ids=set(excluded['ids']);exact=set(excluded['exact']);normalized=set(excluded['normalized']);selected=[]
    for offset,split in enumerate(['test_gen','train_gen']):
        data=datasets[split]
        order=np.random.default_rng(seed+offset).permutation(len(data))
        for index in order:
            row=data[int(index)];pid=str(row['prompt_id']);text=row['prompt']
            if not isinstance(text,str):continue
            h=text_hash(text);n=text_hash(text,True)
            if source_aliases(pid)&ids or h in exact or n in normalized or legacy_text_hash(text) in legacy:continue
            tokens=len(tokenizer.encode(text,add_special_tokens=False))
            if not 16<=tokens<=512:continue
            ids.update(source_aliases(pid));exact.add(h);normalized.add(n)
            selected.append({'prompt_id':pid,'prompt':text,'source_split':split,'source_index':int(index),
                'prompt_text_sha256':h,'normalized_prompt_sha256':n,'prompt_length':tokens})
            if len(selected)==count:return selected
    raise ValueError(f'INSUFFICIENT_ELIGIBLE_FRESH_PROMPTS: {len(selected)} eligible, need {count}')


def make_groups(rows,seed):
    if len(rows)%1000:raise ValueError('Primary cohort must consist of complete1000 groups')
    order=np.random.default_rng(seed).permutation(len(rows))
    output=[]
    for large,start in enumerate(range(0,len(rows),1000)):
        ids=[rows[int(i)]['prompt_id'] for i in order[start:start+1000]]
        group_id=f'g{large:02}_Q1000'
        output.append({'group_id':group_id,'group_index':large,'Q':1000,'prompt_ids':ids,
            'parent_group_id':None,'sha256':digest(ids)})
        for small in range(10):
            subset=ids[small*100:(small+1)*100]
            output.append({'group_id':f'g{large:02}_s{small:02}_Q100','group_index':large,'Q':100,
                'prompt_ids':subset,'parent_group_id':group_id,'sha256':digest(subset)})
    return output


def generation_seed(base,owner,prompt_id,phase):
    # Positive signed64-bit value accepted by both CPU and CUDA generators.
    return int(hashlib.sha256(f'{base}|{owner}|{prompt_id}|{phase}'.encode()).hexdigest()[:15],16)


def kl_selected(base,prompt_id,position,stride):
    return int(hashlib.sha256(f'{base}|{prompt_id}|{position}'.encode()).hexdigest()[:16],16)%stride==0
