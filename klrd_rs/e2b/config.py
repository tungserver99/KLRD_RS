"""Frozen E2B study design. Selection never consumes scores or response text."""
import hashlib

CONDITIONS=tuple(map(str,range(8)))+('clean',)
QUERIES=(20,50,100)
ALPHAS=(.01,.0001)
COUNT=400
SELECTION_DOMAIN='KLRD_RS_E2B_SCW_400_V1'
SOURCE_RUN='experiment2_rs/run_012'
OUTPUT_RUN='experiment2_paraphrase/run_002'
SDK_VERSION='2.54.0'
SCORE_BATCH_SIZE=8


def select_prompt_ids(candidate_ids):
    ids=list(candidate_ids)
    if len(ids)!=1000 or len(set(ids))!=1000 or any(not isinstance(i,str) for i in ids):
        raise ValueError('E2B requires the original 1000 distinct group0 prompt IDs')
    return sorted(ids,key=lambda pid:(hashlib.sha256((SELECTION_DOMAIN+'\0'+pid).encode()).hexdigest(),pid))[:COUNT]


def make_groups(ids):
    if len(ids)!=COUNT or len(set(ids))!=COUNT:raise ValueError('Exactly400 shared prompt IDs required')
    return [{'Q':q,'group_index':i//q,'group_id':f'e2b_Q{q}_g{i//q:02}',
        'prompt_ids':list(ids[i:i+q])} for q in QUERIES for i in range(0,COUNT,q)]


def condition_directory(condition):
    if condition not in CONDITIONS:raise ValueError('Unknown E2B condition')
    return 'clean' if condition=='clean' else f'owner_{int(condition):02}'
