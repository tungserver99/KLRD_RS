"""Execution-only vLLM tuning; never part of the frozen training configuration."""
import math
import os


def tuning():
    """KLRD-style request queue, independently bounded concurrent sequences."""
    values={}
    for field,default in (('request_batch_size',128),('max_num_seqs',64),('max_num_batched_tokens',8192)):
        name='KLRD_RS_VLLM_'+field.upper()
        try:value=int(os.environ.get(name,str(default)))
        except ValueError:raise ValueError(name+' must be a positive integer') from None
        if value<1:raise ValueError(name+' must be a positive integer')
        values[field]=value
    name='KLRD_RS_VLLM_GPU_MEMORY_UTILIZATION'
    try:memory=float(os.environ.get(name,'.85'))
    except ValueError:raise ValueError(name+' must be a finite fraction between zero and one') from None
    if not math.isfinite(memory) or not 0<memory<1:
        raise ValueError(name+' must be a finite fraction between zero and one')
    values['gpu_memory_utilization']=memory
    values['max_num_seqs']=min(values['max_num_seqs'],values['request_batch_size'])
    if values['max_num_batched_tokens']<values['max_num_seqs']:
        raise ValueError('KLRD_RS_VLLM_MAX_NUM_BATCHED_TOKENS must cover concurrent sequences')
    return values


def capture_sizes(max_num_seqs):
    sizes=[];size=1
    while size<=max_num_seqs:
        sizes.append(size);size*=2
    return sorted(set(sizes+[max_num_seqs]))
