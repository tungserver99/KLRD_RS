"""Pinned, import-safe runtime contract shared by E2 provenance and the worker.

Reviewed against the v0.32.0 tag, not the changing latest documentation.
No vLLM import, environment mutation or CUDA initialization here.
"""
TARGET_VERSION='0.32.0'
ARCHITECTURE='KLRDRSQwen2ForCausalLM'
ATTENTION_BACKEND='TRITON_ATTN'  # Tagged backend explicitly supports torch.float32.


def runtime_identity():
    return dict(vllm_version=TARGET_VERSION,engine_version='V1',
        attention_backend=ATTENTION_BACKEND,model_runner='V2',compilation_mode=0,
        logprobs_mode='processed_logprobs',triton_f32_default='ieee',
        cudagraph_mode='FULL_DECODE_ONLY',enable_prefix_caching=False,
        enable_chunked_prefill=True,async_scheduling=False,
        max_num_seqs=8,max_model_len=2048,max_num_batched_tokens=8192,
        gpu_memory_utilization=.40)


def engine_options(settings,*,eager=False):
    # Full graphs work without torch.compile. Keep the exact mutable rotation
    # as tensor operations; no Python owner specialization/constant folding.
    return dict(model=settings['model'],revision=settings['revision'],
        tokenizer=settings['model'],tokenizer_revision=settings['revision'],
        dtype='float32',kv_cache_dtype='auto',tensor_parallel_size=1,
        pipeline_parallel_size=1,quantization=None,
        hf_overrides={'architectures':[ARCHITECTURE]},trust_remote_code=False,
        attention_config={'backend':ATTENTION_BACKEND},
        # V1 defaults to pre-temperature raw logprobs. Probes compare the
        # sampled distribution, so request the post-temperature values explicitly.
        logprobs_mode='processed_logprobs',
        gpu_memory_utilization=.40,max_model_len=2048,max_num_seqs=8,
        max_num_batched_tokens=8192,enforce_eager=bool(eager),
        compilation_config={'mode':0,'cudagraph_mode':'NONE' if eager else 'FULL_DECODE_ONLY',
            'cudagraph_capture_sizes':[] if eager else list(range(1,9))},
        enable_prefix_caching=False,enable_chunked_prefill=not eager,
        async_scheduling=False,generation_config='vllm',
        seed=settings['generation_seed'],distributed_executor_backend='uni')
