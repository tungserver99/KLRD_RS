"""GPU SERVER ONLY. Codex must not execute this script on the local machine.

Optional numerical diagnostics or a bounded throughput benchmark.
These diagnostic outputs are excluded from Experiment 2 primary observations.
"""
from pathlib import Path
import argparse,sys,time,uuid,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['parity','benchmark'],default='parity')
    parser.add_argument('--jobs',type=int,default=2,help='Benchmark batches per clean/key0; no scientific trials')
    parser.add_argument('--numerical-diagnostics',action='store_true',help='Optional non-blocking numerical diagnostics before benchmark; disabled by default')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.jobs<1 or args.jobs>10:parser.error('--jobs must be between 1 and 10')
    # Must precede torch/CUDA initialization; spawned vLLM inherits this value.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from klrd_rs.e2.vllm_backend import require_vllm,VLLMExactRotationTeacherSampler
    from klrd_rs.e2.vllm_gates import validate_vllm_sampler
    from klrd_rs.e2.config import E2Config
    from klrd_rs.e2.frozen import load_frozen
    from klrd_rs.e2.runner import prepare_cohort
    from klrd_rs.e2.sampler import E2ModelAdapter
    from klrd_rs.e2.cohort import generation_seed
    from klrd_rs.config import Config
    from klrd_rs.continuation import tensor_sha256
    from klrd_rs.io import read_json,write_json
    root=Path(__file__).resolve().parents[1]
    out=(args.output or root/'verification/vllm_server'/uuid.uuid4().hex).resolve()
    # Never allow a diagnostic to alter an experiment/frozen-input tree.
    allowed=(root/'verification/vllm_server').resolve()
    if out!=allowed and allowed not in out.parents:parser.error('--output must remain under verification/vllm_server/')
    if out.exists() and any(out.iterdir()):parser.error('Select a fresh diagnostic output directory')
    require_vllm();cfg=E2Config();frozen=load_frozen(root)
    if not torch.cuda.is_available():raise RuntimeError('GPU server required; no CPU substitution')
    if not any((root/'experiment2_rs'/r/'provenance/prompt_manifest.json').exists() for r in ['run_005','run_004','run_003','run_002']):
        raise RuntimeError('Synchronize the existing frozen E2 prompt manifest/private pool; this diagnostic does not select a replacement cohort')
    out.mkdir(parents=True,exist_ok=True)
    adapter=E2ModelAdapter(Config());engine=adapter.readout(frozen['P'],frozen['Q'])
    expected=read_json(root/'experiment1_rs_continuation/numerical/fp64_diagnostic.json')['readout_parameter_sha256']
    actual={'weight':tensor_sha256(engine.weight),'gain':tensor_sha256(engine.gain),
        'bias':None if engine.bias is None else tensor_sha256(engine.bias)}
    if actual!=expected:raise ValueError('Reference model differs from frozen E1')
    rows,manifest=prepare_cohort(root,out,frozen,adapter.tokenizer,cfg)
    sampler=VLLMExactRotationTeacherSampler(adapter,engine,frozen,cfg)
    try:
        if sampler.environment['readout_hashes']!=[expected]:raise ValueError('vLLM model differs from frozen E1')
        requests=[{'prompt_id':r['prompt_id'],'prompt_ids':r['prompt_ids'],
            'seed':generation_seed(cfg.generation_seed,0,r['prompt_id'],'vllm_server_diagnostic')} for r in rows]
        gates=validate_vllm_sampler(sampler,requests,frozen['bank'],cfg,out,
            enabled=args.mode=='parity' or args.numerical_diagnostics)
        if args.mode=='parity':
            write_json(out/'server_validation.json',{'status':gates['status'],'gates':gates,'engine':sampler.environment,
                'scope':'Optional non-blocking numerical diagnostics; not required for generation or E2 scientific acceptance'})
            print('Optional server numerical diagnostics:',gates['status'],out,flush=True)
            return 0
        timings=[]
        for owner in [None,0]:
            for job in range(args.jobs):
                batch=requests[job*8:(job+1)*8]
                begin=time.perf_counter()
                records=sampler.generate(batch,None if owner is None else frozen['bank'][owner],cfg.max_new_tokens)
                elapsed=time.perf_counter()-begin
                tokens=sum(len(r['generated_token_ids']) for r in records)
                timings.append({'owner':'clean' if owner is None else str(owner),'job':job,
                    'responses':len(records),'native_tokens':tokens,'wall_seconds_including_KL_replay':elapsed,
                    'native_tokens_per_second_including_KL_replay':tokens/elapsed,
                    'KL_audit_seconds':sum(r['KL_audit_seconds'] for r in records),
                    'generation_engine_seconds':sum(r['amortized_generation_seconds']-r['KL_audit_seconds'] for r in records)})
                print('vLLM benchmark:',timings[-1],flush=True)
        write_json(out/'benchmark.json',{'status':'SERVER_BENCHMARK_EXECUTED','gates':gates['status'],
            'scope':'Diagnostic requests, never ownership trials; no raw responses included',
            'engine':sampler.environment,'batches':timings,
            'peak_vllm_worker_CUDA_allocated_bytes':sampler.peak_worker_bytes,
            'peak_reference_process_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),
            'memory_note':'Separate process allocator peaks, not a simultaneous device-wide peak; inspect nvidia-smi for total residency'})
        print('Benchmark report:',out/'benchmark.json',flush=True)
        return 0
    except Exception as exc:
        write_json(out/'server_validation_error.json',{'status':'NOT_VALIDATED','error_type':type(exc).__name__,'reason':str(exc)})
        raise
    finally:sampler.close()


if __name__=='__main__':raise SystemExit(main())
