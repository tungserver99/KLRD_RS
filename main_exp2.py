"""Full predeclared45000-response study by default; no training or later stage."""
import argparse
from pathlib import Path
import sys,time,uuid,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--pilot-only',action='store_true',help='Predeclared group0, all8 keys+clean; PILOT_ONLY, not full completion')
    mode.add_argument('--analyze-only',action='store_true',help='Merge/score-report already committed canonical score caches; no generation')
    parser.add_argument('--owner',choices=['clean',*map(str,range(8)),'23','59'],help='Optional deterministic owner shard')
    parser.add_argument('--group',type=int,choices=range(5),help='Optional predetermined Q1000 group shard')
    parser.add_argument('--chunk',type=int,choices=range(125),help='Optional fixed8-response chunk within group')
    parser.add_argument('--max-jobs',type=int,help='Stop after this many newly scored jobs; mark partial study INCOMPLETE')
    args=parser.parse_args(argv)
    if args.max_jobs is not None and args.max_jobs<1:parser.error('--max-jobs must be positive')
    root=Path(__file__).resolve().parent;out=(args.output or root/'experiment2_rs/run_010').resolve()
    for name in ['experiment1_rs','experiment1_rs_continuation','inputs','continuation_inputs','e2_inputs','e2_inputs_v2',*[f'experiment2_rs/run_{i:03}' for i in range(1,10)]]:
        protected=(root/name).resolve()
        if out==protected or protected in out.parents or out in protected.parents:
            parser.error('E1 runs, frozen inputs and historical E2 runs 001 through 009 are read-only; select an independent vLLM output directory')
    from klrd_rs.io import write_json
    from klrd_rs.e2.config import E2Config
    from klrd_rs.e2.frozen import load_frozen,ArtifactError
    from klrd_rs.e2.sampler import SamplerInvalid
    from klrd_rs.e2.runner import execute
    from klrd_rs.e2.utility import redact
    start=time.perf_counter();session=uuid.uuid4().hex
    try:
        print('E2: validating exact E1 artifacts and all64 key hashes...',flush=True)
        frozen=load_frozen(root)
        result=execute(root,out,frozen,E2Config(),args)
        print('Experiment2:',result['status'],flush=True)
        print('Review package:',out/'KLRD_RS_experiment2_review.zip',flush=True)
        exit_code=0 if result['complete'] else 1
    except (OSError,ValueError,RuntimeError,ImportError,KeyError) as exc:
        execution='BLOCKED_MISSING_OR_CHANGED_ARTIFACT' if isinstance(exc,ArtifactError) else 'SAMPLER_INVALID' if isinstance(exc,SamplerInvalid) else 'BLOCKED_TECHNICAL_OR_INCOMPLETE'
        if 'INSUFFICIENT_ELIGIBLE_FRESH_PROMPTS' in str(exc):execution='INSUFFICIENT_ELIGIBLE_FRESH_PROMPTS'
        elif 'GPU_UNAVAILABLE' in str(exc):execution='GPU_UNAVAILABLE'
        result={'status':'E2_INCONCLUSIVE','execution_status':execution,'complete':False,
            'scientific_success':False,'reason':redact(str(exc)),'error_type':type(exc).__name__}
        from filelock import FileLock
        out.mkdir(parents=True,exist_ok=True,mode=0o700)
        with FileLock(str(out/'initialization.lock')):
            write_json(out/'last_error.json',result)
            # Preserve a previous completed report when a later incompatible attempt fails.
            if not (out/'report/summary.json').exists():write_json(out/'report/summary.json',result)
        print(execution+': '+str(exc),file=sys.stderr,flush=True);exit_code=2
    write_json(out/('provenance/session_runs/'+session+'.json'),{'session_wall_seconds':time.perf_counter()-start,
        'execution_status':result['status'],'note':'Recorded session wall time; interrupted uncommitted GPU work may be absent from per-response timing'})
    if (out/'report/experiment2_report.md').exists():
        from filelock import FileLock
        from klrd_rs.e2.report import package
        with FileLock(str(out/'analysis.lock')):package(root,out)
    return exit_code


if __name__=='__main__':raise SystemExit(main())
