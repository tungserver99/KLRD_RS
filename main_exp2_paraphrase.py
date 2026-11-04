"""E2B: fixed3600 GPT-5 high-fidelity rewrites and unchanged spherical verification.

SERVER ONLY. Never execute locally, including tests or prepare/evaluation stages.
"""
import argparse,os,time,uuid
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','paraphrase','verify','report','all'],default='all')
    parser.add_argument('--workers',type=int,default=128,help='Copied transport concurrency1..128; generation protocol unchanged')
    parser.add_argument('--output',type=Path,help='Distinct run under experiment2_paraphrase/; default run_002')
    args=parser.parse_args(argv)
    if not 1<=args.workers<=128:parser.error('--workers must be1..128')
    print(f'E2B: stage={args.stage}; loading libraries.',flush=True)
    from klrd_rs.e2b.source import resolve_output,prepare,load_prepared,load_verifier_inputs
    from klrd_rs.io import write_json
    from filelock import FileLock,Timeout
    root=Path(__file__).resolve().parent
    try:out=resolve_output(root,args.output)
    except ValueError as exc:parser.error(str(exc))
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    session=uuid.uuid4().hex;started=time.perf_counter();acquired=False
    try:
        with FileLock(str(out/'run.lock'),timeout=0):
            acquired=True
            try:
                existing=(out/'provenance/population_manifest.json').exists()
                if not existing and args.stage in ('all','paraphrase','prepare'):
                    state=prepare(root,out)
                else:
                    if not existing:raise RuntimeError('Prepare the immutable E2B population on the server first')
                    print('E2B resume: reading saved3600-response population; no E1/E2 scan.',flush=True)
                    state=load_prepared(root,out)
                if args.stage=='prepare':
                    print('E2B population/protocol prepared; no API call or model inference performed.',flush=True)
                    return 0
                from klrd_rs.e2b.transport import run,load_pairs
                if args.stage in ('all','paraphrase'):pairs=run(state,workers=args.workers)
                else:pairs=load_pairs(state) # Reads the journal only; no client/key initialization.
                if args.stage=='paraphrase':
                    print('E2B paraphrases complete. Run --stage verify on the GPU server.',flush=True)
                    return 0
                print('E2B: loading frozen verifier geometry/key bank.',flush=True)
                state['frozen']=load_verifier_inputs(root)
                if args.stage in ('all','verify'):
                    from klrd_rs.e2b.scoring import score
                    score(state,pairs)
                from klrd_rs.e2b.analysis import analyze
                result=analyze(state,pairs)
                print('E2B:',result['status'],flush=True)
                print('Report:',out/'report/experiment2b_report.md',flush=True)
                print('Semantic-quality review:',result['semantic_quality_review']['status'],flush=True)
                return 0
            finally:
                write_json(out/'provenance/executions'/(session+'.json'),{'stage':args.stage,'workers':args.workers,
                    'session_wall_seconds':time.perf_counter()-started,
                    'note':'Operational session receipt only; no credentials or environment dump'})
    except Timeout:
        print('Another E2B process holds this output run; no concurrent API/scoring process started.',flush=True)
        return 2
    except (OSError,ValueError,RuntimeError,ImportError,KeyError) as exc:
        # The copied transport sanitizes upstream API exceptions; never serialize
        # raw exceptions from OpenAI client initialization or an environment dump.
        reason=str(exc) if isinstance(exc,(ValueError,RuntimeError)) else type(exc).__name__
        if acquired:write_json(out/'report/last_error.json',{'status':'E2B_INCOMPLETE_OR_TECHNICAL_ERROR',
            'complete':False,'error_type':type(exc).__name__,'reason':reason,
            'policy':'Keep durable receipts/checkpoints. No reduced cohort, alternate model or teacher regeneration.'})
        print('E2B stopped:',reason,flush=True)
        return 2


if __name__=='__main__':raise SystemExit(main())
