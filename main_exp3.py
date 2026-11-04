"""Server-only E3. Pilot trains/validates/merges; evaluation and API are opt-in."""
import argparse,os,time,uuid
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
# New CLI runs use vLLM. Keep E3Config's historical constructor unchanged so
# optimizer resume does not mistake a generation-default edit for a loss change.
NEW_RUN_GENERATION_BACKEND='vllm'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('pilot','full','train','merge','evaluate','paraphrase','report'),default='pilot')
    p.add_argument('--output',type=Path,help='Independent experiment3_rs run; default run_001. Full requires an explicit new output.')
    p.add_argument('--legacy-root',help='Old KLRD repository for trusted clean corpus reuse; default sibling ../KLRD')
    p.add_argument('--generation-backend',choices=('hf','vllm'),default=None)
    p.add_argument('--generation-dtype',choices=('float32','bfloat16'),default=None)
    p.add_argument('--logit-chunk',type=int,help='Execution chunk only; full vocabulary remains unchanged')
    p.add_argument('--checkpoint-steps',type=int,help='Save resumable optimizer state every N optimizer steps')
    p.add_argument('--student-backbone-fp32',action='store_true',help='Disable student backbone BF16 autocast; teacher/tails already FP32')
    p.add_argument('--workers',type=int,default=None,help='Paraphrase concurrency1..128; no API unless --mode paraphrase')
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.mode=='full' and args.output is None:p.error('full needs --output experiment3_rs/run_002; pilot is never silently extended')
    print('E3: mode='+args.mode+'; loading run configuration.',flush=True)
    from klrd_rs.e3.config import E3Config
    from klrd_rs.e3.provenance import resolve_output,initialize_run,append_event
    from klrd_rs.io import write_json,read_json
    from filelock import FileLock,Timeout
    root=Path(__file__).resolve().parent
    try:out=resolve_output(root,args.output)
    except ValueError as exc:p.error(str(exc))
    overrides={}
    for name in ('legacy_root','generation_backend','generation_dtype','checkpoint_steps','workers'):
        if getattr(args,name) is not None:overrides[name]=getattr(args,name)
    if args.logit_chunk is not None:overrides['full_vocab_logit_chunk']=args.logit_chunk
    if args.student_backbone_fp32:overrides['student_bf16_autocast']=False
    new_settings={'generation_backend':NEW_RUN_GENERATION_BACKEND,**overrides}
    try:cfg=E3Config(profile='full' if args.mode=='full' else 'pilot',**new_settings)
    except ValueError as exc:p.error(str(exc))
    state={'root':root,'out':out,'cfg':cfg,'mode':args.mode,
        'requested_profile':args.mode if args.mode in ('pilot','full') else None}
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    started=time.monotonic();session=uuid.uuid4().hex;acquired=False
    try:
        with FileLock(str(out/'run.lock'),timeout=0):
            acquired=True
            initialize_run(state,create=args.mode in ('pilot','full','train'))
            # An existing identity owns scientific/execution settings. Do not
            # silently accept CLI overrides that contradict committed checkpoints.
            for name,value in overrides.items():
                if name=='workers' and args.mode=='paraphrase':continue
                if getattr(state['cfg'],name)!=value:
                    raise ValueError('Existing run setting differs for '+name+'; resume without changing it or choose a new output')
            if args.workers is not None and args.mode=='paraphrase':state['api_workers']=args.workers
            print('E3 output: '+str(out),flush=True)
            if args.mode in ('pilot','full','train'):
                from klrd_rs.e3 import data,train,report
                dataset=data.prepare(state)
                train.run(state,dataset)
                report.run(state,package=False)
                print('Training/merge finished. Evaluation command (the shell launcher runs it next):',flush=True)
                print('python main_exp3.py --mode evaluate --output '+str(out),flush=True)
            elif args.mode=='merge':
                from klrd_rs.e3 import train,report
                train.merge(state)
                report.run(state,package=False)
            elif args.mode=='evaluate':
                from klrd_rs.e3 import evaluate,report
                evaluate.run(state)
                report.run(state,package=False)
            elif args.mode=='paraphrase':
                from klrd_rs.e3 import paraphrase,report
                paraphrase.run(state)
                report.run(state,package=False)
            else:
                from klrd_rs.e3 import report
                report.run(state,package=True)
            summary_path=out/'report/summary.json'
            if summary_path.is_file():
                print('E3 status: '+read_json(summary_path)['status'],flush=True)
                print('Report: '+str(out/'report/experiment3_report.md'),flush=True)
            if args.mode=='report' and (out/'KLRD_RS_experiment3_review.zip').is_file():
                print('Review ZIP: '+str(out/'KLRD_RS_experiment3_review.zip'),flush=True)
            append_event(out/'provenance/executions.jsonl',{'session':session,'mode':args.mode,
                'seconds':time.monotonic()-started,'completed':True})
            return 0
    except Timeout:
        print('E3 stopped: another process owns this run. No concurrent work started.',flush=True)
        return 2
    except (Exception,KeyboardInterrupt) as exc:
        if acquired:
            prior=out/'report/summary.json'
            previous=read_json(prior) if prior.exists() else {}
            write_json(prior,dict(previous,status='INCOMPLETE',complete=False,
                previous_status=previous.get('status'),failed_mode=args.mode,
                error_type=type(exc).__name__,reason=str(exc),
                note='Retain committed data/checkpoints/API receipts; resume the same command.'))
            append_event(out/'provenance/executions.jsonl',{'session':session,'mode':args.mode,
                'seconds':time.monotonic()-started,'completed':False,'error_type':type(exc).__name__})
        print('E3 stopped: '+str(exc),flush=True)
        return 130 if isinstance(exc,KeyboardInterrupt) else 2


if __name__=='__main__':raise SystemExit(main())
