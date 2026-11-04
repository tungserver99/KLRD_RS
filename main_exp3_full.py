"""Full10k study phases. Run only on the server; no work happens on import."""
import argparse,os,time,uuid
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('train','evaluate','arc-c','paraphrase','report'),required=True)
    p.add_argument('--output',type=Path,default=Path('experiment3_full/run_001'))
    p.add_argument('--pilot-output',type=Path,default=Path('experiment3_rs/run_001'))
    p.add_argument('--legacy-root',help='Old KLRD root; default uses the saved pilot setting')
    p.add_argument('--max-new-tokens',type=int,choices=(512,1024),default=None,
        help='Evaluation only; default1024. Training remains512. Paraphrase requires1024.')
    p.add_argument('--workers',type=int,default=128,help='Paraphrase concurrency1..128')
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if not 1<=args.workers<=128:p.error('--workers must be1..128')
    if args.max_new_tokens is not None and args.mode not in ('evaluate','paraphrase'):
        p.error('--max-new-tokens belongs to evaluation/paraphrase only; training stays512')
    if args.mode=='paraphrase' and args.max_new_tokens not in (None,1024):
        p.error('This full-study paraphrase phase uses the new1024 originals only')
    root=Path(__file__).resolve().parent
    print('E3 full: phase='+args.mode+'; loading saved protocol.',flush=True)
    from klrd_rs.e3full.common import initialize,project_path,prepare_training,complete
    from klrd_rs.e3.provenance import append_event
    from klrd_rs.io import read_json,write_json
    from filelock import FileLock,Timeout
    out=project_path(root,args.output);allowed=(root/'experiment3_full').resolve()
    if out==allowed or allowed not in out.parents:p.error('--output must be a separate run under experiment3_full/')
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    acquired=False;started=time.monotonic();session=uuid.uuid4().hex
    try:
        with FileLock(str(out/'run.lock'),timeout=0):
            acquired=True
            state=initialize(root,out,args.pilot_output,args.legacy_root,args.mode,args.workers)
            if args.mode=='train':
                from klrd_rs.e3 import train
                if (out/'training/selection.json').exists():
                    print('E3 full: training already selected; checking/resuming merged export.',flush=True)
                    train.merge(state)
                else:
                    train.run(state,prepare_training(state))
                selection=read_json(out/'training/selection.json')
                complete(state,'provenance/training_complete.json',[
                    'training/selection.json','training/baseline.json','training/epochs.json',
                    'training/eligibility.json','provenance/merged_manifest.json'],
                    status='TRAINING_COMPLETE',training_initialization='clean pinned base + fresh adapter',
                    train_sequences=10000,validation_sequences=1000,
                    selected_epoch=selection['selected']['epoch'],black_box_transfer_established=False)
            elif args.mode=='evaluate':
                from klrd_rs.e3full.evaluation import run
                run(state,args.max_new_tokens or 1024)
            elif args.mode=='arc-c':
                from klrd_rs.e3full.arc import run
                run(state)
            elif args.mode=='paraphrase':
                from klrd_rs.e3full.paraphrase import run
                print('E3 full: explicit1024 paraphrase phase; up to1200 new API requests, cached successes reused.',flush=True)
                run(state)
            else:
                from klrd_rs.e3full.report import run
                result=run(state,package=True)
                print('E3 full report status: '+str(result.get('status','see report')),flush=True)
                print('Report: '+str(out/'report/full_experiment_report.md'),flush=True)
                print('Review ZIP: '+str(out/'KLRD_RS_experiment3_full_review.zip'),flush=True)
            append_event(out/'provenance/full_executions.jsonl',{'session':session,'mode':args.mode,
                'max_new_tokens':args.max_new_tokens,'seconds':time.monotonic()-started,'completed':True})
            print('E3 full phase complete: '+args.mode,flush=True)
            return 0
    except Timeout:
        print('E3 full stopped: another process owns this run; no concurrent phase started.',flush=True)
        return 2
    except (Exception,KeyboardInterrupt) as exc:
        if acquired:
            write_json(out/'provenance/last_full_error.json',{'status':'INCOMPLETE','mode':args.mode,
                'max_new_tokens':args.max_new_tokens,'error_type':type(exc).__name__,'reason':str(exc),
                'policy':'Retain committed data/checkpoints/outputs/API receipts; rerun the same command'})
            append_event(out/'provenance/full_executions.jsonl',{'session':session,'mode':args.mode,
                'seconds':time.monotonic()-started,'completed':False,'error_type':type(exc).__name__})
        print('E3 full stopped: '+str(exc),flush=True)
        return 130 if isinstance(exc,KeyboardInterrupt) else 2


if __name__=='__main__':raise SystemExit(main())
