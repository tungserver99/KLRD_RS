"""Read-only progress display; run beside E2 without restarting its process.

Standard library only. Does not load models, score responses or modify artifacts.
ETA covers remaining generation + scoring batches, excluding final aggregation.
"""
import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def duration(seconds):
    seconds=max(0,round(seconds))
    hours,remainder=divmod(seconds,3600)
    minutes,seconds=divmod(remainder,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def planned_jobs(manifest):
    cfg=manifest['E2_config']
    for owner in ['clean',*map(str,cfg['owners']),*map(str,cfg['stress'])]:
        stress=owner in set(map(str,cfg['stress']))
        for group in manifest['groups']:
            index=group['group_index']
            if group['Q']!=1000 or (stress and index!=0):
                continue
            ids=group['prompt_ids']
            for chunk,start in enumerate(range(0,len(ids),cfg['batch_size'])):
                yield f'primary_owner{owner}_g{index:02}_chunk{chunk:04}',len(ids[start:start+cfg['batch_size']])


def snapshot(out,jobs,receipts):
    completed=responses=generated=0
    generated_names={p.stem for p in (out/'private/jobs').glob('primary_*.json')}
    score_names={p.stem for p in (out/'private/scores').glob('primary_*.npz')}
    marker_paths={p.name.removesuffix('.sha256.json'):p
        for p in (out/'private/scores').glob('primary_*.sha256.json')}
    for name,count in jobs:
        if name not in generated_names:
            continue
        generated+=1
        if name not in score_names or name not in marker_paths:
            continue
        try:
            if name not in receipts:
                receipts[name]=read_json(marker_paths[name])
            receipt=receipts[name]
        except (OSError,ValueError):
            continue
        # Completion receipt only; this monitor does not replace E2 checksum
        # verification or scientific evaluation.
        if receipt.get('sha256') and receipt.get('records_sha256'):
            completed+=1
            responses+=count
    return completed,responses,generated


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'experiment2_rs/run_010')
    parser.add_argument('--interval',type=float,default=30.,help='Refresh interval in seconds')
    parser.add_argument('--once',action='store_true',help='Display one snapshot and exit')
    args=parser.parse_args()
    if not math.isfinite(args.interval) or args.interval<1:
        parser.error('--interval must be finite and at least 1 second')
    out=args.output.resolve()
    path=out/'provenance/prompt_manifest.json'
    print(f'E2 progress: {out}',flush=True)
    print('Scope: full study including stress; smoke excluded. ETA covers batches, not final report aggregation.',flush=True)
    jobs=None
    started=None
    baseline=0
    receipts={}
    try:
        while True:
            if jobs is None:
                if not path.is_file():
                    print('Waiting for frozen prompt manifest / model initialization...',flush=True)
                    if args.once:
                        return 0
                    time.sleep(args.interval)
                    continue
                jobs=list(planned_jobs(read_json(path)))
                if not jobs:
                    raise ValueError('Frozen manifest contains no planned E2 jobs')
            done,responses,generated=snapshot(out,jobs,receipts)
            now=time.monotonic()
            if started is None:
                started=now
                baseline=done
            new=done-baseline
            remaining=len(jobs)-done
            elapsed=now-started
            eta=duration(elapsed/new*remaining) if new>0 else 'estimating (waiting for newly completed batches)'
            if not remaining:
                eta='00:00:00'
            print(f'[{datetime.now().strftime("%H:%M:%S")}] E2 {done:,}/{len(jobs):,} batches ({100*done/len(jobs):.2f}%)'
                f' | responses scored {responses:,}/{sum(count for _,count in jobs):,}'
                f' | generated batches {generated:,}/{len(jobs):,}'
                f' | ETA ~{eta} | observing {duration(elapsed)}',flush=True)
            if done==len(jobs):
                print('All planned batches generated and scored. E2 may still be aggregating/writing its final report.',flush=True)
                return 0
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\nProgress monitor stopped; the separate E2 process is unaffected.',flush=True)
        return 0


if __name__=='__main__':
    raise SystemExit(main())
