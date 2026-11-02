"""Run the frozen Experiment 1. The shell launcher needs no configuration."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(description='Frozen KLRD-RS Experiment 1; no training or paraphrase generation.')
    parser.add_argument('--audit-only',action='store_true',help='Verify bundled historical input hashes/cohorts without loading Qwen')
    parser.add_argument('--smoke',action='store_true',help='Run a separately labeled synthetic CPU fixture, never research evidence')
    parser.add_argument('--continue-experiment1',action='store_true',help='Recompute exact-context FP64 diagnostic and gate a new Experiment 1 run')
    parser.add_argument('--output',type=Path,help='Optional new output directory; default experiment1_rs next to this file')
    args=parser.parse_args(argv)
    from klrd_rs.config import Config
    from klrd_rs.io import write_json
    from klrd_rs.pipeline import run_experiment, report
    root=Path(__file__).resolve().parent
    if args.continue_experiment1 and (args.smoke or args.audit_only):parser.error('Continuation cannot be combined with smoke/audit-only')
    out=(args.output or root/('synthetic_smoke' if args.smoke else 'experiment1_rs_continuation' if args.continue_experiment1 else 'experiment1_rs')).resolve()
    for protected in [(root/'experiment1_rs').resolve(),(root/'continuation_inputs').resolve()]:
        if out==protected or protected in out.parents or out in protected.parents:
            parser.error('Historical run and frozen continuation inputs are read-only, including descendants. Use a separate output directory.')
    inputs_root=root/'inputs'
    if out==inputs_root or inputs_root in out.parents or out in inputs_root.parents:
        parser.error('Output must be separate from immutable inputs')
    cfg=Config()
    try:
        if args.smoke:
            import torch
            torch.set_num_threads(min(torch.get_num_threads(),4))
            from klrd_rs.toy import ToyAdapter,toy_inputs
            cfg=replace(cfg,m=3,pca_dim=12,calibration_keys=8,validation_keys=12,
                fisher_context_sample_cap=600,covariance_contexts=120,strength_contexts=64,
                local_contexts=80,diagnostic_random_keys=20000,bootstrap_replicates=100,
                readout_chunk=8,finite_difference_h=.0001,finite_difference_atol=.001,mean_zero_atol=1e-8)
            result=run_experiment(cfg,toy_inputs(),ToyAdapter(),out,synthetic=True)
        else:
            from klrd_rs.inputs import load_inputs
            print('Verifying portable historical input bundle...',flush=True)
            inputs=load_inputs(inputs_root,cfg)
            if args.audit_only:
                result={'status':'AUDIT_ONLY','scientific_success':False,
                    'calibration_responses':len(inputs['calibration']),
                    'diagnostic_original_responses':len(inputs['original']),
                    'diagnostic_paraphrase_responses':len(inputs['paraphrase']) if inputs['paraphrase'] is not None else 0,
                    'basis_shape':list(inputs['basis'].shape),'historical_hashes_verified':True}
                result['unavailable_optional']=inputs['index']['unavailable_optional']
                result['calibration_context_excluded_nonroundtrip']=sum(r['text_roundtrip_equal'] is False for r in inputs['calibration'])
                result['diagnostic_canonical_reencoding_nonroundtrip']=sum(r.get('text_roundtrip_equal') is False for r in inputs['original'])
                write_json(out/'provenance/input_manifest.json',inputs['index'])
                write_json(out/'provenance/resolved_config.json',cfg.resolved())
                report(out,result)
            elif args.continue_experiment1:
                from klrd_rs.continuation import load_anchor,verify_source,run_continuation
                anchor=load_anchor(root/'continuation_inputs',cfg)
                verify_source(root/'experiment1_rs',anchor,include_cache=False)
                from klrd_rs.model import ModelAdapter
                result=run_continuation(cfg,inputs,ModelAdapter(cfg),root,out)
            else:
                from klrd_rs.model import ModelAdapter
                result=run_experiment(cfg,inputs,ModelAdapter(cfg),out,input_root=inputs_root)
        print('Result:',result['status'],'—',out/'report/experiment1_report.md',flush=True)
        return 0 if result['status'] in ('VALID_IMPLEMENTATION','SYNTHETIC_CHECKS_PASSED','AUDIT_ONLY') else 1
    except (OSError,ValueError,RuntimeError,ImportError,KeyError) as exc:
        # Never replace existing successful outputs after an incompatible resume.
        failure={'status':'BLOCKED_TECHNICAL','scientific_success':False,
            'error_type':type(exc).__name__,'reason':str(exc),'synthetic':args.smoke}
        write_json(out/'last_error.json',failure)
        if not (out/'report/summary.json').exists():report(out,failure)
        print('BLOCKED_TECHNICAL:',str(exc),file=sys.stderr,flush=True)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
