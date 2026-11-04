"""SERVER ONLY regression tests. Supplied as source; do not execute locally."""
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_all_stage_launcher_orders_every_expensive_phase_and_report():
    source=(ROOT/'run_experiment3.sh').read_text(encoding='utf-8')
    commands=[line for line in source.splitlines() if line.startswith('python ')]
    prefix='python main_exp3_full.py --mode '
    assert commands==[prefix+suffix+' --output "$output"' for suffix in (
        'train','evaluate --max-new-tokens 512','evaluate --max-new-tokens 1024',
        'arc-c','paraphrase --max-new-tokens 1024','report')]
    assert 'set -euo pipefail' in source
    assert 'experiment3_full/run_001' in source
    assert 'pip install' not in source
    assert '1200 paid requests' in source


def test_output_boundaries_preserve_historical_runs(tmp_path):
    from klrd_rs.e3.provenance import resolve_output
    assert resolve_output(tmp_path,tmp_path/'experiment3_full/run_001')==tmp_path/'experiment3_full/run_001'
    for name in ('experiment3_full','experiment1_rs/run_001',
                 'experiment2_rs/run_012','experiment2_paraphrase/run_002'):
        with pytest.raises(ValueError):resolve_output(tmp_path,tmp_path/name)


def test_cli_training_length_cannot_be_silently_expanded():
    import main_exp3_full
    assert main_exp3_full.parser().parse_args(['--mode','evaluate']).max_new_tokens is None
    # Argument checks run before importing/loading run state, models or clients.
    with pytest.raises(SystemExit):
        main_exp3_full.main(['--mode','train','--max-new-tokens','1024'])
    with pytest.raises(SystemExit):
        main_exp3_full.main(['--mode','paraphrase','--max-new-tokens','512'])


def test_explicit_seed_override_preserves_historical_default():
    from klrd_rs.e3.generation import _request_seed,_seed
    settings={'seed':123,'kind':'untrained'}
    assert _request_seed(settings,{'prompt_id':'a'})==_seed(123,'a','untrained')
    assert _request_seed(settings,{'prompt_id':'a','sampling_seed':45})==45
    with pytest.raises(ValueError):_request_seed(settings,{'prompt_id':'a','sampling_seed':-1})


def test_prepare_recovers_crash_after_selection_without_loading_models(tmp_path,monkeypatch):
    """An existing selection alone must not strand resume without its receipts."""
    from klrd_rs.e3full import common
    from klrd_rs.e3 import data
    from klrd_rs.e3.config import E3Config
    from klrd_rs.io import write_json,read_json,digest,sha256
    cfg=E3Config(profile='full')
    out=tmp_path/'full';pilot=tmp_path/'pilot'
    selected={split:[{'prompt_id':split+str(i),'prompt':split+' prompt '+str(i)}
        for i in range(size)] for split,size in [('train',10000),('validation',1000)]}
    source=pilot/'private/full_selection.json'
    write_json(source,selected)
    write_json(pilot/'provenance/data_selection.json',
        {'identity':'pilot','selection_sha256':sha256(source),'heldout_sources':{}})
    evidence={'identity':'pilot','rows_sha256':{}}
    for split,name,size in [('train','train_manifest.jsonl',2000),('validation','val_manifest.jsonl',200)]:
        rows=[dict(row,source='train_gen',split=split,response='clean context') for row in selected[split][:size]]
        data._write_rows(pilot/'private'/name,rows)
        evidence['rows_sha256'][split]=digest(rows)
        write_json(pilot/'private/data'/(split+'_clean.identity.json'),{'settings':{
            'model':cfg.model,'revision':cfg.revision,'dtype':'float32',
            'temperature':.8,'max_new_tokens':512,'kind':'clean_train'}})
    write_json(pilot/'provenance/data_complete.json',evidence)
    state={'out':out,'identity':'full','cfg':cfg,'pilot':{'out':pilot,'identity':'pilot'}}
    monkeypatch.setattr(data,'_exclusions',lambda state:(None,{}))
    monkeypatch.setattr(data,'_overlap',lambda row,index:False)
    monkeypatch.setattr(data,'prepare',lambda state:SimpleNamespace(prepared=True))
    original=common.ensure_manifest
    def interrupted(path,value):
        if Path(path)==out/'provenance/data_selection.json':raise OSError('preempted')
        return original(path,value)
    monkeypatch.setattr(common,'ensure_manifest',interrupted)
    with pytest.raises(OSError,match='preempted'):common.prepare_training(state)
    assert (out/'private/full_selection.json').exists()
    assert not (out/'provenance/data_selection.json').exists()
    monkeypatch.setattr(common,'ensure_manifest',original)
    assert common.prepare_training(state).prepared
    reuse=read_json(out/'provenance/pilot_clean_reuse.json')
    assert reuse['reused']=={'train':2000,'validation':200}
    assert reuse['new_clean_needed']=={'train':8000,'validation':800}
    assert read_json(out/'provenance/data_selection.json')['identity']=='full'
