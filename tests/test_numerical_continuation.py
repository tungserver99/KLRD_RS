from dataclasses import replace
import numpy as np
import pytest
import torch


def small_engine():
    from klrd_rs.math import planes
    from klrd_rs.readout import RotationReadout
    rng=np.random.default_rng(113)
    p,q=planes(np.eye(12),3,18)
    return RotationReadout(torch.tensor(rng.normal(size=(41,12)),dtype=torch.float32),
        torch.tensor(rng.uniform(.5,1.5,12),dtype=torch.float32),1e-6,
        torch.tensor(p),torch.tensor(q),.8)


def test_fp64_three_methods_and_all_steps_are_reported_without_mutating_weights():
    from klrd_rs.numerical import precision_diagnostic,policy
    from klrd_rs.config import Config
    engine=small_engine();before=engine.weight.clone()
    r=np.random.default_rng(91).normal(size=(2,12)).astype('float32')
    report,rows=precision_diagnostic(engine,r,replace(Config(),m=3))
    assert report['fp64_checks_passed']
    assert report['continuation_allowed']
    assert report['root_cause']=='FP32_FINITE_DIFFERENCE_ROUNDOFF'
    assert report['policy']['fd_steps']==[.002,.005,.01]
    assert len({row['direction_index'] for row in rows})==3
    assert {row['context_index'] for row in rows}=={-1,0,1}
    assert any(row['method']=='autograd_fp64' for row in rows)
    assert all('p99_abs' in row and 'median_scaled' in row and 'worst_abs_token_id' in row for row in rows)
    assert engine.weight.dtype==torch.float32 and torch.equal(before,engine.weight)
    assert policy()['fd_atol']==Config().finite_difference_atol


def test_formula_corruption_is_not_waived_by_more_precision(monkeypatch):
    from klrd_rs.numerical import precision_diagnostic
    from klrd_rs.readout import RotationReadout
    from klrd_rs.config import Config
    original=RotationReadout.full_scores
    def corrupt(self,r):
        lp,phi=original(self,r)
        return lp,phi+.02
    monkeypatch.setattr(RotationReadout,'full_scores',corrupt)
    report,_=precision_diagnostic(small_engine(),np.ones((2,12),dtype='float32'),replace(Config(),m=3))
    assert not report['continuation_allowed']
    assert report['root_cause']=='UNRESOLVED_FORMULA_OR_NUMERICAL_ERROR'


def test_error_summary_identifies_context_token_and_scaled_extremes():
    from klrd_rs.numerical import error_rows
    reference=np.array([[0.,100.,1.],[1.,2.,3.]])
    candidate=reference+np.array([[.01,.2,0.],[0.,0.,.03]])
    rows=error_rows(reference,candidate,4,'test',.003,.03)
    assert rows[0]['worst_abs_token_id']==1
    assert rows[0]['worst_scaled_token_id']==0
    assert rows[0]['direction_index']==4 and rows[1]['context_index']==1
    assert rows[2]['context_index']==-1
    assert rows[2]['p99_abs']==pytest.approx(np.quantile(np.abs(candidate-reference),.99))
    assert rows[2]['worst_abs_context_index']==0


def test_anchor_is_exact_and_frozen_config_cannot_change():
    from pathlib import Path
    from klrd_rs.continuation import load_anchor
    from klrd_rs.config import Config
    root=Path(__file__).resolve().parents[1]
    if not (root/'continuation_inputs/manifest.json').exists():pytest.skip('No frozen historical anchor')
    anchor=load_anchor(root/'continuation_inputs',Config())
    assert anchor['arrays']['residuals'].shape==(2,2048)
    assert anchor['arrays']['positions'].tolist()==[90,105]
    with pytest.raises(ValueError,match='config|frozen'):
        load_anchor(root/'continuation_inputs',replace(Config(),finite_difference_atol=.004))


def test_launcher_runs_gated_continuation():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    assert (root/'run_experiment1.sh').read_bytes()==b'python main.py --continue-experiment1\n'


def test_context_and_plane_mismatch_are_rejected():
    from klrd_rs.continuation import check_continuation_contexts
    a={'residuals':np.ones((2,12)),'prompt_ids':np.array(['a','b']),
        'positions':np.array([4,5]),'targets':np.array([6,7]),'P':np.eye(12)[:,:3],'Q':np.eye(12)[:,3:6]}
    check_continuation_contexts(a,a['P'],a['Q'],{'arrays':a})
    changed=dict(a,residuals=a['residuals']+.01)
    with pytest.raises(ValueError,match='contexts changed'):
        check_continuation_contexts(changed,a['P'],a['Q'],{'arrays':a})
    with pytest.raises(ValueError,match='planes changed'):
        check_continuation_contexts(a,a['P']+.01,a['Q'],{'arrays':a})


def test_cache_inheritance_is_read_only_and_checks_payloads(tmp_path):
    from klrd_rs.continuation import inherit_contexts
    from klrd_rs.io import StageStore,sha256
    source=tmp_path/'old';old=StageStore(source/'private/cache',{'frozen':True})
    names=['contexts_covariance','contexts_fisher','contexts_strength','contexts_local']
    for name in names:old.save(name,{'residuals':np.ones((2,4))})
    inventory={p.relative_to(source).as_posix():{'sha256':sha256(p)} for p in source.rglob('*') if p.is_file()}
    store=StageStore(tmp_path/'new/cache',{'new_code':True})
    inherit_contexts(source,{'source':{'files':inventory}},store)
    assert np.array_equal(store.load('contexts_covariance')['residuals'],np.ones((2,4)))
    assert all(sha256(source/name)==item['sha256'] for name,item in inventory.items())
    (old.root/'contexts_covariance.npz').write_bytes(b'corrupt')
    fresh=StageStore(tmp_path/'other/cache',{})
    with pytest.raises(ValueError,match='payload checksum'):
        inherit_contexts(source,{'source':{'files':inventory}},fresh)


def test_review_package_excludes_secrets_and_contains_prior_no_go(tmp_path):
    from klrd_rs.review_bundle import create_review_bundle
    from klrd_rs.io import write_json
    import zipfile
    root=tmp_path/'project';out=root/'continuation';out.mkdir(parents=True)
    write_json(out/'report/summary.json',{'status':'NO_GO'})
    (out/'private').mkdir();(out/'private/secret.pt').write_text('secret')
    write_json(root/'experiment1_rs/report/summary.json',{'status':'NO_GO'})
    archive=create_review_bundle(root,out,{'manifest_sha256':'anchor'})
    with zipfile.ZipFile(archive) as z:
        assert 'prior_NO_GO/report/summary.json' in z.namelist()
        assert 'SHA256_MANIFEST.json' in z.namelist()
        assert not any('private' in x for x in z.namelist())


def test_continuation_uses_saved_planes_without_regenerating_geometry():
    from klrd_rs.continuation import frozen_planes
    from klrd_rs.config import Config
    arrays={'basis':np.eye(12),'mean':np.zeros(12),'P':np.eye(12)[:,:3],
            'Q':-np.eye(12)[:,3:6]}
    inputs={'basis':arrays['basis'].copy(),'mean':arrays['mean'].copy()}
    p,q=frozen_planes(inputs,{'arrays':arrays})
    assert np.array_equal(p,arrays['P']) and np.array_equal(q,arrays['Q'])
    with pytest.raises(ValueError,match='basis|geometry'):
        frozen_planes(dict(inputs,basis=inputs['basis']+.01),{'arrays':arrays})


@pytest.mark.parametrize('mode',['--continue-experiment1','--audit-only','--smoke'])
def test_main_rejects_every_output_inside_historical_run_before_writing(tmp_path,monkeypatch,mode):
    import main as entry
    project=tmp_path/'project';old=project/'experiment1_rs';old.mkdir(parents=True)
    marker=old/'preserved.txt';marker.write_text('NO_GO')
    monkeypatch.setattr(entry,'__file__',str(project/'main.py'))
    with pytest.raises(SystemExit):entry.main([mode,'--output',str(old/'nested')])
    assert list(old.iterdir())==[marker]


def test_historical_integrity_detects_same_size_cache_corruption_on_resume(tmp_path):
    from klrd_rs.continuation import verify_source
    from klrd_rs.io import sha256
    source=tmp_path/'old';payload=source/'private/cache/contexts_covariance.npz'
    payload.parent.mkdir(parents=True);payload.write_bytes(b'original')
    anchor={'source':{'files':{'private/cache/contexts_covariance.npz':{
        'sha256':sha256(payload),'bytes':payload.stat().st_size}}}}
    verify_source(source,anchor)
    payload.write_bytes(b'modified')
    with pytest.raises(ValueError,match='checksum'):
        verify_source(source,anchor)
