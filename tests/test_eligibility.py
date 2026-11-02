import json
from dataclasses import replace
import numpy as np
import torch


def minimal_bundle(root):
    from klrd_rs.config import Config
    from klrd_rs.io import write_json,sha256
    cfg=replace(Config(),pca_dim=4,m=2)
    rows=[{'prompt_id':'a','prompt':'A','response':'X','source':'train_gen','text_roundtrip_equal':True}]
    diag=[{'prompt_id':'b','prompt':'B','response':'Y','source':'test_gen','text_roundtrip_equal':True}]
    root.mkdir()
    meta={'model_id':cfg.model,'model_revision':cfg.revision,'tokenizer_revision':cfg.revision,'tau':cfg.tau}
    for name in ['calibration_manifest','diagnostic_manifest']:write_json(root/(name+'.json'),meta)
    write_json(root/'calibration_selection.json',{'calibration':rows})
    write_json(root/'diagnostic_selection.json',{'final':diag})
    (root/'clean_calibration.jsonl').write_text(json.dumps(rows[0])+'\n')
    (root/'clean_original.jsonl').write_text(json.dumps(diag[0])+'\n')
    (root/'paraphrases.jsonl').write_text(json.dumps({'split':'clean','prompt_id':'b','source_text':'Y','paraphrased_text':'Z','status':'ok'})+'\n')
    (root/'legacy_scores.jsonl').write_text('unavailable legacy fixture')
    torch.save({'basis':torch.eye(4),'mean':torch.zeros(4)},root/'active_basis.pt')
    files={p.name:{'sha256':sha256(p)} for p in root.iterdir()}
    write_json(root/'bundle_manifest.json',{'files':files,'basis_provenance':{'clean_calibration_only':True}})
    return cfg


def test_optional_bundle_files_missing_do_not_block_core(tmp_path):
    from klrd_rs.inputs import load_inputs
    root=tmp_path/'bundle';cfg=minimal_bundle(root)
    (root/'legacy_scores.jsonl').unlink()
    x=load_inputs(root,cfg)
    assert len(x['original'])==1 and len(x['paraphrase'])==1
    (root/'paraphrases.jsonl').unlink()
    x=load_inputs(root,cfg)
    assert x['paraphrase'] is None
    assert set(x['index']['unavailable_optional'])=={'legacy_scores.jsonl','paraphrases.jsonl'}


def test_unrecoverable_calibration_response_is_never_used_as_context(tmp_path):
    from klrd_rs.replay import contexts
    from klrd_rs.toy import ToyAdapter
    from klrd_rs.io import StageStore
    rows=[{'prompt_id':'known','prompt':'a','response':'long clean text','text_roundtrip_equal':True},
          {'prompt_id':'unrecoverable','prompt':'b','response':'other text','text_roundtrip_equal':False}]
    result=contexts(rows,ToyAdapter(),10,42,StageStore(tmp_path,{}),'sample')
    assert set(result['prompt_ids'])=={'known'}
    assert result['excluded_prompt_ids'].tolist()==['unrecoverable']
