from pathlib import Path
import pytest


def test_unused_reserve_is_not_a_used_cohort_but_consumed_replacement_is():
    from klrd_rs.e2.exclusions import UsedHistoryIndex
    index=UsedHistoryIndex()
    inventory=[{'prompt_id':'unused','prompt':'An unused reserve prompt'},
               {'prompt_id':'consumed','prompt':'A consumed replacement'}]
    index.consume('data/train_reserve_prompts.jsonl',inventory)
    index.consume('data/initial_implant_train_prompts.jsonl',[{'prompt_id':'old','prompt':'Original used prompt'}])
    index.consume('data/prompt_replacements.jsonl',[{'replaced_prompt_id':'old','replacement_prompt_id':'consumed'}])
    index.consume('data/implant_train_prompts.jsonl',[inventory[1]])
    assert 'unused' not in index.ids
    assert {'old','consumed'}<=index.ids
    assert index.inventory_sources==['data/train_reserve_prompts.jsonl']


def test_diagnostic_removed_eligible_metadata_is_not_usage_evidence():
    from klrd_rs.e2.exclusions import UsedHistoryIndex
    index=UsedHistoryIndex()
    index.consume('provenance/prompt_exclusion_manifest.json',{
        'prior_source_ids':['previously_used'],
        'removed':[{'prompt_id':'too_short','prompt':'A rejected unused prompt'}],
        'eligible_prompts':[{'prompt_id':'merely_considered','prompt':'Not selected yet'}],
        'selected_prompt_ids':['selected']})
    assert {'previously_used','selected'}<=index.ids
    assert not {'too_short','merely_considered'} & index.ids


def test_source_aliases_and_legacy_normalized_hashes_are_still_excluded():
    from klrd_rs.e2.exclusions import UsedHistoryIndex,legacy_text_hash
    from klrd_rs.e2.cohort import select_cohort
    index=UsedHistoryIndex()
    index.consume('used.jsonl',[{'source_id':'train_gen:used','prompt':'Old prompt'}])
    index.add_authority({'source_ids':['older'],'prompt_sha256':[legacy_text_hash('HELLO World')]})
    assert {'used','train_gen:used','older'}<=index.ids
    class Tok:
        def encode(self,*a,**kw):return [1]*20
    dataset={'test_gen':[{'prompt_id':'used','prompt':'Changed text'},
        {'prompt_id':'alias','prompt':' HELLO  World '},
        {'prompt_id':'fresh','prompt':'A genuinely unused prompt'}],'train_gen':[]}
    rows=select_cohort(dataset,Tok(),index.as_sets(),1,1)
    assert rows[0]['prompt_id']=='fresh'


def test_genuinely_used_prompt_remains_excluded_even_if_also_in_reserve():
    from klrd_rs.e2.exclusions import UsedHistoryIndex
    from klrd_rs.e2.cohort import text_hash
    index=UsedHistoryIndex();row={'prompt_id':'both','prompt':'Used and reserved duplicate'}
    index.consume('data/final_prompts.jsonl',[row])
    index.consume('data/reserve_prompts.jsonl',[row])
    assert 'both' in index.ids and text_hash(row['prompt'],True) in index.normalized


def test_failed_e2_run_is_preserved_and_default_moves_to_run002(tmp_path,monkeypatch):
    import main_exp2 as entry
    root=tmp_path/'project';old=root/'experiment2_rs/run_001';old.mkdir(parents=True)
    (old/'failure.json').write_text('0 eligible')
    monkeypatch.setattr(entry,'__file__',str(root/'main_exp2.py'))
    with pytest.raises(SystemExit):entry.main(['--output',str(old/'nested')])
    assert [p.name for p in old.iterdir()]==['failure.json']
