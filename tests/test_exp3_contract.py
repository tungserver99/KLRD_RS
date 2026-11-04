"""SERVER ONLY. These contracts are source deliverables; never run locally."""
from pathlib import Path
import pytest
from klrd_rs.e3.config import E3Config
from klrd_rs.e3.provenance import resolve_output,atomic_directory,file_manifest


def test_scientific_constants_and_pilot_sizes_are_frozen():
    pilot=E3Config();full=E3Config(profile='full')
    assert (pilot.train_count,pilot.validation_count)==(2000,200)
    assert (full.train_count,full.validation_count)==(10000,1000)
    assert pilot.owner_key_id==0 and pilot.key_seed==2026102803
    assert pilot.tau==.8 and pilot.m==32 and pilot.strength==.09271215438842773
    assert pilot.lora_r==16 and pilot.lora_alpha==32 and pilot.lora_dropout==0
    assert pilot.gradient_accumulation_steps==16 and pilot.full_vocab_logit_chunk==32
    assert pilot.max_epochs==2 and pilot.learning_rate==1e-5
    assert pilot.resolved()['automatic_api_calls'] is False
    assert pilot.resolved()['automatic_post_training_evaluation'] is False
    with pytest.raises(ValueError):E3Config(owner_key_id=1)


def test_causal_selection_masks_prompt_padding_and_special_response_positions():
    from klrd_rs.data import encode_response
    class Tokenizer:
        all_special_ids=[99]
        def apply_chat_template(self,*args,**kwargs):return [10,11,12]
        def encode(self,text,**kwargs):return [20,99,21]
    encoded=encode_response(Tokenizer(),'original prompt','visible response')
    assert encoded['input_ids']==[10,11,12,20,99,21]
    assert encoded['positions']==[2,4] # y0 from last prompt, not position3.
    assert encoded['targets']==[20,21] # EOS/special time step never added.


def test_e3_output_cannot_overwrite_history(tmp_path):
    root=tmp_path/'project';root.mkdir()
    assert resolve_output(root)==root/'experiment3_rs/run_001'
    for value in (root,root/'experiment3_rs',root/'experiment2_rs/run_012',root/'experiment2_paraphrase/run_002'):
        with pytest.raises(ValueError):resolve_output(root,value)


def test_atomic_directory_does_not_publish_partial_or_replace_committed(tmp_path):
    destination=tmp_path/'epoch_001'
    def broken(temp):
        (temp/'partial').write_text('incomplete')
        raise RuntimeError('simulated preemption')
    with pytest.raises(RuntimeError):atomic_directory(destination,broken)
    assert not destination.exists()
    atomic_directory(destination,lambda temp:(temp/'complete').write_text('committed'))
    before=file_manifest(destination)
    with pytest.raises(FileExistsError):atomic_directory(destination,lambda temp:None)
    assert file_manifest(destination)==before


def test_launcher_trains_then_evaluates_and_full_requires_explicit_output():
    root=Path(__file__).resolve().parents[1]
    launcher=(root/'run_experiment3_pilot.sh').read_text()
    assert 'set -euo pipefail' in launcher
    commands=[line.strip() for line in launcher.splitlines() if line.startswith('python ')]
    assert commands==['python main_exp3.py --mode pilot','python main_exp3.py --mode evaluate']
    assert '--mode paraphrase' not in launcher
    import main_exp3
    with pytest.raises(SystemExit):main_exp3.main(['--mode','full'])
    assert 'paraphrase' in main_exp3.parser().format_help()


def test_cli_mode_dispatch_does_not_enable_downstream_api_implicitly():
    # Inspect source; import/call neither training nor API clients.
    import ast
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'main_exp3.py').read_text())
    api_calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
        and node.func.value.id=='paraphrase']
    assert len(api_calls)==1
    # No dependency installation or shell startup side effects.
    assert 'pip install' not in (root/'run_experiment3.sh').read_text()
