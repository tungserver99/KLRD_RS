"""One-time maintainer utility; normal server invocation never needs this.

Copies exact historical inputs without mutating the historical run.
Usage: python tools/import_legacy.py /path/to/KLRD
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import shutil
from klrd_rs.io import sha256, write_json, read_json


def main(source, destination):
    source, destination = Path(source), Path(destination)
    run = source/'outputs/klrd_vllm'
    files = {
        'calibration_manifest.json':'private/manifest.json',
        'calibration_selection.json':'private/selected_prompts.json',
        'diagnostic_manifest.json':'followup_raw_llr_v1/private/manifest.json',
        'diagnostic_selection.json':'followup_raw_llr_v1/private/selected_prompts.json',
        'diagnostic_protocol.json':'followup_raw_llr_v1/private/protocol.json',
        'clean_calibration.jsonl':'private/clean_calibration.jsonl',
        'clean_original.jsonl':'followup_raw_llr_v1/private/clean_final.jsonl',
        'active_basis.pt':'private/active_basis.pt',
        'paraphrases.jsonl':'paraphrase_scw_table2_v1/paraphrases.jsonl',
        'paraphrase_manifest.json':'paraphrase_scw_table2_v1/manifest.json',
        'generation_engine_clean.json':'private/generation_engine_clean.json',
        'diagnostic_generation_engine_clean.json':'followup_raw_llr_v1/private/generation_engine_clean.json',
        'private/legacy_key.pt':'private/keys/correct.pt',
        'legacy_fingerprint.jsonl':'followup_raw_llr_v1/private/fingerprint_final.jsonl',
        'legacy_scores.jsonl':'followup_raw_llr_v1/private/scores_fingerprint_final.jsonl',
    }
    provenance = {}
    for target, relative in files.items():
        old, new = run/relative, destination/target
        if not old.exists():
            if target.startswith(('private/legacy','legacy_')) or target.startswith('diagnostic_generation_'):
                continue
            raise FileNotFoundError(old)
        new.parent.mkdir(parents=True,exist_ok=True)
        h = sha256(old)
        if new.exists():
            if sha256(new) != h: raise ValueError('Refusing to replace changed bundled input: '+target)
        else: shutil.copyfile(old,new)
        provenance[target] = {'source':'KLRD/outputs/klrd_vllm/'+relative,'sha256':h,'bytes':old.stat().st_size}
        print('Bundled',target,flush=True)
    for filename in ['data.py','model.py','rotation.py','pipeline.py','vllm_backend.py','followup.py']:
        old, new = source/'klrd'/filename, destination/'source_snapshot'/filename
        new.parent.mkdir(parents=True,exist_ok=True)
        if not new.exists(): shutil.copyfile(old,new)
        if sha256(new)!=sha256(old): raise ValueError('Source snapshot changed')
        provenance['source_snapshot/'+filename]={'source':'KLRD/klrd/'+filename,'sha256':sha256(old),'bytes':old.stat().st_size}
    # Public basis was fit from these clean-only activations in the source pipeline.
    # Its hash is retained without requiring an 819 MB input file for model replay.
    activations = run/'private/calibration_activations.pt'
    write_json(destination/'bundle_manifest.json', {
        'schema_version':1, 'files':provenance,
        'basis_provenance':{'clean_calibration_only':True,
            'source':'KLRD/outputs/klrd_vllm/private/calibration_activations.pt',
            'activations_sha256':sha256(activations),
            'activations_bytes':activations.stat().st_size,
            'basis_source':'clean calibration records; source_snapshot/pipeline.py build_keys; no diagnostic text'},
        'diagnostic_cohort':'Historical followup_raw_llr_v1/test_gen/final, exact SCW-clean matched pairs',
        'diagnostic_status':'development data, already inspected',
        'legacy_plane_sign_independence':'Source make_key draws Gaussian planes before independent sign draws; source snapshot retained',
        'private_key_material':'private/legacy_key.pt is excluded from version control; never publish inputs',
    })


if __name__=='__main__':
    main(sys.argv[1],Path(__file__).resolve().parents[1]/'inputs')
