"""Prior exposure evidence, distinct from unused candidate/source inventories.

Role rules adapted from KLRD stage3a_wmle_implantation/data.py. A reserve that
was consumed is still excluded through initial/final cohorts and attempt journals.
"""
from pathlib import Path
import hashlib,unicodedata
from .cohort import text_hash


def legacy_text_hash(text):
    canonical=' '.join(unicodedata.normalize('NFKC',text).split())
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def source_aliases(value):
    text=str(value);aliases={text}
    for prefix in ['train_gen:','test_gen:','ultrachat:','ultrachat_200k:']:
        if text.startswith(prefix):aliases.add(text[len(prefix):])
    return aliases


def unused_inventory_path(path):
    name=Path(path).name.lower()
    return any(x in name for x in ['reserve_prompts','reserve_manifest','reserve_prompt_manifest',
        'eligible_prompts','rejected_prompts','removed_prompts','source_inventory'])


def unused_inventory_field(field):
    return bool(set(field.lower().split('_')) & {'removed','removals','rejected','eligible',
        'reserve','considered','unselected','excluded','inventory'})


class UsedHistoryIndex:
    def __init__(self):
        self.ids=set();self.exact=set();self.normalized=set();self.legacy=set();self.with_text=set()
        self.inventory_sources=[]

    def add_authority(self,value):
        if not isinstance(value.get('source_ids'),list) or not isinstance(value.get('prompt_sha256'),list):
            raise ValueError('Missing frozen used-history IDs/hash authority')
        for pid in value['source_ids']:self.ids.update(source_aliases(pid))
        self.legacy.update(value['prompt_sha256'])

    def consume(self,path,value):
        if unused_inventory_path(path):self.inventory_sources.append(str(path));return
        self._walk(value)

    def _walk(self,value):
        if isinstance(value,list):
            for item in value:self._walk(item)
            return
        if not isinstance(value,dict):return
        rowids=set()
        for field in ['prompt_id','source_id','replaced_prompt_id','replacement_prompt_id']:
            item=value.get(field)
            if isinstance(item,(str,int)) and not isinstance(item,bool):rowids.update(source_aliases(item))
        self.ids.update(rowids)
        for field in ['prompt','prompt_text','instruction']:
            text=value.get(field)
            if isinstance(text,str) and text.strip():
                self.exact.add(text_hash(text));self.normalized.add(text_hash(text,True))
                self.legacy.add(legacy_text_hash(text));self.with_text.update(rowids)
        for field,item in value.items():
            if unused_inventory_field(field):continue
            if isinstance(item,list) and (field.endswith('prompt_ids') or field.endswith('source_ids') or field=='selected_ids'):
                for pid in item:
                    if isinstance(pid,(str,int)) and not isinstance(pid,bool):self.ids.update(source_aliases(pid))
            if field in ['prompt_sha256','prompt_sha','normalized_prompt_sha256','prior_prompt_sha256']:
                hashes=item if isinstance(item,list) else [item]
                for h in hashes:
                    if isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h):self.legacy.add(h)
            if isinstance(item,(list,dict)):self._walk(item)

    def as_sets(self):
        return {'ids':self.ids,'exact':self.exact,'normalized':self.normalized,
            'legacy_normalized':self.legacy}

    def serialized(self):
        return {'prompt_ids':sorted(self.ids),'exact_text_hashes':sorted(self.exact),
            'normalized_text_hashes':sorted(self.normalized),'legacy_normalized_text_hashes':sorted(self.legacy),
            'ids_without_recovered_text':sorted(self.ids-self.with_text),
            'policy_version':2,'inventory_sources_not_counted_as_used':self.inventory_sources}
