"""Verified PC expert-bank relocation with resumable copy/activate/retire phases.

Plans are private: they contain physical paths. Returned receipts are public-safe
file-copy evidence, never model-quality evidence. All old index bytes stay valid.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid

from .bandwidth import EXPERT_BYTES
from .checkpoint import MODEL, REVISION, validate_expert_identity
from .model_artifacts import canonical, digest
from .resources import GiB, MiB, MemoryBudget
from .source_roots import BANKS, CONFIG, MARKER, SourceRoots, _plain, _read, marker, source_lease
from .storage import NativeStore

JOURNAL = '.vesper-bank-migration.json'


def _bytes(value): return (json.dumps(value, indent=2) + '\n').encode()


def _hash(path,maximum):
    h = hashlib.sha256(); count = 0
    with _plain(path).open('rb') as stream:
        for block in iter(lambda: stream.read(MiB), b''):
            count += len(block)
            if count>maximum:raise ValueError('Expert file grew beyond its saved extent')
            h.update(block)
    return count, h.hexdigest()


def _entry_name(name):
    match = re.fullmatch(r'layer([0-9]{1,2})/expert-([0-9]{3})', name)
    if not match: raise ValueError('Unowned expert directory in migration plan')
    layer, expert = map(int, match.groups()); validate_expert_identity(layer, expert)
    return layer, expert


def inventory(root):
    source = _plain(Path(root).resolve(strict=True) / BANKS[0])
    if not source.is_dir(): raise ValueError('Missing closed original PC bank')
    entries = []
    budget = MemoryBudget(cap=64 * MiB, owner='expert_bank_migration_metadata'); budget.acquire(budget.cap)
    try:
        paths = sorted(source.glob('layer*/expert-*'))
        if not 1 <= len(paths) <= (64 * GiB) // EXPERT_BYTES: raise ValueError('Unbounded original PC bank')
        for path in paths:
            name = path.relative_to(source).as_posix(); layer, expert = _entry_name(name)
            _plain(path)
            if set(p.name for p in path.iterdir()) != {'manifest.json', 'experts.bin'}:
                raise ValueError('Unexpected files in an immutable expert object')
            m, raw = _read(path/'manifest.json', MiB); packed = NativeStore(path,budget,max_experts=1)
            if packed.manifest != m or len(packed.entries) != 1 or f'{layer}:{expert}' not in packed.entries:
                raise ValueError('Original expert changed during migration inventory')
            e = packed.entries[f'{layer}:{expert}']
            if m['file_bytes'] != EXPERT_BYTES: raise ValueError('Migration requires single-expert immutable objects')
            entries.append({'directory':name,'manifest_bytes':len(raw),'manifest_sha256':digest(raw),
                'payload_bytes':EXPERT_BYTES,'payload_sha256':e['sha256']})
        _remaining_files(source,entries)
        return entries
    finally: budget.release(budget.cap)


def _remaining_files(source,entries):
    expected = {str(source/e['directory']/name) for e in entries for name in ('manifest.json','experts.bin')}
    if source.exists():
        for count,path in enumerate(source.rglob('*'),1):
            if count > 5 * len(entries) + 100: raise ValueError('Excessive closed-bank contents')
            _plain(path)
            if path.is_file() and str(path) not in expected: raise ValueError('Unowned file in closed PC bank')


def make_plan(root,target):
    if not Path(target).is_absolute(): raise ValueError('Expected a private absolute SSD directory')
    root=Path(root).resolve(strict=True); target=_plain(target)
    target=target.resolve(strict=False)
    if target.is_relative_to(root) or root.is_relative_to(target): raise ValueError('SSD root overlaps artifacts')
    with source_lease(root):
        if (root/CONFIG).exists(): raise FileExistsError('PC source roots are already configured; use the saved plan to resume')
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise FileExistsError('Refusing to adopt a nonempty SSD directory')
        entries=inventory(root)
        p={'schema_version':1,'kind':'vesper_private_pc_bank_migration_plan','model':MODEL,'revision':REVISION,
            'platform':'pc-heterogeneous','profile':'native','artifact_root':str(root),'ssd_directory':str(target),
            'source_bank':BANKS[0],'active_bank':BANKS[1],
            'markers':{name:marker(name) for name in BANKS},'entries':entries,
            'original_packed_payload_bytes':len(entries)*EXPERT_BYTES,
            'total_file_bytes':sum(e['manifest_bytes']+e['payload_bytes'] for e in entries)}
        p['plan_sha256']=canonical(p); return p


def _validate_plan(root,p):
    if (p.get('schema_version'),p.get('kind'),p.get('platform'),p.get('profile'),p.get('model'),p.get('revision'),
            p.get('source_bank'),p.get('active_bank')) != (
            1,'vesper_private_pc_bank_migration_plan','pc-heterogeneous','native',MODEL,REVISION,*BANKS):
        raise ValueError('Different bank-migration scope')
    if p.get('plan_sha256')!=canonical({k:v for k,v in p.items() if k!='plan_sha256'}):raise ValueError('Migration plan checksum differs')
    if Path(p['artifact_root']).resolve(strict=True)!=root:raise ValueError('Migration plan belongs to another artifact root')
    if not Path(p['ssd_directory']).is_absolute():raise ValueError('Nonabsolute migration destination')
    target=_plain(p['ssd_directory']).resolve(strict=False)
    if target.is_relative_to(root) or root.is_relative_to(target):raise ValueError('Migration target overlaps artifacts')
    if not 1<=len(p['entries'])<=(64*GiB)//EXPERT_BYTES:raise ValueError('Unbounded migration inventory')
    names=[]
    for e in p['entries']:
        _entry_name(e['directory']);names.append(e['directory'])
        if (e['payload_bytes']!=EXPERT_BYTES or not 1<=e['manifest_bytes']<=MiB
                or any(not re.fullmatch('[0-9a-f]{64}',e[n]) for n in ('manifest_sha256','payload_sha256'))):
            raise ValueError('Invalid file-bound migration entry')
    if names!=sorted(set(names)) or p['original_packed_payload_bytes']!=len(names)*EXPERT_BYTES:
        raise ValueError('Changed migration object census')
    if p['total_file_bytes']!=sum(e['payload_bytes']+e['manifest_bytes'] for e in p['entries']):raise ValueError('Changed migration byte census')
    from .source_roots import _valid_marker
    if set(p['markers'])!=set(BANKS):raise ValueError('Incomplete migration root identities')
    for name in BANKS:_valid_marker(p['markers'][name],name)
    if p['markers'][BANKS[0]]['store_id']==p['markers'][BANKS[1]]['store_id']:raise ValueError('Duplicate migration store ownership')
    return target


def _new_file(path,raw):
    path=_plain(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.stat().st_size!=len(raw) or path.read_bytes()!=raw:raise FileExistsError('Refusing to replace different owned metadata')
        return
    pending=path.with_name(path.name+'.pending-'+uuid.uuid4().hex)
    try:
        with pending.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        _publish(pending,path)
    finally:pending.unlink(missing_ok=True)


def _publish(pending,target):
    # Windows rename refuses an existing target. POSIX rename would replace
    # it, so use an exclusive hard-link publication there instead.
    if os.name=='nt':os.rename(pending,target)
    else:os.link(pending,target);pending.unlink()


def _verify(path,count,sha):
    if _plain(path).stat().st_size!=count or _hash(path,count)!=(count,sha):
        raise ValueError('Copied expert file bytes differ from the saved plan')


def _copy(source,target,count,sha):
    source=_plain(source);target=_plain(target)
    if target.exists():_verify(target,count,sha);return 0
    target.parent.mkdir(parents=True,exist_ok=True)
    pending=target.with_name(target.name+'.copy-pending')
    if pending.exists():
        _verify(source,count,sha) # the complete source still exists before discarding our interrupted partial
        _plain(pending).unlink()
    h=hashlib.sha256();copied=0
    with source.open('rb') as src,pending.open('xb') as out:
        for block in iter(lambda:src.read(MiB),b''):
            copied+=len(block);h.update(block)
            if copied>count:raise ValueError('Source expert grew beyond its saved extent')
            if out.write(block)!=len(block):raise OSError('Short expert-migration write')
        out.flush();os.fsync(out.fileno())
    if (copied,h.hexdigest())!=(count,sha):raise ValueError('Source payload changed during copy')
    _publish(pending,target);pending.unlink(missing_ok=True)
    return copied


def execute(root,plan,progress=None):
    root=Path(root).resolve(strict=True);target=_validate_plan(root,plan);started=time.perf_counter()
    source=_plain(root/BANKS[0]);entries=plan['entries'];copied=retired=0
    physical={name:target/Path(name).name for name in BANKS}
    config={'schema_version':1,'kind':'vesper_pc_source_roots','platform':'pc-heterogeneous','profile':'native',
        'model':MODEL,'revision':REVISION,'active_bank':BANKS[1],
        'mounts':[{'logical_prefix':name,'physical_directory':str(physical[name]),
            'marker_sha256':digest(_bytes(plan['markers'][name]))} for name in BANKS]}
    config_raw=_bytes(config)
    journal={'schema_version':1,'kind':'vesper_owned_bank_migration','plan_sha256':plan['plan_sha256'],
        'model':MODEL,'revision':REVISION,'store_ids':[plan['markers'][n]['store_id'] for n in BANKS]}
    with source_lease(root):
        active=(root/CONFIG).exists()
        if active and _read(root/CONFIG,64*MiB//1024)[1]!=config_raw:raise ValueError('Another source-root configuration is active')
        if not active and inventory(root)!=entries:raise ValueError('Closed source inventory changed since planning')
        if not active:
            volume=target
            while not volume.exists():
                if volume.parent==volume:raise OSError('SSD destination volume is unavailable')
                volume=volume.parent
            if shutil.disk_usage(volume).free<128*GiB+plan['total_file_bytes']+MiB:
                raise OSError('SSD migration would consume its reserve or fresh-bank admission')
        if target.exists():
            if not target.is_dir():raise ValueError('SSD migration target is not a directory')
            if (target/JOURNAL).exists():
                if _read(target/JOURNAL,4096)[0]!=journal:raise ValueError('SSD directory belongs to another migration')
            elif any(target.iterdir()):raise FileExistsError('Refusing to write into an unowned SSD directory')
        target.mkdir(parents=True,exist_ok=True);_new_file(target/JOURNAL,_bytes(journal))
        for name in BANKS:_new_file(physical[name]/MARKER,_bytes(plan['markers'][name]))
        remaining=plan['total_file_bytes']
        if not active:
            for i,e in enumerate(entries):
                # Retain64GiB free, admit the entire fresh64GiB write bank,
                # and leave space for all remaining verified files and metadata.
                if shutil.disk_usage(target).free<128*GiB+remaining+MiB:
                    raise OSError('SSD migration would consume its reserve or fresh-bank admission')
                for filename,prefix in (('experts.bin','payload'),('manifest.json','manifest')):
                    copied+=_copy(source/e['directory']/filename,physical[BANKS[0]]/e['directory']/filename,
                        e[prefix+'_bytes'],e[prefix+'_sha256'])
                remaining-=e['payload_bytes']+e['manifest_bytes']
                if progress:progress({'stage':'copied','objects':i+1,'total_objects':len(entries)})
            if inventory(root)!=entries:raise ValueError('Source inventory changed before activation')
        for i,e in enumerate(entries):
            for filename,prefix in (('experts.bin','payload'),('manifest.json','manifest')):
                _verify(physical[BANKS[0]]/e['directory']/filename,e[prefix+'_bytes'],e[prefix+'_sha256'])
            if progress:progress({'stage':'verified','objects':i+1,'total_objects':len(entries)})
        _new_file(root/CONFIG,config_raw)
        selected=SourceRoots(root);selected.check_unchanged()
        _remaining_files(source,entries)
        for i,e in enumerate(entries):
            for filename,prefix in (('experts.bin','payload'),('manifest.json','manifest')):
                old=_plain(source/e['directory']/filename)
                if old.exists():
                    # Check both copies again immediately before retiring this
                    # exact source file. Never recursively delete a bank/tree.
                    if not old.resolve(strict=True).is_relative_to(source):raise ValueError('Retirement escaped its original bank')
                    _verify(old,e[prefix+'_bytes'],e[prefix+'_sha256'])
                    _verify(physical[BANKS[0]]/e['directory']/filename,e[prefix+'_bytes'],e[prefix+'_sha256'])
                    old.unlink();retired+=e[prefix+'_bytes']
            for directory in (source/e['directory'],(source/e['directory']).parent):
                try:directory.rmdir()
                except OSError:pass # leave any still-owned or unknown contents intact
            if progress:progress({'stage':'retired','objects':i+1,'total_objects':len(entries)})
        selected.check_unchanged()
        _remaining_files(source,entries)
        if any((source/e['directory']/name).exists() for e in entries for name in ('manifest.json','experts.bin')):
            raise ValueError('Original bank files reappeared during retirement')
        return {'schema_version':1,'probe':'verified_original_expert_bank_relocation','status':'pass',
            'plan_sha256':plan['plan_sha256'],'model':MODEL,'revision':REVISION,'platform':'pc-heterogeneous','profile':'native',
            'objects':len(entries),'files':entries,'original_packed_payload_bytes':plan['original_packed_payload_bytes'],
            'all_files_verified_bytes':plan['total_file_bytes'],'copied_file_bytes_this_process':copied,
            'retired_source_file_bytes_this_process':retired,'source_storage':selected.public(),
            'original_index_bytes_changed':0,'weight_bytes_converted':0,'model_execution_run':False,'quality_scored':False,
            'elapsed_seconds':time.perf_counter()-started}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('plan','execute'))
    p.add_argument('--artifacts',type=Path,default=Path('artifacts'));p.add_argument('--ssd-directory',type=Path)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    from .trajectory_workspace import select,own_evidence
    w=select(a.artifacts);w.check_runtime_module(__file__,'src/vesper/bank_migration.py')
    path=own_evidence(a.artifacts,a.plan)
    if a.operation=='plan':
        if a.ssd_directory is None:p.error('--ssd-directory is required to plan')
        report=make_plan(a.artifacts,a.ssd_directory);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(_bytes(report))
        print(json.dumps({'status':'planned','objects':len(report['entries']),'payload_bytes':report['original_packed_payload_bytes'],
            'plan_sha256':report['plan_sha256']}),flush=True)
    else:
        if a.ssd_directory is not None:p.error('Execution uses the saved physical destination')
        if a.receipt is None:p.error('--receipt is required for execution')
        receipt=own_evidence(a.artifacts,a.receipt)
        if receipt.exists():raise FileExistsError('Refusing to replace a completed migration receipt')
        plan=_read(path,4*MiB)[0]
        context=w.context('pc-bank-migration-'+plan['plan_sha256'][:16])
        report=execute(a.artifacts,plan,lambda v:print(json.dumps(v),flush=True))
        report=w.stamp_result(report,initial_context=context)
        receipt.parent.mkdir(parents=True,exist_ok=True)
        with receipt.open('xb') as stream:stream.write(_bytes(report))
        print(json.dumps({'status':'pass','objects':report['objects'],'model_execution_run':False}),flush=True)


if __name__=='__main__':main()
