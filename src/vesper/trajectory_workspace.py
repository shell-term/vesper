"""PC-only run routing and explicit import of a pre-isolation input snapshot."""

from copy import deepcopy
import json
from pathlib import Path

from .checkpoint import MODEL,REVISION
from .expert_bank import existing_sources
from .model_artifacts import canonical,digest
from .resources import MiB
from .workspace import Workspace,WorkspaceError
from .source_roots import roots,BANKS


def select(root):
    try:root=Path(root).resolve(strict=True);workspace=Workspace.open(root.parent)
    except OSError as error:raise WorkspaceError('Missing or invalid PC artifact root/binding') from error
    profile=workspace.config['profiles'][workspace.profile]
    if (root!=workspace.root/'artifacts' or workspace.platform!='pc-heterogeneous' or workspace.profile!='native'
            or profile!={'model':MODEL,'revision':REVISION}):raise WorkspaceError('This trajectory runner requires the explicitly bound native PC workspace')
    workspace.check_runtime_module(__file__,'src/vesper/trajectory_workspace.py')
    return workspace


def run_path(root,name):
    workspace=select(root)
    return workspace,workspace.artifact_path(f'trajectories/{name}/plan.json',create_parent=True).parent


def bank_sources(root):
    root=Path(root)
    # Legacy immutable inputs remain readable, but all new writes go to the
    # selected platform. Do not move a live legacy bank or duplicate its bytes.
    selected=roots(root)
    return sorted(set(existing_sources(root)+list((root/'expert-bank-v1').glob('layer*/expert-*'))
        +[p for name in BANKS for p in selected.path(name,strict=False).glob('layer*/expert-*')]))


def bank_path(root):
    workspace=select(root);selected=roots(root)
    if selected.mounts:return selected.path(selected.active_bank)
    return workspace.artifact_path('expert-bank-v1/placeholder',create_parent=True).parent


def own_evidence(root,path):
    workspace=select(root);path=Path(path).resolve(strict=False)
    if not any(path.is_relative_to(workspace.root/part) for part in ('artifacts/pc-heterogeneous','results/pc-heterogeneous')):
        raise WorkspaceError('A PC trajectory cannot read foreign or unbound execution evidence')
    return path


def _copy(source,target,limit):
    if not source.is_file() or not 1<=source.stat().st_size<=limit:raise ValueError('Invalid bounded legacy input snapshot')
    with source.open('rb') as stream:raw=stream.read(limit+1)
    if len(raw)>limit:raise ValueError('Legacy input grew beyond its bound')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.read_bytes()!=raw:raise FileExistsError('Refusing to replace a migrated input')
    else:
        with target.open('xb') as stream:stream.write(raw)
    return {'bytes':len(raw),'sha256':digest(raw)}


def migrate_input(root,name):
    from .trajectory_evidence import load_plan,plan_identity
    root=Path(root);workspace,target=run_path(root,name);legacy=root/'trajectories'/name
    if not legacy.resolve(strict=True).is_relative_to(root.resolve(strict=True)):raise ValueError('Legacy input escaped the source root')
    if (target/'plan.json').exists():raise FileExistsError('PC input snapshot already exists')
    plan=load_plan(root,legacy,allow_legacy=True);raw=(legacy/'plan.json').read_bytes();files={}
    for e in plan['assets']['embeddings']:
        if type(e.get('id')) is not int or not 0<=e['id']<129280 or e['file']!=f"embeddings/{e['id']}.bin":raise ValueError('Unanchored legacy embedding input')
        name_=e['file'];files[name_]=_copy(legacy/name_,target/name_,10240)
    for l in (1,14):
        for filename,limit in (('manifest.json',12*MiB),('rows.bin',8*MiB)):
            name_=f'rows/layer{l}/{filename}';files[name_]=_copy(legacy/name_,target/name_,limit)
    changed=deepcopy(plan)
    for l,p in changed['experts'].items():
        for filename,limit in (('manifest.json',4*MiB),):
            name_=f'indices/layer{l}/{filename}';files[name_]=_copy(legacy/name_,target/name_,limit)
        name_=f'indices/layer{l}.json';files[name_]=_copy(legacy/name_,target/name_,4*MiB)
        p['store']=(target.resolve().relative_to(root.resolve())/f'indices/layer{l}').as_posix()
    changed.pop('plan_sha256')
    changed['legacy_input_migration']={'source_plan_raw_sha256':digest(raw),'source_plan_identity_sha256':plan_identity(plan),
        'new_model_execution_run':False,'expert_payload_bytes_copied':0,'source_run':f'trajectories/{name}','copied_files':files}
    changed['plan_sha256']=canonical(changed)
    if plan_identity(changed)!=plan_identity(plan):raise ValueError('Migration changed the logical model input snapshot')
    workspace.check()
    with (target/'plan.json').open('xb') as stream:stream.write((json.dumps(changed,indent=2)+'\n').encode())
    _copy(legacy/'plan.json',target/'legacy-plan.json',64*MiB)
    if (legacy/'preparation.json').exists():_copy(legacy/'preparation.json',target/'legacy-preparation.json',64*MiB)
    load_plan(root,target)
    return {'status':'pass','platform':workspace.platform,'logical_plan_sha256':plan_identity(plan),'expert_payload_bytes_copied':0,
        'new_model_execution_run':False,'migrated_files':len(files)}
