"""Prepare actual longer whole-model trajectories without copying expert unions."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from .backbone import LAYERS
from .backbone_assets import load as load_bindings
from .backbone_prepare import PreparedMoE
from .backbone_runtime import open_backbone, validate_moe_manifests
from .catalog_reader import CatalogReader
from .checkpoint import MODEL, REVISION
from .compile_cache import CompilationCache
from .connected import ConnectedMoE
from .expert_bank import ExpertBank, existing_sources
from .formats import bf16_values
from .gate import GateProjection, load_gate_fixture
from .generation import TextGeneration
from .generation_assets import GenerationAssets, reserve_disk
from .indexed_store import IndexedStore
from .model_artifacts import canonical
from .output_assets import load as load_output
from .resources import GiB, MiB, MemoryBudget
from .router import RouterProbe, load_router_fixture
from .sequence import MAX_EMBEDDINGS, MAX_ROWS, scope
from .sequence_assets import request
from .tokenizer import PinnedTokenizer
from .trajectory_workspace import select,bank_path,bank_sources


class PlanningMoE(PreparedMoE):
    def __init__(self,root,bindings,layer,run,bank,reader,transfers,progress):
        super().__init__(root,bindings,layer)
        self.run,self.bank,self.reader,self.transfers,self.progress=Path(run),bank,reader,transfers,progress
        self.decisions=[]

    def resolve(self,values):
        weight,_=load_gate_fixture(self.files[0]);bias,_=load_router_fixture(self.files[1])
        with GateProjection(weight) as gate,RouterProbe(bias) as router:
            route=router.inspect(gate.project(bf16_values(values)),False)
        ids=sorted(map(int,route['indices']))
        self.bank.prepare(self.layer,ids,self.reader,self.transfers)
        scratch=self.run/'scratch';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='index-',dir=scratch) as temporary:
            target=Path(temporary)/'index';self.bank.index(self.layer,ids,target)
            factory=lambda directory,budget:IndexedStore(directory,self.root,budget,layer=self.layer)
            with ConnectedMoE(target,*self.files,layer=self.layer,profile='backbone_probe',source_factory=factory) as moe:
                validate_moe_manifests(moe,self.bindings,self.layer)
                result=moe.forward(values,False)
        if not np.array_equal(result['routing']['indices'],route['indices']) or not np.array_equal(result['routing']['weights'].view(np.uint32),route['weights'].view(np.uint32)):
            raise RuntimeError('Actual trajectory preparation changed the original route')
        self.available.update(ids)
        self.decisions.append({'indices':route['indices'].tolist(),'weight_bits':route['weights'].view(np.uint32).tolist()})
        if self.progress:self.progress({'prepared_layer':self.layer,'position':len(self.decisions)-1,'distinct_experts':len(self.available)})
        return result


def prepare(root,run,*,case_id='coding-python-expression',progress=None):
    root,run=Path(root),Path(run)
    workspace=select(root)
    if not run.resolve().is_relative_to((root/'pc-heterogeneous/trajectories').resolve()):raise ValueError('Expected a bound PC whole-model trajectory run')
    run.mkdir(parents=True,exist_ok=True)
    if (run/'plan.json').exists():raise FileExistsError('Refusing to replace a complete whole-model preparation')
    reserve_disk(run,8*GiB)
    case,req=request(case_id);bindings=json.loads((root/'backbone-controls.json').read_bytes())
    started=time.perf_counter()
    with ExitStack() as stack:
        allowance=MemoryBudget(cap=12*GiB,owner='whole_trajectory_fixture_preparation');allowance.acquire(allowance.cap)
        stack.callback(allowance.release,allowance.cap)
        row_budget=MemoryBudget(cap=128*MiB,owner='whole_trajectory_input_rows');row_budget.acquire(row_budget.cap)
        stack.callback(row_budget.release,row_budget.cap)
        cache=stack.enter_context(CompilationCache())
        reader=stack.enter_context(CatalogReader(root/'import-source'))
        bank=stack.enter_context(ExpertBank(root,bank_path(root),bank_sources(root)))
        transfers=stack.enter_context(ThreadPoolExecutor(max_workers=8))
        tokenizer=PinnedTokenizer(root/'import-source');prompt=req.encode(tokenizer)
        assets=GenerationAssets(root,run,reader,max_embeddings=MAX_EMBEDDINGS,max_rows=MAX_ROWS)
        for token in prompt:assets.token(token)
        arrays=load_bindings(root,bindings);head_arrays,head_manifest=load_output(root/'output-fixture')
        stack.callback(arrays.clear);stack.callback(head_arrays.clear)
        planners={l:PlanningMoE(root,bindings,l,run,bank,reader,transfers,progress) for l in LAYERS}
        engine=open_backbone(stack,root,arrays,assets.token_rows,planners,head_arrays,engram_rows=assets.rows)
        def step(ids,*,sample):
            assets.token(int(ids[0,0]));candidate=engine.transaction.history.prepare(ids,engine.position);assets.prepare_rows(candidate)
            return engine.transaction.step(ids,lambda c:engine._finish(engine._evaluate(c,ids,lambda l,v:planners[l].resolve(v)),sample=sample))
        session=TextGeneration(tokenizer,step,req);calls=[]
        while not session.done:
            meta,result=session.advance();calls.append({**meta,'routes':{str(l):planners[l].decisions[-1] for l in LAYERS}})
            if progress:progress({'planned_token':meta,'continuation':session.summary(),'elapsed_seconds':time.perf_counter()-started,'bank':bank.stats(),'compiler':cache.stats()})
        prepared_assets=assets.freeze();experts={}
        for l,p in planners.items():
            path=run/f'indices/layer{l}';receipt=bank.index(l,p.available,path)
            with (run/f'indices/layer{l}.json').open('xb') as stream:stream.write((path/'manifest.json').read_bytes())
            experts[str(l)]={'ids':sorted(p.available),'store':path.resolve().relative_to(root.resolve()).as_posix(),**receipt}
        report={'schema_version':1,'probe':'whole_model_trajectory_artifact_preparation','status':'pass','model':MODEL,'revision':REVISION,
            'scope':scope(),'case':case,'request':{'prompt':req.prompt,'max_new_tokens':req.max_new_tokens,'eos_id':req.eos_id},
            'temperature':.7,'seed':2026,'layers':list(LAYERS),'bindings_sha256':bindings['bindings_sha256'],'output_fixture':head_manifest,
            'tokenizer':engine.tokenizer_identity,'assets':prepared_assets,'experts':experts,'calls':calls,'completion':session.summary(),
            'bank':bank.stats(),'reader':reader.stats(),'compiler':cache.stats(),'compiled_images':cache.manifest(),'maximum_parallel_expert_imports':4,
            'maximum_parallel_payload_transfers':8,'artifact_preparation_fetch_between_stages':True,'independent_model_comparison_run':False,
            'original_reference_execution_run':False,'quality_scored':False,'transaction':engine.transaction.stats()}
    report['compiler_after_close']=cache.stats();report['bank_after_close']=bank.stats()
    report['plan_sha256']=canonical(report)
    with (run/'plan.json').open('xb') as stream:stream.write((json.dumps(report,indent=2)+'\n').encode())
    return report
