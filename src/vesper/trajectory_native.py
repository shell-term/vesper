"""Independent offline native execution of a qualified whole-model trajectory."""

from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

from .backbone import LAYERS
from .backbone_assets import load as load_bindings
from .backbone_runtime import open_backbone
from .backbone_validation import native_state
from .compile_cache import CompilationCache
from .engram_rows import EngramRows
from .evaluation import score_completion
from .generation import TextGeneration
from .generation_validation import record
from .indexed_moe import IndexedMoE
from .model_artifacts import digest
from .output_assets import load as load_output
from .resources import GiB, MemoryBudget
from .sequence import MAX_ROWS
from .sequence_assets import request
from .tokenizer import PinnedTokenizer
from .topology_validation import state_digest
from .trajectory_evidence import load_plan,load_tokens,plan_identity,scope,expert_plan,load_case,write_case,validate_original
from .trajectory_workspace import own_evidence
from .source_roots import roots


class InjectedWholeTokenFailure(RuntimeError):pass


def run_native(root,run,original,original_raw,reference_run,progress=None):
    own_evidence(root,run);own_evidence(root,reference_run)
    validate_original(original,reference_run)
    frameworks={n:{'installed':importlib.util.find_spec(n) is not None,'imported':n in sys.modules} for n in ('torch','tilelang','transformers','sympy')}
    if any(any(v.values()) for v in frameworks.values()):raise ValueError('Expected an independent framework-free native environment')
    root,run=Path(root),Path(run);plan=load_plan(root,run);case,req=request(plan['case']['id'])
    if plan_identity(plan)!=original['plan_sha256']:raise ValueError('Different actual whole-model preparation')
    with ExitStack() as stack:
        http=stack.enter_context(patch('urllib.request.urlopen',side_effect=RuntimeError('Native whole-model trajectory must stay offline')))
        allowance=MemoryBudget(cap=12*GiB,owner='longer_native_fixture');allowance.acquire(allowance.cap)
        stack.callback(allowance.release,allowance.cap)
        cache=stack.enter_context(CompilationCache())
        bindings=json.loads((root/'backbone-controls.json').read_bytes());arrays=load_bindings(root,bindings)
        head_arrays,head_manifest=load_output(root/'output-fixture');stack.callback(arrays.clear);stack.callback(head_arrays.clear)
        if head_manifest!=original['output_fixture']:raise ValueError('Different actual original output fixture')
        rows={l:stack.enter_context(EngramRows(run/f'rows/layer{l}',l,max_rows=MAX_ROWS)) for l in (1,14)}
        providers={l:IndexedMoE(root,bindings,l,plan['experts'][str(l)],original['expert_provenance'],index_directory=root/plan['experts'][str(l)]['store']) for l in LAYERS}
        engine=open_backbone(stack,root,arrays,load_tokens(root,run,plan),providers,head_arrays,engram_rows=rows)
        if (bindings['bindings_sha256']!=original['bindings_sha256'] or engine.tokenizer_identity!=original['tokenizer']
                or engine.engram_manifests!=original['Engram_fixtures']
                or {str(l):p.converted_sha256 for l,p in engine.probe_operators['projections'].items()}!=original['original_grouped_conversion']):
            raise ValueError('Native whole-model original control/conversion bindings differ')
        tokenizer=PinnedTokenizer(root/'import-source');fault=None;cases=[]
        def step(ids,*,sample):
            nonlocal fault
            if engine.position==len(original['completion']['prompt_ids'])-1 and fault is None:
                before=state_digest(native_state(engine))
                def refuse(result):raise InjectedWholeTokenFailure('Injected after final logits and sampling')
                try:engine.step(ids,sample=sample,observe=refuse)
                except InjectedWholeTokenFailure:pass
                else:raise RuntimeError('Expected an actual failed whole-model token')
                after=state_digest(native_state(engine))
                if before!=after:raise RuntimeError('Failed longer whole-model token published model state or RNG')
                fault={'position':engine.position,'stage':'after_final_logits_and_sampling','error':'InjectedWholeTokenFailure',
                    'before_state_sha256':before,'after_state_sha256':after,'healthy_retry_compared':True}
            return engine.step(ids,sample=sample)
        session=TextGeneration(tokenizer,step,req)
        while not session.done:
            meta,result=session.advance();captured=record(meta,result,native_state(engine))
            if len(cases)>=len(original['case_files']):raise RuntimeError('Native execution exceeds original completion')
            expected,raw=load_case(reference_run,original['case_files'][len(cases)],'original')
            for key in ('meta','boundaries','state_sha256','rng_offset','ordered_routes'):
                if captured[key]!=expected[key]:raise RuntimeError('Actual longer whole-model native math/state differs: '+key)
            captured['status']='pass';captured['reference_case_sha256']=digest(raw)
            cases.append(write_case(run,'native',captured))
            if progress:progress({'native_token':meta,'boundaries':len(captured['boundaries']),'status':'pass'})
        decoded=tokenizer.decode(list(session.generated));score=score_completion(case,decoded,stop_reason=session.stop_reason)
        report={'schema_version':1,'probe':'native_whole_model_trajectory_recheck','status':'pass','scope':scope(case['id']),
            'source_storage':roots(root).public(),
            'reference_report_sha256':digest(original_raw),'original_identity_sha256':original['identity_sha256'],
            'complete_goldens_sha256':original['complete_goldens_sha256'],'plan_sha256':plan_identity(plan),'expert_plan':expert_plan(plan),
            'case_files':cases,'completion':session.summary(),'decoded_selected_ids':decoded,'quality_score':score,'fault':fault,
            'native_frameworks':frameworks,'python_http_disabled_during_native_execution':True,'HTTP_calls_during_native':http.call_count,
            'weight_payloads_fetched_in_recheck':0,'indexed_sources':{str(l):p.stats() for l,p in providers.items()},
            'Engram_row_caches_before_close':{str(l):r.stats() for l,r in rows.items()},
            'compiler_cache':cache.stats(),'compiled_images':cache.manifest(),'transaction_active':engine.transaction.stats()}
    report['compiler_cache_after_close']=cache.stats();report['transaction_cleanup']=engine.transaction.stats()
    report['Engram_row_caches_after_close']={str(l):r.stats() for l,r in rows.items()}
    return report
