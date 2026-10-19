"""Complete bounded Engram corpus, original bit comparisons and atomic failures."""

from contextlib import ExitStack
import copy
import json
from pathlib import Path

import numpy as np

from .attention_validation import array_digest,check_arrays
from .engram import NativeEngram,GATE_STAGES
from .engram_assets import load_controls,CONTROL_BYTES
from .engram_corpus import hidden
from .engram_rows import EngramRows
from .engram_session import EngramSession
from .model_artifacts import canonical
from .ngram import NgramState,LAYERS,HashSnapshot
from .resources import MemoryBudget,MiB,GiB,physical_memory
from .tokenizer import PinnedTokenizer,compressed_token_map

FAULTS=('host_pressure','row_io_layer1','row_hash_corruption_layer1','after_layer1_output','row_cache_pressure_layer14',
    'layer14_projection_download','layer14_final_download','missing_row_layer14','combined_primary_cap','publication_pressure')
STAGES=('rows_codes_fp8','rows_scales_e8m0','embedding_fp32','embedding_bf16','projection_input_fp8','projection_scale_e8m0',
    'projection_output_fp32','projection_output_bf16','hidden_square_fp32','key_square_fp32','weight_product_fp32','hidden_weight_fp32',
    'dot_product_fp32','hidden_mean_fp32','key_mean_fp32','dot_sum_fp32',*GATE_STAGES,'delta_fp32','residual_fp32','output_bf16')
QUALIFIED_IDENTITY='4deaf66bf1c7684c30ce73639c72d837c45182cd519ad8c71fedf68b1e96802f'
QUALIFIED_GOLDENS='d65af2cbf3dc904f210038866a09f4e9cc4990f6bda27e496ce133e9abb4af35'
NEGATIVES=('normalize_across_HC_copies','remove_signed_sqrt','discard_original_row_scales',
    'drop_signed_sqrt_clamp_at_zero','omit_token_mask','ignore_masked_history','clear_history_between_chunks')


def open_session(stack,source,fixtures):
    allowance=MemoryBudget(cap=GiB); allowance.acquire(allowance.cap); stack.callback(allowance.release,allowance.cap)
    tokenizer=PinnedTokenizer(source); mapping,identity=compressed_token_map(tokenizer)
    ops,manifests={},{}
    for layer in LAYERS:
        rows=stack.enter_context(EngramRows(Path(fixtures)/'rows'/f'layer{layer}',layer))
        arrays,m=load_controls(Path(fixtures)/f'controls{layer}',layer)
        manifests[str(layer)]={'controls':m,'rows':rows.manifest}
        ops[layer]=stack.enter_context(NativeEngram(arrays,rows))
    session=stack.enter_context(EngramSession(NgramState(mapping),ops))
    return session,manifests,identity


def metadata(key,case,variant='original'):
    return {'id':key,'source_case':case['id'],'stream_index':case['stream_index'],'position':case['position'],
        'batch':len(case['ids']),'sequence':len(case['ids'][0]),'variant':variant}


def compare_outputs(expected,actual):
    checks=[]
    for layer in LAYERS:
        if set(expected[layer])!=set(STAGES) or set(actual['layers'][layer])!=set(STAGES): raise ValueError('Incomplete Engram boundary registry')
        checks.extend(check_arrays(expected[layer][n],actual['layers'][layer][n],f'layer{layer}/{n}') for n in STAGES)
    return checks


def fingerprint_outputs(actual):
    return [{'check':f'layer{l}/{n}','elements':a.size,'dtype':str(a.dtype),'shape':list(a.shape),'sha256':array_digest(a)}
        for l in LAYERS for n,a in actual['layers'][l].items()]


def exercise(session,plan,checker,progress=None):
    results,faults,snapshots=[],[],[]; minimum=physical_memory()['available']
    by_id={c['id']:c for c in plan['hash_cases']}
    def record(key,case,variant='original'):
        nonlocal minimum
        x=np.array(case['ids'],np.int64); mask=None if case['mask'] is None else np.array(case['mask'],np.bool_)
        inputs={l:hidden(case['stream_index'],case['position'],x.shape[0],x.shape[1],l) for l in LAYERS}
        if variant=='zero':
            for h in inputs.values(): h.fill(0); h[...,1]=0x8000
        elif variant=='scaled':
            from .formats import bf16_bits,bf16_values
            inputs={l:bf16_bits(bf16_values(h)*np.float32(64)) for l,h in inputs.items()}
        actual=session.step(x,inputs,mask,case['position'])
        if (actual['hash_ids'].tolist()!=case['hash_ids'] or actual['compressed'].tolist()!=case['compressed']
                or actual['lookbacks'].tolist()!=case['lookbacks'] or session.hash.history.tolist()!=case['committed_history']
                or actual['state_sha256']!=case['state_sha256']): raise RuntimeError('Original Engram addresses/history differ')
        meta=metadata(key,case,variant); checked=checker(meta,inputs,mask,actual)
        if checked['status']!='pass': raise RuntimeError('Complete original Engram intermediate comparison differs')
        row={'meta':meta,'status':'pass','input_sha256':{str(l):array_digest(h) for l,h in inputs.items()},
            'hash_ids_sha256':array_digest(actual['hash_ids']),'compressed_sha256':array_digest(actual['compressed']),
            'lookbacks_sha256':array_digest(actual['lookbacks']),'committed_state_sha256':actual['state_sha256'],**checked}
        results.append(row); minimum=min(minimum,physical_memory()['available'])
        if progress: progress({'case':key,'steps':len(results),'status':'pass'})
        return row
    for repeat in range(2):
        for case in plan['hash_cases']:
            if case['position']==0: session.hash.reset()
            record(f'repeat{repeat}/'+case['id'],case)
    initial=by_id['english/decode/0']; following=by_id['english/decode/1']; third=by_id['english/decode/2']
    for variant in ('zero','scaled'):
        session.hash.reset(); record('component/'+variant,initial,variant)
    for mode in FAULTS:
        session.hash.reset(); record('fault/'+mode+'/warm',initial)
        state=session.hash.state_sha256(); counts={l:o.calls_completed for l,o in session.operators.items()}
        prior_cap_refusals=session.combined_cap_refusals
        started={l:o.calls_started for l,o in session.operators.items()}
        o1,o14=[session.operators[l] for l in LAYERS]; r1,r14=o1.rows,o14.rows
        r1.clear(); r14.clear()
        original=(session.budget.acquire,r1.reader.read,o1.forward,r14.budget.acquire,o14.device.download,o14.forward)
        resident={l:dict(o.device.allocations) for l,o in session.operators.items()}
        def pressure(*a,**k): raise MemoryError('Injected bounded Engram pressure')
        def io(*a,**k): raise OSError('Injected required-row I/O failure')
        def corrupt(*a,**k): return bytes(b^1 for b in original[1](*a,**k))
        def after_source(*a,**k): original[2](*a,**k); raise OSError('Injected failure after layer1 output')
        def download(pointer,shape,dtype):
            if (mode=='layer14_projection_download' and shape[-1]==25600 and dtype==np.float32
                    or mode=='layer14_final_download' and tuple(shape[-2:])==(4,5120) and dtype==np.uint16):
                raise OSError('Injected layer14 download failure')
            return original[4](pointer,shape,dtype)
        def late(*a,**k):
            output=original[5](*a,**k)
            if mode=='combined_primary_cap':
                if sum(o14.device.allocations.values())+100*MiB>o14.device.cap: raise RuntimeError('Injected request exceeds the individual ledger')
                pointer=o14.device.allocate(100*MiB); o14.device.free(pointer)
                raise RuntimeError('Combined Engram cap did not refuse allocation')
            session.budget.acquire=pressure
            return output
        removed=None
        if mode=='host_pressure': session.budget.acquire=pressure
        elif mode=='row_io_layer1': r1.reader.read=io
        elif mode=='row_hash_corruption_layer1': r1.reader.read=corrupt
        elif mode=='after_layer1_output': o1.forward=after_source
        elif mode=='row_cache_pressure_layer14': r14.budget.acquire=pressure
        elif mode in ('layer14_projection_download','layer14_final_download'): o14.device.download=download
        elif mode=='missing_row_layer14':
            missing=int(np.array(following['hash_ids'],np.int64)[0,0,1,0]); removed=(missing,r14.entries.pop(missing))
        elif mode in ('combined_primary_cap','publication_pressure'): o14.forward=late
        caught=None
        try:
            x=np.array(following['ids'],np.int64); mask=None
            inputs={l:hidden(0,1,1,1,l) for l in LAYERS}
            session.step(x,inputs,mask,1)
        except (MemoryError,OSError,ValueError) as e: caught=type(e).__name__
        finally:
            (session.budget.acquire,r1.reader.read,o1.forward,r14.budget.acquire,o14.device.download,o14.forward)=original
            if removed is not None: r14.entries[removed[0]]=removed[1]
        completed=[session.operators[l].calls_completed-counts[l] for l in LAYERS]
        expected=[0,0] if mode in FAULTS[:3] else [1,1] if mode in FAULTS[-2:] else [1,0]
        clean=all(session.operators[l].device.allocations==resident[l] for l in LAYERS)
        unchanged=state==session.hash.state_sha256()
        parent_refused=session.combined_cap_refusals-prior_cap_refusals==1
        if mode=='combined_primary_cap' and not parent_refused: raise RuntimeError('Combined Engram parent did not refuse before backend')
        if caught is None or completed!=expected or not clean or not unchanged: raise RuntimeError('Engram failure did not preserve atomic state and cleanup')
        faults.append({'name':mode,'status':'pass','error_type':caught,'no_output_returned':True,'history_and_position_unchanged':unchanged,
            'temporary_GPU_buffers_released':clean,'completed_layers':completed,
            'started_layers':[session.operators[l].calls_started-started[l] for l in LAYERS]})
        if mode=='combined_primary_cap': faults[-1].update(requested_bytes=100*MiB,individual_cap_allows_request=True,combined_parent_refused_before_backend=parent_refused)
        record('fault/'+mode+'/retry',following)
    session.hash.reset(); record('snapshot/warm',initial); before=session.hash.state_sha256(); handle=session.hash.snapshot()
    first=record('snapshot/advance',following); record('snapshot/advance-again',third)
    session.hash.rollback(handle); restored=session.hash.state_sha256()==before
    again=record('snapshot/replayed',following)
    same=first['boundaries']==again['boundaries'] and first['committed_state_sha256']==again['committed_state_sha256']
    refused=0
    for token in (handle,HashSnapshot()):
        try: session.hash.rollback(token)
        except ValueError: refused+=1
    copied=session.hash.inspect(); before=session.hash.state_sha256(); copied['history'].fill(999)
    isolated=before==session.hash.state_sha256()
    handle=session.hash.snapshot(); session.hash.reset()
    try: session.hash.rollback(handle)
    except ValueError: refused+=1
    if not restored or not same or not isolated or refused!=3: raise RuntimeError('Engram history snapshot semantics differ')
    snapshots.append({'name':'whole_token_history','status':'pass','history_restored':restored,'all_boundaries_reproduced':same,
        'diagnostic_copy_isolated':isolated,'consumed_foreign_expired_handle_refusals':refused})
    return results,faults,snapshots,minimum


def report_identity(report):
    return canonical({k:report[k] for k in ('model','revision','plan','fixtures','native_profiles','hash_reference','original_source','original_methods','VMM_transport')})


def golden_digest(report):
    return canonical([{k:r[k] for k in ('meta','input_sha256','hash_ids_sha256','compressed_sha256','lookbacks_sha256','committed_state_sha256','boundaries')}
        for r in report['session_cases']])


def expected_metadata(plan):
    cases=plan['hash_cases']; first,second,third=[next(c for c in cases if c['id']==f'english/decode/{i}') for i in range(3)]
    result=[metadata(f'repeat{r}/'+c['id'],c) for r in range(2) for c in cases]
    result.extend(metadata('component/'+v,first,v) for v in ('zero','scaled'))
    result.extend(metadata('fault/'+f+'/'+label,c) for f in FAULTS for label,c in (('warm',first),('retry',second)))
    result.extend(metadata('snapshot/'+label,c) for label,c in (('warm',first),('advance',second),('advance-again',third),('replayed',second)))
    return result


def validate_faults(faults,snapshots):
    if [f['name'] for f in faults]!=list(FAULTS): raise ValueError('Incomplete Engram failure matrix')
    for i,f in enumerate(faults):
        expected=[0,0] if i<3 else [1,1] if i>=8 else [1,0]
        started=[0,0] if i==0 else [1,0] if i<4 else [1,1]
        if (f['status']!='pass' or f['error_type'] not in ('MemoryError','OSError','ValueError')
                or f['completed_layers']!=expected or f['started_layers']!=started
                or any(f[k] is not True for k in ('no_output_returned','history_and_position_unchanged','temporary_GPU_buffers_released'))):
            raise ValueError('Incomplete Engram failure, state or cleanup evidence')
    parent=faults[8]
    if (parent.get('requested_bytes')!=100*MiB or parent.get('individual_cap_allows_request') is not True
            or parent.get('combined_parent_refused_before_backend') is not True): raise ValueError('Missing combined-parent refusal proof')
    if len(snapshots)!=1: raise ValueError('Incomplete Engram snapshot matrix')
    s=snapshots[0]
    if (s.get('name')!='whole_token_history' or s.get('status')!='pass' or s.get('consumed_foreign_expired_handle_refusals')!=3
            or any(s.get(k) is not True for k in ('history_restored','all_boundaries_reproduced','diagnostic_copy_isolated'))):
        raise ValueError('Incomplete original token-history restoration evidence')


def validate_memory(memory,*,reference):
    if (memory['native_host_allowances_bytes']!=2096*MiB or memory['system_available_floor_bytes']!=32*GiB
            or memory['minimum_sampled_available_bytes']<32*GiB or memory['reservation_is_RSS_limit'] is not False):
        raise ValueError('Engram host resource evidence differs')
    for phase in ('active','cleanup'):
        m=memory[phase]; done=phase=='cleanup'; s=m['session']
        if (s['combined_primary_cap_bytes']!=384*MiB or not 2*CONTROL_BYTES<=s['combined_primary_peak_owned_bytes']<=384*MiB
                or s['host_reserved_bytes']!=(0 if done else 32*MiB) or s['combined_cap_refusals_before_backend']!=1
                or s['history_tokens_per_batch_row']!=3): raise ValueError('Combined Engram allocation evidence differs')
        for layer in ('1','14'):
            op=m['operators'][layer]; rows=m['rows'][layer]
            if (op['host_reserved_bytes']!=(0 if done else 512*MiB) or op['host_allowance_bytes']!=512*MiB
                    or not CONTROL_BYTES<=op['device_peak_owned_bytes']<=256*MiB
                    or op['device_owned_bytes']!=(0 if done else CONTROL_BYTES) or op['owned_context_active'] is done):
                raise ValueError('Engram operator allocation/cleanup evidence differs')
            if (rows['host_allowance_bytes']!=8*MiB or not 0<rows['host_peak_bytes']<=8*MiB or rows['cache_capacity_rows']!=32
                    or rows['cached_rows']!=(0 if done else 32) or (done and rows['host_reserved_bytes']!=0)
                    or rows['model_wide_mapping'] is not False or rows['buffered_row_read_fallback'] is not False):
                raise ValueError('Engram row-cache bounds/cleanup evidence differs')
        if reference:
            v=m['VMM']
            if (v['physical_cap_bytes']!=4*GiB or v['page_bytes']!=2*MiB or v['peak_physical_owned_bytes']!=3888119808
                    or v['peak_virtual_reserved_bytes']!=202761043968 or v['physical_owned_bytes']!=(0 if done else 3888119808)
                    or v['virtual_reserved_bytes']!=(0 if done else 202761043968) or v['owned_context_active'] is done
                    or v['F_embedding_replaced'] is not False or v['original_row_addresses_remapped'] is not False
                    or v['unrequested_pages']!='unmapped' or v['physical_allocations_released'] is not done
                    or v['virtual_reservations_released'] is not done): raise ValueError('Unchanged full-address oracle storage evidence differs')
            from .ngram import TABLE_ROWS
            if ([t['shape'] for t in v['tables']]!=[[n,w] for n in TABLE_ROWS for w in (256,8)]
                    or any(t['original_addressing'] is not True for t in v['tables'])): raise ValueError('Original table addressing differs')
    if reference and memory['reference_host_allowance_bytes']!=2*GiB: raise ValueError('Original reference host allowance differs')


def validate_goldens(report):
    """Reject missing/substituted provenance or arithmetic before opening a GPU.

    Two registered digests bind the entire input/address/fixture/compiler corpus
    and all 6,820 typed/shaped original intermediates, including signed-zero bits.
    Failure, sensitivity and allocation evidence have independent semantic gates.
    """
    from .checkpoint import MODEL,REVISION
    from .ngram import layout
    if (report.get('probe'),report.get('status'),report.get('model'),report.get('revision'))!=(
            'original_Engram_complete_parity','pass',MODEL,REVISION): raise ValueError('Unqualified original Engram report')
    if (report_identity(report)!=QUALIFIED_IDENTITY or report.get('source_fixture_compiler_identity_sha256')!=QUALIFIED_IDENTITY
            or golden_digest(report)!=QUALIFIED_GOLDENS or report.get('complete_goldens_sha256')!=QUALIFIED_GOLDENS):
        raise ValueError('Unqualified Engram source/fixture/compiler or complete goldens')
    plan=report['plan']; cases=report['session_cases']; h=report['hash_reference']
    if ([r['meta'] for r in cases]!=expected_metadata(plan) or len(cases)!=110 or len(plan['hash_cases'])!=42
            or h['plan']!=plan or h['all_original_token_map_entries_compared']!=129280 or len(h['tokenizer_comparisons'])!=6
            or h['status']!='pass' or len(h['hash_comparisons'])!=42): raise ValueError('Incomplete original tokenizer/hash/session matrix')
    for name,a in zip(('primes','offsets','multipliers'),layout()):
        if h[name]!=a.tolist(): raise ValueError('Original N-gram layout differs')
    for c,hc in zip(plan['hash_cases'],h['hash_comparisons']):
        if (hc['case']!=c['id'] or hc['status']!='pass' or hc['unchanged_original_CPU_and_GPU_hashes_equal'] is not True
                or hc['hash_sha256']!=array_digest(np.array(c['hash_ids'],np.int64))
                or hc['active_tail_sha256']!=array_digest(np.array(c['committed_history'],np.int64))):
            raise ValueError('Original CPU/GPU hash or carried history differs')
    controls=report['unchanged_original_controls']
    if len(controls)!=220: raise ValueError('Missing unchanged original forward controls')
    for i,r in enumerate(cases):
        boundaries=r['boundaries']
        if (r['status']!='pass' or [b['check'] for b in boundaries]!=[f'layer{l}/{n}' for l in LAYERS for n in STAGES]
                or any(b['status']!='pass' or b['mismatches']!=0 or b['actual_sha256']!=b['expected_sha256']
                    or b['elements']!=int(np.prod(b['shape'])) for b in boundaries)):
            raise ValueError('Incomplete bitwise original Engram boundary evidence')
        for j,l in enumerate(LAYERS):
            c=controls[2*i+j]
            if (c['layer']!=l or c['status']!='pass' or c['output_sha256']!=boundaries[j*len(STAGES)+len(STAGES)-1]['expected_sha256']
                    or any(c[k] is not True for k in ('embedding_forward_bitwise','projection_FP32_rounds_to_original_linear','Engram_forward_bitwise'))):
                raise ValueError('Unchanged original Engram method control differs')
    validate_faults(report['faults'],report['snapshots'])
    negatives=report['negative_controls']
    if ([c['check'] for c in negatives]!=list(NEGATIVES) or any(c['status']!='fail' or not 0<c['mismatches']<=c['elements']
            or c['expected_sha256']==c['actual_sha256'] for c in negatives)): raise ValueError('Incomplete changed-arithmetic/history sensitivity evidence')
    validate_memory(report['memory'],reference=True)
    if (report['reference_default_dtype']!='bfloat16' or report['weight_payload_bytes_fetched_during_comparison']!=0
            or report['full_model_quality']!='not_run' or report['end_to_end_speed']!='not_measured'):
        raise ValueError('Engram report exceeds its qualified scope')
    return {r['meta']['id']:r for r in cases}
