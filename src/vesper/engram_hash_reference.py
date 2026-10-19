"""Unchanged original tokenizer/hash methods on CPU and CUDA."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from .engram_corpus import row_plan,TEXTS
from .model_artifacts import read_checked,digest
from .tokenizer import PinnedTokenizer


def compare_hashes(source,ordinal=0):
    import torch
    import transformers
    import tokenizers
    source=Path(source); read_checked(source,'inference/engram.py')
    name='vesper_original_ngram_methods'
    spec=importlib.util.spec_from_file_location(name,source/'engram.py'); original=importlib.util.module_from_spec(spec)
    sys.modules[name]=original; spec.loader.exec_module(original)
    tokenizer=transformers.AutoTokenizer.from_pretrained(str(source),local_files_only=True,trust_remote_code=False)
    native=PinnedTokenizer(source); plan=row_plan(source)
    mapping,size=original.build_compressed_token_map(tokenizer)
    if size!=99092 or digest(np.array(mapping,np.int64).tobytes())!=plan['tokenizer']['compressed_map_sha256']:
        raise RuntimeError('All-token original compressed vocabulary differs')
    args=SimpleNamespace(**json.loads(read_checked(source,'inference/config.json')),max_batch_size=2,max_seq_len=128)
    with torch.device('cpu'):
        layout=original.EngramLayout.from_args(args); cpu=original.NgramHashState(args,layout,tokenizer)
        gpu=original.NgramHashState(args,layout,tokenizer).to(f'cuda:{ordinal}')
    encodings=[]
    for text in TEXTS:
        ids=tokenizer.encode(text,add_special_tokens=False)
        decoded=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if ids!=native.encode(text) or decoded!=native.decode(ids): raise RuntimeError('Original token encode/decode differs')
        encodings.append({'text':text,'ids':ids,'decode_sha256':digest(decoded.encode())})
    comparisons=[]
    for case in plan['hash_cases']:
        x=torch.tensor(case['ids'],dtype=torch.int64,device='cpu')
        mask=None if case['mask'] is None else torch.tensor(case['mask'],dtype=torch.bool,device='cpu')
        pos=case['position']; end=pos+x.shape[1]
        expected=cpu(x,pos,mask).numpy(); device=gpu(x.to(f'cuda:{ordinal}'),pos,None if mask is None else mask.to(f'cuda:{ordinal}')).cpu().numpy()
        if not np.array_equal(expected,np.array(case['hash_ids'],np.int64)) or not np.array_equal(device,expected):
            raise RuntimeError('Original CPU/CUDA N-gram addresses differ')
        history=cpu.cache[:x.shape[0],max(0,end-3):end].numpy()
        if history.tolist()!=case['committed_history'] or not np.array_equal(gpu.cache[:x.shape[0],max(0,end-3):end].cpu().numpy(),history):
            raise RuntimeError('Original CPU/CUDA N-gram history differs')
        comparisons.append({'case':case['id'],'status':'pass','hash_sha256':digest(expected.tobytes()),'active_tail_sha256':digest(history.tobytes()),
            'unchanged_original_CPU_and_GPU_hashes_equal':True})
    return {'status':'pass','plan':plan,'all_original_token_map_entries_compared':len(mapping),'tokenizer_comparisons':encodings,
        'hash_comparisons':comparisons,'primes':cpu.primes.tolist(),'offsets':cpu.offsets.tolist(),'multipliers':cpu.multipliers.tolist(),
        'engram_source_sha256':digest(read_checked(source,'inference/engram.py')),
        'reference_versions':{'torch':torch.__version__,'torch_git':torch.version.git_version,'torch_cuda':torch.version.cuda,
            'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'numpy':np.__version__}}
