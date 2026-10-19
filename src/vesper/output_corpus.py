"""Deterministic output-component inputs, distinct from model-level hidden states."""

import numpy as np
from .formats import bf16_bits
from .output_assets import EMBED_IDS,WIDTH


def cases():
    return [{'id':f'embedding/{token}/batch{b}','batch':b,'embedding_token':token,'scale':None,'zero':False}
        for token in EMBED_IDS for b in (1,2)]+[
        {'id':f'generated/{i}/batch{b}','batch':b,'embedding_token':None,'scale':s,'zero':False}
        for i,s in enumerate((2**-16,1.,2**12)) for b in (1,2)]+[
        {'id':f'zero/batch{b}','batch':b,'embedding_token':None,'scale':0.,'zero':True} for b in (1,2)]


def inputs(case,embedding):
    b=case['batch']
    if case['embedding_token'] is not None:
        row=embedding[EMBED_IDS.index(case['embedding_token'])]; x=np.broadcast_to(row,(b,1,4,WIDTH)).copy()
        pre=np.full((b,1,4),.25,np.float32)
    else:
        k=np.arange(b*4*WIDTH,dtype=np.int64).reshape(b,1,4,WIDTH)
        values=((k*37%257)-128).astype(np.float32)*np.float32(case['scale']/32)
        x=bf16_bits(values); pre=np.broadcast_to(np.array([.125,.25,.5,.75],np.float32),(b,1,4)).copy()
    return np.ascontiguousarray(x),pre
