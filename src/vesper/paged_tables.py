"""Bounded physical residency at unchanged original CUDA virtual row addresses.

This optional oracle transport never replaces F.embedding or remaps logical
row IDs. Only the exact rows for the next lookup are materialized. Other
logical IDs are rejected even if their physical page happens to be mapped.
"""

from copy import deepcopy
import ctypes as C

import numpy as np

from .engram_reference_storage import VirtualTables, AccessDesc, Location
from .resources import MiB, GiB


class PagedVirtualTables(VirtualTables):
    max_active_rows = 24

    def __init__(self,ordinal=0,*,max_rows=12288):
        super().__init__(ordinal,max_rows=max_rows)
        self.cap=512*MiB
        self.paged={};self.activations=self.maps_total=self.evictions_total=0

    def paged_table(self,shape,rows):
        import torch
        if (shape not in ((384006168,256),(384006168,8),(384016682,256),(384016682,8),(129280,10240))
                or not 1<=len(rows)<=self.max_rows or any(type(i) is not int or not 0<=i<shape[0] or type(raw) is not bytes or len(raw)!=shape[1] for i,raw in rows.items())):
            raise ValueError("Expected bounded original-address paged rows")
        self.device.current();size=((int(np.prod(shape))+self.page-1)//self.page)*self.page
        address=C.c_uint64();self.driver.check(self.reserve(C.byref(address),size,0,0,0));base=address.value
        self.virtual+=size;self.virtual_peak=max(self.virtual_peak,self.virtual);self.stack.callback(self._drop_address,base,size)
        state={'base':base,'shape':shape,'rows':dict(rows),'pages':{},'active_rows':set()}
        self.paged[base]=state;self.stack.callback(self._destroy_table,state)
        class Interface:pass
        interface=Interface();interface.__cuda_array_interface__={'shape':shape,'strides':None,'typestr':'|u1','data':(base,False),'version':3}
        tensor=torch.as_tensor(interface,device=f'cuda:{self.ordinal}')
        if tensor.data_ptr()!=base or tuple(tensor.shape)!=tuple(shape) or tensor.stride()!=(shape[1],1):raise RuntimeError("Paged original table interoperability changed its address/shape")
        self.tables.append({'shape':list(shape),'original_addressing':True,'virtual_bytes':size,'mapped_physical_pages':0})
        return tensor

    def _close_table(self,state):
        for page in list(state['pages']):
            handle=state['pages'][page];self._drop_page(state['base']+page*self.page,handle)
            del state['pages'][page];self.evictions_total+=1
        state['active_rows'].clear()

    def _destroy_table(self,state):
        self._close_table(state);state['rows'].clear()

    def activate(self,tensor,ids):
        state=self.paged.get(tensor.data_ptr())
        if state is None:raise ValueError("Unknown original virtual table")
        values=np.asarray(ids)
        if values.dtype!=np.int64:raise ValueError("Expected original integer row IDs")
        requested=set(map(int,values.reshape(-1)))
        if not 1<=len(requested)<=self.max_active_rows or not requested.issubset(state['rows']):raise ValueError("Unprepared/excessive logical row lookup")
        self.device.current();self.driver.check(self.driver.cuCtxSynchronize())
        width=state['shape'][1]
        pages={p for row in requested for p in range(row*width//self.page,((row+1)*width-1)//self.page+1)}
        old=set(state['pages']);added=pages-old;removed=old-pages
        next_physical=self.physical+(len(added)-len(removed))*self.page
        if next_physical>self.cap or self.device.memory()['free']-len(added)*self.page<2*GiB:
            raise MemoryError("Paged original rows exceed the physical cap/GPU reserve")
        state['active_rows'].clear()
        for page in sorted(removed):
            self._drop_page(state['base']+page*self.page,state['pages'][page]);del state['pages'][page];self.evictions_total+=1
        # A single per-table close callback owns current mappings. The callback
        # count never grows with the number of logical prefix/model calls.
        for page in sorted(added):
            address=state['base']+page*self.page;handle=C.c_uint64()
            self.driver.check(self.create(C.byref(handle),self.page,C.byref(self.prop),0))
            try:self.driver.check(self.map(address,self.page,0,handle,0))
            except BaseException:self.driver.check(self.release(handle));raise
            state['pages'][page]=handle.value;self.physical+=self.page;self.peak=max(self.peak,self.physical);self.maps_total+=1
            desc=AccessDesc(Location(1,self.ordinal),3)
            self.driver.check(self.access(address,self.page,C.byref(desc),1));self.driver.check(self.zero(address,0,self.page))
        for row in sorted(requested):
            data=np.frombuffer(state['rows'][row],np.uint8)
            self.driver.check(self.driver.cuMemcpyHtoD_v2(state['base']+row*width,data.ctypes.data,data.nbytes))
        state['active_rows']=requested;self.activations+=1

    def deactivate(self,tensor):
        state=self.paged.get(tensor.data_ptr())
        if state is None:raise ValueError("Unknown original virtual table")
        self.device.current();self.driver.check(self.driver.cuCtxSynchronize());self._close_table(state)

    def stats(self):
        result=deepcopy(super().stats())
        for table in result['tables']:
            table['initial_mapped_physical_pages']=table['mapped_physical_pages']
            state=next((s for s in self.paged.values() if list(s['shape'])==table['shape']),None)
            table['mapped_physical_pages']=(len(state['pages']) if state is not None else table['mapped_physical_pages']) if self.device.context else 0
        result.update(row_transport='paged_original_virtual_addresses',maximum_active_rows_per_table=self.max_active_rows,
            activation_calls=self.activations,total_pages_mapped=self.maps_total,total_pages_evicted=self.evictions_total,
            current_paged_table_pages=sum(len(s['pages']) for s in self.paged.values()),
            current_logical_active_rows=sum(len(s['active_rows']) for s in self.paged.values()),
            table_close_callbacks=len(self.paged))
        return result
