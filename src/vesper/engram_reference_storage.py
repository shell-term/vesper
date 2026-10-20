"""Optional full-shape original tables with bounded CUDA virtual-memory backing.

Only verified requested rows receive physical pages. Original F.embedding sees
the actual original shape, strides and row addresses; it is not replaced.
"""

from contextlib import ExitStack
import ctypes as C

import numpy as np

from .cuda import Driver,Device,_bind
from .resources import GiB,MiB


class Location(C.Structure): _fields_=[('type',C.c_int),('id',C.c_int)]
class AllocationFlags(C.Structure):
    _fields_=[('compressionType',C.c_ubyte),('gpuDirectRDMACapable',C.c_ubyte),('usage',C.c_ushort),('reserved',C.c_ubyte*4)]
class AllocationProp(C.Structure):
    _fields_=[('type',C.c_int),('requestedHandleTypes',C.c_int),('location',Location),('win32HandleMetaData',C.c_void_p),('allocFlags',AllocationFlags)]
class AccessDesc(C.Structure): _fields_=[('location',Location),('flags',C.c_int)]


class VirtualTables:
    def __init__(self,ordinal=0,*,max_rows=1024):
        from .sequence import MAX_ROWS
        if type(max_rows) is not int or max_rows not in (1024,MAX_ROWS): raise ValueError("Unknown explicit original-row transport profile")
        self.max_rows=max_rows
        self.ordinal=ordinal; self.stack=ExitStack(); self.physical=0; self.peak=0; self.cap=4*GiB; self.virtual=0; self.virtual_peak=0; self.tables=[]

    def __enter__(self):
        try: return self._open()
        except BaseException: self.stack.close(); raise

    def _open(self):
        self.driver=Driver(); self.device=self.stack.enter_context(Device(self.driver,self.ordinal,cap=MiB,free_floor=2*GiB))
        lib=self.driver.lib; P=C.POINTER; ptr=C.c_uint64; size=C.c_size_t
        self.granularity=_bind(lib,'cuMemGetAllocationGranularity',[P(size),P(AllocationProp),C.c_int])
        self.reserve=_bind(lib,'cuMemAddressReserve',[P(ptr),size,size,ptr,C.c_ulonglong])
        self.create=_bind(lib,'cuMemCreate',[P(ptr),size,P(AllocationProp),C.c_ulonglong])
        self.map=_bind(lib,'cuMemMap',[ptr,size,size,ptr,C.c_ulonglong]); self.access=_bind(lib,'cuMemSetAccess',[ptr,size,P(AccessDesc),size])
        self.unmap=_bind(lib,'cuMemUnmap',[ptr,size]); self.release=_bind(lib,'cuMemRelease',[ptr]); self.free=_bind(lib,'cuMemAddressFree',[ptr,size])
        self.zero=_bind(lib,'cuMemsetD8_v2',[ptr,C.c_ubyte,size]); self.prop=AllocationProp(type=1,location=Location(1,self.ordinal))
        g=size(); self.driver.check(self.granularity(C.byref(g),C.byref(self.prop),0)); self.page=g.value
        if not 0<self.page<=2*MiB: raise ValueError("Unregistered bounded CUDA VMM granularity")
        return self

    def _drop_page(self,address,handle):
        self.device.current(); self.driver.check(self.unmap(address,self.page)); self.driver.check(self.release(handle)); self.physical-=self.page

    def _drop_address(self,base,size):
        self.device.current(); self.driver.check(self.free(base,size)); self.virtual-=size

    def table(self,shape,rows):
        import torch
        if (shape not in ((384006168,256),(384006168,8),(384016682,256),(384016682,8),(129280,10240)) or not 1<=len(rows)<=self.max_rows
                or any(type(i) is not int or not 0<=i<shape[0] or type(raw) is not bytes or len(raw)!=shape[1] for i,raw in rows.items())):
            raise ValueError("Expected bounded original virtual table rows")
        self.device.current(); total=int(np.prod(shape)); size=((total+self.page-1)//self.page)*self.page
        address=C.c_uint64(); self.driver.check(self.reserve(C.byref(address),size,0,0,0)); base=address.value
        self.virtual+=size; self.virtual_peak=max(self.virtual_peak,self.virtual)
        self.stack.callback(self._drop_address,base,size)
        pages=sorted({page for row in rows for page in range((int(row)*shape[1])//self.page,
            ((int(row)+1)*shape[1]-1)//self.page+1)})
        for page in pages:
            if self.physical+self.page>self.cap or self.device.memory()['free']-self.page<2*GiB:
                raise MemoryError("Original-shape Engram oracle exceeds its physical VMM cap/reserve")
            handle=C.c_uint64(); self.driver.check(self.create(C.byref(handle),self.page,C.byref(self.prop),0))
            try:
                self.driver.check(self.map(base+page*self.page,self.page,0,handle,0))
            except BaseException:
                self.driver.check(self.release(handle)); raise
            self.physical+=self.page; self.peak=max(self.peak,self.physical)
            self.stack.callback(self._drop_page,base+page*self.page,handle.value)
            desc=AccessDesc(Location(1,self.ordinal),3)
            self.driver.check(self.access(base+page*self.page,self.page,C.byref(desc),1))
            self.driver.check(self.zero(base+page*self.page,0,self.page))
        for row,raw in rows.items():
            if len(raw)!=shape[1]: raise ValueError("Original virtual row width differs")
            data=np.frombuffer(raw,np.uint8)
            self.driver.check(self.driver.cuMemcpyHtoD_v2(base+int(row)*shape[1],data.ctypes.data,data.nbytes))
        class Interface: pass
        interface=Interface(); interface.__cuda_array_interface__={'shape':shape,'strides':None,'typestr':'|u1','data':(base,False),'version':3}
        tensor=torch.as_tensor(interface,device=f'cuda:{self.ordinal}')
        if tensor.data_ptr()!=base or tuple(tensor.shape)!=tuple(shape) or tensor.stride()!=(shape[1],1):
            raise RuntimeError("Original-shape CUDA table interoperability copied or changed its layout")
        self.tables.append({'shape':list(shape),'original_addressing':True,'virtual_bytes':size,'mapped_physical_pages':len(pages)})
        return tensor

    def stats(self):
        return {'physical_cap_bytes':self.cap,'physical_owned_bytes':self.physical,'peak_physical_owned_bytes':self.peak,
            'virtual_reserved_bytes':self.virtual,'peak_virtual_reserved_bytes':self.virtual_peak,'page_bytes':self.page,'tables':self.tables,
            'F_embedding_replaced':False,'original_row_addresses_remapped':False,'unrequested_pages':'unmapped',
            'physical_allocations_released':self.physical==0,'virtual_reservations_released':self.virtual==0,
            'owned_context_active':bool(self.device.context)}

    def __exit__(self,*_):
        import torch
        torch.cuda.synchronize(self.ordinal); self.stack.close()
