from contextlib import ExitStack
import ctypes as C
from types import SimpleNamespace
import unittest

import numpy as np

from vesper.paged_tables import PagedVirtualTables
from vesper.resources import GiB,MiB


class Driver:
    def __init__(self):self.events=[]
    def check(self,result):
        if result:raise RuntimeError('owned CUDA test fault')
    def cuCtxSynchronize(self):return 0
    def cuMemcpyHtoD_v2(self,address,pointer,size):
        self.events.append(('copy',address,C.string_at(pointer,size)));return 0


class PagedTableTests(unittest.TestCase):
    def table(self):
        # Fake CUDA calls test page accounting, eviction/guard order and exact
        # original offsets. The separate GPU gate uses original F.embedding.
        v=PagedVirtualTables.__new__(PagedVirtualTables);v.driver=Driver();v.page=2*MiB;v.cap=512*MiB
        v.physical=v.peak=0;v.ordinal=0;v.prop=C.c_int(0);v.stack=ExitStack();v.activations=v.maps_total=v.evictions_total=0
        v.device=SimpleNamespace(current=lambda:None,memory=lambda:{'free':8*GiB},context=1)
        sequence=[0]
        def create(pointer,*args):sequence[0]+=1;C.cast(pointer,C.POINTER(C.c_uint64)).contents.value=sequence[0];return 0
        v.create=create;v.map=lambda *a:0;v.unmap=lambda *a:0;v.release=lambda *a:0;v.access=lambda *a:0;v.zero=lambda *a:0
        base=16*MiB;state={'base':base,'shape':(384006168,256),
            'rows':{i*8192:bytes([i%256])*256 for i in range(1496)},'pages':{},'active_rows':set()}
        v.paged={base:state};v.stack.callback(v._destroy_table,state)
        return v,state,SimpleNamespace(data_ptr=lambda:base)

    def test_many_original_rows_keep_only_current_pages_and_constant_callbacks(self):
        v,state,tensor=self.table();callbacks=len(v.stack._exit_callbacks)
        for i in range(1000):
            selected=np.array([(i+j)*8192 for j in range(24)],np.int64)
            v.activate(tensor,selected)
            self.assertEqual((len(state['pages']),v.physical),(24,48*MiB))
            self.assertEqual(len(v.stack._exit_callbacks),callbacks)
        self.assertEqual(v.peak,48*MiB)
        v.stack.close();self.assertEqual(v.physical,0);self.assertFalse(state['rows'])
        self.assertEqual(v.maps_total,v.evictions_total)

    def test_unprepared_row_in_an_already_mapped_page_is_rejected(self):
        v,state,tensor=self.table();v.activate(tensor,np.array([0],np.int64));events=len(v.driver.events)
        # Row 1 shares row 0's physical page; it is not an available source row.
        with self.assertRaises(ValueError):v.activate(tensor,np.array([1],np.int64))
        self.assertEqual(len(v.driver.events),events);self.assertEqual(state['active_rows'],{0})
        v.stack.close();self.assertEqual(v.physical,0)

    def test_original_row_offsets_and_bytes_are_preserved(self):
        v,state,tensor=self.table();selected=np.array([8192,3*8192],np.int64);v.activate(tensor,selected)
        self.assertEqual(v.driver.events,[('copy',state['base']+int(r)*256,state['rows'][int(r)]) for r in selected])
        v.deactivate(tensor);self.assertEqual(v.physical,0);self.assertEqual(state['active_rows'],set())
        v.stack.close()

    def test_cap_refusal_happens_before_old_residency_changes(self):
        v,state,tensor=self.table();v.activate(tensor,np.array([0,8192],np.int64));old=dict(state['pages']);events=len(v.driver.events)
        v.cap=MiB
        with self.assertRaises(MemoryError):v.activate(tensor,np.array([2*8192,3*8192],np.int64))
        self.assertEqual(state['pages'],old);self.assertEqual(len(v.driver.events),events)
        v.stack.close();self.assertEqual(v.physical,0)
