from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vesper.generation_assets import checked_cache
from vesper.resources import GiB,MiB
from vesper.source_roots import SourceRoots
from vesper.trajectory_space import preparation_file_bounds,preparation_admission,capture_admission,disk_checked_progress
from test_source_roots import mounted


class TrajectorySpaceTests(unittest.TestCase):
    def test_separate_volume_preparation_admission_covers_all_bounded_run_file_categories(self):
        bounds=preparation_file_bounds()
        self.assertEqual(bounds['runtime_and_flat_indexes'],320*MiB)
        self.assertEqual(bounds['plan_and_preparation_report'],128*MiB)
        self.assertEqual(bounds['original_row_cache'],2*12288*(264+4096))
        self.assertLess(sum(bounds.values()),GiB)

    def test_only_an_explicit_separate_bank_filesystem_uses_run_only_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'artifacts';root.mkdir()
            self.assertEqual(preparation_admission(root),8*GiB)
            mounted(root,base/'ssd');r=SourceRoots(root)
            with patch('vesper.trajectory_space._device',return_value=1):
                self.assertEqual(preparation_admission(root),8*GiB)
            with patch('vesper.trajectory_space._device',side_effect=lambda p:1 if p==r.root else 2):
                self.assertEqual(preparation_admission(root),GiB)

    def test_capture_admits_complete_sealed_call_count_and_refuses_unbounded_or_changed_census(self):
        for count in (1,190,388,512):
            plan={'completion':{'calls':count},'calls':[{}]*count}
            self.assertEqual(capture_admission(plan),(count*4+68)*MiB)
        for count,actual in ((0,0),(513,513),(190,189),(True,1)):
            with self.assertRaises(ValueError):capture_admission({'completion':{'calls':count},'calls':[{}]*actual})

    def test_reference_runtime_startup_is_admitted_separately_from_record_bytes(self):
        plan={'completion':{'calls':104},'calls':[{}]*104}
        self.assertEqual(capture_admission(plan,reference=True),8*GiB+484*MiB)
        self.assertEqual(capture_admission(plan,reference=False),484*MiB)
        with self.assertRaises(ValueError):capture_admission(plan,reference='reference')

    def test_disk_consumed_after_admission_stops_before_the_next_progress_publication(self):
        from collections import namedtuple
        Usage=namedtuple('Usage','total used free');written=[]
        progress=disk_checked_progress(Path('.'),written.append)
        with patch('vesper.generation_assets.shutil.disk_usage',side_effect=[
                Usage(128*GiB,0,64*GiB+MiB),Usage(128*GiB,0,64*GiB-1),Usage(128*GiB,0,64*GiB+MiB)]):
            progress({'original_layer':0,'position':0})
            with self.assertRaisesRegex(OSError,'64 GiB disk reserve'):
                progress({'original_layer':1,'position':0})
            self.assertEqual(written,[{'original_layer':0,'position':0}])
            progress({'original_layer':2,'position':0})
        self.assertEqual(written,[{'original_layer':0,'position':0},{'original_layer':2,'position':0}])

    def test_oversized_original_cache_sidecar_fails_before_either_file_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'row.bin'
            with self.assertRaisesRegex(ValueError,'metadata exceeds'):
                checked_cache(path,{'bytes':1,'large':'x'*4096},lambda:b'\x00')
            self.assertFalse(path.exists());self.assertFalse(path.with_suffix('.json').exists())
