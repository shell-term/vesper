from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vesper.benchmark_assets import MissingBenchmarkAsset
from vesper.gpqa_native import DynamicMoE


class NativeGPQATests(unittest.TestCase):
    def test_missing_expert_in_offline_mode_refuses_before_import_or_execution(self):
        route = {'indices': np.arange(6, dtype=np.int64), 'weights': np.ones(6, np.float32)}
        bank = Mock(); bank.has.return_value = False
        reader = Mock(); reader.maximum_fetch_bytes = 0
        with patch('vesper.backbone_runtime.moe_files', return_value=(Path('gate'), Path('router'), Path('shared'))), \
             patch('vesper.gpqa_native.load_gate_fixture', return_value=(None, None)), \
             patch('vesper.gpqa_native.load_router_fixture', return_value=(None, None)), \
             patch('vesper.gpqa_native.GateProjection'), patch('vesper.gpqa_native.RouterProbe') as router:
            router.return_value.__enter__.return_value.inspect.return_value = route
            p = DynamicMoE(Path('.'), {}, 0, bank, reader, None, None, None)
            with self.assertRaises(MissingBenchmarkAsset): p.forward(np.zeros(5120, np.uint16))
            bank.prepare.assert_not_called(); bank.index.assert_not_called()

    def test_actual_route_is_resolved_and_executed_without_a_saved_trajectory(self):
        route = {'indices': np.arange(6, dtype=np.int64), 'weights': np.ones(6, np.float32)}
        bank = Mock(); bank.has.return_value = True
        reader = Mock(); reader.maximum_fetch_bytes = 0
        with tempfile.TemporaryDirectory() as tmp, \
             patch('vesper.backbone_runtime.moe_files', return_value=(Path('gate'), Path('router'), Path('shared'))), \
             patch('vesper.gpqa_native.load_gate_fixture', return_value=(None, None)), \
             patch('vesper.gpqa_native.load_router_fixture', return_value=(None, None)), \
             patch('vesper.gpqa_native.GateProjection'), patch('vesper.gpqa_native.RouterProbe') as router, \
             patch('vesper.gpqa_native.ConnectedMoE') as connected, \
             patch('vesper.gpqa_native.validate_moe_manifests'):
            router.return_value.__enter__.return_value.inspect.return_value = route
            connected.return_value.__enter__.return_value.forward.return_value = {'routing': route}
            p = DynamicMoE(Path('.'), {}, 3, bank, reader, None, Path(tmp), None)
            p.forward(np.zeros(5120, np.uint16))
            self.assertEqual(bank.prepare.call_args.args[0:2], (3, [0, 1, 2, 3, 4, 5]))
            self.assertEqual(p.calls, 1)
            connected.return_value.__enter__.return_value.forward.return_value = {
                'routing': {'indices': route['indices'], 'weights': np.zeros(6, np.float32)}}
            with self.assertRaisesRegex(RuntimeError, 'routing'): p.forward(np.zeros(5120, np.uint16))
