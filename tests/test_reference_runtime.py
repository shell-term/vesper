from types import SimpleNamespace
import unittest

from vesper.reference_runtime import isolate_nvrtc_handles


class RuntimeTests(unittest.TestCase):
    def test_compiling_another_shape_does_not_replace_a_live_launch_handle(self):
        class NVRTCKernelAdapter:
            kernels = {}

        compiled = {}

        def factory(shape):
            if shape not in compiled:
                adapter = NVRTCKernelAdapter()
                adapter.kernels["same_generated_name"] = shape
                compiled[shape] = SimpleNamespace(adapter=adapter)
            return compiled[shape]

        guarded = isolate_nvrtc_handles(factory)
        first = guarded((2304, 5120))
        second = guarded((5120, 2304))
        self.assertEqual(first.adapter.kernels["same_generated_name"], (2304, 5120))
        self.assertEqual(second.adapter.kernels["same_generated_name"], (5120, 2304))
        self.assertIs(guarded((2304, 5120)), first)
        self.assertEqual(first.adapter.kernels["same_generated_name"], (2304, 5120))

    def test_an_existing_instance_table_is_preserved(self):
        class NVRTCKernelAdapter:
            def __init__(self):
                self.kernels = {"kernel": object()}

        kernel = SimpleNamespace(adapter=NVRTCKernelAdapter())
        handles = kernel.adapter.kernels
        self.assertIs(isolate_nvrtc_handles(lambda: kernel)().adapter.kernels, handles)
