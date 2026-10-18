import ast
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vesper.attention_storage import _query_allocation, load_variant, trace_shapes, variant_source
from vesper.checkpoint import REFERENCE_HASHES


SOURCE = b'''pass_configs = {}
@tilelang.jit(pass_configs=pass_configs)
def sparse_attn_kernel(h, d, scale=None):
    q_shared = T.alloc_shared((h, d), BF16)
    kv_shared = T.alloc_shared((64, d), BF16)
    scale = d ** -0.5
    return q_shared, kv_shared, scale
'''


class AttentionStorageTests(unittest.TestCase):
    def test_only_query_storage_changes_and_the_input_is_not_modified(self):
        with patch.dict(REFERENCE_HASHES, {"kernel.py": hashlib.sha256(SOURCE).hexdigest()}):
            rendered, metadata = variant_source(SOURCE)
        original = ast.parse(SOURCE).body[-1]
        candidate = ast.parse(rendered).body[-1]
        allocation = candidate.body[0].value
        self.assertEqual(allocation.func.attr, "alloc_fragment")
        allocation.func.attr = "alloc_shared"
        self.assertEqual(ast.dump(original), ast.dump(candidate))
        self.assertEqual(original.body[1].value.func.attr, "alloc_shared")
        self.assertEqual(metadata["query_storage_only_ast_check"], "pass")
        self.assertFalse(metadata["head_partition_changed"])
        self.assertIn(b"T.alloc_shared((h, d), BF16)", SOURCE)

    def test_shared_baseline_retains_the_complete_original_function_ast(self):
        with patch.dict(REFERENCE_HASHES, {"kernel.py": hashlib.sha256(SOURCE).hexdigest()}):
            rendered, _ = variant_source(SOURCE, "shared")
        self.assertEqual(ast.dump(ast.parse(SOURCE).body[-1]), ast.dump(ast.parse(rendered).body[-1]))

    def test_source_hash_is_required_before_parsing_or_execution(self):
        with patch("vesper.attention_storage.ast.parse") as parse:
            with self.assertRaisesRegex(ValueError, "pinned hash"):
                variant_source(SOURCE + b"\n# changed")
            parse.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            variant_source(SOURCE, "head-split")

    def test_changed_or_ambiguous_query_layout_is_rejected(self):
        for source in ("q_shared=T.alloc_shared((h,d//2),BF16)",
                       "q_shared=T.alloc_fragment((h,d),BF16)",
                       "q_shared=T.alloc_shared((h,d),BF16)\nq_shared=T.alloc_shared((h,d),BF16)"):
            with self.assertRaises(ValueError):
                _query_allocation(ast.parse(source))

    def test_existing_generated_cache_tampering_is_rejected_before_import(self):
        with patch.dict(REFERENCE_HASHES, {"kernel.py": hashlib.sha256(SOURCE).hexdigest()}), tempfile.TemporaryDirectory() as directory:
            _, metadata = variant_source(SOURCE)
            path = Path(directory) / ("attention_" + metadata["generated_python_sha256"] + ".py")
            path.write_bytes(b"changed executable content")
            with patch("vesper.attention_storage.importlib.util.spec_from_file_location") as importer:
                with self.assertRaisesRegex(ValueError, "modified"):
                    load_variant(SOURCE, Path(directory))
                importer.assert_not_called()

    def test_trace_allocations_are_bounded_before_allocating(self):
        shapes = trace_shapes(2, 3, 64, 512, 193)
        self.assertEqual(shapes[0], (2, 3, 4, 64, 64))
        self.assertEqual(shapes[-1], (2, 3, 64, 512))
        self.assertLess(sum(math.prod(shape) * 4 for shape in shapes), 8 * 1024**2)
        for args in ((2**40, 1, 64, 512, 128), (1, 4, 64, 512, 128), (1, 1, 64, 512, 194),
                     (1, 1, 64, 512, 0), (True, 1, 64, 512, 128), (1, 1, 64, 512.0, 128)):
            with self.assertRaisesRegex(ValueError, "bounded"):
                trace_shapes(*args)


if __name__ == "__main__":
    unittest.main()
