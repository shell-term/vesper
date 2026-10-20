from copy import deepcopy
import json
from pathlib import Path
import unittest

from vesper.compile_cache_validation import validate_reuse,image_identity,QUALIFIED_CUBIN_IMAGE_SET
from vesper.model_artifacts import digest

ROOT=Path(__file__).resolve().parents[1]


class CompilationReuseEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=ROOT/'results/2026-10-20-compilation-cache'
        cls.native_raw=(cls.directory/'native.json').read_bytes();cls.native=json.loads(cls.native_raw)
        cls.compiler=json.loads((cls.directory/'compiler.json').read_bytes())
        cls.original_raw=(ROOT/'results/2026-10-19-generation/original.json').read_bytes();cls.original=json.loads(cls.original_raw)
        cls.indices={str(l):(ROOT/f'results/2026-10-20-indexed-generation/indices/layer{l}.json').read_bytes() for l in range(40)}

    def test_complete_model_comparison_and_all_fourteen_exact_images(self):
        self.assertTrue(validate_reuse(self.native,self.native_raw,self.compiler,self.original,self.original_raw,self.indices))
        self.assertEqual(image_identity(self.native['compiled_images']),QUALIFIED_CUBIN_IMAGE_SET)
        self.assertEqual(self.native['compiler_cache']['hits'],1476)
        self.assertEqual(sum(len(c['boundaries']) for c in self.native['cases']),26250)

    def test_raw_published_evidence_files_remain_size_and_hash_bound(self):
        summary=json.loads((self.directory/'summary.json').read_bytes())
        for name,entry in summary['files'].items():
            raw=(self.directory/name).read_bytes();self.assertEqual((len(raw),digest(raw)),(entry['bytes'],entry['sha256']))

    def test_different_images_compiler_reference_or_live_allocations_are_refused(self):
        for mutate in (lambda r:r['compiled_images'][0].update(sha256='0'*64),
                lambda r:r.update(whole_model_replay_reference_sha256='0'*64),
                lambda r:r['cache_after_close'].update(host_reserved_bytes=1),
                lambda r:r['compiled_images'][0]['compiler'].update(nvrtc_version=[13,4]),
                lambda r:r['memory_after_policy_close'].update(charged_host_bytes=1)):
            r=deepcopy(self.compiler);mutate(r)
            with self.assertRaises(ValueError):validate_reuse(self.native,self.native_raw,r,self.original,self.original_raw,self.indices)

    def test_claiming_hits_without_complete_model_math_cannot_pass(self):
        r=deepcopy(self.native);r['cases'][-1]['boundaries'].pop()
        with self.assertRaises(ValueError):validate_reuse(r,self.native_raw,self.compiler,self.original,self.original_raw,self.indices)
        r=deepcopy(self.native);r['compiler_cache']['hits']=0
        with self.assertRaises(ValueError):validate_reuse(r,self.native_raw,self.compiler,self.original,self.original_raw,self.indices)
