from copy import deepcopy
import json
from pathlib import Path
import unittest

from vesper.model_artifacts import canonical,digest
from vesper.trajectory_evidence import identity,validate_original,validate_native

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/pc-heterogeneous/2026-10-20-coding-trajectory'


class PublishedCodingTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=(RUN/'original.json').read_bytes();cls.original=json.loads(cls.raw)
        cls.native=json.loads((RUN/'native.json').read_bytes())

    def test_complete_real_coding_capture_and_every_published_file_remain_bound(self):
        summary=json.loads((RUN/'summary.json').read_bytes())
        for name,entry in summary['files'].items():
            self.assertFalse(Path(name).is_absolute());self.assertNotIn('..',Path(name).parts)
            payload=(RUN/name).read_bytes()
            self.assertEqual((len(payload),digest(payload)),(entry['bytes'],entry['sha256']),name)
        self.assertTrue(validate_original(self.original,RUN))
        self.assertTrue(validate_native(self.native,RUN,self.original,self.raw,RUN))
        self.assertEqual(summary['successful_model_calls'],34)
        self.assertEqual(summary['typed_boundaries'],178423)
        self.assertEqual(summary['complete_FP32_logits'],4395520)
        self.assertEqual(summary['actual_original_routes'],1360)
        self.assertEqual(summary['completion']['selected_ids_including_eos'],[38753,26913,19,63,1])
        self.assertEqual(summary['completion']['completion_text'],'xs[-1]')
        self.assertEqual(summary['completion']['stop_reason'],'eos')
        self.assertEqual(summary['quality_score'],{'status':'pass','passed':True,'reason':'declared_python_expression_cases','actual':[9,7,0]})
        self.assertEqual(summary['other_declared_quality_tasks'],'not_run')
        self.assertEqual(summary['native_failed_and_restored_calls'],1)
        self.assertEqual(summary['storage_elements_compared'],962450386)
        self.assertEqual(summary['distinct_original_experts'],3067)
        for kind,report in (('original',self.original),('native',self.native)):
            memory=summary['sampled_memory'][kind]
            self.assertEqual(memory,{k:report['memory_after_policy_close'][k] for k in memory})
        for operation,name,head in (
                ('prepare','preparation.json','89a6cbef6b324e3bcf13dee62af346712295ef97'),
                ('reference','original.json','89a6cbef6b324e3bcf13dee62af346712295ef97'),
                ('native','native.json','7596d2b8e45db2d82d1eb65ff2e6323d1abe0534')):
            receipt=json.loads((RUN/(operation+'-context.json')).read_bytes())
            payload=(RUN/name).read_bytes();context=receipt['workspace_context']
            self.assertEqual(receipt['operation'],operation)
            self.assertEqual(receipt['raw_report'],{'file':name,'bytes':len(payload),'sha256':digest(payload)})
            self.assertEqual((context['platform'],context['profile'],context['git_head']),('pc-heterogeneous','native',head))
        self.assertFalse(summary['new_model_execution_run_in_summary_writer'])

    def test_rehashed_changed_original_completion_cannot_qualify_itself(self):
        altered=deepcopy(self.original)
        altered['completion']['completion_text']='xs[0]'
        altered['identity_sha256']=identity(altered)
        altered['complete_goldens_sha256']=canonical(altered['case_files'])
        with self.assertRaisesRegex(ValueError,'independently qualified'):
            validate_original(altered,RUN)

    def test_changed_failed_token_recovery_is_refused_despite_matching_successful_math(self):
        altered=deepcopy(self.native);altered['fault']['after_state_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'state restoration'):
            validate_native(altered,RUN,self.original,self.raw,RUN)


if __name__=='__main__':unittest.main()
