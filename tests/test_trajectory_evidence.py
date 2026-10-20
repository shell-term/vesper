from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from vesper.trajectory_evidence import read,write_case,load_case,check_trajectory
from vesper.trajectory_evidence import scope,INPUT_PINS,ORIGINAL_FORMAT_RAW_SHA256
from vesper.model_artifacts import canonical,digest
from vesper.sequence_assets import request

ROOT=Path(__file__).resolve().parents[1]


class TrajectoryEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The already qualified short corpus tests the streaming container.
        # It is never presented as a new longer model execution.
        cls.legacy=json.loads((ROOT/'results/2026-10-19-generation/original.json').read_bytes())

    def fixture(self,run):
        legacy=self.legacy
        return {'scope':{'request':legacy['scope']['request']},'completion':deepcopy(legacy['completion']),
            'expert_plan':deepcopy(legacy['expert_plan']),
            'case_files':[write_case(run,'original',c) for c in legacy['cases']]}

    def replace(self,run,report,pos,mutate):
        descriptor=report['case_files'][pos];case,_=load_case(run,descriptor,'original');mutate(case)
        (run/descriptor['file']).unlink()
        report['case_files'][pos]=write_case(run,'original',case)

    def test_complete_typed_math_history_and_feedback_survive_streaming(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);report=self.fixture(run);self.assertTrue(check_trajectory(report,run,'original'))
            self.assertEqual(sum(d['typed_boundaries'] for d in report['case_files']),26250)
            for d,c in zip(report['case_files'],self.legacy['cases']):self.assertEqual(load_case(run,d,'original')[0],c)

    def test_all_complete_task_inputs_are_bound_to_the_independent_original_CPU_gate(self):
        raw=(ROOT/'results/2026-10-20-message-format/original.json').read_bytes()
        self.assertEqual(digest(raw),ORIGINAL_FORMAT_RAW_SHA256)
        from vesper.message_validation import validate_reference
        original=json.loads(raw);self.assertTrue(validate_reference(original))
        for c in original['quality_preflight']:
            case,req=request(c['id']);count,prompt_sha,ids_sha,case_sha=INPUT_PINS[c['id']]
            self.assertEqual((count,prompt_sha,ids_sha,case_sha),(len(c['prompt_token_ids']),digest(c['prompt'].encode()),canonical(c['prompt_token_ids']),canonical(case)))
            self.assertEqual(req.prompt,c['prompt']);self.assertEqual(scope(c['id'])['original_prompt_tokens'],count)

    def test_missing_call_or_early_termination_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);report=self.fixture(run);report['case_files'].pop()
            with self.assertRaises(ValueError):check_trajectory(report,run,'original')
            report['completion']['calls']-=1;report['completion']['selected_ids_including_eos'].pop();report['completion']['completion_ids'].pop()
            with self.assertRaisesRegex(ValueError,'Premature'):check_trajectory(report,run,'original')

    def test_rehashed_wrong_feedback_and_rng_progression_are_refused(self):
        for field in ('feedback','rng'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                run=Path(tmp);report=self.fixture(run)
                def mutate(c):
                    if field=='feedback':c['meta']['token_id']=1
                    else:c['rng_offset']+=4
                self.replace(run,report,3,mutate)
                with self.assertRaises(ValueError):check_trajectory(report,run,'original')

    def test_file_bytes_path_and_typed_boundary_census_remain_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);report=self.fixture(run);d=deepcopy(report['case_files'][0]);d['file']='../escape.json'
            with self.assertRaises(ValueError):load_case(run,d,'original')
            with self.assertRaises(ValueError):write_case(run,'../escape',self.legacy['cases'][0])
            invalid=deepcopy(self.legacy['cases'][0]);invalid['meta']['position']=512
            with self.assertRaises(ValueError):write_case(run,'native',invalid)
            d=report['case_files'][0];path=run/d['file'];path.write_bytes(path.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'bytes'):load_case(run,d,'original')
            path.write_bytes(path.read_bytes()[:-1]);self.replace(run,report,0,lambda c:c['boundaries'].pop())
            with self.assertRaisesRegex(ValueError,'Missing'):check_trajectory(report,run,'original')

    def test_oversized_call_is_refused_before_read_and_completed_call_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);write_case(run,'original',self.legacy['cases'][0])
            with self.assertRaises(FileExistsError):write_case(run,'original',self.legacy['cases'][0])
            path=run/'oversized.json'
            with path.open('wb') as stream:stream.truncate(4*1024**2+1)
            with self.assertRaisesRegex(ValueError,'Excessive'):read(path,4*1024**2)
