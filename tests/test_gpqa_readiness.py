import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vesper.checkpoint import MODEL, REVISION
from vesper.gpqa_subset import ARCHIVE_SHA256, CSV_SHA256, FIELDS, POPULATION, PROTOCOL, SOURCE_REVISION, make_cases
from vesper.gpqa_readiness import input_requirements, load_subset
from vesper.model_artifacts import canonical


class GPQAReadinessTests(unittest.TestCase):
    def test_unchanged_source_selection_and_prior_plan_are_required(self):
        rows = [dict(zip(FIELDS, (f'id{i}', 'Synthetic?', 'yes', 'no', 'other', 'none'))) for i in range(5)]
        plan = {'benchmark': 'gpqa-diamond', 'population': POPULATION, 'protocol': PROTOCOL,
                'source_revision': SOURCE_REVISION, 'archive_sha256': ARCHIVE_SHA256, 'csv_sha256': CSV_SHA256,
                'temperature': 1., 'top_p': 1., 'selection_seed': 1, 'case_count': 3,
                'cases': make_cases(rows, 3, seed=1, max_new_tokens=262144), 'full_evaluation_run': False}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            workspace = SimpleNamespace(platform='pc-heterogeneous', profile='native', artifact_path=lambda name: path)
            def save():
                body = {k: v for k, v in plan.items() if k not in ('workspace_context', 'plan_sha256')}
                plan['plan_sha256'] = canonical(body)
                plan['workspace_context'] = {'platform': workspace.platform, 'profile': workspace.profile,
                                             'model': MODEL, 'revision': REVISION}
                path.write_text(json.dumps(plan))
            save()
            with patch('vesper.gpqa_readiness.load_archive', return_value=rows):
                self.assertEqual(load_subset(workspace, 'original'), plan)
                plan['cases'][0]['messages'][0]['content'] += ' changed'; save()
                with self.assertRaisesRegex(ValueError, 'source selection'): load_subset(workspace, 'original')
                plan['temperature'] = .7; save()
                with self.assertRaisesRegex(ValueError, 'source identity'): load_subset(workspace, 'original')
            with self.assertRaises(ValueError): load_subset(workspace, '../escape')

    def test_input_requirements_count_distinct_rows_without_model_execution(self):
        tokenizer = SimpleNamespace(encode=lambda text: [100, 100, 101])
        hashes = np.zeros((1, 3, 2, 24), np.int64)
        hashes[:, 1:, 0, :] = 1
        hashes[:, :, 1, :] = 2
        with patch('vesper.gpqa_readiness.prompt_hashes', return_value=(hashes, {})):
            row = input_requirements([{'id': 'synthetic', 'encoded_prompt': 'whole prompt'}], tokenizer, None)[0]
        self.assertEqual(row['distinct_prompt_embedding_rows'], 2)
        self.assertEqual(row['distinct_prompt_Engram_rows'], {'1': 2, '14': 1})
        self.assertEqual(row['prompt_Engram_payload_bytes'], 3 * 264)
        self.assertEqual(row['expert_route_coverage'], 'unknown_without_model_execution')
