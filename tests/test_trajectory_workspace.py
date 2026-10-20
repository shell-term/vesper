import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from vesper.checkpoint import MODEL,REVISION
from vesper.model_artifacts import canonical,digest
from vesper.trajectory_evidence import plan_identity
from vesper.trajectory_workspace import select,run_path,own_evidence,migrate_input
from vesper.workspace import Workspace,WorkspaceError
import test_workspace

ROOT=Path(__file__).resolve().parents[1]


class TrajectoryWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.artifacts=self.root/'artifacts';self.artifacts.mkdir()
        git=lambda *args:test_workspace.git(self.root,*args)
        git('init','--initial-branch=main');git('config','user.email','test@example.invalid');git('config','user.name','Test')
        git('remote','add','origin','https://github.com/shell-term/vesper.git')
        (self.root/'.gitignore').write_text('.vesper-workspace.json\nartifacts/\n')
        (self.root/'platforms').mkdir();shutil.copyfile(ROOT/'platforms/registry.json',self.root/'platforms/registry.json')
        module=self.root/'src/vesper/trajectory_workspace.py';module.parent.mkdir(parents=True);module.write_text('# synthetic runtime fixture\n')
        git('add','.');git('commit','-m','fixture');Workspace.bind(self.root,platform='pc-heterogeneous')
        p=patch('vesper.trajectory_workspace.__file__',str(module));p.start();self.addCleanup(p.stop)

    def test_explicit_PC_binding_routes_only_own_private_and_public_evidence(self):
        self.assertEqual(select(self.artifacts).platform,'pc-heterogeneous')
        _,path=run_path(self.artifacts,'coding-fixture')
        self.assertEqual(path,self.artifacts/'pc-heterogeneous/trajectories/coding-fixture')
        self.assertEqual(own_evidence(self.artifacts,self.root/'results/pc-heterogeneous/run'),self.root/'results/pc-heterogeneous/run')
        for p in (self.root/'results/dgx-spark/run',self.artifacts/'trajectories/legacy',self.root/'outside'):
            with self.assertRaises(WorkspaceError):own_evidence(self.artifacts,p)

    def test_wrong_branch_runtime_or_artifact_root_cannot_be_silently_rebound(self):
        with self.assertRaises(WorkspaceError):select(self.artifacts/'pc-heterogeneous')
        with patch('vesper.trajectory_workspace.__file__',str(ROOT/'src/vesper/trajectory_workspace.py')):
            with self.assertRaises(WorkspaceError):select(self.artifacts)
        test_workspace.git(self.root,'switch','-c','feature/dgx-spark-fixture')
        with self.assertRaises(WorkspaceError):select(self.artifacts)

    def test_explicit_legacy_input_import_preserves_semantic_identity_without_copying_expert_payloads(self):
        legacy=self.artifacts/'trajectories/coding-fixture';legacy.mkdir(parents=True)
        payloads={'embeddings/0.bin':b'original-embedding',
            'rows/layer1/manifest.json':b'{}','rows/layer1/rows.bin':b'row1',
            'rows/layer14/manifest.json':b'{}','rows/layer14/rows.bin':b'row14',
            'indices/layer0/manifest.json':b'{}','indices/layer0.json':b'{}'}
        for name,raw in payloads.items():
            path=legacy/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        plan={'model':MODEL,'revision':REVISION,'scope':{},'case':{},'request':{},'temperature':.7,'seed':2026,'layers':[0],
            'bindings_sha256':'fixture','output_fixture':{},'tokenizer':{},'assets':{'embeddings':[{'id':0,'file':'embeddings/0.bin'}]},
            'calls':[],'completion':{},'experts':{'0':{'ids':[0],'store':'trajectories/coding-fixture/indices/layer0','manifest_sha256':digest(b'{}')}}}
        plan['plan_sha256']=canonical(plan);(legacy/'plan.json').write_text(json.dumps(plan))
        def load(root,run,**kwargs):return json.loads((run/'plan.json').read_bytes())
        with patch('vesper.trajectory_evidence.load_plan',side_effect=load):
            result=migrate_input(self.artifacts,'coding-fixture')
            self.assertEqual(result['expert_payload_bytes_copied'],0);self.assertFalse(result['new_model_execution_run'])
            target=self.artifacts/'pc-heterogeneous/trajectories/coding-fixture';new=json.loads((target/'plan.json').read_bytes())
            self.assertEqual(plan_identity(new),plan_identity(plan))
            for name,raw in payloads.items():self.assertEqual((target/name).read_bytes(),raw)
            self.assertTrue((legacy/'plan.json').exists());self.assertFalse(list(target.rglob('experts.bin')))
            with self.assertRaises(FileExistsError):migrate_input(self.artifacts,'coding-fixture')
