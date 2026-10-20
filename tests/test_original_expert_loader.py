from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vesper.connected_reference import OfficialConnected


class OriginalExpertLoaderTests(unittest.TestCase):
    def owner(self,loader=None):
        value=OfficialConnected.__new__(OfficialConnected)
        value.layer=0;value.ordinal=0;value.directory=Path('reference');value.expert_directory=Path('experts')
        value.experts=OrderedDict();value.provenance={};value.expert_loader=loader
        return value

    def test_explicit_offline_loader_keeps_actual_ids_arrays_and_six_entry_LRU(self):
        calls=[];arrays=object()
        def load(layer,expert):calls.append((layer,expert));return arrays,{'layer':layer,'expert':expert}
        owner=self.owner(load)
        with patch('vesper.connected_reference.load_expert_fixture',side_effect=AssertionError('No NPZ fallback')),patch('vesper.connected_reference.OfficialExpert',side_effect=lambda a,m,*args:SimpleNamespace(arrays=a,expert=m['expert'])):
            first=owner._expert(0);self.assertIs(owner._expert(0),first)
            for expert in range(1,7):self.assertIs(owner._expert(expert).arrays,arrays)
            self.assertEqual(list(owner.experts),[1,2,3,4,5,6]);self.assertEqual(calls,[(0,i) for i in range(7)])
            self.assertEqual(owner.provenance['0:6']['expert'],6)

    def test_default_fixture_path_remains_the_qualified_loader(self):
        owner=self.owner();arrays=object();m={'layer':0,'expert':3}
        with patch('vesper.connected_reference.load_expert_fixture',return_value=(arrays,m)) as load,patch('vesper.connected_reference.OfficialExpert',return_value=object()) as expert:
            owner._expert(3)
            load.assert_called_once_with(Path('experts/expert-003/native-expert.npz'))
            self.assertIs(expert.call_args.args[0],arrays)

    def test_loader_cannot_substitute_a_different_original_expert(self):
        owner=self.owner(lambda layer,expert:({}, {'layer':layer,'expert':expert+1}))
        with patch('vesper.connected_reference.OfficialExpert',side_effect=AssertionError('Wrong identity cannot reach arithmetic')):
            with self.assertRaisesRegex(ValueError,'identity'):owner._expert(3)
        self.assertFalse(owner.experts);self.assertFalse(owner.provenance)
