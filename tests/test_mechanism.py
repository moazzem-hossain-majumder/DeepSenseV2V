import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.mechanism_sample import choose
from src import mechanism_memorize as worker
from src.mechanism_review import benchmark
from src.publication_data import digest
from src.acquire_scenarios import save_json

class MechanismTests(unittest.TestCase):
    def test_hard_examples_are_training_and_block_separated(self):
        m=pd.DataFrame(dict(scenario=np.repeat([36,37,38,39],4),split=['train']*16,target_beam=[10]*16,block_key=[str(i) for i in range(16)]))
        prior=np.tile(-np.arange(256),(16,1));power=np.full((16,256),-20.);power[:,10]=0
        ids=choose(m,power,prior);self.assertEqual(len(ids),16);self.assertEqual(len(set(m.block_key.iloc[ids])),16)
        m.loc[0,'split']='test'
        with self.assertRaises(ValueError):choose(m,power,prior)
    def test_snapshotted_baseline_checks(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);benchmark(folder,lambda **v:None)
            r=json.loads((folder/'benchmark_checks.json').read_text())
            self.assertTrue(r['previous_sample_dependency_detected']);self.assertEqual(r['shifted_index_unique_count'],255)

@unittest.skipUnless(torch.cuda.is_available(),'CUDA integration')
class MemorizationGPU(unittest.TestCase):
    def test_all_modes_resume_and_positive_control_learns(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sample=root/'stage117';sample.mkdir();rng=np.random.default_rng(117)
            prior=np.tile(np.linspace(0,-.5,20,dtype=np.float32),(4,1));power=np.full((4,256),-20.,np.float32);power[:,19]=0
            np.savez(sample/'sample.npz',descriptors=rng.normal(size=(4,20,569)).astype(np.float32),candidates=np.tile(np.arange(20),(4,1)),prior=prior,power=power)
            head=torch.nn.Sequential(torch.nn.Linear(569,128),torch.nn.GELU(),torch.nn.Linear(128,1));torch.nn.init.zeros_(head[-1].weight);torch.nn.init.zeros_(head[-1].bias)
            torch.save(head.state_dict(),sample/'initial_head.pt')
            save_json(sample/'sample.json',dict(sample_sha256=digest(sample/'sample.npz'),head_sha256=digest(sample/'initial_head.pt')))
            with patch.object(worker,'OUT',root):
                for mode in ('current_loss','bounded_ce','unbounded_ce','per_example_control'):
                    folder=root/mode;folder.mkdir();plan=dict(mode=mode);steps=500 if mode=='per_example_control' else 100
                    worker.run(plan,folder,lambda **v:None,steps);before=(folder/'result.json').read_text()
                    worker.run(plan,folder,lambda **v:None,steps);self.assertEqual(before,(folder/'result.json').read_text())
                    if mode=='per_example_control':self.assertEqual(json.loads(before)['final']['success5'],1.)

if __name__=='__main__':unittest.main()
