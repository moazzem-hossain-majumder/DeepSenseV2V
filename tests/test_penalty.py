import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from src import factorial_training as worker
from src import penalty_worker as stage_worker
from src import penalty_training as adaptation
from src import publication_training as engine
from src.conservative_training import ContextLoader
from src.penalty_plan import PLAN
from src.acquire_scenarios import save_json

class TinyEncoder(torch.nn.Module):
    def forward(self,x):return x.mean((1,2,3))[:,None].expand(-1,512)

class PenaltyTests(unittest.TestCase):
    def test_hard_mask_with_extremely_negative_unbounded_scores(self):
        model=worker.FactorialReranker(20,False,False,TinyEncoder())
        with torch.no_grad():model.head[-1].bias.fill_(-1e6)
        gps=torch.randn(2,5,273);prior=torch.linspace(-20,0,256).repeat(2,1);gps[:,-1,17:]=prior
        out=model(None,gps);actual=torch.argsort(out['logits'],descending=True)[:,:20]
        expected=torch.argsort(prior,descending=True)[:,:20]
        torch.testing.assert_close(actual.sort(1).values,expected.sort(1).values)
        self.assertEqual(int((out['logits']==-1000).sum()),2*236)
        self.assertTrue(torch.isfinite(out['logits']).all())
        for kind in ('soft','hard'):self.assertTrue(torch.isfinite(worker.objective(out,None,torch.randn(2,256),kind)))
    def test_window_and_factorial_completeness(self):
        self.assertEqual([p['id'] for p in PLAN],list(range(134,140)))
        self.assertEqual(len({(p['penalty'],p['vision']) for p in PLAN if p['kind']=='train'}),4)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'window.json';save_json(path,dict(ends_utc='2000-01-01T00:00:00+00:00'))
            with patch.object(stage_worker,'WINDOW',path):self.assertFalse(stage_worker.within_window())

@unittest.skipUnless(torch.cuda.is_available(),'CUDA integration')
class PenaltyGPU(unittest.TestCase):
    def test_four_arms_train_checkpoint_and_resume(self):
        rng=np.random.default_rng(124);gps=rng.normal(size=(4,5,9)).astype(np.float32);power=rng.normal(size=(4,256)).astype(np.float32)
        m=pd.DataFrame(dict(scenario=[36,37,36,37],split=['train','train','val','val']));images=torch.randint(0,256,(4,5,96,192,3),dtype=torch.uint8)
        for bounded in (False,):
            for objective,penalty in (('hard',0.),('hard',.1)):
                for vision in (False,True):
                    with self.subTest(bounded=bounded,objective=objective,vision=vision),tempfile.TemporaryDirectory() as d:
                        root=Path(d);audit=root/'stage44';audit.mkdir();folder=root/'run'
                        np.save(audit/'context.npy',rng.normal(size=(4,8)).astype(np.float32));np.save(audit/'prior_pooled.npy',rng.normal(size=(4,256)).astype(np.float32))
                        cfg={**worker.COMMON,'variant':'fusion_ce','epochs':2,'seed':42,'holdout':None,'camera':'both','k':20,'vision':vision,'bounded':bounded,'objective':objective,'workers':0,'batch':2}
                        def batches(m,g,p,idx,c,context,prior,train=False):
                            base=DataLoader(TensorDataset(images[idx],torch.from_numpy(g[idx]),torch.from_numpy(power.argmax(1)[idx]),torch.from_numpy(power[idx]),torch.tensor([36,37,36,37])[idx],torch.tensor(idx)),batch_size=2,shuffle=train)
                            return ContextLoader(base,context,prior)
                        with patch.object(worker,'PREVIOUS',root),patch.object(worker,'verify',return_value={}),patch.object(worker,'configuration',return_value=cfg),patch.object(worker,'make_loader',side_effect=batches),patch.object(engine,'load',return_value=(m,gps,power,{'synthetic':True})):
                            plan=dict(directory=str(folder),penalty=penalty);progress=lambda **v:save_json(folder/'status.json',v)
                            adaptation.run(plan,progress);before=(folder/'result.json').read_text();adaptation.run(plan,progress)
                            self.assertEqual(before,(folder/'result.json').read_text())
                            with np.load(folder/'val_predictions.npz') as a:self.assertTrue(((a['logits']>=0).sum(1)==20).all())

if __name__=='__main__':unittest.main()
