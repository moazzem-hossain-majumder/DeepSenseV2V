import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from src import rerank_training as worker
from src import publication_training as engine
from src.conservative_training import ContextLoader
from src.rerank_plan import PLAN
from src.rerank_evidence import select
from src.acquire_scenarios import save_json

class TinyEncoder(torch.nn.Module):
    def forward(self,x):return x.mean((1,2,3))[:,None].expand(-1,512)

class RerankTests(unittest.TestCase):
    def test_shortlist_constraint_zero_initialization_and_finite_gradient(self):
        for k in (10,20):
            model=worker.Reranker(k,False,TinyEncoder());gps=torch.randn(3,5,273);prior=torch.linspace(-20,0,256).repeat(3,1)
            gps[:,-1,17:]=prior;expected=torch.argsort(prior,descending=True,stable=True)[:,:k]
            out=model(None,gps)
            torch.testing.assert_close(out['selected'],prior.gather(1,expected))
            with torch.no_grad():model.head[-1].weight.fill_(1000)
            out=model(None,gps);actual=torch.argsort(out['logits'],descending=True)[:,:k]
            torch.testing.assert_close(actual.sort(1).values,expected.sort(1).values)
            loss=worker.loss(out,torch.zeros(3,dtype=torch.long),torch.randn(3,256),'fusion_ce');loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertLessEqual(float(out['residual'].abs().max()),2.)
    def test_visual_gate_must_beat_both_controls(self):
        records=[dict(k=10,vision=False,score=.8),dict(k=10,vision=True,score=.801)]
        self.assertIsNone(select(records,.78));records[1]['score']=.81
        self.assertEqual(select(records,.78)['k'],10)
        self.assertIsNone(select(records,.82))
        self.assertEqual([p['id'] for p in PLAN],list(range(96,117)))

@unittest.skipUnless(torch.cuda.is_available(),'CUDA integration')
class RerankGPU(unittest.TestCase):
    def test_matched_models_train_and_resume(self):
        rng=np.random.default_rng(96);gps=rng.normal(size=(4,5,9)).astype(np.float32);power=rng.normal(size=(4,256)).astype(np.float32)
        m=pd.DataFrame(dict(scenario=[36,37,36,37],split=['train','train','val','val']))
        images=torch.randint(0,256,(4,5,96,192,3),dtype=torch.uint8)
        for k,vision in ((10,False),(10,True),(20,False),(20,True)):
            with self.subTest(k=k,vision=vision),tempfile.TemporaryDirectory() as d:
                root=Path(d);audit=root/'stage44';audit.mkdir();folder=root/'run'
                np.save(audit/'context.npy',rng.normal(size=(4,8)).astype(np.float32));np.save(audit/'prior_pooled.npy',rng.normal(size=(4,256)).astype(np.float32))
                cfg={**worker.COMMON,'variant':'fusion_ce','epochs':2,'seed':42,'holdout':None,'camera':'both','k':k,'vision':vision,'workers':0,'batch':2}
                def batches(m,g,p,idx,c,context,prior,train=False):
                    base=DataLoader(TensorDataset(images[idx],torch.from_numpy(g[idx]),torch.from_numpy(power.argmax(1)[idx]),torch.from_numpy(power[idx]),torch.tensor([36,37,36,37])[idx],torch.tensor(idx)),batch_size=2,shuffle=train)
                    return ContextLoader(base,context,prior)
                with patch.object(worker,'PREVIOUS',root),patch.object(worker,'verify',return_value={}),patch.object(worker,'configuration',return_value=cfg),patch.object(worker,'make_loader',side_effect=batches),patch.object(engine,'load',return_value=(m,gps,power,{'synthetic':True})):
                    plan=dict(directory=str(folder));progress=lambda **v:save_json(folder/'status.json',v)
                    worker.run(plan,progress);before=(folder/'result.json').read_text();worker.run(plan,progress)
                    self.assertEqual(before,(folder/'result.json').read_text())
                    saved=torch.load(folder/'best.pt',weights_only=False,map_location='cpu')
                    self.assertTrue(all(torch.isfinite(t).all() for t in saved['model'].values()))

if __name__=='__main__':unittest.main()
