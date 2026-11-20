import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from src import conservative_training as worker
from src import publication_training as engine
from src.acquire_scenarios import save_json


@unittest.skipUnless(torch.cuda.is_available(),'CUDA integration')
class ConservativeGPU(unittest.TestCase):
    def test_models_train_save_select_fallback_and_resume(self):
        rng=np.random.default_rng(44);gps=rng.normal(size=(4,5,9)).astype(np.float32);profiles=rng.normal(size=(4,256)).astype(np.float32)
        m=pd.DataFrame({'scenario':[36,37,36,37],'split':['train','train','val','val']})
        images=torch.zeros(4,5,96,96,3,dtype=torch.uint8);labels=torch.from_numpy(profiles.argmax(1))
        def tiny_loader(m,g,p,idx,cfg,train=False):
            rgb=torch.zeros(4,1,dtype=torch.uint8) if cfg['variant']=='gps' else images
            return DataLoader(TensorDataset(rgb[idx],torch.from_numpy(g[idx]),labels[idx],torch.from_numpy(p[idx]),torch.tensor([36,37,36,37])[idx],torch.tensor(idx)),batch_size=2,shuffle=train)
        for method in ('compact_gps','bounded_multimodal'):
            with self.subTest(method=method),tempfile.TemporaryDirectory(prefix='deepsense_conservative_check_') as d:
                out=Path(d);audit=out/'stage44';audit.mkdir();folder=out/'run'
                np.save(audit/'context.npy',rng.normal(size=(4,8)).astype(np.float32))
                np.save(audit/'prior_pooled.npy',np.zeros((4,256),np.float32))
                save_json(audit/'report.json',{'configurations':{'pooled':{'chosen':'synthetic','selection_val':{'macro_success5':1.}}}})
                np.savez_compressed(audit/'baseline_pooled_val.npz',indices=np.array([2,3]),logits=np.zeros((2,256)),profiles=profiles[2:],labels=labels[2:].numpy(),scenarios=np.array([36,37]))
                plan=dict(directory=str(folder),method=method,seed=42,holdout=None)
                with patch.object(worker,'OUT',out),patch.object(worker,'COMMON',{**worker.COMMON,'epochs':1}),patch.object(engine,'load',return_value=(m,gps,profiles,{'synthetic':True})),patch.object(engine,'loader',side_effect=tiny_loader):
                    progress=lambda **kw:save_json(folder/'status.json',kw)
                    worker.run(plan,progress)
                    self.assertFalse(json.loads((folder/'selection.json').read_text())['correction_enabled'])
                    before=(folder/'selection.json').read_text();worker.run(plan,progress)
                    self.assertEqual(before,(folder/'selection.json').read_text())


if __name__=='__main__':unittest.main()
