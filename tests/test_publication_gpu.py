"""Tiny synthetic integration check: optimizer, checkpoint, inference and resume."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from src import publication_training as training
from src.publication_plan import COMMON


@unittest.skipUnless(torch.cuda.is_available(),'CUDA integration check')
class GPUIntegration(unittest.TestCase):
    def test_train_save_and_resume_without_extra_epochs(self):
        torch.manual_seed(42)
        manifest=pd.DataFrame({'scenario':[36,37,36,37], 'split':['train','train','val','val']})
        gps=np.random.default_rng(1).normal(size=(4,5,9)).astype(np.float32)
        profiles=np.random.default_rng(2).normal(size=(4,256)).astype(np.float32)
        images=torch.randint(0,255,(4,5,96,96,3),dtype=torch.uint8)
        labels=torch.from_numpy(profiles.argmax(1))
        def tiny_loader(m,g,p,idx,cfg,train=False):
            ds=TensorDataset(images[idx],torch.from_numpy(g[idx]),labels[idx],torch.from_numpy(p[idx]),
                torch.tensor([36,37,36,37])[idx],torch.tensor(idx))
            return DataLoader(ds,batch_size=2,shuffle=train)
        with tempfile.TemporaryDirectory(prefix='deepsense_gpu_check_') as directory:
            folder=Path(directory);cfg={**COMMON,'variant':'power_aware','seed':42,'holdout':None,'epochs':1}
            plan={'directory':str(folder)}
            with patch.object(training,'config',return_value=cfg),patch.object(training,'load',return_value=(manifest,gps,profiles,{'synthetic':True})),patch.object(training,'loader',side_effect=tiny_loader):
                training.train(plan,lambda **values:None)
                first=(folder/'result.json').read_text()
                self.assertTrue((folder/'best.pt').exists())
                training.train(plan,lambda **values:None)
                self.assertEqual(first,(folder/'result.json').read_text())
                self.assertEqual(len(json.loads((folder/'power_aware_history.json').read_text())),1)


if __name__=='__main__':unittest.main()
