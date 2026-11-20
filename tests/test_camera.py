import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from src import camera_training as worker
from src import camera_data as data
from src import publication_data as old_data
from src import publication_training as engine
from src.camera_plan import PLAN
from src.camera_evidence import eligible
from src.acquire_scenarios import save_json
from PIL import Image

class CameraTests(unittest.TestCase):
    def test_rear_cache_build_and_verified_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);front=root/'front';front.mkdir();rear=root/'rear';hashes={}
            for s in (36,37,38,39):
                raw=root/f'data/raw/scenario{s}';raw.mkdir(parents=True)
                Image.fromarray(np.full((4,4,3),s,np.uint8)).save(raw/'rear.png')
                csv=raw/f'scenario{s}.csv';pd.DataFrame({'unit1_rgb6':['rear.png']}).to_csv(csv,index=False)
                hashes[str(s)]=old_data.digest(csv)
            save_json(front/'receipt.json',dict(csv_sha256=hashes))
            with patch.object(data,'ROOT',root),patch.object(data,'FRONT',front),patch.object(data,'CACHE',rear):
                data.prepare(lambda **v:None);receipts=data.verify();self.assertEqual(len(receipts),4)
                data.prepare(lambda **v:self.fail('Completed cache unexpectedly rebuilt'))
                self.assertTrue((np.load(rear/'rear38.npy')==38).all())
    def test_gate_and_finite_plan(self):
        self.assertEqual([p['id'] for p in PLAN],list(range(81,96)))
        h=dict(finite=True,rgb_gradient_norm=.01,gray_delta=.001)
        self.assertTrue(eligible(.8,.78,h));self.assertFalse(eligible(.78,.78,h))
        self.assertFalse(eligible(.8,.78,{**h,'gray_delta':0}))
    def test_camera_order_and_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);np.save(root/'rgb36.npy',np.full((5,2,2,3),11,np.uint8));np.save(root/'rear36.npy',np.full((5,2,2,3),29,np.uint8))
            m=pd.DataFrame(dict(scenario=[36],target_beam=[7],x_indices=[[0,1,2,3,4]]))
            with patch.object(data,'CACHE',root),patch.object(old_data,'CACHE',root):
                for camera in ('front','rear','both'):
                    ds=data.CameraClips(m,np.zeros((1,5,9)),np.zeros((1,256)),[0],camera)
                    rgb,g,label,p,s,i=ds[0];self.assertEqual(label,7)
                    self.assertEqual(rgb.shape,(5,2,4 if camera=='both' else 2,3))
                    self.assertTrue((rgb[:,:,:2]==(29 if camera=='rear' else 11)).all())
                    if camera=='both':self.assertTrue((rgb[:,:,2:]==29).all())
                    for mm in list(ds.maps.values())+list(ds.rear_maps.values()):mm._mmap.close()

@unittest.skipUnless(torch.cuda.is_available(),'CUDA integration')
class CameraGPU(unittest.TestCase):
    def test_training_checkpoint_resume_all_pilot_modes(self):
        rng=np.random.default_rng(81);g=rng.normal(size=(4,5,9)).astype(np.float32);power=rng.normal(size=(4,256)).astype(np.float32)
        m=pd.DataFrame(dict(scenario=[36,37,36,37],split=['train','train','val','val']))
        for camera,mode in (('front','tanh_bf16'),('front','tanh_fp32'),('front','linear'),('rear','linear'),('both','linear')):
            with self.subTest(camera=camera,mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d);audit=root/'stage44';audit.mkdir();folder=root/'run'
                np.save(audit/'context.npy',rng.normal(size=(4,8)).astype(np.float32));np.save(audit/'prior_pooled.npy',np.zeros((4,256),np.float32))
                cfg={**worker.COMMON,'variant':'fusion_ce','epochs':2,'seed':42,'holdout':None,'camera':camera,'correction':mode,'workers':0,'batch':2}
                images=torch.randint(0,256,(4,5,96,192 if camera=='both' else 96,3),dtype=torch.uint8)
                def batches(m,g,p,idx,c,context,prior,train=False):
                    base=DataLoader(TensorDataset(images[idx],torch.from_numpy(g[idx]),torch.from_numpy(power.argmax(1)[idx]),torch.from_numpy(power[idx]),torch.tensor([36,37,36,37])[idx],torch.tensor(idx)),batch_size=2,shuffle=train)
                    return worker.ContextLoader(base,context,prior)
                with patch.object(worker,'PREVIOUS',root),patch.object(worker,'verify',return_value={}),patch.object(worker,'configuration',return_value=cfg),patch.object(worker,'make_loader',side_effect=batches),patch.object(engine,'load',return_value=(m,g,power,{'synthetic':True})):
                    plan=dict(directory=str(folder));progress=lambda **v:save_json(folder/'status.json',v)
                    worker.run(plan,progress)
                    before=(folder/'result.json').read_text();worker.run(plan,progress)
                    self.assertEqual(before,(folder/'result.json').read_text())
                    self.assertEqual(json.loads((folder/'status.json').read_text())['status'],'complete')
                    if camera=='both':
                        health=worker.health(plan)
                        self.assertTrue(health['finite'])
                        self.assertGreater(health['rgb_gradient_norm'],0)

if __name__=='__main__':unittest.main()
