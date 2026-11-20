import unittest
import numpy as np
import torch
from src.conservative_plan import PLAN
from src.conservative_geometry import observed_geometry,fit_offset
from src.conservative_training import BoundedCorrection,ContextLoader,use_correction,correction_loss


class ConservativeTests(unittest.TestCase):
    def test_fixed_plan_contains_replication_and_transfer(self):
        self.assertEqual([p['id'] for p in PLAN],list(range(44,62)))
        self.assertEqual(sum(p['kind']=='train' for p in PLAN),14)
        for method in ('compact_gps','bounded_multimodal'):
            self.assertEqual([p['seed'] for p in PLAN if p['method']==method and p['holdout'] is None],[42,7,13])
            self.assertEqual([p['holdout'] for p in PLAN if p['method']==method and p['holdout']],[36,37,38,39])

    def test_initial_prediction_equals_geometry_and_correction_is_bounded(self):
        model=BoundedCorrection();gps=torch.randn(3,5,273)
        torch.testing.assert_close(model(None,gps)['logits'],gps[:,-1,17:])
        with torch.no_grad():model.head[-1].bias.fill_(2.)
        output=model(None,gps);self.assertLessEqual(float(output['correction'].abs().max()),.25)
        power=torch.randn(3,256);loss=correction_loss(output,power.argmax(1),power,'gps');loss.backward()
        self.assertTrue(torch.isfinite(model.head[-1].bias.grad).all())
        model.disable_correction=True
        torch.testing.assert_close(model(None,gps)['logits'],gps[:,-1,17:])

    def test_fallback_requires_predeclared_margin(self):
        self.assertFalse(use_correction(.8,.8));self.assertFalse(use_correction(.801,.8));self.assertTrue(use_correction(.803,.8))

    def test_offset_fit_uses_only_training_indices(self):
        d=np.tile(np.arange(256,dtype=float),(3,1));p=np.full((3,256),-20.)
        p[:2,3]=0;p[2,17]=100
        self.assertEqual(fit_offset(d,p,np.array([0,1])),3)

    def test_enu_velocity_uses_observed_time_in_seconds(self):
        times=np.array([[0,.1,.3,.5,.8]])
        p1=np.zeros((1,5,2));p1[:,:,0]=45
        p1[:,:,1]=np.rad2deg(times*10/(6371000*np.cos(np.deg2rad(45))))
        p2=p1.copy();p2[:,:,0]+=.001
        legacy,physical,context=observed_geometry(p1,p2,times)
        self.assertAlmostEqual(context[0,4],10,places=3)
        self.assertAlmostEqual(context[0,7],.3,places=5)
        self.assertTrue(np.isfinite(physical).all())

    def test_context_loader_alignment(self):
        gps=torch.randn(2,5,9);idx=torch.tensor([2,0])
        batch=(torch.zeros(2,1),gps,torch.zeros(2),torch.zeros(2,256),torch.ones(2),idx)
        context=np.arange(24,dtype=np.float32).reshape(3,8);prior=np.arange(768,dtype=np.float32).reshape(3,256)
        result=next(iter(ContextLoader([batch],context,prior)))[1]
        torch.testing.assert_close(result[:,:,:9],gps)
        np.testing.assert_equal(result[:,-1,9:17],context[[2,0]])
        np.testing.assert_equal(result[:,-1,17:],prior[[2,0]])


if __name__=='__main__':unittest.main()
