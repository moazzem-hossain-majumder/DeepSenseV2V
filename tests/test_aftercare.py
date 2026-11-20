import unittest
import numpy as np
import torch
from src.aftercare_plan import choose_branch,plan_for
from src.aftercare_geometry import GeometryResidual,PriorLoader


class AftercareTests(unittest.TestCase):
    def test_test_exposure_always_prohibits_retraining(self):
        for passed in (False,True):
            branch=choose_branch(passed,True)
            self.assertEqual(branch,'evidence_review')
            self.assertFalse(any(p['kind']=='train' for p in plan_for(branch)))

    def test_weak_branch_is_bounded_and_validation_only(self):
        branch=choose_branch(False,False);plan=plan_for(branch)
        self.assertEqual([p['id'] for p in plan],list(range(37,44)))
        self.assertEqual([p['seed'] for p in plan if p['kind']=='train'],[42,7,13])
        self.assertFalse(any(p['kind'] in ('test','calibrate') for p in plan))

    def test_inconsistent_gate_does_not_trigger_training(self):
        with self.assertRaises(ValueError): plan_for(choose_branch(True,False))

    def test_geometry_wrapper_preserves_core_gps_and_has_finite_gradients(self):
        class Backbone(torch.nn.Module):
            def forward(self,rgb,gps):
                self.last=gps
                return {'logits':torch.zeros(len(gps),256),'pred_profile':torch.zeros(len(gps),256)}
        base=Backbone();model=GeometryResidual(base)
        x=torch.randn(2,5,265);out=model(None,x)
        torch.testing.assert_close(base.last,x[:,:,:9])
        torch.testing.assert_close(out['logits'],x[:,-1,9:])
        out['logits'].square().mean().backward()
        self.assertTrue(torch.isfinite(model.geometry_log_gain.grad))

    def test_loader_keeps_prior_aligned_after_shuffle(self):
        indices=torch.tensor([2,0]);gps=torch.randn(2,5,9)
        batch=(torch.zeros(2,1),gps,torch.zeros(2),torch.zeros(2,256),torch.tensor([36,36]),indices)
        prior=np.arange(3*256,dtype=np.float32).reshape(3,256)
        output=next(iter(PriorLoader([batch],prior)))
        torch.testing.assert_close(output[1][:,:,:9],gps)
        np.testing.assert_equal(output[1][:,-1,9:].numpy(),prior[[2,0]])


if __name__=='__main__':unittest.main()
