import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.publication_plan import PLAN,SEEDS,VARIANTS
from src.publication_data import partitions,scale
from src.publication_metrics import objective,calibrate,candidate_outcomes,metrics
from src.publication_training import frozen_bn
from src import publication_evidence
from src.acquire_scenarios import save_json


class PublicationTests(unittest.TestCase):
    def test_matched_plan_and_final_test_gate_order(self):
        self.assertEqual([p['id'] for p in PLAN],list(range(8,37)))
        for variant in VARIANTS:
            self.assertEqual([p['seed'] for p in PLAN if p.get('variant')==variant and p.get('holdout','missing') is None],list(SEEDS))
        self.assertEqual([p['kind'] for p in PLAN[-4:]],['gate','calibrate','test','report'])

    def test_unseen_scenario_is_absent_from_train_and_selection(self):
        m=pd.DataFrame({'scenario':[36,37,36,37,36,37], 'split':['train','train','val','val','test','calib']})
        train,val,unseen=partitions(m,36)
        self.assertEqual(train.tolist(),[1]);self.assertEqual(val.tolist(),[3]);self.assertEqual(unseen.tolist(),[2])

    def test_scaling_ignores_heldout_extrema(self):
        gps=np.array([[[0.,1.]],[[2.,1.]],[[100.,999.]]])
        transformed,minimum,span=scale(gps,np.array([0,1]))
        np.testing.assert_equal(minimum,[0.,1.]);np.testing.assert_equal(span,[2.,1.])
        self.assertGreater(transformed[-1,0,0],1)

    def test_relative_power_objective_is_offset_invariant_and_finite(self):
        logits=torch.randn(3,256,requires_grad=True);profile=torch.randn(3,256,requires_grad=True)
        power=torch.randn(3,256);labels=power.argmax(1)
        a=objective({'logits':logits,'pred_profile':profile},labels,power,'power_aware')
        b=objective({'logits':logits,'pred_profile':profile},labels,power+20,'power_aware')
        torch.testing.assert_close(a,b);a.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_calibration_includes_first_acceptable_candidate(self):
        logits=np.tile(np.array([4.,3.,2.,1.]),(20,1));power=np.tile(np.array([-20.,-10.,0.,-20.]),(20,1))
        par=calibrate(logits,power)
        self.assertEqual(par['fixed_k'],3)
        for adaptive in (False,True):
            counts,regret=candidate_outcomes(logits,power,par,adaptive)
            self.assertTrue((counts>=3).all());self.assertTrue((regret==0).all())

    def test_frozen_batchnorm_does_not_update_running_stats(self):
        bn=torch.nn.BatchNorm2d(3);model=torch.nn.Sequential(bn);model.train();frozen_bn(model)
        before=bn.running_mean.clone();model(torch.randn(8,3,4,4));torch.testing.assert_close(before,bn.running_mean)

    def test_failed_gate_never_loads_heldout_data(self):
        with tempfile.TemporaryDirectory(prefix='deepsense_gate_check_') as d:
            root=Path(d);save_json(root/'stage33/decision.json',{'passed':False})
            destination=root/'stage35';destination.mkdir()
            with patch.object(publication_evidence,'OUT',root),patch.object(publication_evidence,'load') as load:
                publication_evidence.heldout(destination,lambda **kw:None,'test')
                load.assert_not_called()
                self.assertTrue((destination/'skipped.json').exists())


if __name__=='__main__': unittest.main()
