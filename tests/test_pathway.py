import unittest
import numpy as np
import pandas as pd
import torch
from src.pathway_probe import sample_training
from src.pathway_plan import PLAN

class PathwayTests(unittest.TestCase):
    def test_training_sample_excludes_holdout_and_reserved(self):
        m=pd.DataFrame(dict(scenario=[36,36,37,37,38,38,39,39],split=['train','val','train','test','train','calib','train','val']))
        ids=sample_training(m,38,32)
        np.testing.assert_array_equal(ids,[0,2,6])
        self.assertEqual([p['id'] for p in PLAN],list(range(72,81)))
    def test_saturation_blocks_local_gradient(self):
        raw=torch.tensor([0.,4.,20.],requires_grad=True);(.25*torch.tanh(raw)).sum().backward()
        self.assertAlmostEqual(float(raw.grad[0]),.25)
        self.assertLess(float(raw.grad[1]),.001)
        self.assertEqual(float(raw.grad[2]),0.)

if __name__=='__main__':unittest.main()
