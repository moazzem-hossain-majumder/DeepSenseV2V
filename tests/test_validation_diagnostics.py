import unittest
import numpy as np
from src.validation_diagnostics import summarize


class DiagnosticsTests(unittest.TestCase):
    def test_shortlist_power_and_accuracy_are_distinct_and_monotone(self):
        logits = -np.arange(256,dtype=float)[None,:].repeat(2,0)
        powers = np.full((2,256),-20.)
        powers[:,2]=0
        powers[:,0]=-2
        result = summarize(logits,powers,np.array([2,2]))
        self.assertEqual(result['top1'],0)
        self.assertEqual(result['top3'],1)
        self.assertEqual(result['within_3db_at_1'],1)
        self.assertEqual(result['power_loss_db'],2)
        self.assertEqual(result['mean_power_loss_db_at_3'],0)


if __name__=='__main__': unittest.main()
