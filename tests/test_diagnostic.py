import unittest
import numpy as np
import pandas as pd
from src.diagnostic_analysis import clock_seconds,clock_delta,failure_summary
from src.diagnostic_intervention import donors
from src.diagnostic_plan import PLAN

class DiagnosticTests(unittest.TestCase):
    def test_sensor_clock_and_midnight(self):
        self.assertAlmostEqual(clock_seconds('unit1/rgb5/frame_12-26-03.749002.jpg'),44763.749002)
        self.assertAlmostEqual(float(clock_delta(1,86399)),2)
        with self.assertRaises(ValueError):clock_seconds('missing')
    def test_permutation_keeps_scenarios_and_excludes_reserved(self):
        m=pd.DataFrame({'scenario':[36,36,36,37,37,37,38],'split':['val']*6+['test']})
        ids=np.arange(6);d=donors(m,ids)
        np.testing.assert_array_equal(np.sort(d),ids)
        self.assertTrue((d!=ids).all())
        np.testing.assert_array_equal(m.scenario.to_numpy()[d],m.scenario.to_numpy()[ids])
        np.testing.assert_array_equal(d,donors(m,ids))
    def test_failure_direction_and_mask(self):
        power=np.full((2,6),-20.);power[:,5]=0
        model=np.array([[0,1,2,3,4,5],[5,4,3,2,1,0.]])
        base=model[::-1];result=failure_summary(model,base,power,np.array([5,5]),np.array([True,False]))
        self.assertEqual(result['helped'],1);self.assertEqual(result['harmed'],0)
        self.assertEqual([p['id'] for p in PLAN],list(range(62,72)))

if __name__=='__main__':unittest.main()
