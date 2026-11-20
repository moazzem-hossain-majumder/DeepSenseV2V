import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src import aftercare_analysis as analysis


class ImageEDAPlotTest(unittest.TestCase):
    def test_actual_installed_matplotlib_renders_four_scenario_eda(self):
        with tempfile.TemporaryDirectory(prefix='deepsense_plot_check_') as d:
            folder=Path(d);rng=np.random.default_rng(38)
            manifest=pd.DataFrame([dict(scenario=s,split='train',x_indices=[0,1,2,3,4]) for s in (36,37,38,39)])
            for s in (36,37,38,39):
                np.save(folder/f'rgb{s}.npy',rng.integers(0,256,(5,96,96,3),dtype=np.uint8))
            with patch.object(analysis,'CACHE',folder),patch.object(analysis.pd,'read_json',return_value=manifest):
                analysis.image_eda(folder,lambda **values:None)
            report=json.loads((folder/'image_quality.json').read_text())
            self.assertEqual(set(report['scenarios']),{'36','37','38','39'})
            for scenario in report['scenarios'].values(): self.assertEqual(scenario['sampled_frames'],5)
            self.assertGreater((folder/'image_quality.png').stat().st_size,1000)


if __name__=='__main__': unittest.main()
