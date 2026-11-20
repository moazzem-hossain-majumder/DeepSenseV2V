import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from src.readiness_audit import inspect_file
from src.multiscenario_windows import build_windows


def frame_fixture():
    n=160
    frame=pd.DataFrame({'abs_index':np.concatenate([np.arange(20)+i*100 for i in range(8)]),
        'seq_index':np.repeat(np.arange(8),20),'unit1_overall-beam':np.zeros(n,dtype=int)})
    frame['timestamp']=[f'00-00-{i//10:02d}.{i%10}00000' for i in range(n)]
    return frame


class ReadinessTests(unittest.TestCase):
    def test_windows_stay_inside_sequences_and_splits(self):
        frame=frame_fixture()
        windows=build_windows(frame,36)
        self.assertEqual(len(windows),8*7)
        first=windows.iloc[0]
        self.assertEqual(first.x_indices,[0,2,4,6,8])
        self.assertEqual(first.y_index,13)
        self.assertAlmostEqual(first.horizon_seconds,.5)
        self.assertTrue((windows.groupby('block_key').split.nunique()==1).all())
        for row in windows.itertuples():
            self.assertEqual(frame.seq_index.iloc[row.x_indices+[row.y_index]].nunique(),1)
        other=build_windows(frame,37)
        self.assertFalse(set(windows.block_key)&set(other.block_key))

    def test_windows_reject_missing_sample_spans(self):
        frame=frame_fixture()
        frame.loc[10:19,'abs_index']+=1
        windows=build_windows(frame,36)
        self.assertFalse(((windows.seq_index==0)).any())

    def test_numeric_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'power.txt'
            np.savetxt(path,np.ones(64))
            self.assertNotIn('error',inspect_file((path,np.ones(64),'pwr')))
            self.assertIn('error',inspect_file((path,np.zeros(64),'pwr')))
            path.write_text('bad data')
            self.assertIn('error',inspect_file((path,None,'pwr')))

    def test_full_jpeg_decode_detects_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'image.jpg'
            Image.new('RGB',(128,128),'red').save(path)
            self.assertNotIn('error',inspect_file((path,None,'rgb180')))
            raw=path.read_bytes()
            path.write_bytes(raw[:len(raw)//2])
            self.assertIn('error',inspect_file((path,None,'rgb180')))

    def test_legacy_loader_does_not_substitute_black_images(self):
        from src.dataset import load_or_build_rgb_cache
        with tempfile.TemporaryDirectory() as directory, patch('os.getcwd',return_value=directory):
            with self.assertRaises(FileNotFoundError):
                load_or_build_rgb_cache({'unit1_rgb5':['unit1/rgb5/not_here.jpg']},None,
                                       data_root=directory,img_size=(8,8))


if __name__=='__main__': unittest.main()
