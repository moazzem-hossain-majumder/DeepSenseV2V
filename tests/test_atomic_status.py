import ctypes
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from src.acquire_scenarios import save_json


class AtomicStatusTests(unittest.TestCase):
    def test_transient_lock_preserves_old_json_until_replace(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'status.json'
            save_json(target,{'epoch':3})
            original=Path.replace;attempts=[]
            def blocked(path,destination):
                attempts.append(1)
                if len(attempts)<4:
                    self.assertEqual(json.loads(target.read_text()),{'epoch':3})
                    raise PermissionError('transient reader lock')
                return original(path,destination)
            with patch.object(Path,'replace',blocked),patch('src.acquire_scenarios.time.sleep'):
                save_json(target,{'epoch':4})
            self.assertEqual(json.loads(target.read_text()),{'epoch':4})
            self.assertEqual(len(list(Path(d).glob('*.tmp'))),0)

    def test_persistent_permission_error_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'status.json';save_json(target,{'old':True})
            with patch.object(Path,'replace',side_effect=PermissionError('denied')),patch('src.acquire_scenarios.time.sleep'):
                with self.assertRaises(PermissionError): save_json(target,{'new':True})
            self.assertEqual(json.loads(target.read_text()),{'old':True})

    @unittest.skipUnless(os.name=='nt','Windows sharing behavior')
    def test_real_windows_reader_lock_recovers(self):
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
        kernel.CreateFileW.restype=ctypes.c_void_p
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'status.json';save_json(target,{'old':True})
            handle=kernel.CreateFileW(str(target),0x80000000,3,None,3,0,None)
            self.assertNotEqual(handle,ctypes.c_void_p(-1).value)
            release=threading.Timer(.35,lambda:kernel.CloseHandle(handle));release.start()
            try: save_json(target,{'new':True})
            finally: release.join()
            self.assertEqual(json.loads(target.read_text()),{'new':True})


if __name__=='__main__': unittest.main()
