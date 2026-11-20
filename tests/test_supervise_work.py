import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.supervise_work import command_for, supervise


class SupervisorTests(unittest.TestCase):
    def test_pilot_uses_separate_multiscenario_runner(self):
        self.assertEqual(command_for('extended', [])[-2:], ['-m', 'src.multiscenario_extended'])
        command = command_for('pilot', [])
        self.assertEqual(command[-2:], ['-m', 'src.multiscenario_pilot'])
        self.assertNotIn('train.py', command)
        with self.assertRaises(ValueError):
            command_for('pilot', ['--model', 'P3'])

    def test_training_requires_explicit_arguments_and_resume(self):
        with self.assertRaises(ValueError):
            command_for('train', [])
        self.assertEqual(command_for('train', ['--model', 'P3'])[-1], '--resume')
        self.assertEqual(command_for('train', ['--model', 'P3', '--resume']).count('--resume'), 1)

    def test_unexpected_worker_exit_restarts_after_fifteen_seconds(self):
        first = MagicMock(pid=111)
        first.wait.return_value = 9
        second = MagicMock(pid=222)
        second.wait.return_value = 0
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            with patch('src.supervise_work.REPORTS', reports), \
                 patch('src.supervise_work.subprocess.Popen', side_effect=[first, second]) as popen, \
                 patch('src.supervise_work.time.sleep') as sleep:
                supervise('download', [])
            sleep.assert_called_once_with(15)
            self.assertEqual(popen.call_count, 2)
            status = json.loads((reports / 'download_supervisor.json').read_text())
            self.assertEqual(status['status'], 'finished')


if __name__ == '__main__':
    unittest.main()
