import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch
from src import unattended_queue as queue
from src.acquire_scenarios import save_json


class QueueTests(unittest.TestCase):
    def run_case(self, success):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            save_json(root/'results/validation_diagnostics_v1/status.json',{'status':'complete'})
            plan=[dict(id=4,title='A',directory='results/a',models=['analysis'],epochs=1,unit='passes'),
                  dict(id=5,title='B',directory='results/b',models=['analysis'],epochs=1,unit='passes')]
            fake=MagicMock();fake.pid=123;fake.create_time.return_value=100;fake.is_running.return_value=True
            stage=[0]
            def wait(timeout):
                save_json(root/plan[stage[0]]['directory']/'status.json',{'status':'complete' if success else 'failed'})
                stage[0]+=1
                return 0 if success else 1
            fake.wait.side_effect=wait
            with patch.object(queue,'ROOT',root),patch.object(queue,'QUEUE',root/'queue.json'), \
                 patch('src.unattended_stages.PLAN',plan),patch.object(queue.subprocess,'Popen',return_value=fake) as launch, \
                 patch.object(queue.psutil,'Process',return_value=fake):
                code=queue.run()
                state=json.loads((root/'queue.json').read_text())
                return code,state,launch.call_count

    def test_success_advances_and_completes(self):
        code,state,calls=self.run_case(True)
        self.assertEqual((code,calls,state['status']),(0,2,'complete'))

    def test_failure_does_not_start_next_stage(self):
        code,state,calls=self.run_case(False)
        self.assertEqual((code,calls),(1,1))
        self.assertEqual(state['stages']['5']['status'],'queued')
        self.assertEqual(state['stages']['4']['failures'],1)


if __name__=='__main__': unittest.main()
