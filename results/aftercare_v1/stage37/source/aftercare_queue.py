import json
import sys
from src.aftercare_plan import OUT,plan_for
from src.unattended_queue import run

if __name__=='__main__':
    route=json.loads((OUT/'route.json').read_text())
    sys.exit(run(plan_for(route['branch']),'src.aftercare_worker','aftercare',OUT/'queue.json'))
