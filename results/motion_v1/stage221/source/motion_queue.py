import sys
from src.motion_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/variation_v1/stage220'):raise ValueError('Stage 220 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.motion_worker','motion',OUT/'queue.json'))
