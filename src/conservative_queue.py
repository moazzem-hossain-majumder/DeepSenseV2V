import sys
from src.conservative_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed

if __name__=='__main__':
    if not completed(ROOT/'results/aftercare_v1/stage43'):raise ValueError('Stage 43 is not complete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.conservative_worker','conservative',OUT/'queue.json'))
