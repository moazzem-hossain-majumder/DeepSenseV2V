import sys
from src.anchor_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/steady_v1/stage210'):raise ValueError('Stage 210 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.anchor_worker','anchor',OUT/'queue.json'))
