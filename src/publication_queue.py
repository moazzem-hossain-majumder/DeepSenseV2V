import sys
from src.publication_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed

if __name__=='__main__':
    if not completed(ROOT/'results/stage7_comparison'):
        raise ValueError('Previous research queue has not completed')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.publication_worker','publication',OUT/'queue.json'))
