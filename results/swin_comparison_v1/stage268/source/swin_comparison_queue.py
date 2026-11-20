import sys
from src.swin_comparison_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/widecheck_v1/stage267'):raise ValueError('Stage 267 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.swin_comparison_worker','swin_comparison',OUT/'queue.json'))

