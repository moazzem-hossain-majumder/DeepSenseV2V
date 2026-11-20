"""Change only the temporal summary consumed by the frozen 192px feature head."""
from src import robust_training as base
from src.robust_training import RobustReranker

class EndpointReranker(RobustReranker):
    def forward(self, rgb, gps):
        delta = rgb[:, -1:] - rgb[:, :1]
        return super().forward(delta, gps)

def run(plan, progress):
    old = base.RobustReranker
    base.RobustReranker = EndpointReranker
    try:
        base.run(plan, progress)
    finally:
        base.RobustReranker = old
