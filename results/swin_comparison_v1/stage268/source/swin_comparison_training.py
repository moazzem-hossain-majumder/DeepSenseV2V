"""Matched GPS ablation of a paper-inspired two-scale shifted-window fusion model."""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.swin_transformer import SwinTransformerBlock
from src import publication_training as engine
from src.publication_plan import COMMON
from src.swin_comparison_data import CHANNELS, SpatialClips


class Fusion(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.vision = vision
        dim = 32
        self.gps = nn.Linear(9, dim)
        self.position = nn.Parameter(torch.zeros(1, 5, 9, dim))
        nn.init.normal_(self.position, std=.02)
        self.project = nn.ModuleList(nn.Linear(c, dim, bias=False) for c in CHANNELS)
        self.stages = nn.ModuleList(
            nn.Sequential(*(SwinTransformerBlock(dim, 4, [5, 5], [0, 2 if i % 2 else 0], mlp_ratio=2.)
                            for i in range(4))) for _ in CHANNELS)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 256))

    def forward(self, rgb, gps):
        location = self.gps(gps.float()).unsqueeze(2)
        grid = self.position.expand(len(gps), -1, -1, -1)
        views = rgb.split(CHANNELS, dim=-1) if self.vision else None
        for i, stage in enumerate(self.stages):
            visual = self.project[i](views[i].float()) if self.vision else location.new_zeros(len(gps), 5, 8, 32)
            grid = stage(grid + torch.cat((location, visual), dim=2))
        return {"logits": self.head(grid.mean((1, 2)))}


def train(plan, progress):
    config = {**COMMON, "epochs": plan["epochs"], "batch": 96, "workers": 0, "brightness": 0.,
              "size": 224, "seed": plan["seed"], "holdout": plan["holdout"], "variant": "fusion_ce",
              "vision": plan["vision"], "lr": .0004, "weight_decay": .001, "dropout": 0.}
    def loader(m, g, p, ids, cfg, train=False):
        dataset = SpatialClips(m, g, p, ids, cfg["vision"])
        return DataLoader(dataset, batch_size=cfg["batch"], shuffle=train, pin_memory=True)
    # Each queue worker owns its Python process.
    engine.config = lambda _: config
    engine.build_model = lambda cfg: Fusion(cfg["vision"]).cuda()
    engine.loader = loader
    engine.inputs = lambda rgb, gps, *args: (rgb.cuda(), gps.cuda())
    engine.train(plan, progress)
