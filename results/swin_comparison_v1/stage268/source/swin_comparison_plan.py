from src.publication_plan import ROOT

OUT = ROOT / "results/swin_comparison_v1"
PLAN = []


def entry(kind, title, **config):
    sid = 268 + len(PLAN)
    PLAN.append(dict(id=sid, kind=kind, title=f"Stage {sid} - {title}",
        directory=f"results/swin_comparison_v1/stage{sid}",
        models=["fusion_ce" if kind == "train" else "analysis"],
        epochs=12 if kind == "train" else 1, unit="epochs" if kind == "train" else "analysis passes", **config))


entry("prepare", "Freeze protocol and extract EfficientNet-B3 features")
for seed in (42, 7, 13):
    for held in (None, 36, 37, 38, 39):
        for vision in (False, True):
            label = "GPS + RGB" if vision else "GPS"
            split = f"excluded S{held}" if held else "pooled"
            entry("train", f"Shifted-window {label}, {split}, seed {seed}", seed=seed, holdout=held, vision=vision)
entry("report", "Paired three-seed comparison with GPS and geometry")
