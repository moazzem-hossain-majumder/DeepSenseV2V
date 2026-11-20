"""Spatial EfficientNet features from only the images used by development windows."""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from src.acquire_scenarios import save_json
from src.publication_data import Clips, digest
from src.publication_plan import CACHE as SOURCE, ROOT

CACHE = ROOT / "data/processed/swin_comparison_v1"
CHANNELS = (48, 232)


def prepare(progress):
    CACHE.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((SOURCE / "receipt.json").read_text())
    manifest_path = SOURCE / "manifest.jsonl"
    assert digest(manifest_path) == receipt["manifest_sha256"]
    manifest = pd.read_json(manifest_path, lines=True)
    encoder = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1).features[:7].cuda().eval()
    mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    for scenario in (36, 37, 38, 39):
        csv = ROOT / f"data/raw/scenario{scenario}/scenario{scenario}.csv"
        assert digest(csv) == receipt["csv_sha256"][str(scenario)]
        frame = pd.read_csv(csv)
        subset = manifest[manifest.scenario.eq(scenario) & manifest.split.isin(["train", "val"])]
        ids = np.unique(np.concatenate(subset.x_indices.to_numpy()))
        path = CACHE / f"s{scenario}.npy"
        state = CACHE / f"s{scenario}.json"
        meta = json.loads(state.read_text()) if state.exists() else {"done": 0}
        if "sha256" in meta:
            assert digest(path) == meta["sha256"]
            continue
        features = np.lib.format.open_memmap(path, mode="r+" if meta["done"] else "w+",
            shape=(len(frame), 8, sum(CHANNELS)), dtype=np.float16)

        def decode(i):
            views = []
            for col in ("unit1_rgb5", "unit1_rgb6"):
                with Image.open(csv.parent / frame.at[i, col]) as image:
                    views.append(np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)))
            return views

        with ThreadPoolExecutor(max_workers=6) as pool, torch.inference_mode():
            for first in range(meta["done"], len(ids), 24):
                indices = ids[first:first + 24]
                batch = np.asarray(list(pool.map(decode, indices))).reshape(-1, 224, 224, 3)
                x = torch.from_numpy(batch).cuda().permute(0, 3, 1, 2).float() / 255
                x = (x - mean) / std
                maps = []
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for layer, block in enumerate(encoder):
                        x = block(x)
                        if layer in (3, 6):
                            maps.append(adaptive_avg_pool2d(x, (2, 2)).flatten(2).transpose(1, 2))
                values = torch.cat(maps, dim=-1).reshape(len(indices), 8, sum(CHANNELS)).half()
                if not torch.isfinite(values).all():
                    raise ValueError("Nonfinite image features")
                features[indices] = values.cpu().numpy()
                features.flush()
                done = first + len(indices)
                save_json(state, {"done": done})
                progress(status=f"extracting_S{scenario}", model="analysis", batch=done, batches=len(ids))
        del features
        save_json(state, {"done": len(ids), "sha256": digest(path)})


class SpatialClips(Clips):
    def __init__(self, manifest, gps, power, ids, vision):
        super().__init__(manifest, gps, power, ids, images=False)
        if not manifest.split.iloc[ids].isin(["train", "val"]).all():
            raise ValueError("Development partitions only")
        self.vision = vision

    def __getitem__(self, i):
        blank, *rest = super().__getitem__(i)
        if not self.vision:
            return blank, *rest
        j = self.indices[i]
        scenario = self.scenarios[j]
        if scenario not in self.maps:
            self.maps[scenario] = np.load(CACHE / f"s{scenario}.npy", mmap_mode="r")
        return np.array(self.maps[scenario][self.x[j]], copy=True), *rest
