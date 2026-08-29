import os
import sys
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torchvision.io as io
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from src.gps_features import extract_sequence_gps_features, GPSFeatureScaler
from src.beam_reconstruction import linear_to_db, compute_power_gap_db
from src.paths import resolve_raw_data_root, scenario36_csv_path, scenario36_pkl_path

class DeepSenseDataset(Dataset):
    """
    High-Performance Multimodal PyTorch Dataset for DeepSense V2V beam tracking.
    Uses unified in-memory tensor caching to eliminate disk bottlenecks and saturate RTX 5070 GPU.
    Yields:
      - 'rgb': (5, 3, H, W) float32 normalized tensor
      - 'gps': (5, 9) float32 normalized kinematic tensor
      - 'beam_label': int target beam index (0-255)
      - 'profile_db': (256,) float32 tensor of power profile in dB
      - 'gap_db': (256,) float32 tensor of power gaps in dB
      - 'seq_index': int trajectory run ID

    Training augmentation (is_training=True):
      - ColorJitter: brightness ±10%, contrast ±10% applied identically to all 5 frames
        in a sequence so temporal consistency is preserved.
      - NOTE: Horizontal flips are intentionally omitted — flipping would change the
        apparent Tx-Rx bearing and invalidate the GPS-derived beam label geometry.
    """
    def __init__(self, seq_df, gps_array, raw_df, p_dict, rgb_cache, data_root=".", is_training=True):
        self.seq_df = seq_df.reset_index(drop=True)
        self.gps_array = gps_array  # (N_seq, 5, 9)
        self.raw_df = raw_df
        self.p_dict = p_dict
        self.rgb_cache = rgb_cache  # (N_frames, 3, H, W) uint8 in RAM
        self.data_root = data_root
        self.is_training = is_training

        # Precompute full 256-beam power profiles
        pwr1 = np.array(self.p_dict["unit1_pwr1"])
        pwr2 = np.array(self.p_dict["unit1_pwr2"])
        pwr3 = np.array(self.p_dict["unit1_pwr3"])
        pwr4 = np.array(self.p_dict["unit1_pwr4"])
        self.all_pwr_linear = np.concatenate([pwr1, pwr2, pwr3, pwr4], axis=1)
        self.all_pwr_db = linear_to_db(self.all_pwr_linear)
        self.all_gap_db = compute_power_gap_db(self.all_pwr_db)

        # ImageNet normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Train-only ColorJitter (same params sampled once per sequence, applied to all 5 frames)
        self._jitter = T.ColorJitter(brightness=0.1, contrast=0.1)

    def __len__(self):
        return len(self.seq_df)

    def _apply_train_augmentation(self, rgb_float: torch.Tensor) -> torch.Tensor:
        """
        Apply ColorJitter with a single set of parameters sampled once and applied to
        every frame in the 5-frame sequence so temporal consistency is preserved.
        Input/output shape: (5, 3, H, W) float32 in [0, 1].
        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            T.ColorJitter.get_params(
                self._jitter.brightness,
                self._jitter.contrast,
                self._jitter.saturation,
                self._jitter.hue,
            )
        augmented = []
        for frame in rgb_float:  # iterate over 5 frames
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    frame = TF.adjust_brightness(frame, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    frame = TF.adjust_contrast(frame, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    frame = TF.adjust_saturation(frame, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    frame = TF.adjust_hue(frame, hue_factor)
            augmented.append(frame)
        return torch.stack(augmented, dim=0)

    def __getitem__(self, idx):
        row = self.seq_df.iloc[idx]
        x_indices = row["x_indices"]
        y_idx = row["y_index"]

        gps_tensor = torch.from_numpy(self.gps_array[idx]).float()

        # Pure in-memory slice (Zero disk I/O -> 100% GPU saturation)
        raw_rgb = self.rgb_cache[x_indices]  # (5, 3, H, W) uint8
        rgb_float = raw_rgb.float() / 255.0   # scale to [0, 1] before augmentation

        # Apply train-time ColorJitter consistently across all 5 frames
        if self.is_training:
            rgb_float = self._apply_train_augmentation(rgb_float)

        rgb_tensor = (rgb_float - self.mean) / self.std

        target_beam = int(row["target_beam"])
        profile_db = torch.from_numpy(self.all_pwr_db[y_idx]).float()
        gap_db = torch.from_numpy(self.all_gap_db[y_idx]).float()
        seq_idx = int(row["seq_index"])

        return {
            "rgb": rgb_tensor,
            "gps": gps_tensor,
            "beam_label": target_beam,
            "profile_db": profile_db,
            "gap_db": gap_db,
            "seq_index": seq_idx,
        }

def load_or_build_rgb_cache(p_dict, raw_df, data_root=".", img_size=(96, 96)):
    """
    Load or build unified in-memory RGB tensor cache for all 24,799 frames.
    Saves binary cache file to data/processed/rgb_cache_HxW.pt for instantaneous subsequent loading.
    """
    cache_file = os.path.join(os.path.abspath("."), "data", "processed", f"rgb_cache_{img_size[0]}x{img_size[1]}.pt")
    if os.path.exists(cache_file):
        print(f"Loading precomputed RGB tensor cache from {cache_file} (Instant DMA into RAM)...", flush=True)
        cached_tensor = torch.load(cache_file, weights_only=True)
        print(f"Loaded {len(cached_tensor):,} RGB frames into RAM ({cached_tensor.element_size() * cached_tensor.nelement() / (1024**2):.1f} MB)", flush=True)
        return cached_tensor

    if "unit1_rgb5" in p_dict:
        rgb_paths_all = list(p_dict["unit1_rgb5"])
    else:
        rgb_paths_all = list(raw_df["unit1_rgb5"].values)

    N = len(rgb_paths_all)
    print(f"Building high-speed RGB tensor cache ({img_size[0]}x{img_size[1]}) for {N:,} frames using 16 parallel threads...", flush=True)
    all_tensors = [None] * N
    resize_op = T.Resize(img_size, antialias=True)

    def load_frame(i):
        rel_p = rgb_paths_all[i]
        candidates = [
            os.path.join(data_root, "scenario36", rel_p),
            os.path.join(data_root, rel_p),
            os.path.join(data_root, "scenario36", rel_p.replace("scenario36/", "")),
            rel_p
        ]
        chosen = None
        for c in candidates:
            if os.path.exists(c) and os.path.isfile(c):
                chosen = c
                break
        if chosen is not None:
            try:
                t = io.read_image(chosen)
                t = resize_op(t)
                all_tensors[i] = t
                return
            except Exception:
                pass
        all_tensors[i] = torch.zeros((3, img_size[0], img_size[1]), dtype=torch.uint8)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(tqdm(pool.map(load_frame, range(N)), total=N, desc="Preloading RGB frames to RAM"))

    stacked = torch.stack(all_tensors, dim=0)  # (24799, 3, H, W) uint8
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    torch.save(stacked, cache_file)
    print(f"Saved RGB tensor cache to {cache_file} ({stacked.element_size() * stacked.nelement() / (1024**2):.1f} MB)", flush=True)
    return stacked

def prepare_multimodal_data(data_root=".", img_size=(96, 96), processed_dir=None):
    """
    Loads raw metadata, builds sequences, computes/loads cached GPS features,
    preloads RGB image cache into RAM, and returns high-speed datasets for train/val/calib/test.
    Raw files may live outside the git repo; processed caches stay under this project.
    """
    raw_root = resolve_raw_data_root(data_root)
    project_root = os.path.abspath(".")
    if processed_dir is None:
        processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    csv_path = scenario36_csv_path(raw_root)
    pkl_path = scenario36_pkl_path(raw_root)
    manifest_path = os.path.join(processed_dir, "split_manifest.csv")
    cache_gps_path = os.path.join(processed_dir, "gps_features_scaled.npy")
    data_root = raw_root

    print(f"Loading metadata from {csv_path} and {pkl_path}...", flush=True)
    raw_df = pd.read_csv(csv_path)
    with open(pkl_path, "rb") as fp:
        p_dict = pickle.load(fp)

    from src.partitioning import build_partition_manifest
    if os.path.exists(manifest_path):
        seq_df = pd.read_csv(manifest_path)
        if "x_indices_str" in seq_df.columns:
            seq_df["x_indices"] = seq_df["x_indices_str"].apply(lambda s: [int(x) for x in str(s).split(",")])
    else:
        seq_df = build_partition_manifest(data_root=raw_root, output_dir=processed_dir)

    if os.path.exists(cache_gps_path):
        print(f"Loading precomputed scaled GPS features from cache ({cache_gps_path})...", flush=True)
        all_gps_scaled = np.load(cache_gps_path)
    else:
        print("Computing engineered GPS features across all sequences...", flush=True)
        all_gps_feats = []
        for idx_list in tqdm(seq_df["x_indices"], desc="Extracting GPS features"):
            feat = extract_sequence_gps_features(raw_df, idx_list)
            all_gps_feats.append(feat)
        all_gps_feats = np.stack(all_gps_feats, axis=0)

        train_mask = (seq_df["split"] == "train").values
        scaler = GPSFeatureScaler()
        scaler.fit(all_gps_feats[train_mask])
        all_gps_scaled = scaler.transform(all_gps_feats)
        os.makedirs(os.path.dirname(cache_gps_path), exist_ok=True)
        np.save(cache_gps_path, all_gps_scaled)
        print(f"Cached scaled GPS features to {cache_gps_path}", flush=True)

    # Load / build RAM RGB cache
    rgb_cache = load_or_build_rgb_cache(p_dict, raw_df, data_root=data_root, img_size=img_size)

    datasets = {}
    for split_name in ["train", "val", "calib", "test"]:
        mask = (seq_df["split"] == split_name).values
        sub_seq_df = seq_df[mask].reset_index(drop=True)
        sub_gps = all_gps_scaled[mask]
        datasets[split_name] = DeepSenseDataset(
            sub_seq_df, sub_gps, raw_df, p_dict,
            rgb_cache=rgb_cache,
            data_root=data_root,
            is_training=(split_name == "train")
        )

    return datasets, raw_df, p_dict

def get_dataloaders(datasets, batch_size=64, num_workers=0, pin_memory=True):
    loaders = {}
    safe_pin_memory = bool(pin_memory and torch.cuda.is_available())
    for split, ds in datasets.items():
        is_train = (split == "train")
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=safe_pin_memory,
            drop_last=False
        )
    return loaders
