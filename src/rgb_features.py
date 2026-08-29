import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.io as io

# Standard ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_rgb_transforms(img_size=(128, 128), is_training=True):
    """
    Returns torchvision transforms for RGB frames (accepting tensor or PIL).
    Note: Horizontal flips are avoided to preserve V2V spatial beam angular geometry.
    """
    if is_training:
        return T.Compose([
            T.Resize(img_size, antialias=True),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize(img_size, antialias=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

_RESOLVED_PATH_CACHE = {}

def resolve_image_path(rel_path, data_root="."):
    """Resolve and cache full image path to avoid redundant filesystem syscalls."""
    if rel_path in _RESOLVED_PATH_CACHE:
        return _RESOLVED_PATH_CACHE[rel_path]

    candidates = [
        os.path.join(data_root, "scenario36", rel_path),
        os.path.join(data_root, rel_path),
        os.path.join(data_root, "scenario36", rel_path.replace("scenario36/", "")),
        rel_path
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            _RESOLVED_PATH_CACHE[rel_path] = c
            return c
    _RESOLVED_PATH_CACHE[rel_path] = None
    return None

def load_rgb_frame(rel_path, data_root=".", transform=None):
    """
    Fast and safe load of an RGB image into a normalized float32 tensor (3, H, W).
    """
    img_path = resolve_image_path(rel_path, data_root=data_root)
    if img_path is None:
        return torch.zeros((3, 96, 96), dtype=torch.float32)

    try:
        # High-performance C++ libjpeg direct decode to tensor
        raw_tensor = io.read_image(img_path)  # (3, H, W) uint8
        float_tensor = raw_tensor.float() / 255.0
        if transform is not None:
            return transform(float_tensor)
        return float_tensor
    except Exception:
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                t_img = T.ToTensor()(img)
                if transform is not None:
                    return transform(t_img)
                return t_img
        except Exception:
            return torch.zeros((3, 96, 96), dtype=torch.float32)

def load_sequence_rgb_frames(rel_paths, data_root=".", transform=None):
    """
    Load a sequence of RGB frames (e.g. 5 historical frames) into a tensor of shape (5, 3, H, W).
    """
    tensors = [load_rgb_frame(p, data_root=data_root, transform=transform) for p in rel_paths]
    return torch.stack(tensors, dim=0)  # Shape: (seq_len, 3, H, W)

