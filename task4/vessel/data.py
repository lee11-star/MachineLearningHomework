from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset


def image_ids(folder):
    return sorted((p.stem for p in Path(folder).iterdir() if p.suffix.lower() in {'.jpg', '.png'}), key=int)


def split_ids(ids, seed=42):
    shuffled = np.random.default_rng(seed).permutation(ids).tolist()
    n_val = max(1, len(ids) // 5)
    return sorted(shuffled[n_val:], key=int), sorted(shuffled[:n_val], key=int)


def _image_path(folder, case_id):
    return next(p for p in Path(folder).iterdir() if p.stem == str(case_id) and p.suffix.lower() in {'.jpg', '.png'})


def load_case(root, split, case_id):
    root = Path(root)
    rgb = np.asarray(Image.open(_image_path(root / split / 'image', case_id)).convert('RGB'), dtype=np.float32) / 255
    field = rgb.max(axis=2) > (20 / 255)
    pixels = rgb[field]
    normalized = (rgb - pixels.mean(axis=0)) / np.maximum(pixels.std(axis=0), 0.02) / 3
    green = rgb[:, :, 1]
    contrast = np.clip((green - gaussian_filter(green, sigma=8)) * 8, -1, 1)
    features = np.concatenate([np.clip(normalized, -1, 1), contrast[:, :, None]], axis=2)
    features[~field] = 0
    label = None
    if split == 'train':
        label = (np.asarray(Image.open(_image_path(root / split / 'label', case_id)).convert('L')) > 128).astype(np.float32)
        if label.shape != field.shape:
            raise ValueError(f'Image/label size mismatch for {case_id}')
    return np.ascontiguousarray(features.transpose(2, 0, 1)), field, label


class PatchDataset(Dataset):
    def __init__(self, cases, length, patch_size=128):
        self.cases = cases
        self.length = length
        self.patch_size = patch_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        features, _, label = self.cases[np.random.randint(len(self.cases))]
        size = self.patch_size
        height, width = label.shape
        y = np.random.randint(height - size + 1)
        x = np.random.randint(width - size + 1)
        patch = features[:, y:y + size, x:x + size].copy()
        target = label[None, y:y + size, x:x + size].copy()
        turns = np.random.randint(4)
        patch = np.rot90(patch, turns, axes=(1, 2))
        target = np.rot90(target, turns, axes=(1, 2))
        if np.random.random() < 0.5:
            patch, target = patch[:, :, ::-1], target[:, :, ::-1]
        patch = patch * np.random.uniform(0.85, 1.15, (4, 1, 1)).astype(np.float32)
        patch[:3] += np.random.uniform(-0.04, 0.04, (3, 1, 1)).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(patch)), torch.from_numpy(np.ascontiguousarray(target))
