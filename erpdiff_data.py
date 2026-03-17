"""
ERPDiff — Data loading and dataset construction.

Handles all three datasets:
  Dataset 1 (RSVP):         [N, 1, 1, 63, 128]
  Dataset 2 (row-column):   [E, R, F, 32, 128]
  Dataset 3 (BCI Comp III): [E, 15, 12, 64, 160]
"""

import glob
import os
import pickle
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from erpdiff_config import TrainConfig


def discover_subject_pkls(predata_dir: str) -> List[Tuple[str, str]]:
    pkls = sorted(glob.glob(os.path.join(predata_dir, "*.pkl")))
    if not pkls:
        raise FileNotFoundError(f"No .pkl files found under: {predata_dir}")
    return [(os.path.splitext(os.path.basename(p))[0], p) for p in pkls]


def _coerce_array(arr) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.dtype != object:
        return arr
    try:
        return np.stack(list(arr), axis=0)
    except ValueError:
        pass
    parts = []
    for item in arr.ravel():
        coerced = _coerce_array(item)
        parts.append(coerced)
    try:
        return np.concatenate(parts, axis=0)
    except ValueError as e:
        raise ValueError(
            f"Cannot coerce object array, sub-shapes={[p.shape for p in parts]}: {e}"
        )


def load_subject_pkl(pkl_path: str) -> Dict[str, np.ndarray]:
    with open(pkl_path, "rb") as f:
        eeg = pickle.load(f)
    if "data" not in eeg or "label" not in eeg:
        raise ValueError(f"Bad pkl format: {pkl_path}, keys={list(eeg.keys())}")
    eeg["data"]  = _coerce_array(eeg["data"])
    eeg["label"] = _coerce_array(eeg["label"])
    return eeg


def split_epoch_indices(
    n_epochs: int, seed: int, train_ratio: float, val_ratio_in_train: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n_epochs)
    rng.shuffle(idx)
    n_train   = int(n_epochs * train_ratio)
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]
    rng.shuffle(train_idx)
    n_val     = int(len(train_idx) * val_ratio_in_train)
    val_idx   = train_idx[:n_val]
    train_idx = train_idx[n_val:]
    return train_idx, val_idx, test_idx


class PreDataTrialDataset(Dataset):
    """Trial-level dataset from 5D [E, R, F, C, T] preprocessed data."""

    def __init__(self, subject_id: str, eeg: Dict[str, np.ndarray], epoch_indices: Sequence[int]):
        self.subject_id = subject_id
        self.data  = eeg["data"]
        self.label = eeg["label"]

        if self.data.ndim != 5:
            raise ValueError(f"Expected data ndim=5, got {self.data.shape}")
        if self.label.ndim != 3:
            raise ValueError(f"Expected label ndim=3, got {self.label.shape}")

        E, R, F, C, T = self.data.shape
        if self.label.shape != (E, R, F):
            raise ValueError(f"Label shape mismatch: {self.label.shape} vs data {self.data.shape}")

        self.samples: List[Tuple[np.ndarray, int]] = []
        skipped = 0
        for e in epoch_indices:
            for r in range(R):
                for f in range(F):
                    y = int(self.label[e, r, f])
                    if y < 0:
                        skipped += 1
                        continue
                    x = self.data[e, r, f].astype(np.float32)
                    self.samples.append((x, y))

        if skipped > 0:
            print(f"  [Dataset] {subject_id}: skipped {skipped} samples with label=-1")

        self.C = C
        self.T = T

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        x_tensor = torch.from_numpy(x).unsqueeze(0)
        return x_tensor, torch.tensor(y, dtype=torch.long), self.subject_id


def build_subject_splits(subject_id: str, pkl_path: str, cfg: TrainConfig):
    eeg      = load_subject_pkl(pkl_path)
    n_epochs = eeg["data"].shape[0]
    train_idx, val_idx, test_idx = split_epoch_indices(
        n_epochs, cfg.seed, cfg.train_ratio, cfg.val_ratio_in_train
    )
    train_ds = PreDataTrialDataset(subject_id, eeg, train_idx)
    val_ds   = PreDataTrialDataset(subject_id, eeg, val_idx)
    test_ds  = PreDataTrialDataset(subject_id, eeg, test_idx)
    return train_ds, val_ds, test_ds


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, cfg: TrainConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
