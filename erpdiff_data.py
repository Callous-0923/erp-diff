"""
ERPDiff — Data loading and dataset construction.

Design philosophy: preprocessing scripts do the hard work (filtering, label
binarization, phase separation). This module just loads clean data and slices.

Supports:
  Dataset 1 (RSVP):         pkl with {"offline": {...}, "online": {...}}
  Dataset 2 (row-column):   pkl with {"data": 5D, "label": 3D}
  Dataset 3 (BCI Comp III): pkl with {"data": 5D, "label": 3D}

Interface consumed by erpdiff_train.py:
  discover_subject_pkls(dir) -> [(sid, path), ...]
  build_subject_splits(sid, path, cfg) -> (train_ds, val_ds, test_ds)
  make_loader(dataset, batch_size, shuffle, cfg) -> DataLoader
  dataset.C, dataset.T  (channel count, time steps)
  __getitem__ returns (x_tensor, y_tensor, ...)  where batch[0]=x, batch[1]=y
"""

import glob
import os
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from erpdiff_config import TrainConfig


# ============================================================
# Discovery
# ============================================================

def discover_subject_pkls(predata_dir: str) -> List[Tuple[str, str]]:
    pkls = sorted(glob.glob(os.path.join(predata_dir, "*.pkl")))
    if not pkls:
        raise FileNotFoundError(f"No .pkl files found under: {predata_dir}")
    return [(os.path.splitext(os.path.basename(p))[0], p) for p in pkls]


# ============================================================
# Dataset class
# ============================================================

class EEGTrialsDataset(Dataset):
    """
    Simple trial dataset. Stores data as torch.Tensor (not numpy reference).

    Each sample returns (x [1, C, T], y scalar, subject_id).
    Pickle-safe: only contains self-owned Tensors, no external array references.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, subject_id: str = ""):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        if x.ndim != 3:
            raise ValueError(f"Expected x [N, C, T], got {x.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x/y length mismatch: {x.shape[0]} vs {y.shape[0]}")
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.subject_id = subject_id
        self.C = int(x.shape[1])
        self.T = int(x.shape[2])

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx].unsqueeze(0), self.y[idx], self.subject_id


# ============================================================
# Pickle loading + format detection
# ============================================================

def _load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in {path}, got {type(obj)}")
    return obj


def _is_dataset1(obj: dict) -> bool:
    """Dataset 1 dual-format: has 'offline' and 'online' sub-dicts."""
    return "offline" in obj and "online" in obj


def _looks_like_dataset2(obj: dict, data: Optional[np.ndarray] = None) -> bool:
    """Heuristic for Hoffmann dataset2 after preprocessing."""
    if "com" in obj:
        return True
    if data is None and "data" in obj:
        data = _coerce_array(obj["data"])
    return data is not None and data.ndim == 5 and data.shape[1] == 20 and data.shape[2] == 6


# ============================================================
# Dataset 1: official character-level split
# ============================================================

def _build_dataset1_splits(
    subject_id: str, obj: dict, seed: int, n_train_chars: int = 20,
) -> Tuple[EEGTrialsDataset, EEGTrialsDataset, EEGTrialsDataset]:
    """
    Official split per paper:
      train: offline, 20 randomly chosen characters
      val:   offline, remaining 4 characters
      test:  online, all data
    """
    offline = obj["offline"]
    online  = obj["online"]

    x_off = np.asarray(offline["data"], dtype=np.float32)    # [N, C, T]
    y_off = np.asarray(offline["label"], dtype=np.int64)      # [N]
    idx_off = np.asarray(offline["trial_idx"]).ravel()         # [N]

    x_on = np.asarray(online["data"], dtype=np.float32)       # [N, C, T]
    y_on = np.asarray(online["label"], dtype=np.int64)         # [N]

    # Character-level split
    chars = np.unique(idx_off)
    if len(chars) < n_train_chars + 1:
        raise ValueError(
            f"[{subject_id}] only {len(chars)} offline chars, "
            f"need at least {n_train_chars + 1}"
        )

    rng = np.random.RandomState(seed)
    shuffled = chars.copy()
    rng.shuffle(shuffled)
    train_chars = set(shuffled[:n_train_chars].tolist())
    val_chars   = set(shuffled[n_train_chars:].tolist())

    train_mask = np.isin(idx_off, list(train_chars))
    val_mask   = np.isin(idx_off, list(val_chars))

    train_ds = EEGTrialsDataset(x_off[train_mask], y_off[train_mask], subject_id)
    val_ds   = EEGTrialsDataset(x_off[val_mask],   y_off[val_mask],   subject_id)
    test_ds  = EEGTrialsDataset(x_on,              y_on,              subject_id)

    print(
        f"  [{subject_id}] D1 split: "
        f"train={len(train_ds)}(tar={int(train_ds.y.sum())}) "
        f"val={len(val_ds)}(tar={int(val_ds.y.sum())}) "
        f"test={len(test_ds)}(tar={int(test_ds.y.sum())})"
    )
    return train_ds, val_ds, test_ds


# ============================================================
# Dataset 2/3: generic 5D epoch-level split
# ============================================================

def _coerce_array(arr) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.dtype != object:
        return arr
    try:
        return np.stack(list(arr), axis=0)
    except ValueError:
        pass
    parts = [_coerce_array(item) for item in arr.ravel()]
    return np.concatenate(parts, axis=0)


def _flatten_5d(data_5d: np.ndarray, label_3d: np.ndarray,
                epoch_indices: Sequence[int],
                return_meta: bool = False,
                ):
    """
    Flatten selected epochs from [E, R, F, C, T] / [E, R, F] to [N, C, T] / [N].
    Skips label < 0 (placeholder).

    If return_meta=True, also returns (runs, reps, flashes) arrays for
    command accuracy computation in the benchmark.
    """
    E, R, F, C, T = data_5d.shape
    x_list, y_list = [], []
    run_list, rep_list, flash_list = [], [], []
    for e in epoch_indices:
        for r in range(R):
            for f in range(F):
                y = int(label_3d[e, r, f])
                if y < 0:
                    continue
                x_list.append(data_5d[e, r, f])
                y_list.append(y)
                if return_meta:
                    run_list.append(e)
                    rep_list.append(r)
                    flash_list.append(f)
    if not x_list:
        raise ValueError("No valid trials after flattening")
    x_out = np.stack(x_list, axis=0).astype(np.float32)
    y_out = np.asarray(y_list, dtype=np.int64)
    if return_meta:
        return (
            x_out, y_out,
            np.asarray(run_list, dtype=np.int64),
            np.asarray(rep_list, dtype=np.int64),
            np.asarray(flash_list, dtype=np.int64),
        )
    return x_out, y_out


class EEGTrialsWithMetaDataset(Dataset):
    """
    Trial dataset with (run, rep, flash) metadata for command accuracy.

    Each sample returns (x [1, C, T], y scalar, run, rep, flash).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 runs: np.ndarray, reps: np.ndarray, flashes: np.ndarray,
                 subject_id: str = ""):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.runs = torch.from_numpy(np.asarray(runs, dtype=np.int64))
        self.reps = torch.from_numpy(np.asarray(reps, dtype=np.int64))
        self.flashes = torch.from_numpy(np.asarray(flashes, dtype=np.int64))
        self.subject_id = subject_id
        self.C = int(x.shape[1])
        self.T = int(x.shape[2])

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return (self.x[idx].unsqueeze(0), self.y[idx],
                self.runs[idx], self.reps[idx], self.flashes[idx])


def _build_generic_splits(
    subject_id: str, obj: dict, seed: int,
    train_ratio: float, val_ratio_in_train: float,
) -> Tuple[EEGTrialsDataset, EEGTrialsDataset, EEGTrialsDataset]:
    """Dataset 2/3: random epoch-level split on 5D data."""
    data  = _coerce_array(obj["data"])
    label = _coerce_array(obj["label"])

    if data.ndim != 5:
        raise ValueError(f"Expected 5D data, got {data.shape}")
    if label.ndim != 3:
        raise ValueError(f"Expected 3D label, got {label.shape}")

    n_epochs = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(n_epochs)
    rng.shuffle(idx)

    n_train = int(n_epochs * train_ratio)
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]
    rng.shuffle(train_idx)
    n_val = int(len(train_idx) * val_ratio_in_train)
    val_idx   = train_idx[:n_val]
    train_idx = train_idx[n_val:]

    x_tr, y_tr = _flatten_5d(data, label, train_idx)
    x_va, y_va = _flatten_5d(data, label, val_idx)
    x_te, y_te = _flatten_5d(data, label, test_idx)

    return (
        EEGTrialsDataset(x_tr, y_tr, subject_id),
        EEGTrialsDataset(x_va, y_va, subject_id),
        EEGTrialsDataset(x_te, y_te, subject_id),
    )


def _build_dataset2_splits(
    subject_id: str,
    obj: dict,
    seed: int,
    n_train_runs: int = 10,
    n_val_runs: int = 4,
    n_test_runs: int = 10,
) -> Tuple[EEGTrialsDataset, EEGTrialsDataset, EEGTrialsDataset]:
    """
    Dataset2 split aligned to Jin et al. (2024) / MOCNN:
      10 runs train, 4 runs val, 10 runs test.
    """
    data = _coerce_array(obj["data"])
    label = _coerce_array(obj["label"])

    if data.ndim != 5:
        raise ValueError(f"Expected 5D data, got {data.shape}")
    if label.ndim != 3:
        raise ValueError(f"Expected 3D label, got {label.shape}")

    n_epochs = data.shape[0]
    total_needed = n_train_runs + n_val_runs + n_test_runs
    if n_epochs < total_needed:
        raise ValueError(
            f"[{subject_id}] dataset2 requires at least {total_needed} runs for 10/4/10 split, got {n_epochs}"
        )

    rng = np.random.RandomState(seed)
    idx = np.arange(n_epochs)
    rng.shuffle(idx)

    train_idx = idx[:n_train_runs]
    val_idx = idx[n_train_runs : n_train_runs + n_val_runs]
    test_idx = idx[n_train_runs + n_val_runs : n_train_runs + n_val_runs + n_test_runs]

    x_tr, y_tr = _flatten_5d(data, label, train_idx)
    x_va, y_va = _flatten_5d(data, label, val_idx)
    x_te, y_te = _flatten_5d(data, label, test_idx)

    print(
        f"  [{subject_id}] D2 split (MOCNN 10/4/10): "
        f"train_runs={len(train_idx)} val_runs={len(val_idx)} test_runs={len(test_idx)} | "
        f"train={len(x_tr)} val={len(x_va)} test={len(x_te)}"
    )
    return (
        EEGTrialsDataset(x_tr, y_tr, subject_id),
        EEGTrialsDataset(x_va, y_va, subject_id),
        EEGTrialsDataset(x_te, y_te, subject_id),
    )


def build_generic_splits_with_meta(
    subject_id: str, pkl_path: str, seed: int,
    train_ratio: float = 0.8, val_ratio_in_train: float = 0.2,
    dataset_name: Optional[str] = None,
) -> Tuple[EEGTrialsWithMetaDataset, EEGTrialsWithMetaDataset, EEGTrialsWithMetaDataset]:
    """
    Build splits with (run, rep, flash) metadata for command accuracy.
    Only applicable to dataset2/3 (5D format).
    """
    obj = _load_pkl(pkl_path)
    if _is_dataset1(obj):
        raise ValueError("Dataset1 does not support command accuracy (no 5D structure)")

    data  = _coerce_array(obj["data"])
    label = _coerce_array(obj["label"])

    if dataset_name == "dataset2" or (dataset_name is None and _looks_like_dataset2(obj, data)):
        n_epochs = data.shape[0]
        if n_epochs < 24:
            raise ValueError(f"[{subject_id}] dataset2 requires 24 runs, got {n_epochs}")

        rng = np.random.RandomState(seed)
        idx = np.arange(n_epochs)
        rng.shuffle(idx)

        train_idx = idx[:10]
        val_idx = idx[10:14]
        test_idx = idx[14:24]

        x_tr, y_tr, r_tr, rp_tr, f_tr = _flatten_5d(data, label, train_idx, return_meta=True)
        x_va, y_va, r_va, rp_va, f_va = _flatten_5d(data, label, val_idx, return_meta=True)
        x_te, y_te, r_te, rp_te, f_te = _flatten_5d(data, label, test_idx, return_meta=True)

        return (
            EEGTrialsWithMetaDataset(x_tr, y_tr, r_tr, rp_tr, f_tr, subject_id),
            EEGTrialsWithMetaDataset(x_va, y_va, r_va, rp_va, f_va, subject_id),
            EEGTrialsWithMetaDataset(x_te, y_te, r_te, rp_te, f_te, subject_id),
        )

    n_epochs = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(n_epochs)
    rng.shuffle(idx)

    n_train = int(n_epochs * train_ratio)
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]
    rng.shuffle(train_idx)
    n_val = int(len(train_idx) * val_ratio_in_train)
    val_idx   = train_idx[:n_val]
    train_idx = train_idx[n_val:]

    x_tr, y_tr, r_tr, rp_tr, f_tr = _flatten_5d(data, label, train_idx, return_meta=True)
    x_va, y_va, r_va, rp_va, f_va = _flatten_5d(data, label, val_idx, return_meta=True)
    x_te, y_te, r_te, rp_te, f_te = _flatten_5d(data, label, test_idx, return_meta=True)

    return (
        EEGTrialsWithMetaDataset(x_tr, y_tr, r_tr, rp_tr, f_tr, subject_id),
        EEGTrialsWithMetaDataset(x_va, y_va, r_va, rp_va, f_va, subject_id),
        EEGTrialsWithMetaDataset(x_te, y_te, r_te, rp_te, f_te, subject_id),
    )


# ============================================================
# Unified entry point
# ============================================================

def build_subject_splits(
    subject_id: str, pkl_path: str, cfg: TrainConfig,
) -> Tuple[EEGTrialsDataset, EEGTrialsDataset, EEGTrialsDataset]:
    """
    Auto-detect pkl format and build train/val/test splits.
    Interface matches erpdiff_train.py expectations.
    """
    obj = _load_pkl(pkl_path)

    if _is_dataset1(obj):
        return _build_dataset1_splits(subject_id, obj, cfg.seed)

    if cfg.dataset == "dataset2" or _looks_like_dataset2(obj):
        return _build_dataset2_splits(subject_id, obj, cfg.seed)

    return _build_generic_splits(
        subject_id, obj, cfg.seed,
        cfg.train_ratio, cfg.val_ratio_in_train,
    )


# ============================================================
# DataLoader with optional balanced sampling
# ============================================================

_IMBALANCE_THRESHOLD = 0.10


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    cfg: TrainConfig,
    balance: bool | None = None,
) -> DataLoader:
    """
    Create DataLoader. Balanced sampling auto-enabled when minority class < 10%.

    Parameters
    ----------
    balance : True=force on, False=force off, None=auto-detect
    """
    sampler = None

    if shuffle and balance is not False:
        labels = _extract_labels(dataset)
        if labels is not None:
            counts = torch.bincount(labels, minlength=2).float().clamp_min(1.0)
            minority_ratio = float(counts.min() / counts.sum())

            if balance is True or minority_ratio < _IMBALANCE_THRESHOLD:
                inv_freq = 1.0 / counts
                sample_weights = inv_freq[labels].double()
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(labels),
                    replacement=True,
                )
                shuffle = False
                minority_cls = int(counts.argmin())
                print(
                    f"  [Loader] Balanced sampling ON: "
                    f"class{minority_cls}={int(counts[minority_cls])}/"
                    f"{int(counts.sum())} ({minority_ratio:.1%})"
                )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=getattr(cfg, "num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )


def _extract_labels(dataset: Dataset):
    """Extract label tensor from dataset or ConcatDataset."""
    if hasattr(dataset, "y"):
        y = dataset.y
        return y.long() if isinstance(y, torch.Tensor) else torch.as_tensor(y, dtype=torch.long)
    if isinstance(dataset, ConcatDataset):
        parts = []
        for ds in dataset.datasets:
            if not hasattr(ds, "y"):
                return None
            y = ds.y
            parts.append(y.long() if isinstance(y, torch.Tensor) else torch.as_tensor(y, dtype=torch.long))
        return torch.cat(parts)
    return None
