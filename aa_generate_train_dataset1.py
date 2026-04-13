"""
Dataset I (RSVP) preprocessing — outputs dual-format pkl.

Output structure per subject:
    {
        "offline": {
            "data":      np.float32 [N_off, 63, 128],   # [trials, channels, time]
            "label":     np.int64   [N_off],             # 0=nontarget, 1=target
            "trial_idx": np.int64   [N_off],             # character epoch id
            "stim_code": np.int64   [N_off],             # RSVP symbol code
        },
        "online": {
            "data":      np.float32 [N_on, 63, 128],
            "label":     np.int64   [N_on],
            "trial_idx": np.int64   [N_on],
            "stim_code": np.int64   [N_on],
        },
        "meta": { ... },
    }

Label convention:
    The MATLAB field phase_obj.y may use {0,1} or {1,2} encoding depending on
    the subject. We determine the target value by counting: the minority class
    is always target. This is binarized to {0=nontarget, 1=target} at
    preprocessing time so training code never needs to guess.

Data split (per official paper description):
    - train:  offline, 20 randomly chosen characters
    - val:    offline, remaining 4 characters
    - test:   online, all characters
    Split is done in erpdiff_data.py, not here. We just provide the trial_idx
    so the training code can group by character.
"""

import os
import pickle
import traceback
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
from scipy import signal

# =========================
# Configuration
# =========================
INPUT_DIR  = r"D:\files\datasets\erp-dataset1_clean"
OUTPUT_DIR = r"D:\files\datasets\erp-dataset1_preprocessed"

REQUIRED_CHANNELS = [
    'Fp1', 'Fp2',
    'AF3', 'AF4',
    'Fz',
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
    'FCz',
    'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6',
    'FT7', 'FT8',
    'T7', 'T8',
    'Cz',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
    'TP7', 'TP8',
    'CPz',
    'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6',
    'Pz',
    'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
    'POz',
    'PO3', 'PO4', 'PO7', 'PO8', 'PO9', 'PO10',
    'Oz', 'O1', 'O2'
]

TARGET_FS    = 128
LOWCUT       = 0.5
HIGHCUT      = 30.0
FILTER_ORDER = 4
WINDOW_SEC   = 1.0


# =========================
# Utilities
# =========================

def matlab_str(x) -> str:
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore").strip()
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return matlab_str(x.item())
        if x.dtype.kind in {"U", "S"}:
            return "".join(x.tolist()).strip()
        return " ".join([matlab_str(i) for i in x.ravel()]).strip()
    return str(x).strip()


def load_subject_data(mat_path: str):
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "data" not in mat:
        raise KeyError(f"{os.path.basename(mat_path)}: missing 'data'")
    phases = np.atleast_1d(mat["data"]).ravel().tolist()
    if len(phases) < 2:
        raise ValueError(f"{os.path.basename(mat_path)}: expected 1x2 cell")
    return phases[0], phases[1]


def get_channels_from_phase(phase_obj) -> List[str]:
    arr = np.atleast_1d(phase_obj.channels).ravel()
    return [c for c in (matlab_str(x) for x in arr) if c != ""]


def ensure_time_channel_shape(X, num_channels: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X shape unexpected: {X.shape}")
    if X.shape[1] == num_channels:
        return X
    if X.shape[0] == num_channels:
        return X.T
    raise ValueError(f"X shape {X.shape} doesn't match {num_channels} channels")


def bandpass_filter(X_tc: np.ndarray, fs: int,
                    low: float = LOWCUT, high: float = HIGHCUT,
                    order: int = FILTER_ORDER) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="bandpass")
    return signal.filtfilt(b, a, X_tc, axis=0)


def zscore_per_channel(epoch_ct: np.ndarray) -> np.ndarray:
    mean = epoch_ct.mean(axis=1, keepdims=True)
    std  = epoch_ct.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    return (epoch_ct - mean) / std


def _binarize_labels(y_raw: np.ndarray) -> np.ndarray:
    """
    Convert any binary encoding to {0=nontarget, 1=target}.

    Strategy: the minority class is always target.
    Handles {0,1}, {1,2}, {-1,1}, or any other 2-value encoding.
    """
    y = np.asarray(y_raw).ravel()
    uniq = np.unique(y)

    if uniq.size == 1:
        # Degenerate: all same label
        return np.zeros_like(y, dtype=np.int64)

    if uniq.size != 2:
        raise ValueError(f"Expected binary labels, got unique: {uniq.tolist()}")

    val_a, val_b = uniq[0], uniq[1]
    count_a = (y == val_a).sum()
    count_b = (y == val_b).sum()

    # Minority = target (1), majority = nontarget (0)
    target_val = val_a if count_a < count_b else val_b
    return (y == target_val).astype(np.int64)


# =========================
# Core preprocessing
# =========================

def preprocess_phase(phase_obj, phase_name: str = "offline") -> Dict:
    """
    Process one phase (offline or online).

    Returns dict with:
        data:      [N, C, T] float32
        label:     [N] int64, 0=nontarget, 1=target
        trial_idx: [N] int64, character epoch id
        stim_code: [N] int64, RSVP symbol code
    """
    channels = get_channels_from_phase(phase_obj)

    missing = [c for c in REQUIRED_CHANNELS if c not in channels]
    extra   = [c for c in channels if c not in REQUIRED_CHANNELS]
    if missing or extra or len(channels) != 63:
        raise ValueError(
            f"[{phase_name}] channel error: num={len(channels)}, "
            f"missing={missing}, extra={extra}"
        )

    X = ensure_time_channel_shape(phase_obj.X, len(channels))
    ch2idx      = {ch: i for i, ch in enumerate(channels)}
    reorder_idx = [ch2idx[ch] for ch in REQUIRED_CHANNELS]
    X = X[:, reorder_idx]

    fs = int(np.asarray(phase_obj.fs).squeeze())
    X = bandpass_filter(X, fs, low=LOWCUT, high=HIGHCUT, order=FILTER_ORDER)

    if fs != TARGET_FS:
        X = signal.resample_poly(X, up=TARGET_FS, down=fs, axis=0)

    trial      = np.asarray(phase_obj.trial).ravel().astype(np.int64)
    y_raw      = np.asarray(phase_obj.y).ravel()
    y_trialIdx = np.asarray(phase_obj.y_trialIdx).ravel()
    if not hasattr(phase_obj, "y_stim"):
        raise ValueError(f"[{phase_name}] missing field 'y_stim' for stimulus codes")
    y_stim = np.asarray(phase_obj.y_stim).ravel()

    trial0   = trial - 1
    trial_ds = np.round(trial0 * TARGET_FS / fs).astype(np.int64)
    win_len  = int(WINDOW_SEC * TARGET_FS)

    data_list  = []
    label_raw_list = []
    tidx_list  = []
    stim_list  = []
    dropped    = 0

    for i, start in enumerate(trial_ds):
        end = start + win_len
        if start < 0 or end > X.shape[0]:
            dropped += 1
            continue
        epoch = X[start:end, :].T
        epoch = zscore_per_channel(epoch)
        data_list.append(epoch.astype(np.float32))
        label_raw_list.append(float(y_raw[i]))
        tidx_list.append(int(np.asarray(y_trialIdx[i]).squeeze()))
        stim_list.append(int(np.asarray(y_stim[i]).squeeze()))

    if not data_list:
        raise RuntimeError(f"[{phase_name}] no events extracted")

    data_3d   = np.stack(data_list, axis=0)                    # [N, C, T]
    label_raw = np.asarray(label_raw_list)
    label_bin = _binarize_labels(label_raw)                    # [N] {0, 1}
    trial_idx = np.asarray(tidx_list, dtype=np.int64)          # [N]
    stim_code = np.asarray(stim_list, dtype=np.int64)          # [N]

    n_target = int(label_bin.sum())
    n_total  = len(label_bin)
    raw_uniq = np.unique(label_raw)
    print(f"    {phase_name}: {n_total} events (dropped={dropped}), "
          f"raw labels={raw_uniq.tolist()} -> target={n_target}, "
          f"nontarget={n_total - n_target}")

    return {
        "data":      data_3d,
        "label":     label_bin,
        "trial_idx": trial_idx,
        "stim_code": stim_code,
    }


# =========================
# Per-subject entry point
# =========================

def preprocess_one_subject(mat_path: str) -> Dict:
    offline_obj, online_obj = load_subject_data(mat_path)
    off = preprocess_phase(offline_obj, phase_name="offline")
    on  = preprocess_phase(online_obj,  phase_name="online")

    return {
        "offline": off,
        "online":  on,
        "meta": {
            "dataset_name":     "dataset1",
            "source_file":      os.path.basename(mat_path),
            "paradigm":         "RSVP",
            "num_channels":     len(REQUIRED_CHANNELS),
            "sample_rate":      TARGET_FS,
            "window_sec":       WINDOW_SEC,
            "window_points":    int(WINDOW_SEC * TARGET_FS),
            "filtering":        f"{LOWCUT}-{HIGHCUT}Hz bandpass, order={FILTER_ORDER}",
            "normalization":    "per-trial per-channel zscore",
            "label_convention": "0=nontarget, 1=target (binarized from raw at preprocessing time)",
            "n_offline":        off["data"].shape[0],
            "n_online":         on["data"].shape[0],
            "offline_chars":    int(np.unique(off["trial_idx"]).size),
            "online_chars":     int(np.unique(on["trial_idx"]).size),
            "offline_stim_codes": int(np.unique(off["stim_code"]).size),
            "online_stim_codes":  int(np.unique(on["stim_code"]).size),
        },
    }


# =========================
# Main
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if os.path.isfile(os.path.join(INPUT_DIR, f))
    )
    if not files:
        print(f"[ERROR] No files found in {INPUT_DIR}")
        return

    for fp in files:
        subject_name = os.path.splitext(os.path.basename(fp))[0]
        print(f"\n[PROCESS] {subject_name}")
        try:
            result = preprocess_one_subject(fp)
            out_path = os.path.join(OUTPUT_DIR, subject_name + ".pkl")
            with open(out_path, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

            m = result["meta"]
            print(f"  offline: {m['n_offline']} events, {m['offline_chars']} chars")
            print(f"  online:  {m['n_online']} events, {m['online_chars']} chars")
            print(f"  saved -> {out_path}")
        except Exception as e:
            print(f"  [ERROR] {subject_name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
