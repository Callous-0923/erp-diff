import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.io as sio
from scipy import signal as sig

# ============================================================
# Dataset III-II (BCI Competition III Wadsworth 2004)
# Revised for the actual MAT structure:
#   Signal       : [n_epochs, n_samples, 64]
#   Flashing     : [n_epochs, n_samples]
#   StimulusCode : [n_epochs, n_samples]
#   StimulusType : [n_epochs, n_samples]   (train only)
#   TargetChar   : [n_epochs, n_samples]   (train only)
#
# Official extraction rule:
# - Use stimulus onset when Flashing changes from 0 -> 1.
# - The first sample of each epoch is also a valid onset when Flashing[0] == 1.
# - Accumulate trials into 12 stimulus-code buffers.
# - Each code should appear 15 times in one character epoch.
# - We keep the 0-667 ms window used by the ITSEF paper: 160 samples at 240 Hz.
#
# Preprocessing pipeline (per epoch, before trial extraction):
#   1. 60 Hz notch filter (US power line, Q=35)
#   2. 0.5-30 Hz Butterworth bandpass (order=4, zero-phase filtfilt)
#   3. Per-trial per-channel z-score normalization
# ============================================================

INPUT_DIR = r"D:\files\datasets\BCI_Comp_III_Wads_2004"
OUTPUT_DIR = r"D:\files\datasets\BCI_Comp_III_Wads_2004_preprocessed"

SAMPLE_RATE = 240
WINDOW_SAMPLES = 160  # 0-667 ms at 240 Hz
NUM_CHANNELS = 64
NUM_CODES = 12
NUM_REPEATS = 15
EPS = 1e-8

# --- Filtering parameters ---
LOWCUT = 0.5          # Hz, high-pass cutoff
HIGHCUT = 30.0        # Hz, low-pass cutoff
FILTER_ORDER = 4      # Butterworth order
NOTCH_FREQ = 60.0     # Hz, US power line frequency
NOTCH_Q = 35.0        # Quality factor for notch filter


def _bandpass_filter(X: np.ndarray, fs: int,
                     low: float = LOWCUT, high: float = HIGHCUT,
                     order: int = FILTER_ORDER) -> np.ndarray:
    """
    带通滤波，X: [n_samples, n_channels]。
    使用 zero-phase filtfilt 避免相位偏移。
    """
    nyq = 0.5 * fs
    b, a = sig.butter(order, [low / nyq, high / nyq], btype="bandpass")
    return sig.filtfilt(b, a, X, axis=0).astype(X.dtype)


def _notch_filter(X: np.ndarray, fs: int,
                  freq: float = NOTCH_FREQ, Q: float = NOTCH_Q) -> np.ndarray:
    """
    陷波滤波去除工频噪声，X: [n_samples, n_channels]。
    BCI Competition III 数据在美国采集，工频 60 Hz。
    """
    b, a = sig.iirnotch(freq, Q, fs)
    return sig.filtfilt(b, a, X, axis=0).astype(X.dtype)


def _safe_scalar_str(x: Any) -> str:
    arr = np.asarray(x)
    if arr.size == 0:
        return ""
    if arr.dtype.kind in {"U", "S"}:
        return "".join(arr.astype(str).ravel().tolist()).strip()
    if arr.dtype == object:
        vals = []
        for item in arr.ravel():
            vals.append(_safe_scalar_str(item))
        return "".join(vals).strip()
    if arr.size == 1:
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="ignore").strip()
        return str(item).strip()
    return str(arr.squeeze()).strip()


def _extract_target_chars(target_char_mat: np.ndarray, n_epochs: int) -> np.ndarray:
    """
    Normalize TargetChar into shape [n_epochs] with one character string per epoch.

    Known layouts in this dataset include:
    - shape (1,), single concatenated string of length n_epochs
    - shape (n_epochs,), one char per epoch
    - shape (n_epochs, n_samples), repeated char in each epoch row
    """
    tc = np.asarray(target_char_mat, dtype=object)

    if tc.size == 0:
        return np.asarray([''] * int(n_epochs), dtype=object)

    flat = tc.reshape(-1)
    if flat.size == 1:
        s = _safe_scalar_str(flat[0])
        if len(s) >= int(n_epochs):
            return np.asarray(list(s[:int(n_epochs)]), dtype=object)
        return np.asarray([s[:1] if s else ''] * int(n_epochs), dtype=object)

    if tc.ndim >= 2 and tc.shape[0] == int(n_epochs):
        vals: List[str] = []
        for i in range(int(n_epochs)):
            row = np.asarray(tc[i]).ravel()
            chosen = ''
            for item in row:
                s = _safe_scalar_str(item)
                if s:
                    chosen = s[0]
                    break
            vals.append(chosen)
        return np.asarray(vals, dtype=object)

    vals = []
    for item in flat.tolist()[:int(n_epochs)]:
        s = _safe_scalar_str(item)
        vals.append(s[0] if s else '')
    if len(vals) < int(n_epochs):
        vals.extend([''] * (int(n_epochs) - len(vals)))
    return np.asarray(vals, dtype=object)


def _zscore_per_channel(x_ct: np.ndarray) -> np.ndarray:
    x = np.asarray(x_ct, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < EPS, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def _rising_edges(flashing_1d: np.ndarray) -> np.ndarray:
    f = np.asarray(flashing_1d).astype(np.int32).reshape(-1)
    if f.size == 0:
        return np.empty((0,), dtype=np.int64)
    edges = np.flatnonzero((f[1:] == 1) & (f[:-1] == 0)) + 1
    if f[0] == 1:
        edges = np.concatenate(([0], edges))
    return edges.astype(np.int64)


def _validate_train_mat_shapes(mat: Dict[str, Any], mat_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ["Signal", "Flashing", "StimulusCode", "StimulusType"]
    for key in required:
        if key not in mat:
            raise KeyError(f"{os.path.basename(mat_path)} missing required variable: {key}")

    signal = np.asarray(mat["Signal"], dtype=np.float32)
    flashing = np.asarray(mat["Flashing"])
    stim_code = np.asarray(mat["StimulusCode"])
    stim_type = np.asarray(mat["StimulusType"])
    target_char = np.asarray(mat["TargetChar"]) if "TargetChar" in mat else None

    if signal.ndim != 3:
        raise ValueError(f"Signal must be [epochs, samples, channels], got {signal.shape}")
    if signal.shape[2] != NUM_CHANNELS:
        raise ValueError(f"Signal last dim must be {NUM_CHANNELS}, got {signal.shape}")
    if flashing.shape != signal.shape[:2]:
        raise ValueError(f"Flashing shape mismatch: {flashing.shape} vs {signal.shape[:2]}")
    if stim_code.shape != signal.shape[:2]:
        raise ValueError(f"StimulusCode shape mismatch: {stim_code.shape} vs {signal.shape[:2]}")
    if stim_type.shape != signal.shape[:2]:
        raise ValueError(f"StimulusType shape mismatch: {stim_type.shape} vs {signal.shape[:2]}")

    if target_char is None:
        target_char_vec = np.asarray([""] * signal.shape[0], dtype=object)
    else:
        target_char_vec = _extract_target_chars(target_char, int(signal.shape[0]))
    return signal, flashing, stim_code, stim_type, target_char_vec


def _extract_one_epoch(
    signal_epoch: np.ndarray,
    flashing_epoch: np.ndarray,
    stim_code_epoch: np.ndarray,
    stim_type_epoch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    x_epoch: [15, 12, 64, 160]
    y_epoch: [15, 12]
    onset_epoch: [15, 12]   original onset sample indices within the epoch
    flat_order_codes: [180] observed stimulus codes in temporal order
    """
    onsets = _rising_edges(flashing_epoch)

    x_epoch = np.zeros((NUM_REPEATS, NUM_CODES, NUM_CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
    y_epoch = np.zeros((NUM_REPEATS, NUM_CODES), dtype=np.int64)
    onset_epoch = -np.ones((NUM_REPEATS, NUM_CODES), dtype=np.int64)
    code_counts = np.zeros((NUM_CODES,), dtype=np.int64)
    observed_codes: List[int] = []

    n_samples = int(signal_epoch.shape[0])
    for onset in onsets.tolist():
        code = int(stim_code_epoch[onset])
        if code < 1 or code > NUM_CODES:
            continue
        end = onset + WINDOW_SAMPLES
        if end > n_samples:
            continue

        code_idx = code - 1
        rep_idx = int(code_counts[code_idx])
        if rep_idx >= NUM_REPEATS:
            raise ValueError(f"StimulusCode {code} appears more than {NUM_REPEATS} times in one epoch")

        seg = signal_epoch[onset:end, :].T  # [C, T]
        seg = _zscore_per_channel(seg)

        x_epoch[rep_idx, code_idx] = seg
        y_epoch[rep_idx, code_idx] = 1 if int(stim_type_epoch[onset]) > 0 else 0
        onset_epoch[rep_idx, code_idx] = onset
        code_counts[code_idx] += 1
        observed_codes.append(code)

    if not np.all(code_counts == NUM_REPEATS):
        raise ValueError(f"Each code should appear {NUM_REPEATS} times, got {code_counts.tolist()}")

    target_count = int(y_epoch.sum())
    expected_target_count = 2 * NUM_REPEATS
    if target_count != expected_target_count:
        raise ValueError(
            f"Each epoch should contain {expected_target_count} target flashes, got {target_count}"
        )

    return x_epoch, y_epoch, onset_epoch, np.asarray(observed_codes, dtype=np.int64)


def preprocess_subject_train(mat_path: str) -> Dict[str, Any]:
    mat = sio.loadmat(mat_path)
    signal_raw, flashing, stim_code, stim_type, target_char = _validate_train_mat_shapes(mat, mat_path)

    n_epochs, n_samples, n_channels = signal_raw.shape

    # --- 逐 epoch 滤波 ---
    # signal shape: [n_epochs, n_samples, 64]
    # 每个 epoch 是一段连续的 EEG 记录，在截取 trial 之前滤波
    # 避免短 segment 滤波的边界伪影
    signal = np.empty_like(signal_raw)
    for ep in range(n_epochs):
        epoch_data = signal_raw[ep]                       # [n_samples, 64]
        epoch_data = _notch_filter(epoch_data, SAMPLE_RATE)  # 去 60Hz 工频
        epoch_data = _bandpass_filter(epoch_data, SAMPLE_RATE)  # 0.5-30Hz 带通
        signal[ep] = epoch_data
    print(f"  [filter] {os.path.basename(mat_path)}: "
          f"notch={NOTCH_FREQ}Hz + bandpass={LOWCUT}-{HIGHCUT}Hz applied to {n_epochs} epochs")

    data = np.zeros((n_epochs, NUM_REPEATS, NUM_CODES, n_channels, WINDOW_SAMPLES), dtype=np.float32)
    label = np.zeros((n_epochs, NUM_REPEATS, NUM_CODES), dtype=np.int64)
    onset_idx = np.zeros((n_epochs, NUM_REPEATS, NUM_CODES), dtype=np.int64)
    code_order = np.zeros((n_epochs, NUM_REPEATS * NUM_CODES), dtype=np.int64)

    for ep in range(n_epochs):
        x_ep, y_ep, on_ep, seq_ep = _extract_one_epoch(
            signal[ep], flashing[ep], stim_code[ep], stim_type[ep]
        )
        data[ep] = x_ep
        label[ep] = y_ep
        onset_idx[ep] = on_ep
        if seq_ep.size != NUM_REPEATS * NUM_CODES:
            raise ValueError(f"Epoch {ep}: expected 180 valid onsets, got {seq_ep.size}")
        code_order[ep] = seq_ep

    return {
        "dataset_name": "dataset3",
        "data": data,                      # [epoch, repeat, code, C, T]
        "label": label,                    # [epoch, repeat, code]
        "target_char": target_char,        # [epoch]
        "epoch_onset_idx": onset_idx,      # [epoch, repeat, code]
        "epoch_code_order": code_order,    # [epoch, 180]
        "meta": {
            "dataset_name": "dataset3",
            "source_dataset": "BCI Competition III Dataset III-II",
            "source_file": os.path.basename(mat_path),
            "original_signal_shape": tuple(int(v) for v in signal_raw.shape),
            "sample_rate": SAMPLE_RATE,
            "window_samples": WINDOW_SAMPLES,
            "window_ms": float(WINDOW_SAMPLES / SAMPLE_RATE * 1000.0),
            "num_epochs": int(n_epochs),
            "num_samples_per_epoch": int(n_samples),
            "num_channels": int(n_channels),
            "num_codes": NUM_CODES,
            "num_repeats": NUM_REPEATS,
            "label_definition": {"target": 1, "nontarget": 0},
            "trial_layout": "[epoch, repeat, code, channel, time]",
            "split_unit_recommendation": "epoch",
            "normalization": "per-trial per-channel zscore",
            "filtering": {
                "notch": f"{NOTCH_FREQ}Hz (Q={NOTCH_Q})",
                "bandpass": f"{LOWCUT}-{HIGHCUT}Hz (order={FILTER_ORDER})",
                "method": "zero-phase filtfilt, applied per-epoch before trial extraction",
            },
            "onset_rule": "Flashing rising edge with first-sample Flashing==1 included",
        },
    }


def _default_output_name(mat_path: str) -> str:
    base = os.path.splitext(os.path.basename(mat_path))[0]
    base = base.replace("Subject_", "subject_")
    return f"{base}.pkl"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_files = [
        os.path.join(INPUT_DIR, "Subject_A_Train.mat"),
        os.path.join(INPUT_DIR, "Subject_B_Train.mat"),
    ]

    for mat_path in train_files:
        if not os.path.isfile(mat_path):
            print(f"[skip] not found: {mat_path}")
            continue
        print(f"[load] {mat_path}")
        obj = preprocess_subject_train(mat_path)
        out_path = os.path.join(OUTPUT_DIR, _default_output_name(mat_path))
        with open(out_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"[save] {out_path} | data={obj['data'].shape} | label={obj['label'].shape} | "
            f"targets={int(obj['label'].sum())}"
        )


if __name__ == "__main__":
    main()
