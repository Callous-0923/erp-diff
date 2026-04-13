import os
import pickle
from datetime import datetime

import numpy as np
import scipy.io as scio
from scipy import signal


RAW_DATA_DIR = r"D:\files\datasets\erp"
OUT_DIR = r"D:\files\codes\EEGInception\PreData_MOCNN"
SRATE = 2048
DOWN_SAMPLE = 128
N_SAMPLES = 128
FLASH_NUM = 6
REPEAT_NUM = 20
DOWNSAMPLE_RATE = SRATE // DOWN_SAMPLE
ALL_CHANNELS = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5",
    "P7", "P3", "Pz", "PO3", "O1", "Oz", "O2", "PO4", "P4", "P8",
    "CP6", "CP2", "C4", "T8", "FC6", "FC2", "F4", "F8", "AF4", "Fp2",
    "Fz", "Cz", "MA1", "MA2",
]
REFERENCE_CHANNELS = ["T7", "T8"]
REFERENCE_CH_ID = [ALL_CHANNELS.index(ch) for ch in REFERENCE_CHANNELS]


def bandpass_filter(data: np.ndarray, srate: int) -> np.ndarray:
    """Butterworth bandpass aligned to the MOCNN paper: 0.5-30 Hz."""
    order = 4
    wn = [0.5 * 2 / srate, 30.0 * 2 / srate]
    sos = signal.butter(order, wn, btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, data, axis=-1)


def _event_datetime(row: np.ndarray) -> datetime:
    row = row.astype(int)
    return datetime(row[0], row[1], row[2], row[3], row[4], row[5], row[6])


def _zscore_per_channel(trial: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(trial, axis=1, keepdims=True)
    std = np.std(trial, axis=1, keepdims=True)
    std = np.maximum(std, eps)
    return (trial - mean) / std


def _preprocess_run(mat_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    eeg = scio.loadmat(mat_path)
    data = np.asarray(eeg["data"], dtype=np.float64)
    events = np.asarray(eeg["events"], dtype=np.float64)
    target = int(np.squeeze(eeg["target"]))
    stimuli = np.asarray(np.squeeze(eeg["stimuli"]), dtype=np.int64)

    data = data - np.mean(data[REFERENCE_CH_ID, :], axis=0, keepdims=True)
    data = data[:32, :]
    data = bandpass_filter(data, SRATE)
    data = data[:, ::DOWNSAMPLE_RATE]

    min_events = REPEAT_NUM * FLASH_NUM
    if stimuli.shape[0] < min_events:
        raise ValueError(f"{mat_path}: expected at least {min_events} stimuli, got {stimuli.shape[0]}")
    if events.shape[0] < min_events:
        raise ValueError(f"{mat_path}: expected at least {min_events} events, got {events.shape[0]}")

    events = np.concatenate((events, np.zeros((events.shape[0], 1))), axis=1)
    events[:, -1] = 1000000 * (events[:, -2] - np.floor(events[:, -2]))
    events[:, -2] = np.floor(events[:, -2])
    t0 = _event_datetime(events[0])

    trials_data = np.zeros((REPEAT_NUM, FLASH_NUM, data.shape[0], N_SAMPLES), dtype=np.float32)
    trials_label = np.zeros((REPEAT_NUM, FLASH_NUM), dtype=np.int32)

    for rep_idx in range(REPEAT_NUM):
        for flash_order in range(FLASH_NUM):
            event_idx = rep_idx * FLASH_NUM + flash_order
            flash_id = int(stimuli[event_idx])
            event_time = _event_datetime(events[event_idx])
            pos = round((event_time - t0).total_seconds() * DOWN_SAMPLE + 0.4 * DOWN_SAMPLE)
            trial = data[:, pos : pos + N_SAMPLES]
            if trial.shape[-1] != N_SAMPLES:
                raise ValueError(
                    f"{mat_path}: invalid trial window at event {event_idx}, got shape {trial.shape}"
                )
            trial = _zscore_per_channel(trial).astype(np.float32)
            trials_data[rep_idx, flash_id - 1, :, :] = trial
            trials_label[rep_idx, flash_id - 1] = int(flash_id == target)

    return trials_data, trials_label, target + 100


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for sub in sorted(os.listdir(RAW_DATA_DIR)):
        sub_dir = os.path.join(RAW_DATA_DIR, sub)
        if not os.path.isdir(sub_dir):
            continue
        print(sub)
        ses_eeg = {"data": [], "label": [], "com": []}
        for session in sorted(os.listdir(sub_dir)):
            session_dir = os.path.join(sub_dir, session)
            if not os.path.isdir(session_dir):
                continue
            print(f"  {session}")
            for epoch in sorted(os.listdir(session_dir)):
                if not epoch.endswith(".mat"):
                    continue
                mat_path = os.path.join(session_dir, epoch)
                print(f"    {epoch}")
                trials_data, trials_label, command_id = _preprocess_run(mat_path)
                ses_eeg["data"].append(trials_data)
                ses_eeg["label"].append(trials_label)
                ses_eeg["com"].append(command_id)

        data_by_flash = np.asarray(ses_eeg["data"], dtype=np.float32)
        label_by_flash = np.asarray(ses_eeg["label"], dtype=np.int32)
        com_trig = np.asarray(ses_eeg["com"], dtype=np.int32)
        eeg = {"data": data_by_flash, "label": label_by_flash, "com": com_trig}

        print(f"  [OUT] data={data_by_flash.shape} label={label_by_flash.shape} com={com_trig.shape}")
        out_path = os.path.join(OUT_DIR, f"{sub}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(eeg, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
