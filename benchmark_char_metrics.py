"""
Character-level accuracy utilities for P300 datasets (B1 protocol).

This module is intentionally independent from training logic:
- it only consumes model outputs / metadata and computes extra metrics
- it does not change optimization, early stopping, or trial-level metrics
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_utils import forward_logits


# Standard 6x6 matrix used by BCI Competition III (dataset3).
# rows: StimulusCode 7..12 (0-based flashes 6..11)
# cols: StimulusCode 1..6  (0-based flashes 0..5)
DATASET3_6X6_GRID: Tuple[Tuple[str, ...], ...] = (
    ("A", "B", "C", "D", "E", "F"),
    ("G", "H", "I", "J", "K", "L"),
    ("M", "N", "O", "P", "Q", "R"),
    ("S", "T", "U", "V", "W", "X"),
    ("Y", "Z", "1", "2", "3", "4"),
    ("5", "6", "7", "8", "9", "_"),
)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_char(x: Any) -> str:
    s = str(x).strip()
    return s[:1].upper() if s else ""


@torch.no_grad()
def collect_probs_labels_and_meta(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Collect positive-class probabilities and optional metadata from a loader.

    Supported sample layouts:
    - (x, y, subject_id)
    - (x, y, run, rep, flash)
    """
    model.eval()
    probs_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
    runs_all: List[torch.Tensor] = []
    reps_all: List[torch.Tensor] = []
    flashes_all: List[torch.Tensor] = []

    has_meta = False
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True).long()
        logits = forward_logits(model, x)
        probs = F.softmax(logits, dim=1)[:, 1]

        probs_all.append(probs)
        labels_all.append(y)

        if len(batch) >= 5:
            has_meta = True
            runs_all.append(batch[2].detach().cpu().long())
            reps_all.append(batch[3].detach().cpu().long())
            flashes_all.append(batch[4].detach().cpu().long())

    out: Dict[str, Optional[np.ndarray]] = {
        "probs_pos": _to_numpy(torch.cat(probs_all) if probs_all else np.empty((0,), dtype=np.float32)),
        "labels": _to_numpy(torch.cat(labels_all) if labels_all else np.empty((0,), dtype=np.int64)),
        "runs": None,
        "reps": None,
        "flashes": None,
    }
    if has_meta and runs_all:
        out["runs"] = _to_numpy(torch.cat(runs_all))
        out["reps"] = _to_numpy(torch.cat(reps_all))
        out["flashes"] = _to_numpy(torch.cat(flashes_all))
    return out


def compute_char_acc_dataset1_rsvp(
    probs_pos: Any,
    trial_idx: Any,
    stim_code: Any,
    labels: Optional[Any] = None,
    target_stim_by_trial: Optional[Mapping[int, int]] = None,
) -> Dict[str, Any]:
    """
    Character accuracy for dataset1 RSVP.

    For each character epoch (trial_idx), average probability per stimulus code.
    Predicted character = argmax stim code score.
    """
    probs = _to_numpy(probs_pos).astype(np.float64).reshape(-1)
    trials = _to_numpy(trial_idx).astype(np.int64).reshape(-1)
    stims = _to_numpy(stim_code).astype(np.int64).reshape(-1)
    if not (len(probs) == len(trials) == len(stims)):
        raise ValueError("probs_pos/trial_idx/stim_code length mismatch")

    y = None if labels is None else _to_numpy(labels).astype(np.int64).reshape(-1)
    if y is not None and len(y) != len(probs):
        raise ValueError("labels length mismatch")

    return compute_char_acc_dataset1_rsvp_curve(
        probs_pos=probs,
        trial_idx=trials,
        stim_code=stims,
        labels=y,
        target_stim_by_trial=target_stim_by_trial,
    )


def compute_char_acc_dataset1_rsvp_curve(
    probs_pos: Any,
    trial_idx: Any,
    stim_code: Any,
    labels: Optional[Any] = None,
    target_stim_by_trial: Optional[Mapping[int, int]] = None,
    max_reps: int = 10,
) -> Dict[str, Any]:
    """
    Character accuracy curve for dataset1 RSVP.

    For each trial_idx (character epoch), restore repetition order by counting
    occurrences of each stimulus code in the original event sequence. Then,
    for each k in [1, max_reps], aggregate probabilities using only the first
    k repetitions of each stimulus code and decode the character by argmax.
    """
    probs = _to_numpy(probs_pos).astype(np.float64).reshape(-1)
    trials = _to_numpy(trial_idx).astype(np.int64).reshape(-1)
    stims = _to_numpy(stim_code).astype(np.int64).reshape(-1)
    if not (len(probs) == len(trials) == len(stims)):
        raise ValueError("probs_pos/trial_idx/stim_code length mismatch")

    y = None if labels is None else _to_numpy(labels).astype(np.int64).reshape(-1)
    if y is not None and len(y) != len(probs):
        raise ValueError("labels length mismatch")

    rep_idx = np.zeros_like(trials, dtype=np.int64)
    rep_counter: Dict[Tuple[int, int], int] = {}
    for i, (tid, stim) in enumerate(zip(trials.tolist(), stims.tolist())):
        key = (int(tid), int(stim))
        rep_counter[key] = rep_counter.get(key, 0) + 1
        rep_idx[i] = rep_counter[key]

    valid_trials: List[int] = []
    skipped = 0
    gt_by_trial: Dict[int, int] = {}
    pred_by_trial_by_k: Dict[str, Dict[int, int]] = {str(k): {} for k in range(1, int(max_reps) + 1)}
    curve_by_k: Dict[str, float] = {}

    for tid in np.unique(trials):
        mask_t = (trials == tid)
        stim_candidates = np.unique(stims[mask_t])
        if stim_candidates.size == 0:
            skipped += 1
            continue

        if target_stim_by_trial is not None and int(tid) in target_stim_by_trial:
            gt_code = int(target_stim_by_trial[int(tid)])
        elif y is not None:
            pos_codes = np.unique(stims[mask_t & (y == 1)])
            if pos_codes.size != 1:
                skipped += 1
                continue
            gt_code = int(pos_codes[0])
        else:
            skipped += 1
            continue

        valid_trials.append(int(tid))
        gt_by_trial[int(tid)] = gt_code

    for k in range(1, int(max_reps) + 1):
        correct = 0
        evaluated = 0
        pred_map = pred_by_trial_by_k[str(k)]

        for tid in valid_trials:
            mask_t = (trials == tid)
            stim_candidates = np.unique(stims[mask_t])

            best_score = -np.inf
            pred_code = int(stim_candidates[0])
            for c in stim_candidates:
                mask_c = mask_t & (stims == c) & (rep_idx <= k)
                if not np.any(mask_c):
                    continue
                score = float(np.sum(probs[mask_c]))
                if score > best_score:
                    best_score = score
                    pred_code = int(c)

            pred_map[int(tid)] = pred_code
            evaluated += 1
            correct += int(pred_code == gt_by_trial[int(tid)])

        curve_by_k[str(k)] = round((correct / evaluated) if evaluated > 0 else 0.0, 6)

    final_key = str(int(max_reps))
    return {
        "metric": "char_acc_dataset1",
        "char_acc": curve_by_k.get(final_key, 0.0),
        "char_acc_at_k": curve_by_k,
        "main_k": int(max_reps),
        "n_char_epochs": int(len(valid_trials)),
        "n_skipped": int(skipped),
        "pred_by_trial": pred_by_trial_by_k[final_key],
        "pred_by_trial_at_k": pred_by_trial_by_k,
        "gt_by_trial": gt_by_trial,
    }


def _infer_single_target_per_run(
    labels: np.ndarray,
    runs: np.ndarray,
    flashes: np.ndarray,
) -> Tuple[Dict[int, int], int]:
    target_by_run: Dict[int, int] = {}
    skipped = 0
    for run_id in np.unique(runs):
        mask_r = (runs == run_id)
        pos = np.unique(flashes[mask_r & (labels == 1)])
        if pos.size != 1:
            skipped += 1
            continue
        target_by_run[int(run_id)] = int(pos[0])
    return target_by_run, skipped


def compute_char_acc_dataset2(
    probs_pos: Any,
    labels: Any,
    runs: Any,
    reps: Any,
    flashes: Any,
    max_k: int = 20,
    report_max_k: Optional[int] = 10,
    target_flash_by_run: Optional[Mapping[int, int]] = None,
) -> Dict[str, Any]:
    """
    Character/command accuracy for dataset2 (6 candidates, single target).
    """
    probs = _to_numpy(probs_pos).astype(np.float64).reshape(-1)
    y = _to_numpy(labels).astype(np.int64).reshape(-1)
    run = _to_numpy(runs).astype(np.int64).reshape(-1)
    rep = _to_numpy(reps).astype(np.int64).reshape(-1)
    fl = _to_numpy(flashes).astype(np.int64).reshape(-1)
    if not (len(probs) == len(y) == len(run) == len(rep) == len(fl)):
        raise ValueError("dataset2 inputs length mismatch")

    if target_flash_by_run is None:
        target_flash_by_run, skipped_infer = _infer_single_target_per_run(y, run, fl)
    else:
        target_flash_by_run = {int(k): int(v) for k, v in target_flash_by_run.items()}
        skipped_infer = 0

    ca_by_k_full: Dict[str, float] = {}
    valid_runs = sorted(target_flash_by_run.keys())

    for k in range(1, int(max_k) + 1):
        correct = 0
        total = 0
        for rid in valid_runs:
            mask_r = (run == rid)
            if not np.any(mask_r):
                continue
            scores = np.full((6,), -np.inf, dtype=np.float64)
            for c in range(6):
                mask = mask_r & (fl == c) & (rep < k)
                if np.any(mask):
                    scores[c] = float(np.mean(probs[mask]))
            pred = int(np.argmax(scores))
            gt = int(target_flash_by_run[rid])
            total += 1
            correct += int(pred == gt)
        ca_by_k_full[str(k)] = round((correct / total) if total > 0 else 0.0, 6)

    if report_max_k is None:
        main_k = int(max_k)
    else:
        main_k = min(int(max_k), int(report_max_k))
    ca_by_k = {str(k): ca_by_k_full[str(k)] for k in range(1, main_k + 1)}
    return {
        "metric": "command_acc_dataset2",
        "ca_at_k": ca_by_k,
        "ca_main": ca_by_k.get(str(main_k), 0.0),
        "command_acc_at_k": ca_by_k,
        "command_acc_main": ca_by_k.get(str(main_k), 0.0),
        "full_ca_at_k": ca_by_k_full,
        "main_k": main_k,
        "full_max_k": int(max_k),
        "n_epochs": len(valid_runs),
        "n_skipped": int(skipped_infer),
    }


def _decode_dataset3_char(
    row_flash_0based: int,
    col_flash_0based: int,
    grid: Sequence[Sequence[str]],
) -> str:
    # row flashes are 6..11 -> row indices 0..5
    row_idx = int(row_flash_0based) - 6
    col_idx = int(col_flash_0based)
    if row_idx < 0 or row_idx >= 6 or col_idx < 0 or col_idx >= 6:
        return ""
    return _safe_char(grid[row_idx][col_idx])


def compute_char_acc_dataset3(
    probs_pos: Any,
    labels: Any,
    runs: Any,
    reps: Any,
    flashes: Any,
    max_k: int = 15,
    report_max_k: Optional[int] = 10,
    target_chars: Optional[Any] = None,
    grid: Sequence[Sequence[str]] = DATASET3_6X6_GRID,
) -> Dict[str, Any]:
    """
    Character accuracy for dataset3 (12 flashes: 6 columns + 6 rows, 2 targets).
    """
    probs = _to_numpy(probs_pos).astype(np.float64).reshape(-1)
    y = _to_numpy(labels).astype(np.int64).reshape(-1)
    run = _to_numpy(runs).astype(np.int64).reshape(-1)
    rep = _to_numpy(reps).astype(np.int64).reshape(-1)
    fl = _to_numpy(flashes).astype(np.int64).reshape(-1)
    if not (len(probs) == len(y) == len(run) == len(rep) == len(fl)):
        raise ValueError("dataset3 inputs length mismatch")

    tchars = None if target_chars is None else _to_numpy(target_chars).reshape(-1)

    # infer ground-truth row/col flashes from labels per run
    gt_pair_by_run: Dict[int, Tuple[int, int]] = {}
    skipped = 0
    for rid in np.unique(run):
        mask_r = (run == rid)
        pos = np.unique(fl[mask_r & (y == 1)])
        col_pos = [int(v) for v in pos if 0 <= int(v) <= 5]
        row_pos = [int(v) for v in pos if 6 <= int(v) <= 11]
        if len(col_pos) != 1 or len(row_pos) != 1:
            skipped += 1
            continue
        gt_pair_by_run[int(rid)] = (row_pos[0], col_pos[0])

    valid_runs = sorted(gt_pair_by_run.keys())
    pair_by_k_full: Dict[str, float] = {}
    char_by_k_full: Dict[str, float] = {}

    for k in range(1, int(max_k) + 1):
        pair_correct = 0
        char_correct = 0
        total = 0
        char_total = 0

        for rid in valid_runs:
            mask_r = (run == rid)
            if not np.any(mask_r):
                continue

            # mean score per flash code (0..11)
            scores = np.full((12,), -np.inf, dtype=np.float64)
            for c in range(12):
                mask = mask_r & (fl == c) & (rep < k)
                if np.any(mask):
                    scores[c] = float(np.mean(probs[mask]))

            pred_col = int(np.argmax(scores[0:6]))
            pred_row = int(np.argmax(scores[6:12])) + 6
            gt_row, gt_col = gt_pair_by_run[rid]

            total += 1
            pair_correct += int((pred_row == gt_row) and (pred_col == gt_col))

            if tchars is not None and rid < len(tchars):
                gt_char = _safe_char(tchars[rid])
                pred_char = _decode_dataset3_char(pred_row, pred_col, grid)
                if gt_char:
                    char_total += 1
                    char_correct += int(pred_char == gt_char)

        pair_by_k_full[str(k)] = round((pair_correct / total) if total > 0 else 0.0, 6)
        if char_total > 0:
            char_by_k_full[str(k)] = round(char_correct / char_total, 6)

    if report_max_k is None:
        main_k = int(max_k)
    else:
        main_k = min(int(max_k), int(report_max_k))
    pair_by_k = {str(k): pair_by_k_full[str(k)] for k in range(1, main_k + 1)}
    char_by_k = {str(k): char_by_k_full[str(k)] for k in range(1, main_k + 1) if str(k) in char_by_k_full}
    out: Dict[str, Any] = {
        "metric": "char_acc_dataset3",
        "ca_pair_at_k": pair_by_k,
        "ca_pair_main": pair_by_k.get(str(main_k), 0.0),
        "pair_acc_at_k": pair_by_k,
        "pair_acc_main": pair_by_k.get(str(main_k), 0.0),
        "full_ca_pair_at_k": pair_by_k_full,
        "main_k": main_k,
        "full_max_k": int(max_k),
        "n_epochs": len(valid_runs),
        "n_skipped": int(skipped),
    }
    if char_by_k:
        out["ca_char_at_k"] = char_by_k
        out["ca_char_main"] = char_by_k.get(str(main_k), 0.0)
        out["char_acc_at_k"] = char_by_k
        out["char_acc_main"] = char_by_k.get(str(main_k), 0.0)
        out["full_ca_char_at_k"] = char_by_k_full
    return out


__all__ = [
    "DATASET3_6X6_GRID",
    "collect_probs_labels_and_meta",
    "compute_char_acc_dataset1_rsvp",
    "compute_char_acc_dataset1_rsvp_curve",
    "compute_char_acc_dataset2",
    "compute_char_acc_dataset3",
]
