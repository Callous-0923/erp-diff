"""
Benchmark utilities — unified metrics, TTA, snapshot ensemble, threshold search.

Shared by both ERPDiff and baseline models for fair comparison.
All metric computation uses identical functions regardless of model type.
"""

import copy
import heapq
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


DEFAULT_INFERENCE_BATCH_SIZE = 128


# ============================================================
# Confusion matrix helpers
# ============================================================

def confusion_update(tp, fp, tn, fn, preds, labels):
    """Update confusion matrix counts from predicted class indices and labels."""
    tp += int(((preds == 1) & (labels == 1)).sum().item())
    tn += int(((preds == 0) & (labels == 0)).sum().item())
    fp += int(((preds == 1) & (labels == 0)).sum().item())
    fn += int(((preds == 0) & (labels == 1)).sum().item())
    return tp, fp, tn, fn


def confusion_from_probs(tp, fp, tn, fn, probs_pos, labels, threshold=0.5):
    """Update confusion matrix from positive-class probabilities and a threshold."""
    preds = (probs_pos >= threshold).long()
    return confusion_update(tp, fp, tn, fn, preds, labels)


def confusion_from_logits(tp, fp, tn, fn, logits, labels):
    """Update confusion matrix from logits (argmax decision)."""
    preds = logits.argmax(dim=1)
    return confusion_update(tp, fp, tn, fn, preds, labels)


# ============================================================
# Unified metrics (identical to erpdiff_train.py _compute_full_metrics)
# ============================================================

def compute_full_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, Any]:
    """
    Compute all metrics from confusion matrix counts.

    Returns: acc, macro_pre, macro_rec (=balanced_acc), macro_f1,
             per-class pre/rec/f1, raw confusion counts.
    """
    eps = 1e-9
    total = tp + tn + fp + fn
    acc = (tp + tn) / (total + eps)

    pre_pos = tp / (tp + fp + eps)
    pre_neg = tn / (tn + fn + eps)
    rec_pos = tp / (tp + fn + eps)
    rec_neg = tn / (tn + fp + eps)

    f1_pos = 2 * pre_pos * rec_pos / (pre_pos + rec_pos + eps)
    f1_neg = 2 * pre_neg * rec_neg / (pre_neg + rec_neg + eps)

    return {
        "acc":       round(acc, 6),
        "macro_pre": round(0.5 * (pre_pos + pre_neg), 6),
        "macro_rec": round(0.5 * (rec_pos + rec_neg), 6),
        "macro_f1":  round(0.5 * (f1_pos + f1_neg), 6),
        "pre_pos":   round(pre_pos, 6),
        "pre_neg":   round(pre_neg, 6),
        "rec_pos":   round(rec_pos, 6),
        "rec_neg":   round(rec_neg, 6),
        "f1_pos":    round(f1_pos, 6),
        "f1_neg":    round(f1_neg, 6),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ============================================================
# Collect logits / probs from a standard model (single output)
# ============================================================

def _extract_logits(output: Any) -> torch.Tensor:
    """Handle models that return logits directly or wrap them in a tuple/list."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim >= 2 and item.shape[-1] == 2:
                return item
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Unsupported model output type for logits extraction: {type(output)!r}")


def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward helper that normalizes baseline model outputs to logits."""
    return _extract_logits(model(x))


def _stack_loader_inputs(
    loader: DataLoader,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_x, all_y = [], []
    for batch in loader:
        all_x.append(batch[0].to(device))
        all_y.append(batch[1].to(device).long())
    return torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)


@torch.no_grad()
def collect_average_probs(
    model: nn.Module,
    all_x: torch.Tensor,
    states: Optional[List[dict]] = None,
    use_tta: bool = False,
    n_augments: int = 20,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
) -> torch.Tensor:
    """
    Average softmax probabilities across optional ensemble states and TTA views.

    Returns probabilities with shape [N, 2].
    """
    model.eval()
    state_list = states or [None]
    num_augments = max(int(n_augments), 1)
    total_probs = torch.zeros(all_x.size(0), 2, device=all_x.device)

    for state in state_list:
        if state is not None:
            model.load_state_dict(state)
        model.eval()

        aug_iter = range(num_augments) if use_tta else (0,)
        for aug_idx in aug_iter:
            x_view = tta_augment(all_x, aug_idx) if use_tta else all_x
            for start in range(0, all_x.size(0), batch_size):
                end = min(start + batch_size, all_x.size(0))
                logits = forward_logits(model, x_view[start:end])
                total_probs[start:end] += F.softmax(logits, dim=1)

    n_total_views = len(state_list) * (num_augments if use_tta else 1)
    return total_probs / max(n_total_views, 1)

@torch.no_grad()
def collect_logits_and_labels(
    model: nn.Module, loader: DataLoader, device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect all logits and labels from a standard model (output = logits [B,2])."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True).long()
        logits = forward_logits(model, x)
        all_logits.append(logits)
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


@torch.no_grad()
def collect_probs_and_labels(
    model: nn.Module, loader: DataLoader, device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect positive-class probabilities and labels."""
    logits, labels = collect_logits_and_labels(model, loader, device)
    probs_pos = F.softmax(logits, dim=1)[:, 1]
    return probs_pos, labels


@torch.no_grad()
def collect_probs_with_tta(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n_augments: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect positive-class probabilities with TTA averaging."""
    all_x, all_y = _stack_loader_inputs(loader, device)
    avg_probs = collect_average_probs(
        model,
        all_x,
        use_tta=True,
        n_augments=n_augments,
        batch_size=DEFAULT_INFERENCE_BATCH_SIZE,
    )
    return avg_probs[:, 1], all_y


@torch.no_grad()
def collect_probs_with_ensemble(
    model: nn.Module,
    ensemble: Any,
    loader: DataLoader,
    device: str,
    use_tta: bool = False,
    n_augments: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect positive-class probabilities from snapshot ensemble predictions."""
    states = ensemble.get_states()
    if not states:
        raise ValueError("Ensemble is empty")
    all_x, all_y = _stack_loader_inputs(loader, device)
    avg_probs = collect_average_probs(
        model,
        all_x,
        states=states,
        use_tta=use_tta,
        n_augments=n_augments,
        batch_size=DEFAULT_INFERENCE_BATCH_SIZE,
    )
    return avg_probs[:, 1], all_y


# ============================================================
# Evaluate a standard model (single forward pass)
# ============================================================

@torch.no_grad()
def evaluate_standard(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate a standard single-output model with optional threshold.

    If threshold == 0.5, uses argmax (equivalent). Otherwise uses probability
    thresholding on positive class.
    """
    model.eval()
    total_loss = 0.0
    total = 0
    tp = fp = tn = fn = 0

    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True).long()
        logits = forward_logits(model, x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        probs_pos = F.softmax(logits, dim=1)[:, 1]
        tp, fp, tn, fn = confusion_from_probs(tp, fp, tn, fn, probs_pos, y, threshold)

    metrics = compute_full_metrics(tp, fp, tn, fn)
    metrics["test_loss"] = round(total_loss / max(total, 1), 6)
    metrics["threshold"] = threshold
    return metrics


# ============================================================
# Threshold search
# ============================================================

@torch.no_grad()
def find_optimal_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    metric: str = "balanced_acc",
) -> float:
    """
    Search for optimal decision threshold on validation set.
    Grid search from 0.10 to 0.88 in steps of 0.02.

    Args:
        metric: 'acc', 'balanced_acc', or 'macro_f1'
    """
    probs_pos, labels = collect_probs_and_labels(model, loader, device)
    best_thr, best_score = 0.5, -1.0

    for thr_int in range(10, 90, 2):
        thr = thr_int / 100.0
        tp = fp = tn = fn = 0
        tp, fp, tn, fn = confusion_from_probs(tp, fp, tn, fn, probs_pos, labels, thr)
        eps = 1e-9
        total = tp + fp + tn + fn
        acc = (tp + tn) / (total + eps)
        rec_pos = tp / (tp + fn + eps)
        rec_neg = tn / (tn + fp + eps)
        bal_acc = 0.5 * (rec_pos + rec_neg)

        if metric == "macro_f1":
            pre_pos = tp / (tp + fp + eps)
            pre_neg = tn / (tn + fn + eps)
            f1_pos = 2 * pre_pos * rec_pos / (pre_pos + rec_pos + eps)
            f1_neg = 2 * pre_neg * rec_neg / (pre_neg + rec_neg + eps)
            score = 0.5 * (f1_pos + f1_neg)
        elif metric == "balanced_acc":
            score = bal_acc
        else:
            score = acc

        if score > best_score:
            best_score = score
            best_thr = thr

    return best_thr


# ============================================================
# Test-time augmentation (TTA)
# ============================================================

@torch.no_grad()
def tta_augment(x: torch.Tensor, aug_idx: int) -> torch.Tensor:
    """
    Apply stochastic augmentation for TTA.
    Identical to erpdiff_train.py _tta_augment.
    aug_idx=0 returns original (no augmentation).
    """
    if aug_idx == 0:
        return x
    x = x.clone()
    # Time shift
    shift = torch.randint(-3, 4, (1,)).item()
    if shift != 0:
        x = torch.roll(x, shifts=shift, dims=-1)
    # Amplitude scaling [0.95, 1.05]
    scale = 0.95 + torch.rand(1).item() * 0.10
    x = x * scale
    # Small Gaussian noise (2% of per-sample std)
    std = x.flatten(1).std(dim=1, keepdim=True).clamp_min(1e-6).view(-1, 1, 1, 1)
    x = x + torch.randn_like(x) * (0.02 * std)
    return x


@torch.no_grad()
def evaluate_with_tta(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    n_augments: int = 20,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate a standard model with test-time augmentation.
    Averages softmax probabilities over n_augments views, then applies threshold.
    """
    model.eval()
    all_x, all_y = _stack_loader_inputs(loader, device)
    avg_probs = collect_average_probs(
        model,
        all_x,
        use_tta=True,
        n_augments=n_augments,
        batch_size=DEFAULT_INFERENCE_BATCH_SIZE,
    )
    probs_pos = avg_probs[:, 1]

    tp = fp = tn = fn = 0
    tp, fp, tn, fn = confusion_from_probs(tp, fp, tn, fn, probs_pos, all_y, threshold)
    metrics = compute_full_metrics(tp, fp, tn, fn)

    # Compute loss on original (non-augmented) data
    total_loss = 0.0
    for start in range(0, all_x.size(0), DEFAULT_INFERENCE_BATCH_SIZE):
        end = min(start + DEFAULT_INFERENCE_BATCH_SIZE, all_x.size(0))
        logits = forward_logits(model, all_x[start:end])
        loss = criterion(logits, all_y[start:end])
        total_loss += loss.item() * (end - start)
    metrics["test_loss"] = round(total_loss / max(all_x.size(0), 1), 6)
    metrics["threshold"] = threshold
    return metrics


# ============================================================
# Snapshot ensemble
# ============================================================

class SnapshotEnsemble:
    """
    Maintains top-K model checkpoints by balanced accuracy.
    Uses a min-heap so the weakest model can be efficiently replaced.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._heap: List[Tuple[float, int, dict]] = []  # (score, counter, state_dict)
        self._counter = 0

    def update(self, model: nn.Module, bal_acc: float):
        """Consider adding this model to the ensemble."""
        state = copy.deepcopy(model.state_dict())
        self._counter += 1
        if len(self._heap) < self.top_k:
            heapq.heappush(self._heap, (bal_acc, self._counter, state))
        elif bal_acc > self._heap[0][0]:
            heapq.heapreplace(self._heap, (bal_acc, self._counter, state))

    def get_states(self) -> List[dict]:
        """Return all saved state dicts, sorted best-first."""
        return [s for _, _, s in sorted(self._heap, reverse=True)]

    def __len__(self):
        return len(self._heap)


@torch.no_grad()
def evaluate_ensemble(
    model: nn.Module,
    ensemble: SnapshotEnsemble,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    threshold: float = 0.5,
    use_tta: bool = False,
    n_augments: int = 20,
) -> Dict[str, Any]:
    """
    Evaluate using snapshot ensemble: average softmax outputs of top-K models.
    Optionally combines with TTA.
    """
    model.eval()
    states = ensemble.get_states()
    if not states:
        raise ValueError("Ensemble is empty")

    all_x, all_y = _stack_loader_inputs(loader, device)
    avg_probs = collect_average_probs(
        model,
        all_x,
        states=states,
        use_tta=use_tta,
        n_augments=n_augments,
        batch_size=DEFAULT_INFERENCE_BATCH_SIZE,
    )
    probs_pos = avg_probs[:, 1]

    tp = fp = tn = fn = 0
    tp, fp, tn, fn = confusion_from_probs(tp, fp, tn, fn, probs_pos, all_y, threshold)
    metrics = compute_full_metrics(tp, fp, tn, fn)
    metrics["threshold"] = threshold
    metrics["ensemble_size"] = len(states)
    return metrics


@torch.no_grad()
def find_optimal_threshold_ensemble(
    model: nn.Module,
    ensemble: SnapshotEnsemble,
    loader: DataLoader,
    device: str,
    metric: str = "balanced_acc",
) -> float:
    """Search optimal threshold using ensemble average probabilities on val set."""
    model.eval()
    states = ensemble.get_states()
    if not states:
        return 0.5

    all_x, all_y = _stack_loader_inputs(loader, device)
    avg_probs = collect_average_probs(
        model,
        all_x,
        states=states,
        use_tta=False,
        n_augments=1,
        batch_size=DEFAULT_INFERENCE_BATCH_SIZE,
    )
    probs_pos = avg_probs[:, 1]

    best_thr, best_score = 0.5, -1.0
    for thr_int in range(10, 90, 2):
        thr = thr_int / 100.0
        tp = fp = tn = fn = 0
        tp, fp, tn, fn = confusion_from_probs(tp, fp, tn, fn, probs_pos, all_y, thr)
        eps = 1e-9
        rec_pos = tp / (tp + fn + eps)
        rec_neg = tn / (tn + fp + eps)
        bal_acc = 0.5 * (rec_pos + rec_neg)
        total = tp + fp + tn + fn
        acc = (tp + tn) / (total + eps)

        if metric == "macro_f1":
            pre_pos = tp / (tp + fp + eps)
            pre_neg = tn / (tn + fn + eps)
            f1_pos = 2 * pre_pos * rec_pos / (pre_pos + rec_pos + eps)
            f1_neg = 2 * pre_neg * rec_neg / (pre_neg + rec_neg + eps)
            score = 0.5 * (f1_pos + f1_neg)
        elif metric == "balanced_acc":
            score = bal_acc
        else:
            score = acc

        if score > best_score:
            best_score = score
            best_thr = thr

    return best_thr


# ============================================================
# Command accuracy (for dataset2/3 with run/rep/flash metadata)
# ============================================================

@torch.no_grad()
def compute_command_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_k: int = 15,
) -> Dict[int, float]:
    """
    Command-level accuracy at k=1..max_k repetitions.

    The loader must yield (x, y, run, rep, flash) tuples.
    Compatible with EEGInception/kfold_train.py's SubjectFlashDataset.
    """
    model.eval()
    all_probs, all_labels = [], []
    all_runs, all_reps, all_flashes = [], [], []

    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1]
        logits = forward_logits(model, x)
        probs = F.softmax(logits, dim=1)[:, 1].cpu()
        all_probs.append(probs)
        all_labels.append(y)
        all_runs.append(batch[2])
        all_reps.append(batch[3])
        all_flashes.append(batch[4])

    probs_np = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    runs_np = torch.cat(all_runs).numpy()
    reps_np = torch.cat(all_reps).numpy()
    flashes_np = torch.cat(all_flashes).numpy()

    acc_per_k: Dict[int, float] = {}
    for k in range(1, max_k + 1):
        correct = 0
        total = 0
        for run_id in np.unique(runs_np):
            mask_run = runs_np == run_id
            target_flash_candidates = flashes_np[mask_run & (labels_np == 1)]
            if len(target_flash_candidates) == 0:
                continue
            target_flash = int(np.unique(target_flash_candidates)[0])
            avg_prob_per_flash = []
            for fl in np.unique(flashes_np[mask_run]):
                mask = mask_run & (flashes_np == fl) & (reps_np < k)
                if not np.any(mask):
                    avg_prob = -np.inf
                else:
                    avg_prob = probs_np[mask].mean()
                avg_prob_per_flash.append((avg_prob, fl))
            pred_flash = max(avg_prob_per_flash, key=lambda t: t[0])[1]
            total += 1
            if pred_flash == target_flash:
                correct += 1
        acc_per_k[k] = correct / total if total > 0 else 0.0
    return acc_per_k


# ============================================================
# Compute model parameter count
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
