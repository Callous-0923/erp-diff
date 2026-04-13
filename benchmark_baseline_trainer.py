"""
Benchmark baseline trainer — trains baseline models under Protocol B1 and B2.

Protocol B1: Standard training (CE loss, single model, no tricks)
Protocol B2: Enhanced training (FocalLoss, snapshot ensemble, TTA, threshold opt)

All baseline models follow the same interface: model(x) -> logits [B, 2].
"""

import copy
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from benchmark_utils import (
    SnapshotEnsemble,
    collect_probs_and_labels,
    compute_full_metrics,
    confusion_from_probs,
    count_parameters,
    evaluate_ensemble,
    evaluate_standard,
    evaluate_with_tta,
    find_optimal_threshold,
    find_optimal_threshold_ensemble,
)

# Add EEGInception to path for model imports
EEGINCEPTION_DIR = os.path.join(os.path.dirname(__file__), "..", "EEGInception")
if os.path.isdir(EEGINCEPTION_DIR) and EEGINCEPTION_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(EEGINCEPTION_DIR))


def get_baseline_model(name: str, C: int, T: int, device: str) -> nn.Module:
    """
    Factory function for baseline models.
    Mirrors EEGInception/kfold_train.py get_model().
    """
    from EEGInception import EEGInception
    from eegnet import EEGNet
    from mocnn import MOCNN
    from ppnn import PPNN
    from hcann import HCANN
    from tsformer_sa import TSFormerSA
    from transformer import BasicTransformer
    from multi_transformer import MultiHeadTransformerClassifier
    from ICNN import ICNN
    from eeg_dbnet_v2 import EEGDBNetV2

    name = name.lower()
    if name == "eegnet":
        model = EEGNet(input_shape=(C, T), num_classes=2)
    elif name == "eeginception":
        model = EEGInception(input_channels=C, input_samples=T)
    elif name == "mocnn":
        model = MOCNN(C=C, T=T)
    elif name == "ppnn":
        model = PPNN(C=C, T=T)
    elif name == "hcann":
        model = HCANN(C=C, T=T)
    elif name == "tsformer-sa":
        model = TSFormerSA(C=C, T=T)
    elif name == "transformer":
        model = BasicTransformer(C=C, T=T)
    elif name == "multi-transformer":
        model = MultiHeadTransformerClassifier(C=C, T=T)
    elif name == "icnn":
        model = ICNN(in_channels=C, n_samples=T)
    elif name == "eeg-dbnet-v2":
        model = EEGDBNetV2(C=C, T=T)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model.to(device)


ALL_BASELINE_MODELS = [
    "eegnet", "eeginception", "mocnn", "ppnn", "hcann",
    "tsformer-sa", "transformer", "multi-transformer", "icnn", "eeg-dbnet-v2",
]


# ============================================================
# Training helpers
# ============================================================

def compute_class_weights(loader: DataLoader, max_weight: float = 0) -> torch.Tensor:
    """
    Compute class weights from a DataLoader (inverse frequency for pos class).

    Args:
        max_weight: Upper bound for the positive class weight.
                    0 means no limit (use raw ratio).
                    E.g. max_weight=10.0 caps pos_weight at 10.0 even if
                    the true neg/pos ratio is 29.0.
    """
    labels = []
    for batch in loader:
        labels.append(batch[1])
    all_labels = torch.cat(labels)
    counts = torch.bincount(all_labels.long(), minlength=2).float().clamp_min(1.0)
    pos_weight = (counts[0] / counts[1]).item()
    if max_weight > 0:
        pos_weight = min(pos_weight, max_weight)
    print(f"  [ClassWeights] neg={int(counts[0])}, pos={int(counts[1])}, ratio=1:{counts[0]/counts[1]:.1f}, pos_weight={pos_weight:.2f}")
    return torch.tensor([1.0, pos_weight], dtype=torch.float32)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """Standard training epoch (identical to kfold_train.py train_one_epoch)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True).long()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def _eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
) -> Tuple[float, float, float, float]:
    """Validation epoch. Returns (loss, accuracy, balanced_accuracy, rec_pos)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True).long()
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            tp += int(((preds == 1) & (y == 1)).sum().item())
            tn += int(((preds == 0) & (y == 0)).sum().item())
            fp += int(((preds == 1) & (y == 0)).sum().item())
            fn += int(((preds == 0) & (y == 1)).sum().item())
    eps = 1e-9
    rec_pos = tp / (tp + fn + eps)
    rec_neg = tn / (tn + fp + eps)
    bal_acc = 0.5 * (rec_pos + rec_neg)
    return total_loss / max(total, 1), correct / max(total, 1), bal_acc, rec_pos


# ============================================================
# Protocol B1: Standard training
# ============================================================

def train_b1(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    lr: float = 0.001,
    epochs: int = 500,
    patience: int = 20,
    model_name: str = "",
    subject_id: str = "",
    max_class_weight: float = 0,
) -> Dict[str, Any]:
    """
    Protocol B1: Standard baseline training.

    - CrossEntropyLoss with class weights
    - Adam optimizer
    - Early stopping on val_loss
    - Returns best model state dict and training info
    """
    tag = f"[B1][{model_name}][{subject_id}]" if model_name else "[B1]"

    class_weights = compute_class_weights(train_loader, max_weight=max_class_weight).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    stopped_epoch = epochs
    history = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, val_acc, val_bal_acc, val_rec_pos = _eval_one_epoch(model, val_loader, device, criterion)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_bal_acc": val_bal_acc,
            "val_rec_pos": val_rec_pos,
        })

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"{tag}[Epoch {epoch:>3d}] "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} bal_acc={val_bal_acc:.4f} rec_pos={val_rec_pos:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stopped_epoch = epoch
                print(f"{tag} Early stopping at epoch {epoch} (patience={patience})")
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    print(f"{tag} Training done in {train_time:.1f}s, stopped at epoch {stopped_epoch}, best_val_loss={best_val_loss:.4f}")

    return {
        "best_state": best_state,
        "best_val_loss": best_val_loss,
        "stopped_epoch": stopped_epoch,
        "train_time_s": round(train_time, 2),
        "history": history,
    }


# ============================================================
# Protocol B2: Enhanced training
# ============================================================

def train_b2(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    lr: float = 0.001,
    epochs: int = 500,
    patience: int = 20,
    use_focal_loss: bool = True,
    focal_alpha: float = 0.70,
    focal_gamma: float = 2.0,
    use_ensemble: bool = True,
    ensemble_top_k: int = 5,
    model_name: str = "",
    subject_id: str = "",
    max_class_weight: float = 0,
) -> Dict[str, Any]:
    """
    Protocol B2: Enhanced baseline training.

    Compared to B1:
    - Optional FocalLoss instead of CE (B2a)
    - Snapshot ensemble tracking by balanced accuracy (B2c)
    - Threshold optimization and TTA are applied at evaluation time, not here
    """
    from erpdiff_losses import FocalLoss

    tag = f"[B2][{model_name}][{subject_id}]" if model_name else "[B2]"
    loss_type = "FocalLoss" if use_focal_loss else "CE"
    print(f"{tag} Loss={loss_type}, ensemble_top_k={ensemble_top_k}")

    if use_focal_loss:
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma).to(device)
    else:
        class_weights = compute_class_weights(train_loader, max_weight=max_class_weight).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = Adam(model.parameters(), lr=lr)
    ensemble = SnapshotEnsemble(top_k=ensemble_top_k) if use_ensemble else None

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    stopped_epoch = epochs
    history = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, val_acc, val_bal_acc, val_rec_pos = _eval_one_epoch(model, val_loader, device, criterion)

        if ensemble is not None:
            ensemble.update(model, val_bal_acc)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_bal_acc": val_bal_acc,
            "val_rec_pos": val_rec_pos,
        })

        if epoch % 10 == 0 or epoch == 1:
            ens_info = f" ens={len(ensemble)}" if ensemble is not None else ""
            print(
                f"{tag}[Epoch {epoch:>3d}] "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} bal_acc={val_bal_acc:.4f} rec_pos={val_rec_pos:.4f}{ens_info}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stopped_epoch = epoch
                print(f"{tag} Early stopping at epoch {epoch} (patience={patience})")
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    print(f"{tag} Training done in {train_time:.1f}s, stopped at epoch {stopped_epoch}, best_val_loss={best_val_loss:.4f}")

    result = {
        "best_state": best_state,
        "best_val_loss": best_val_loss,
        "stopped_epoch": stopped_epoch,
        "train_time_s": round(train_time, 2),
        "history": history,
    }
    if ensemble is not None:
        result["ensemble"] = ensemble
    return result


# ============================================================
# Full evaluation pipeline for baselines
# ============================================================

def evaluate_baseline_all_protocols(
    model: nn.Module,
    train_result: Dict[str, Any],
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    protocol: str = "B1",
    model_name: str = "",
    subject_id: str = "",
) -> Dict[str, Any]:
    """
    Run evaluation for a baseline model under the specified protocol.

    Protocol B1: single model, threshold=0.5, no TTA
    Protocol B2-Full: ensemble + TTA + threshold optimization

    Returns dict with all metrics.
    """
    tag = f"[{protocol}][{model_name}][{subject_id}]" if model_name else f"[{protocol}]"

    # Use CE for evaluation loss computation (standard, regardless of training loss)
    eval_criterion = nn.CrossEntropyLoss()

    results = {
        "protocol": protocol,
        "train_time_s": train_result["train_time_s"],
        "stopped_epoch": train_result["stopped_epoch"],
        "n_params": count_parameters(model),
    }

    if protocol == "B1":
        # Standard evaluation: single model, argmax (threshold=0.5)
        model.load_state_dict(train_result["best_state"])
        metrics = evaluate_standard(model, test_loader, device, eval_criterion, threshold=0.5)
        results["test"] = metrics

    elif protocol == "B2-Full":
        ensemble = train_result.get("ensemble")

        if ensemble is not None and len(ensemble) > 0:
            print(f"{tag} Searching optimal threshold with ensemble ({len(ensemble)} models) on val set...")
            # Threshold search with ensemble on val set
            opt_thr = find_optimal_threshold_ensemble(
                model, ensemble, val_loader, device, metric="balanced_acc"
            )
            print(f"{tag} Optimal threshold={opt_thr:.2f}, evaluating with ensemble + TTA...")
            # Ensemble + TTA evaluation
            metrics = evaluate_ensemble(
                model, ensemble, test_loader, device, eval_criterion,
                threshold=opt_thr, use_tta=True, n_augments=20,
            )
        else:
            # Fallback: single model with threshold opt + TTA
            model.load_state_dict(train_result["best_state"])
            opt_thr = find_optimal_threshold(model, val_loader, device, metric="balanced_acc")
            print(f"{tag} Optimal threshold={opt_thr:.2f}, evaluating with TTA...")
            metrics = evaluate_with_tta(
                model, test_loader, device, eval_criterion,
                n_augments=20, threshold=opt_thr,
            )
        results["test"] = metrics

    # Print test results (like kfold_train.py)
    m = results.get("test", {})
    print(
        f"{tag}[TEST] "
        f"loss={m.get('test_loss', 0):.4f} | "
        f"acc={m.get('acc', 0):.4f} | "
        f"bal_acc={m.get('macro_rec', 0):.4f} | "
        f"macro_f1={m.get('macro_f1', 0):.4f} | "
        f"macro_pre={m.get('macro_pre', 0):.4f} | "
        f"rec_pos={m.get('rec_pos', 0):.4f} | "
        f"TP={m.get('tp', 0)} FP={m.get('fp', 0)} TN={m.get('tn', 0)} FN={m.get('fn', 0)} | "
        f"thr={m.get('threshold', 0.5):.2f}"
    )

    return results
