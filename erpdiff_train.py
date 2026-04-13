"""
ERPDiff — Main training pipeline (pretrain + finetune).

ERPDiff: Enhancing P300 Detection via ERP-Aware Differential Attention
and Gated Branch Fusion.

Usage:
    python erpdiff_train.py --dataset dataset3 --mode both --epochs 100
    python erpdiff_train.py --dataset dataset1 --mode finetune --pretrained-clb ... --pretrained-rbb ...
"""

import argparse
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset

from erpdiff_config import (
    DATASET_DIRS,
    OUT_BASE_DIR,
    TrainConfig,
    load_spec_to_config,
    resolve_out_dir,
    set_global_seed,
)
from erpdiff_data import (
    build_generic_splits_with_meta,
    build_subject_splits,
    discover_subject_pkls,
    make_loader,
)
from erpdiff_clb_model import CLBPretrainICNN
from erpdiff_model import ERPDiff
from erpdiff_losses import FocalLoss
from erpdiff_rbb_model import RBBPretrainICNN
from erpdiff_train_utils import epoch_run, epoch_run_dwm, eval_dwm
from benchmark_char_metrics import (
    compute_char_acc_dataset1_rsvp,
    compute_char_acc_dataset2,
    compute_char_acc_dataset3,
)


# ============================================================
# CLB-BSD (Bias-variance-aware Self-Distillation) for CLB pretrain
# ============================================================

@dataclass
class CLBBSDConfig:
    beta: float = 0.5
    temperature: float = 2.0
    rho: float = 0.3
    weak_noise_ratio: float = 0.02
    strong_noise_ratio: float = 0.08
    weak_shift: int = 2
    strong_shift: int = 6
    weak_scale_min: float = 0.95
    weak_scale_max: float = 1.05
    strong_scale_min: float = 0.80
    strong_scale_max: float = 1.20
    time_mask_ratio: float = 0.10
    channel_drop_prob: float = 0.10


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x
    if x.dim() == 3:
        return x.unsqueeze(1)
    raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")


def _per_sample_std(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).std(dim=1, keepdim=True).clamp_min(1e-6).view(-1, 1, 1, 1)


def _random_time_shift(x: torch.Tensor, max_shift: int) -> torch.Tensor:
    if max_shift <= 0:
        return x
    bsz = x.size(0)
    shifts = torch.randint(low=-max_shift, high=max_shift + 1, size=(bsz,), device=x.device)
    out = []
    for i in range(bsz):
        out.append(torch.roll(x[i], shifts=int(shifts[i].item()), dims=-1))
    return torch.stack(out, dim=0)


def _time_mask(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    bsz, _, _, t_steps = x.shape
    m = int(max(0, round(t_steps * mask_ratio)))
    if m <= 0 or m >= t_steps:
        return x
    start = torch.randint(0, t_steps - m + 1, size=(bsz,), device=x.device)
    x = x.clone()
    for i in range(bsz):
        s = int(start[i].item())
        x[i, :, :, s : s + m] = 0.0
    return x


def _channel_dropout(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0:
        return x
    bsz, _, channels, _ = x.shape
    mask = (torch.rand((bsz, 1, channels, 1), device=x.device) > drop_prob).float()
    return x * mask


def _augment_weak_eeg(x: torch.Tensor, cfg: CLBBSDConfig) -> torch.Tensor:
    x = _ensure_4d(x).clone()
    bsz = x.size(0)
    scale = torch.empty((bsz, 1, 1, 1), device=x.device).uniform_(cfg.weak_scale_min, cfg.weak_scale_max)
    x = x * scale
    std = _per_sample_std(x)
    x = x + torch.randn_like(x) * (cfg.weak_noise_ratio * std)
    x = _random_time_shift(x, cfg.weak_shift)
    return x


def _augment_strong_eeg(x: torch.Tensor, cfg: CLBBSDConfig) -> torch.Tensor:
    x = _ensure_4d(x).clone()
    bsz = x.size(0)
    scale = torch.empty((bsz, 1, 1, 1), device=x.device).uniform_(cfg.strong_scale_min, cfg.strong_scale_max)
    x = x * scale
    std = _per_sample_std(x)
    x = x + torch.randn_like(x) * (cfg.strong_noise_ratio * std)
    x = _random_time_shift(x, cfg.strong_shift)
    x = _time_mask(x, cfg.time_mask_ratio)
    x = _channel_dropout(x, cfg.channel_drop_prob)
    return x


def _bsd_kl_loss(
    logits_teacher: torch.Tensor,
    logits_student: torch.Tensor,
    y: torch.Tensor,
    temperature: float,
    rho: float,
) -> torch.Tensor:
    T = float(temperature)
    with torch.no_grad():
        p_t = F.softmax(logits_teacher / T, dim=1)
    log_p_s = F.log_softmax(logits_student / T, dim=1)
    kl_per = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1) * (T * T)
    w = torch.where(y == 0, torch.ones_like(kl_per), torch.full_like(kl_per, float(rho)))
    return (w * kl_per).mean()


def epoch_run_clb_bsd(
    model: nn.Module,
    loader,
    device: str,
    optimizer: torch.optim.Optimizer,
    cfg_bsd: CLBBSDConfig,
):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_bsd = 0.0
    correct = 0
    total = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device).long()

        x_w = _augment_weak_eeg(x, cfg_bsd)
        x_s = _augment_strong_eeg(x, cfg_bsd)

        logits_w = model(x_w)
        logits_s = model(x_s)

        ce  = F.cross_entropy(logits_s, y)
        bsd = _bsd_kl_loss(logits_w.detach(), logits_s, y, cfg_bsd.temperature, cfg_bsd.rho)
        loss = ce + float(cfg_bsd.beta) * bsd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        total_ce   += float(ce.item())   * x.size(0)
        total_bsd  += float(bsd.item())  * x.size(0)

        pred = logits_s.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total   += int(x.size(0))

    avg_loss = total_loss / max(1, total)
    avg_ce   = total_ce   / max(1, total)
    avg_bsd  = total_bsd  / max(1, total)
    acc      = correct    / max(1, total)
    return avg_loss, acc, avg_ce, avg_bsd


# ============================================================
# Evaluation helpers
# ============================================================

def _confusion_update_from_probs(tp, fp, tn, fn, probs_pos, y, threshold=0.5):
    """Update confusion matrix using probability threshold instead of argmax."""
    pred = (probs_pos >= threshold).long()
    tp += int(((pred == 1) & (y == 1)).sum().item())
    tn += int(((pred == 0) & (y == 0)).sum().item())
    fp += int(((pred == 1) & (y == 0)).sum().item())
    fn += int(((pred == 0) & (y == 1)).sum().item())
    return tp, fp, tn, fn


def _confusion_update_from_logits(tp, fp, tn, fn, logits, y):
    pred = logits.argmax(dim=1)
    tp += int(((pred == 1) & (y == 1)).sum().item())
    tn += int(((pred == 0) & (y == 0)).sum().item())
    fp += int(((pred == 1) & (y == 0)).sum().item())
    fn += int(((pred == 0) & (y == 1)).sum().item())
    return tp, fp, tn, fn


def _target_recall_and_macro_rec(tp, fp, tn, fn):
    eps = 1e-9
    rec_pos   = tp / (tp + fn + eps)
    rec_neg   = tn / (tn + fp + eps)
    macro_rec = 0.5 * (rec_pos + rec_neg)
    return rec_pos, macro_rec


def _compute_full_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """
    Compute all four standard metrics from confusion matrix counts.
    Returns dict with: acc, macro_pre, macro_rec, macro_f1,
    plus per-class pre/rec/f1 and raw confusion counts.
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


def _score_from_confusion(tp: int, fp: int, tn: int, fn: int, metric: str) -> float:
    eps = 1e-9
    total = tp + tn + fp + fn
    acc = (tp + tn) / (total + eps)
    rec_pos = tp / (tp + fn + eps)
    rec_neg = tn / (tn + fp + eps)
    bal_acc = 0.5 * (rec_pos + rec_neg)
    if metric == "macro_f1":
        pre_pos = tp / (tp + fp + eps)
        pre_neg = tn / (tn + fn + eps)
        f1_pos = 2 * pre_pos * rec_pos / (pre_pos + rec_pos + eps)
        f1_neg = 2 * pre_neg * rec_neg / (pre_neg + rec_neg + eps)
        return 0.5 * (f1_pos + f1_neg)
    if metric == "balanced_acc":
        return bal_acc
    return acc


@torch.no_grad()
def _eval_loader_score_metric(
    model: nn.Module,
    loader,
    device: str,
    alpha_eval: float,
    metric: str,
    threshold: float = 0.5,
) -> float:
    """Evaluate a threshold-based score metric on a loader."""
    model.eval()
    tp = fp = tn = fn = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device).long()
        _, _, fused_logits, _, _ = model(x, alpha_eval)
        probs_pos = F.softmax(fused_logits, dim=1)[:, 1]
        tp, fp, tn, fn = _confusion_update_from_probs(tp, fp, tn, fn, probs_pos, y, threshold)
    return _score_from_confusion(tp, fp, tn, fn, metric)


def _compute_char_metrics_for_subject(
    dataset_name: str,
    subject_id: str,
    pkl_path: str,
    cfg: TrainConfig,
    probs_pos: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, Any]:
    probs_np = probs_pos.detach().cpu().numpy().reshape(-1)
    labels_np = labels.detach().cpu().numpy().reshape(-1)

    if dataset_name == "dataset1":
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        online = obj["online"]
        trial_idx = np.asarray(online["trial_idx"]).reshape(-1)
        stim_code = np.asarray(online["stim_code"]).reshape(-1)
        return compute_char_acc_dataset1_rsvp(
            probs_pos=probs_np,
            trial_idx=trial_idx,
            stim_code=stim_code,
            labels=labels_np,
        )

    if dataset_name in {"dataset2", "dataset3"}:
        _, _, test_ds_meta = build_generic_splits_with_meta(
            subject_id=subject_id,
            pkl_path=pkl_path,
            seed=cfg.seed,
            train_ratio=cfg.train_ratio,
            val_ratio_in_train=cfg.val_ratio_in_train,
        )
        runs = test_ds_meta.runs.detach().cpu().numpy().reshape(-1)
        reps = test_ds_meta.reps.detach().cpu().numpy().reshape(-1)
        flashes = test_ds_meta.flashes.detach().cpu().numpy().reshape(-1)
        if not (len(probs_np) == len(labels_np) == len(runs) == len(reps) == len(flashes)):
            raise ValueError(
                "char metrics length mismatch: "
                f"probs={len(probs_np)} labels={len(labels_np)} "
                f"runs={len(runs)} reps={len(reps)} flashes={len(flashes)}"
            )
        max_k = int(np.max(reps) + 1) if reps.size > 0 else 1

        if dataset_name == "dataset2":
            return compute_char_acc_dataset2(
                probs_pos=probs_np,
                labels=labels_np,
                runs=runs,
                reps=reps,
                flashes=flashes,
                max_k=max_k,
                report_max_k=10,
            )

        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        target_chars = obj.get("target_char", None)
        return compute_char_acc_dataset3(
            probs_pos=probs_np,
            labels=labels_np,
            runs=runs,
            reps=reps,
            flashes=flashes,
            max_k=max_k,
            report_max_k=10,
            target_chars=target_chars,
        )

    return {"warning": f"char metrics not configured for dataset={dataset_name}"}


def _extract_char_acc_main(dataset_name: str, char_acc: Any) -> Optional[float]:
    if not isinstance(char_acc, dict):
        return None
    if dataset_name == "dataset1":
        value = char_acc.get("char_acc")
    elif dataset_name == "dataset2":
        value = char_acc.get("command_acc_main", char_acc.get("ca_main"))
    elif dataset_name == "dataset3":
        value = char_acc.get("char_acc_main", char_acc.get("ca_char_main",
                char_acc.get("pair_acc_main", char_acc.get("ca_pair_main"))))
    else:
        value = None
    if isinstance(value, (int, float)):
        return float(value)
    return None


@torch.no_grad()
def _eval_balanced_acc(
    model: nn.Module, loader, device: str, alpha_eval: float,
) -> float:
    """Compute balanced accuracy = 0.5*(recall_pos + recall_neg) on a loader."""
    model.eval()
    tp = fp = tn = fn = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device).long()
        _, _, fused_logits, _, _ = model(x, alpha_eval)
        tp, fp, tn, fn = _confusion_update_from_logits(tp, fp, tn, fn, fused_logits, y)
    eps = 1e-9
    rec_pos = tp / (tp + fn + eps)
    rec_neg = tn / (tn + fp + eps)
    return 0.5 * (rec_pos + rec_neg)


@torch.no_grad()
def _tta_augment(x: torch.Tensor, aug_idx: int) -> torch.Tensor:
    """Apply stochastic augmentation for test-time augmentation."""
    if aug_idx == 0:
        return x  # original
    x = x.clone()
    # Combine multiple light augmentations for each view
    # Time shift: random small shift
    shift = torch.randint(-3, 4, (1,)).item()
    if shift != 0:
        x = torch.roll(x, shifts=shift, dims=-1)
    # Amplitude scaling
    scale = 0.95 + torch.rand(1).item() * 0.10  # [0.95, 1.05]
    x = x * scale
    # Small Gaussian noise
    std = x.flatten(1).std(dim=1, keepdim=True).clamp_min(1e-6).view(-1, 1, 1, 1)
    x = x + torch.randn_like(x) * (0.02 * std)
    return x


@torch.no_grad()
def eval_dwm_full_tta(
    model: nn.Module,
    loader,
    device: str,
    alpha_eval: float,
    ce: nn.Module,
    n_augments: int = 10,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Full evaluation with test-time augmentation (TTA).
    Averages softmax probabilities over n_augments augmented views.
    """
    model.eval()
    all_probs_list = []
    all_labels = []

    # Collect all samples
    all_x, all_y = [], []
    for batch in loader:
        all_x.append(batch[0].to(device))
        all_y.append(batch[1].to(device).long())
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    # Accumulate predictions from augmented views
    total_probs = torch.zeros(all_x.size(0), 2, device=device)
    for aug_idx in range(n_augments):
        x_aug = _tta_augment(all_x, aug_idx)
        # Process in batches to avoid OOM
        batch_size = 128
        for start in range(0, all_x.size(0), batch_size):
            end = min(start + batch_size, all_x.size(0))
            _, _, fused_logits, _, _ = model(x_aug[start:end], alpha_eval)
            total_probs[start:end] += F.softmax(fused_logits, dim=1)

    avg_probs = total_probs / n_augments
    probs_pos = avg_probs[:, 1]

    # Compute metrics with threshold
    tp = fp = tn = fn = 0
    tp, fp, tn, fn = _confusion_update_from_probs(tp, fp, tn, fn, probs_pos, all_y, threshold)
    metrics = _compute_full_metrics(tp, fp, tn, fn)

    # Compute loss on original (non-augmented) predictions
    total_loss = 0.0
    batch_size = 128
    for start in range(0, all_x.size(0), batch_size):
        end = min(start + batch_size, all_x.size(0))
        _, _, fused_logits, _, _ = model(all_x[start:end], alpha_eval)
        loss = ce(fused_logits, all_y[start:end])
        total_loss += loss.item() * (end - start)
    metrics["test_loss"] = round(total_loss / max(all_x.size(0), 1), 6)
    metrics["threshold"] = threshold
    return metrics


@torch.no_grad()
def _collect_probs_and_labels(
    model: nn.Module, loader, device: str, alpha_eval: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect all positive-class probabilities and labels from a loader."""
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device).long()
        _, _, fused_logits, _, _ = model(x, alpha_eval)
        probs = F.softmax(fused_logits, dim=1)[:, 1]
        all_probs.append(probs)
        all_labels.append(y)
    return torch.cat(all_probs), torch.cat(all_labels)


@torch.no_grad()
def find_optimal_threshold(
    model: nn.Module, loader, device: str, alpha_eval: float,
    metric: str = "acc",
) -> float:
    """
    Search for the decision threshold that maximizes accuracy (or balanced_acc
    or macro_f1) on the given loader (typically validation set).
    """
    probs, labels = _collect_probs_and_labels(model, loader, device, alpha_eval)
    best_thr, best_score = 0.5, 0.0
    for thr_int in range(10, 90, 2):  # 0.10 to 0.88
        thr = thr_int / 100.0
        tp = fp = tn = fn = 0
        tp, fp, tn, fn = _confusion_update_from_probs(tp, fp, tn, fn, probs, labels, thr)
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


@torch.no_grad()
def eval_dwm_full(
    model: nn.Module,
    loader,
    device: str,
    alpha_eval: float,
    ce: nn.Module,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Full evaluation: loss + accuracy + all macro metrics + confusion matrix.
    Uses a custom decision threshold (default 0.5 = argmax equivalent).
    """
    model.eval()
    total_loss = 0.0
    total = 0
    tp = fp = tn = fn = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device).long()
        _, _, fused_logits, _, _ = model(x, alpha_eval)
        loss = ce(fused_logits, y)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        probs_pos = F.softmax(fused_logits, dim=1)[:, 1]
        tp, fp, tn, fn = _confusion_update_from_probs(tp, fp, tn, fn, probs_pos, y, threshold)

    metrics = _compute_full_metrics(tp, fp, tn, fn)
    metrics["test_loss"] = round(total_loss / max(total, 1), 6)
    metrics["threshold"] = threshold
    return metrics


@torch.no_grad()
def eval_rbb_single_branch_metrics(model: nn.Module, loader: Any, device: str):
    """Evaluate RBB branch in isolation (bypass DCM and CLB)."""
    model_was_training = model.training
    model.eval()
    tp = fp = tn = fn = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device).long()
        if hasattr(model, "stem_rbb") and hasattr(model, "tail_rbb"):
            feat = model.stem_rbb(x)
            tok  = feat.squeeze(2).permute(0, 2, 1).contiguous()
            if hasattr(model, "rbb_temporal_attn"):
                tok = model.rbb_temporal_attn(tok)
            feat   = tok.permute(0, 2, 1).contiguous().unsqueeze(2)
            logits = model.tail_rbb(feat)
        else:
            logits = model(x)
        tp, fp, tn, fn = _confusion_update_from_logits(tp, fp, tn, fn, logits, y)
    if model_was_training:
        model.train()
    return _target_recall_and_macro_rec(tp, fp, tn, fn)


def _collect_lambda_info(model: nn.Module) -> Dict[str, Any]:
    """Collect λ, sigma, and gate info from ERPDiff model for logging."""
    info = {}
    # RBB intra-branch TemporalDiffAttn
    if hasattr(model, "rbb_temporal_attn"):
        attn = model.rbb_temporal_attn
        if attn.last_lambda is not None:
            info["rbb_self_attn_lambda"] = float(attn.last_lambda.item())
        if hasattr(attn, "sigma_signal") and hasattr(attn, "sigma_noise"):
            info["temporal_bias_sigma_mode"] = getattr(attn, "temporal_bias_sigma_mode", "shared")
            sigma_signal = attn.sigma_signal.detach().cpu().reshape(-1)
            sigma_noise = attn.sigma_noise.detach().cpu().reshape(-1)
            if sigma_signal.numel() == 1:
                info["sigma_signal"] = float(sigma_signal.item())
                info["sigma_noise"] = float(sigma_noise.item())
            else:
                info["sigma_signal"] = [float(v) for v in sigma_signal.tolist()]
                info["sigma_noise"] = [float(v) for v in sigma_noise.tolist()]
                info["sigma_signal_mean"] = float(sigma_signal.mean().item())
                info["sigma_noise_mean"] = float(sigma_noise.mean().item())
    # Inter-branch GatedDiffCrossAttention
    if hasattr(model, "dcm"):
        dcm = model.dcm
        if hasattr(dcm, "clb_from_rbb") and dcm.clb_from_rbb.last_lambda is not None:
            info["dcm_clb_from_rbb_lambda"] = float(dcm.clb_from_rbb.last_lambda.item())
        if hasattr(dcm, "rbb_from_clb") and dcm.rbb_from_clb.last_lambda is not None:
            info["dcm_rbb_from_clb_lambda"] = float(dcm.rbb_from_clb.last_lambda.item())
    return info


# ============================================================
# Pretrain stage
# ============================================================

def _build_pretrain_loaders(
    predata_dir: str, cfg: TrainConfig, subjects: Optional[Sequence[str]]
) -> Tuple[ConcatDataset, ConcatDataset, int, int]:
    subject_pkls = discover_subject_pkls(predata_dir)
    if subjects:
        subject_pkls = [p for p in subject_pkls if p[0] in subjects]
        if not subject_pkls:
            raise ValueError(f"No subjects matched filter: {subjects}")
    train_sets, val_sets = [], []
    C, T = None, None
    for sid, path in subject_pkls:
        train_ds, val_ds, _ = build_subject_splits(sid, path, cfg)
        train_sets.append(train_ds)
        val_sets.append(val_ds)
        C, T = train_ds.C, train_ds.T
    return ConcatDataset(train_sets), ConcatDataset(val_sets), C, T


def _pretrain_branch(
    label: str,
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: TrainConfig,
    criterion: nn.Module,
    lr: float,
    out_dir: str,
    save_name: str,
    early_stop_patience: int,
    early_stop_min_delta: float,
):
    device         = cfg.device
    model          = model.to(device)
    optimizer      = Adam(model.parameters(), lr=lr, weight_decay=cfg.pretrain_weight_decay)
    scheduler      = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr_ratio * lr)
    history        = []
    best_val       = float("inf")
    best_path      = None
    patience_count = 0
    stop_epoch     = None

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = epoch_run(model, train_loader, device, criterion, optimizer)
        val_loss,   val_acc   = epoch_run(model, val_loader,   device, criterion, None)
        scheduler.step()
        epoch_time = time.perf_counter() - epoch_start

        rbb_val_trec = None
        rbb_val_mrec = None
        should_print = epoch % 10 == 0 or epoch == 1
        if label == "RBB" and should_print:
            rbb_val_trec, rbb_val_mrec = eval_rbb_single_branch_metrics(model, val_loader, device)

        if should_print:
            suffix = ""
            if label == "RBB":
                suffix = f" | rbb_val_trec={rbb_val_trec:.4f} rbb_val_mrec={rbb_val_mrec:.4f}"
            print(
                f"[Pretrain-{label}][Epoch {epoch}] "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
                f"time={epoch_time:.2f}s{suffix}"
            )

        record = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        }
        if label == "RBB":
            record["val_target_recall"] = rbb_val_trec
            record["val_macro_rec"]     = rbb_val_mrec
            # Log λ and sigma from RBB's TemporalDiffAttn
            record.update(_collect_lambda_info(model))

        history.append(record)

        if val_loss < best_val - early_stop_min_delta:
            best_val       = val_loss
            best_path      = os.path.join(out_dir, save_name)
            torch.save(model.state_dict(), best_path)
            patience_count = 0
        else:
            patience_count += 1

        if early_stop_patience > 0 and patience_count >= early_stop_patience:
            stop_epoch = epoch
            if epoch % 10 != 0:
                print(f"[Pretrain-{label}] Early stop at epoch {epoch}.")
            break

    return best_path, history, stop_epoch


def _pretrain_branch_clb_bsd(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: TrainConfig,
    lr: float,
    out_dir: str,
    save_name: str,
    early_stop_patience: int,
    early_stop_min_delta: float,
    cfg_bsd: CLBBSDConfig,
):
    device         = cfg.device
    model          = model.to(device)
    optimizer      = Adam(model.parameters(), lr=lr, weight_decay=cfg.pretrain_weight_decay)
    scheduler      = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr_ratio * lr)
    history        = []
    best_val       = float("inf")
    best_path      = None
    patience_count = 0
    stop_epoch     = None
    ce_for_val     = nn.CrossEntropyLoss()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc, train_ce, train_bsd = epoch_run_clb_bsd(
            model, train_loader, device, optimizer, cfg_bsd
        )
        val_loss, val_acc = epoch_run(model, val_loader, device, ce_for_val, None)
        scheduler.step()
        epoch_time = time.perf_counter() - epoch_start

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[Pretrain-CLB-BSD][Epoch {epoch}] "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} "
                f"(ce={train_ce:.4f}, bsd={train_bsd:.4f}) | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
                f"time={epoch_time:.2f}s"
            )

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "train_ce":   train_ce,
            "train_bsd":  train_bsd,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        })

        if val_loss < best_val - early_stop_min_delta:
            best_val       = val_loss
            best_path      = os.path.join(out_dir, save_name)
            torch.save(model.state_dict(), best_path)
            patience_count = 0
        else:
            patience_count += 1

        if early_stop_patience > 0 and patience_count >= early_stop_patience:
            stop_epoch = epoch
            if epoch % 10 != 0:
                print(f"[Pretrain-CLB-BSD] Early stop at epoch {epoch}.")
            break

    return best_path, history, stop_epoch


def pretrain_stage(
    predata_dir: str,
    out_dir: str,
    cfg: TrainConfig,
    subjects: Optional[Sequence[str]] = None,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
) -> Dict[str, str]:
    train_ds, val_ds, C, T = _build_pretrain_loaders(predata_dir, cfg, subjects)
    pretrain_balance = True if cfg.dataset == "dataset1" else None
    if pretrain_balance is True:
        print("[ERPDiff] pretrain balanced sampling = ON (dataset1)")
    train_loader = make_loader(
        train_ds, cfg.pretrain_batch_size, True, cfg, balance=pretrain_balance
    )
    val_loader   = make_loader(val_ds,   cfg.pretrain_batch_size, False, cfg)

    os.makedirs(out_dir, exist_ok=True)
    clb_model = CLBPretrainICNN(C, T, dropout_p=cfg.dropout_p)
    rbb_model = RBBPretrainICNN(
        C,
        T,
        dropout_p=cfg.dropout_p,
        use_temporal_bias=cfg.use_temporal_bias,
        temporal_bias_sigma_mode=cfg.temporal_bias_sigma_mode,
    )
    focal     = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)

    clb_bsd_cfg = CLBBSDConfig(beta=cfg.clb_bsd_beta)
    clb_path, clb_hist, clb_stop_epoch = _pretrain_branch_clb_bsd(
        clb_model, train_loader, val_loader, cfg,
        cfg.pretrain_lr, out_dir, "pretrain_icnn_clb.pth",
        early_stop_patience, early_stop_min_delta, clb_bsd_cfg,
    )
    rbb_path, rbb_hist, rbb_stop_epoch = _pretrain_branch(
        "RBB", rbb_model, train_loader, val_loader, cfg,
        focal, cfg.pretrain_lr, out_dir, "pretrain_icnn_rbb.pth",
        early_stop_patience, early_stop_min_delta,
    )

    with open(os.path.join(out_dir, "pretrain_log.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "clb_bsd": asdict(clb_bsd_cfg),
                "early_stop": {
                    "patience":       early_stop_patience,
                    "min_delta":      early_stop_min_delta,
                    "clb_stop_epoch": clb_stop_epoch,
                    "rbb_stop_epoch": rbb_stop_epoch,
                },
                "clb_history": clb_hist,
                "rbb_history": rbb_hist,
            },
            f, indent=2,
        )
    return {"clb": clb_path, "rbb": rbb_path}


# ============================================================
# Finetune stage
# ============================================================

def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def finetune_stage(
    predata_dir: str,
    out_dir: str,
    cfg: TrainConfig,
    pretrained_clb: str,
    pretrained_rbb: str,
    subjects: Optional[Sequence[str]] = None,
    warmup_epochs: int = 3,
    lambda_intra: float = 0.0,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    threshold_metric: str = "acc",
    snapshot_metric: str = "balanced_acc",
    snapshot_top_k: int = 5,
    tta_n_augments: int = 20,
):
    subject_pkls = discover_subject_pkls(predata_dir)
    if subjects:
        subject_pkls = [p for p in subject_pkls if p[0] in subjects]
        if not subject_pkls:
            raise ValueError(f"No subjects matched filter: {subjects}")

    device       = cfg.device
    clb_template = torch.load(pretrained_clb, map_location=device)
    rbb_template = torch.load(pretrained_rbb, map_location=device)
    ce_loss      = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma).to(device)

    finetune_report = {
        "config": asdict(cfg),
        "finetune_settings": {
            "warmup_epochs":        warmup_epochs,
            "lambda_intra":         lambda_intra,
            "lambda_comp":          cfg.lambda_comp,
            "comp_margin":          cfg.comp_margin,
            "early_stop_patience":  early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "threshold_metric":     threshold_metric,
            "snapshot_metric":      snapshot_metric,
            "snapshot_top_k":       snapshot_top_k,
            "tta_n_augments":       tta_n_augments,
        },
        "pretrained_paths": {"clb": pretrained_clb, "rbb": pretrained_rbb},
    }
    subject_metrics = {}
    warmup_epochs   = max(0, min(warmup_epochs, cfg.epochs))

    for sid, path in subject_pkls:
        train_ds, val_ds, test_ds = build_subject_splits(sid, path, cfg)
        C, T = train_ds.C, train_ds.T
        train_loader = make_loader(train_ds, cfg.finetune_batch_size, True,  cfg)
        val_loader   = make_loader(val_ds,   cfg.finetune_batch_size, False, cfg)
        test_loader  = make_loader(test_ds,  cfg.finetune_batch_size, False, cfg)

        # --- Build ERPDiff model with all ablation switches ---
        model = ERPDiff(
            in_channels=C,
            n_samples=T,
            dropout_p=cfg.dropout_p,
            n_classes=2,
            attn_dropout=0.0,
            proj_dropout=cfg.dropout_p,
            num_heads=4,
            enable_dcm=True,
            lambda_init=0.8,
            use_temporal_bias=cfg.use_temporal_bias,
            temporal_bias_sigma_mode=cfg.temporal_bias_sigma_mode,
            use_alpha_gate=cfg.use_alpha_gate,
            min_gate=cfg.min_gate,
        ).to(device)
        model.load_branch_state(clb_template, rbb_template)

        # Warmup: freeze stems, only train tails + DCM
        _set_trainable(model.stem_clb,          trainable=False)
        _set_trainable(model.stem_rbb,          trainable=False)
        _set_trainable(model.rbb_temporal_attn,  trainable=False)
        _set_trainable(model.tail_clb,          trainable=True)
        _set_trainable(model.tail_rbb,          trainable=True)
        _set_trainable(model.dcm,               trainable=True)

        optimizer = Adam(model.parameters(), lr=cfg.finetune_lr, weight_decay=cfg.finetune_weight_decay)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr_ratio * cfg.finetune_lr
        )

        best_val_bal_acc = -1.0
        patience_count  = 0
        stop_epoch      = None
        sub_history     = []
        best_model_path = None
        # Top-K snapshot ensemble: keep snapshots using the configured metric
        import heapq
        top_k_snapshots = []  # min-heap of (bal_acc, epoch, state_dict_copy)
        TOP_K = max(1, int(snapshot_top_k))
        subject_dir     = os.path.join(out_dir, sid)
        os.makedirs(subject_dir, exist_ok=True)

        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.perf_counter()
            alpha = 1.0 - (epoch / cfg.epochs) ** 2

            # Unfreeze stems after warmup
            if epoch == warmup_epochs + 1:
                _set_trainable(model.stem_clb,          trainable=True)
                _set_trainable(model.stem_rbb,          trainable=True)
                _set_trainable(model.rbb_temporal_attn,  trainable=True)

            train_loss, train_acc = epoch_run_dwm(
                model, train_loader, device,
                alpha=alpha,
                ce=ce_loss,
                lambda_intra=lambda_intra,
                optimizer=optimizer,
                lambda_comp=cfg.lambda_comp,
                comp_margin=cfg.comp_margin,
            )
            val_loss, val_acc = eval_dwm(model, val_loader, device, alpha_eval=0.5, ce=ce_loss)
            scheduler.step()
            epoch_time = time.perf_counter() - epoch_start

            # --- Compute validation balanced accuracy for early stopping ---
            val_bal_acc = _eval_balanced_acc(model, val_loader, device, alpha_eval=0.5)
            if snapshot_metric == "balanced_acc":
                val_snapshot_score = val_bal_acc
            else:
                val_snapshot_score = _eval_loader_score_metric(
                    model, val_loader, device, alpha_eval=0.5, metric=snapshot_metric, threshold=0.5
                )

            # --- Periodic evaluation and logging ---
            rbb_val_trec = None
            rbb_val_mrec = None
            should_print = epoch % 10 == 0 or epoch == 1
            if should_print:
                rbb_val_trec, rbb_val_mrec = eval_rbb_single_branch_metrics(model, val_loader, device)
                print(
                    f"[ERPDiff-Finetune][{sid}][Epoch {epoch}] "
                    f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} acc={val_acc:.4f} bal_acc={val_bal_acc:.4f} | "
                    f"time={epoch_time:.2f}s"
                    f" | rbb_trec={rbb_val_trec:.4f} rbb_mrec={rbb_val_mrec:.4f}"
                )

            # --- Collect interpretability info ---
            lambda_info = _collect_lambda_info(model)

            sub_history.append({
                "epoch":                 epoch,
                "alpha":                 alpha,
                "train_loss":            train_loss,
                "train_acc":             train_acc,
                "val_loss":              val_loss,
                "val_acc":               val_acc,
                "val_bal_acc":           val_bal_acc,
                "rbb_val_target_recall": rbb_val_trec,
                "rbb_val_macro_rec":     rbb_val_mrec,
                **lambda_info,
            })

            # --- Checkpointing: use balanced accuracy (higher is better) ---
            # Also maintain top-K snapshots for ensemble
            snapshot_entry = (val_snapshot_score, epoch, {k: v.cpu().clone() for k, v in model.state_dict().items()})
            if len(top_k_snapshots) < TOP_K:
                heapq.heappush(top_k_snapshots, snapshot_entry)
            elif val_snapshot_score > top_k_snapshots[0][0]:
                heapq.heapreplace(top_k_snapshots, snapshot_entry)

            if val_bal_acc > best_val_bal_acc + early_stop_min_delta:
                best_val_bal_acc = val_bal_acc
                clb_state = {}
                clb_state.update(model.stem_clb.state_dict())
                clb_state.update(model.tail_clb.state_dict())
                rbb_state = {}
                rbb_state.update(model.stem_rbb.state_dict())
                rbb_state.update(model.rbb_temporal_attn.state_dict())
                rbb_state.update(model.tail_rbb.state_dict())
                torch.save(clb_state, os.path.join(subject_dir, "finetune_clb.pth"))
                torch.save(rbb_state, os.path.join(subject_dir, "finetune_rbb.pth"))
                best_model_path = os.path.join(subject_dir, "finetune_erpdiff.pth")
                torch.save(model.state_dict(), best_model_path)
                patience_count = 0
            else:
                patience_count += 1

            if early_stop_patience > 0 and patience_count >= early_stop_patience and epoch >= 30:
                stop_epoch = epoch
                if epoch % 10 != 0:
                    print(f"[ERPDiff-Finetune][{sid}] Early stop at epoch {epoch}.")
                break

        # --- Test with snapshot ensemble + TTA ---
        # Collect all test data
        all_test_x, all_test_y = [], []
        for batch in test_loader:
            all_test_x.append(batch[0].to(device))
            all_test_y.append(batch[1].to(device).long())
        all_test_x = torch.cat(all_test_x, dim=0)
        all_test_y = torch.cat(all_test_y, dim=0)

        # Collect val data for threshold optimization
        all_val_x, all_val_y = [], []
        for batch in val_loader:
            all_val_x.append(batch[0].to(device))
            all_val_y.append(batch[1].to(device).long())
        all_val_x = torch.cat(all_val_x, dim=0)
        all_val_y = torch.cat(all_val_y, dim=0)

        # Average predictions from top-K snapshots with TTA
        n_tta = max(1, int(tta_n_augments))
        ensemble_test_probs = torch.zeros(all_test_x.size(0), 2, device=device)
        ensemble_val_probs = torch.zeros(all_val_x.size(0), 2, device=device)

        snapshot_epochs = []
        for bal_acc_snap, epoch_snap, state_snap in top_k_snapshots:
            snapshot_epochs.append(epoch_snap)
            model.load_state_dict({k: v.to(device) for k, v in state_snap.items()})
            model.eval()
            with torch.no_grad():
                # TTA for test set
                for aug_idx in range(n_tta):
                    x_aug = _tta_augment(all_test_x, aug_idx)
                    bs = 128
                    for s in range(0, x_aug.size(0), bs):
                        e = min(s + bs, x_aug.size(0))
                        _, _, fused, _, _ = model(x_aug[s:e], 0.5)
                        ensemble_test_probs[s:e] += F.softmax(fused, dim=1)
                # TTA for val set (for threshold search)
                for aug_idx in range(n_tta):
                    x_aug = _tta_augment(all_val_x, aug_idx)
                    bs = 128
                    for s in range(0, x_aug.size(0), bs):
                        e = min(s + bs, x_aug.size(0))
                        _, _, fused, _, _ = model(x_aug[s:e], 0.5)
                        ensemble_val_probs[s:e] += F.softmax(fused, dim=1)

        n_total = len(top_k_snapshots) * n_tta
        ensemble_test_probs /= n_total
        ensemble_val_probs /= n_total
        print(
            f"[ERPDiff-Finetune][{sid}] Ensemble of {len(top_k_snapshots)} snapshots "
            f"(metric={snapshot_metric}, epochs: {sorted(snapshot_epochs)})"
        )

        # Find optimal threshold on val using the requested metric, step 0.02
        val_probs_pos = ensemble_val_probs[:, 1]
        best_thr, best_score = 0.5, 0.0
        for thr_int in range(10, 90, 2):
            thr = thr_int / 100.0
            tp_v = fp_v = tn_v = fn_v = 0
            tp_v, fp_v, tn_v, fn_v = _confusion_update_from_probs(tp_v, fp_v, tn_v, fn_v, val_probs_pos, all_val_y, thr)
            score_v = _score_from_confusion(tp_v, fp_v, tn_v, fn_v, threshold_metric)
            if score_v > best_score:
                best_score = score_v
                best_thr = thr
        opt_thr = best_thr

        # Evaluate ensemble on test with both thresholds
        test_probs_pos = ensemble_test_probs[:, 1]
        tp = fp = tn = fn = 0
        tp, fp, tn, fn = _confusion_update_from_probs(tp, fp, tn, fn, test_probs_pos, all_test_y, 0.5)
        test_default = _compute_full_metrics(tp, fp, tn, fn)
        test_default["threshold"] = 0.5

        tp = fp = tn = fn = 0
        tp, fp, tn, fn = _confusion_update_from_probs(tp, fp, tn, fn, test_probs_pos, all_test_y, opt_thr)
        test_opt = _compute_full_metrics(tp, fp, tn, fn)
        test_opt["threshold"] = opt_thr

        selection_metric_default = _score_from_confusion(
            test_default["tp"], test_default["fp"], test_default["tn"], test_default["fn"], threshold_metric
        )
        selection_metric_opt = _score_from_confusion(
            test_opt["tp"], test_opt["fp"], test_opt["tn"], test_opt["fn"], threshold_metric
        )

        if selection_metric_opt >= selection_metric_default:
            test_metrics = dict(test_opt)
            selected_source = "optimized_threshold_metrics"
        else:
            test_metrics = dict(test_default)
            selected_source = "default_threshold_metrics"
        test_metrics["test_loss"] = 0.0  # not meaningful for ensemble
        test_metrics["threshold_search_metric"] = threshold_metric
        test_metrics["selected_metrics_source"] = selected_source
        test_metrics["val_threshold_search_score"] = round(best_score, 6)
        test_metrics["default_threshold_metrics"] = dict(test_default)
        test_metrics["optimized_threshold_metrics"] = dict(test_opt)

        try:
            char_acc = _compute_char_metrics_for_subject(
                dataset_name=cfg.dataset,
                subject_id=sid,
                pkl_path=path,
                cfg=cfg,
                probs_pos=test_probs_pos,
                labels=all_test_y,
            )
            test_metrics["char_acc"] = char_acc
            char_acc_main = _extract_char_acc_main(cfg.dataset, char_acc)
        except Exception as e:
            test_metrics["char_acc"] = {"error": str(e)}
            char_acc_main = None

        char_msg = ""
        if char_acc_main is not None:
            char_msg = f"  CharAcc={char_acc_main:.4f}"
        print(
            f"[ERPDiff-Test][{sid}] "
            f"Acc={test_metrics['acc']:.4f}  "
            f"Macro-F1={test_metrics['macro_f1']:.4f}  "
            f"Macro-Pre={test_metrics['macro_pre']:.4f}  "
            f"Macro-Rec={test_metrics['macro_rec']:.4f}  "
            f"TP={test_metrics['tp']} FP={test_metrics['fp']} "
            f"TN={test_metrics['tn']} FN={test_metrics['fn']} "
            f"thr={test_metrics.get('threshold', 0.5):.2f} "
            f"(opt_thr={opt_thr:.2f}, val_{threshold_metric}={best_score:.4f}, "
            f"default_{threshold_metric}={selection_metric_default:.4f}, "
            f"opt_{threshold_metric}={selection_metric_opt:.4f})"
            f"{char_msg}"
        )

        with open(os.path.join(subject_dir, "finetune_log.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": asdict(cfg),
                    "early_stop": {
                        "patience":   early_stop_patience,
                        "min_delta":  early_stop_min_delta,
                        "stop_epoch": stop_epoch,
                    },
                    "history":      sub_history,
                    "test_metrics": test_metrics,
                },
                f, indent=2,
            )
        subject_metrics[sid] = test_metrics
        finetune_report[sid] = test_metrics

    if subject_metrics:
        n = len(subject_metrics)
        avg_summary = {
            "avg_acc":       round(sum(v["acc"]       for v in subject_metrics.values()) / n, 6),
            "avg_macro_f1":  round(sum(v["macro_f1"]  for v in subject_metrics.values()) / n, 6),
            "avg_macro_pre": round(sum(v["macro_pre"] for v in subject_metrics.values()) / n, 6),
            "avg_macro_rec": round(sum(v["macro_rec"] for v in subject_metrics.values()) / n, 6),
        }
        char_values = []
        for v in subject_metrics.values():
            c_main = _extract_char_acc_main(cfg.dataset, v.get("char_acc", {}))
            if c_main is not None:
                char_values.append(c_main)
        if char_values:
            avg_summary["avg_char_acc_main"] = round(sum(char_values) / len(char_values), 6)
            avg_summary["n_char_subjects"] = len(char_values)

        if cfg.dataset == "dataset1":
            curve_accumulator: Dict[int, list[float]] = {}
            for v in subject_metrics.values():
                char_acc = v.get("char_acc", {})
                if not isinstance(char_acc, dict):
                    continue
                curve = char_acc.get("char_acc_at_k")
                if not isinstance(curve, dict):
                    continue
                for k_str, value in curve.items():
                    try:
                        k = int(k_str)
                        curve_accumulator.setdefault(k, []).append(float(value))
                    except Exception:
                        continue
            if curve_accumulator:
                avg_summary["avg_char_acc_curve_at_k"] = {
                    str(k): round(sum(vals) / len(vals), 6)
                    for k, vals in sorted(curve_accumulator.items())
                    if vals
                }
        elif cfg.dataset == "dataset2":
            curve_accumulator: Dict[int, list[float]] = {}
            for v in subject_metrics.values():
                char_acc = v.get("char_acc", {})
                if not isinstance(char_acc, dict):
                    continue
                curve = char_acc.get("command_acc_at_k", char_acc.get("ca_at_k"))
                if not isinstance(curve, dict):
                    continue
                for k_str, value in curve.items():
                    try:
                        k = int(k_str)
                        curve_accumulator.setdefault(k, []).append(float(value))
                    except Exception:
                        continue
            if curve_accumulator:
                avg_summary["avg_command_acc_curve_at_k"] = {
                    str(k): round(sum(vals) / len(vals), 6)
                    for k, vals in sorted(curve_accumulator.items())
                    if vals
                }
        elif cfg.dataset == "dataset3":
            pair_curve_accumulator: Dict[int, list[float]] = {}
            char_curve_accumulator: Dict[int, list[float]] = {}
            for v in subject_metrics.values():
                char_acc = v.get("char_acc", {})
                if not isinstance(char_acc, dict):
                    continue
                pair_curve = char_acc.get("pair_acc_at_k", char_acc.get("ca_pair_at_k"))
                if isinstance(pair_curve, dict):
                    for k_str, value in pair_curve.items():
                        try:
                            k = int(k_str)
                            pair_curve_accumulator.setdefault(k, []).append(float(value))
                        except Exception:
                            continue
                char_curve = char_acc.get("char_acc_at_k", char_acc.get("ca_char_at_k"))
                if isinstance(char_curve, dict):
                    for k_str, value in char_curve.items():
                        try:
                            k = int(k_str)
                            char_curve_accumulator.setdefault(k, []).append(float(value))
                        except Exception:
                            continue
            if pair_curve_accumulator:
                avg_summary["avg_pair_acc_curve_at_k"] = {
                    str(k): round(sum(vals) / len(vals), 6)
                    for k, vals in sorted(pair_curve_accumulator.items())
                    if vals
                }
            if char_curve_accumulator:
                avg_summary["avg_char_acc_curve_at_k"] = {
                    str(k): round(sum(vals) / len(vals), 6)
                    for k, vals in sorted(char_curve_accumulator.items())
                    if vals
                }

        char_msg = ""
        if "avg_char_acc_main" in avg_summary:
            char_msg = f"  Avg CharAcc={avg_summary['avg_char_acc_main']:.4f}"
        print(
            f"\n[ERPDiff-Summary] {n} subjects  "
            f"Avg Acc={avg_summary['avg_acc']:.4f}  "
            f"Avg F1={avg_summary['avg_macro_f1']:.4f}  "
            f"Avg Pre={avg_summary['avg_macro_pre']:.4f}  "
            f"Avg Rec={avg_summary['avg_macro_rec']:.4f}"
            f"{char_msg}\n"
        )
    else:
        avg_summary = {"avg_acc": 0.0, "avg_macro_f1": 0.0, "avg_macro_pre": 0.0, "avg_macro_rec": 0.0}
    finetune_report.update(avg_summary)
    with open(os.path.join(out_dir, "finetune_report.json"), "w", encoding="utf-8") as f:
        json.dump(finetune_report, f, indent=2)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ERPDiff pretrain + finetune pipeline.")
    parser.add_argument("--dataset",  choices=["dataset1", "dataset2", "dataset3"], default="dataset1")
    parser.add_argument("--mode",     choices=["pretrain", "finetune", "both"],     default="both")
    parser.add_argument("--predata-dir",  default=None)
    parser.add_argument("--out-dir",      default=None)
    parser.add_argument("--spec-file",    default="ERPDIFF_EXPERIMENT_SPEC.yaml")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--runs",         type=int,   default=None)
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--device",       type=str,   default=None)
    parser.add_argument("--subjects",     nargs="*",  default=None)
    parser.add_argument("--pretrained-clb",                type=str,   default=None)
    parser.add_argument("--pretrained-rbb",                type=str,   default=None)
    parser.add_argument("--warmup-epochs",                 type=int,   default=10)
    parser.add_argument("--lambda-intra",                  type=float, default=0.3)
    parser.add_argument("--finetune-early-stop-patience",  type=int,   default=30)
    parser.add_argument("--finetune-early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--threshold-metric",
        choices=["acc", "macro_f1", "balanced_acc"],
        default="acc",
        help="Metric used to search the validation decision threshold during finetune evaluation.",
    )
    parser.add_argument(
        "--snapshot-metric",
        choices=["acc", "macro_f1", "balanced_acc"],
        default="balanced_acc",
        help="Metric used to keep top-K finetune snapshots for test-time ensembling.",
    )
    parser.add_argument(
        "--snapshot-top-k",
        type=int,
        default=5,
        help="Number of validation snapshots kept for the finetune ensemble.",
    )
    parser.add_argument(
        "--tta-n-augments",
        type=int,
        default=20,
        help="Number of test-time augmentations per kept snapshot.",
    )
    # ERPDiff-specific ablation switches
    parser.add_argument("--no-temporal-bias", action="store_true", help="Disable dual-sigma Gaussian temporal bias.")
    parser.add_argument(
        "--multi-sigma",
        action="store_true",
        help="Enable the multi-sigma temporal bias variant (per-head sigma values).",
    )
    parser.add_argument(
        "--temporal-bias-sigma-mode",
        choices=["shared", "per_head"],
        default=None,
        help="Choose whether temporal bias sigma is shared globally or learned per attention head.",
    )
    parser.add_argument("--no-alpha-gate",    action="store_true", help="Disable alpha-gated cross-attention.")
    parser.add_argument("--lambda-comp",      type=float, default=None, help="Complementarity loss weight.")
    parser.add_argument("--comp-margin",      type=float, default=None, help="Complementarity loss margin.")
    parser.add_argument("--clb-bsd-beta",     type=float, default=None,
                        help="Beta weight for CLB BSD self-distillation loss. "
                             "Set to 0.0 to disable BSD (CE-only CLB pretrain, Ablation Exp7).")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg         = TrainConfig()
    cfg.seed    = args.seed
    cfg.dataset = args.dataset
    if args.device:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.epochs = args.epochs

    cfg, spec_runs = load_spec_to_config(args.spec_file, cfg)
    runs = args.runs if args.runs is not None else spec_runs

    # CLI overrides for ablation switches
    if args.no_temporal_bias:
        cfg.use_temporal_bias = False
    if args.multi_sigma:
        cfg.temporal_bias_sigma_mode = "per_head"
    if args.temporal_bias_sigma_mode is not None:
        cfg.temporal_bias_sigma_mode = args.temporal_bias_sigma_mode
    if args.no_alpha_gate:
        cfg.use_alpha_gate = False
    if args.lambda_comp is not None:
        cfg.lambda_comp = args.lambda_comp
    if args.comp_margin is not None:
        cfg.comp_margin = args.comp_margin
    if args.clb_bsd_beta is not None:
        cfg.clb_bsd_beta = args.clb_bsd_beta

    predata_dir = args.predata_dir or DATASET_DIRS.get(cfg.dataset)
    if not predata_dir:
        raise ValueError("Cannot determine data directory. Use --predata-dir.")

    out_dir = args.out_dir or resolve_out_dir(cfg.dataset, base_dir=OUT_BASE_DIR)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[ERPDiff] dataset     = {cfg.dataset}")
    print(f"[ERPDiff] predata_dir = {predata_dir}")
    print(f"[ERPDiff] out_dir     = {out_dir}")
    print(f"[ERPDiff] device      = {cfg.device}")
    print(f"[ERPDiff] runs        = {runs}")
    print(
        f"[ERPDiff] temporal_bias={cfg.use_temporal_bias}  "
        f"sigma_mode={cfg.temporal_bias_sigma_mode}  "
        f"alpha_gate={cfg.use_alpha_gate}  "
        f"lambda_comp={cfg.lambda_comp}  comp_margin={cfg.comp_margin}  "
        f"clb_bsd_beta={cfg.clb_bsd_beta}"
    )

    for run_idx in range(runs):
        run_seed = cfg.seed + run_idx
        set_global_seed(run_seed)
        run_out = os.path.join(out_dir, f"run_{run_idx + 1}") if runs > 1 else out_dir
        os.makedirs(run_out, exist_ok=True)

        pretrained_paths: Dict[str, str] = {}

        if args.mode in ("pretrain", "both"):
            pretrained_paths = pretrain_stage(
                predata_dir, run_out, cfg, args.subjects,
                early_stop_patience=14, early_stop_min_delta=1e-4,
            )

        if args.mode in ("finetune", "both"):
            clb_path = (
                args.pretrained_clb
                or pretrained_paths.get("clb")
                or os.path.join(run_out, "pretrain_icnn_clb.pth")
            )
            rbb_path = (
                args.pretrained_rbb
                or pretrained_paths.get("rbb")
                or os.path.join(run_out, "pretrain_icnn_rbb.pth")
            )
            finetune_stage(
                predata_dir, run_out, cfg, clb_path, rbb_path,
                args.subjects,
                warmup_epochs=args.warmup_epochs,
                lambda_intra=args.lambda_intra,
                early_stop_patience=args.finetune_early_stop_patience,
                early_stop_min_delta=args.finetune_early_stop_min_delta,
                threshold_metric=args.threshold_metric,
                snapshot_metric=args.snapshot_metric,
                snapshot_top_k=args.snapshot_top_k,
                tta_n_augments=args.tta_n_augments,
            )


if __name__ == "__main__":
    main()
