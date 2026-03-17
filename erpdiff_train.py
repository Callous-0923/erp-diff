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
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

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
from erpdiff_data import build_subject_splits, discover_subject_pkls, make_loader
from erpdiff_clb_model import CLBPretrainICNN
from erpdiff_model import ERPDiff
from erpdiff_losses import FocalLoss
from erpdiff_rbb_model import RBBPretrainICNN
from erpdiff_train_utils import epoch_run, epoch_run_dwm, eval_dwm


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


def _collect_lambda_info(model: nn.Module) -> Dict[str, float]:
    """Collect λ, sigma, and gate info from ERPDiff model for logging."""
    info = {}
    # RBB intra-branch TemporalDiffAttn
    if hasattr(model, "rbb_temporal_attn"):
        attn = model.rbb_temporal_attn
        if attn.last_lambda is not None:
            info["rbb_self_attn_lambda"] = float(attn.last_lambda.item())
        if hasattr(attn, "sigma_signal") and hasattr(attn, "sigma_noise"):
            info["sigma_signal"] = float(attn.sigma_signal.item())
            info["sigma_noise"]  = float(attn.sigma_noise.item())
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
    train_loader = make_loader(train_ds, cfg.pretrain_batch_size, True,  cfg)
    val_loader   = make_loader(val_ds,   cfg.pretrain_batch_size, False, cfg)

    os.makedirs(out_dir, exist_ok=True)
    clb_model = CLBPretrainICNN(C, T, dropout_p=cfg.dropout_p)
    rbb_model = RBBPretrainICNN(C, T, dropout_p=cfg.dropout_p, use_temporal_bias=cfg.use_temporal_bias)
    focal     = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)

    clb_bsd_cfg = CLBBSDConfig()
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
):
    subject_pkls = discover_subject_pkls(predata_dir)
    if subjects:
        subject_pkls = [p for p in subject_pkls if p[0] in subjects]
        if not subject_pkls:
            raise ValueError(f"No subjects matched filter: {subjects}")

    device       = cfg.device
    clb_template = torch.load(pretrained_clb, map_location=device)
    rbb_template = torch.load(pretrained_rbb, map_location=device)
    ce_loss      = nn.CrossEntropyLoss()

    finetune_report = {
        "config": asdict(cfg),
        "finetune_settings": {
            "warmup_epochs":        warmup_epochs,
            "lambda_intra":         lambda_intra,
            "lambda_comp":          cfg.lambda_comp,
            "comp_margin":          cfg.comp_margin,
            "early_stop_patience":  early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
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

        best_val        = float("inf")
        patience_count  = 0
        stop_epoch      = None
        sub_history     = []
        best_model_path = None
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

            # --- Periodic evaluation and logging ---
            rbb_val_trec = None
            rbb_val_mrec = None
            should_print = epoch % 10 == 0 or epoch == 1
            if should_print:
                rbb_val_trec, rbb_val_mrec = eval_rbb_single_branch_metrics(model, val_loader, device)
                print(
                    f"[ERPDiff-Finetune][{sid}][Epoch {epoch}] "
                    f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
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
                "rbb_val_target_recall": rbb_val_trec,
                "rbb_val_macro_rec":     rbb_val_mrec,
                **lambda_info,
            })

            # --- Checkpointing ---
            if val_loss < best_val - early_stop_min_delta:
                best_val = val_loss
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

            if early_stop_patience > 0 and patience_count >= early_stop_patience:
                stop_epoch = epoch
                if epoch % 10 != 0:
                    print(f"[ERPDiff-Finetune][{sid}] Early stop at epoch {epoch}.")
                break

        # --- Test with best model ---
        if best_model_path and os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_loss, test_acc = eval_dwm(model, test_loader, device, alpha_eval=0.5, ce=ce_loss)

        with open(os.path.join(subject_dir, "finetune_log.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": asdict(cfg),
                    "early_stop": {
                        "patience":   early_stop_patience,
                        "min_delta":  early_stop_min_delta,
                        "stop_epoch": stop_epoch,
                    },
                    "history":   sub_history,
                    "test_loss": test_loss,
                    "test_acc":  test_acc,
                },
                f, indent=2,
            )
        subject_metrics[sid] = {"test_loss": test_loss, "test_acc": test_acc}
        finetune_report[sid] = subject_metrics[sid]

    if subject_metrics:
        avg_acc = sum(v["test_acc"] for v in subject_metrics.values()) / len(subject_metrics)
    else:
        avg_acc = 0.0
    finetune_report["avg_test_acc"] = avg_acc
    with open(os.path.join(out_dir, "finetune_report.json"), "w", encoding="utf-8") as f:
        json.dump(finetune_report, f, indent=2)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ERPDiff pretrain + finetune pipeline.")
    parser.add_argument("--dataset",  choices=["dataset1", "dataset2", "dataset3"], default="dataset3")
    parser.add_argument("--mode",     choices=["pretrain", "finetune", "both"],     default="both")
    parser.add_argument("--predata-dir",  default=None)
    parser.add_argument("--out-dir",      default=None)
    parser.add_argument("--spec-file",    default="ERPDIFF_EXPERIMENT_SPEC.yaml")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--runs",         type=int,   default=None)
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--device",       type=str,   default=None)
    parser.add_argument("--subjects",     nargs="*",  default=None)
    parser.add_argument("--pretrained-clb",                type=str,   default=None)
    parser.add_argument("--pretrained-rbb",                type=str,   default=None)
    parser.add_argument("--warmup-epochs",                 type=int,   default=10)
    parser.add_argument("--lambda-intra",                  type=float, default=0.05)
    parser.add_argument("--finetune-early-stop-patience",  type=int,   default=10)
    parser.add_argument("--finetune-early-stop-min-delta", type=float, default=1e-4)
    # ERPDiff-specific ablation switches
    parser.add_argument("--no-temporal-bias", action="store_true", help="Disable dual-sigma Gaussian temporal bias.")
    parser.add_argument("--no-alpha-gate",    action="store_true", help="Disable alpha-gated cross-attention.")
    parser.add_argument("--lambda-comp",      type=float, default=None, help="Complementarity loss weight.")
    parser.add_argument("--comp-margin",      type=float, default=None, help="Complementarity loss margin.")
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
    if args.no_alpha_gate:
        cfg.use_alpha_gate = False
    if args.lambda_comp is not None:
        cfg.lambda_comp = args.lambda_comp
    if args.comp_margin is not None:
        cfg.comp_margin = args.comp_margin

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
    print(f"[ERPDiff] temporal_bias={cfg.use_temporal_bias}  alpha_gate={cfg.use_alpha_gate}  "
          f"lambda_comp={cfg.lambda_comp}  comp_margin={cfg.comp_margin}")

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
            )


if __name__ == "__main__":
    main()
