"""
ERPDiff — Training utility functions.

epoch_run:      Standard single-branch train/eval loop (pretrain stage).
epoch_run_dwm:  Fine-tuning loop with dynamic weighting + complementarity loss.
eval_dwm:       Fine-tuning evaluation loop.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from erpdiff_losses import complementarity_loss


def _mixup_data(x, y, alpha=0.3):
    """Mixup: create convex combinations of training examples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    lam = max(lam, 1.0 - lam)  # ensure lam >= 0.5 so original sample dominates
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def epoch_run(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    optimizer: Adam = None,
) -> Tuple[float, float]:
    """Standard single-branch training/eval loop (used in pretrain stage)."""
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train_mode:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train_mode:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def _mixup_ce_loss(ce, logits, y_a, y_b, lam):
    """Compute mixup-compatible cross-entropy loss."""
    return lam * ce(logits, y_a) + (1 - lam) * ce(logits, y_b)


def epoch_run_dwm(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    alpha: float,
    ce: nn.Module,
    lambda_intra: float,
    optimizer: Adam,
    lambda_comp: float = 0.0,
    comp_margin: float = 0.5,
    mixup_alpha: float = 0.4,
) -> Tuple[float, float]:
    """
    ERPDiff fine-tuning training loop with Mixup augmentation.

    Unpacks 5 values from model.forward():
        logits_clb, logits_rbb, fused_logits, tokens_clb, tokens_rbb

    Args:
        lambda_comp:   Weight for complementarity preservation loss (0 = disabled).
        comp_margin:   Margin for the soft decorrelation loss.
        mixup_alpha:   Mixup interpolation strength (0 = disabled).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()

        # Apply Mixup
        if mixup_alpha > 0:
            mixed_x, y_a, y_b, lam = _mixup_data(x, y, mixup_alpha)
        else:
            mixed_x, y_a, y_b, lam = x, y, y, 1.0

        logits_c, logits_f, fused_logits, tok_clb, tok_rbb = model(mixed_x, alpha)

        loss_fusion = _mixup_ce_loss(ce, fused_logits, y_a, y_b, lam)
        loss_total = loss_fusion

        if lambda_intra > 0:
            loss_aux_c = _mixup_ce_loss(ce, logits_c, y_a, y_b, lam)
            loss_aux_f = _mixup_ce_loss(ce, logits_f, y_a, y_b, lam)
            loss_total = loss_total + lambda_intra * (loss_aux_c + loss_aux_f)

        if lambda_comp > 0:
            loss_comp = complementarity_loss(tok_clb, tok_rbb, margin=comp_margin)
            loss_total = loss_total + lambda_comp * loss_comp

        loss_total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss_total.item() * x.size(0)
        # For accuracy tracking, use original labels (not mixed)
        correct += (fused_logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def eval_dwm(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    alpha_eval: float,
    ce: nn.Module,
) -> Tuple[float, float]:
    """ERPDiff fine-tuning evaluation loop."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _, _, fused_logits, _, _ = model(x, alpha_eval)
            loss = ce(fused_logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (fused_logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)
