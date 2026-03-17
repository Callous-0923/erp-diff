"""
ERPDiff — Training utility functions.

epoch_run:      Standard single-branch train/eval loop (pretrain stage).
epoch_run_dwm:  Fine-tuning loop with dynamic weighting + complementarity loss.
eval_dwm:       Fine-tuning evaluation loop.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from erpdiff_losses import complementarity_loss


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
) -> Tuple[float, float]:
    """
    ERPDiff fine-tuning training loop.

    Unpacks 5 values from model.forward():
        logits_clb, logits_rbb, fused_logits, tokens_clb, tokens_rbb

    Args:
        lambda_comp:   Weight for complementarity preservation loss (0 = disabled).
        comp_margin:   Margin for the soft decorrelation loss.
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

        logits_c, logits_f, fused_logits, tok_clb, tok_rbb = model(x, alpha)

        loss_fusion = ce(fused_logits, y)
        loss_total = loss_fusion

        if lambda_intra > 0:
            loss_aux_c = ce(logits_c, y)
            loss_aux_f = ce(logits_f, y)
            loss_total = loss_total + lambda_intra * (loss_aux_c + loss_aux_f)

        if lambda_comp > 0:
            loss_comp = complementarity_loss(tok_clb, tok_rbb, margin=comp_margin)
            loss_total = loss_total + lambda_comp * loss_comp

        loss_total.backward()
        optimizer.step()
        total_loss += loss_total.item() * x.size(0)
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
