import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logpt = F.log_softmax(logits, dim=1)
        pt = torch.exp(logpt)
        target_one_hot = F.one_hot(target, num_classes=logits.shape[1]).float()
        # 按类别加权: 真实类别位置用 alpha, 非真实类别位置用 (1-alpha)
        alpha_t = target_one_hot * self.alpha + (1 - target_one_hot) * (1 - self.alpha)
        loss = -alpha_t * torch.pow(1 - pt, self.gamma) * logpt
        loss = loss.sum(dim=1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
