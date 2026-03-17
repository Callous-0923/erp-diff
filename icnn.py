from typing import Tuple

import torch
import torch.nn as nn


def _same_pad_1xk(k: int) -> Tuple[int, int]:
    if k % 2 == 1:
        left = right = k // 2
    else:
        left = k // 2 - 1
        right = k // 2
    return left, right


class Conv2dSame1xK(nn.Module):
    """Conv2d with manual 'same' padding along time (width) for kernel (1, k)."""

    def __init__(self, in_ch: int, out_ch: int, k: int, groups: int = 1, bias: bool = False):
        super().__init__()
        pad_l, pad_r = _same_pad_1xk(k)
        self.pad = nn.ZeroPad2d((pad_l, pad_r, 0, 0))
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=(1, k), stride=(1, 1), padding=0, groups=groups, bias=bias
        )

    def forward(self, x):
        return self.conv(self.pad(x))


class ConvBN(nn.Module):
    """Conv + BN (+ optional ReLU)."""

    def __init__(self, conv: nn.Module, out_ch: int, activation: bool = False):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if activation else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class InceptionBlock(nn.Module):
    """Inception module with 4 branches, output channels=32."""

    def __init__(self, in_ch: int, k_b: int, k_c: int, pool_k: int):
        super().__init__()
        self.a1 = ConvBN(nn.Conv2d(in_ch, 8, kernel_size=(1, 1), bias=False), 8, activation=True)
        self.b1 = ConvBN(nn.Conv2d(in_ch, 4, kernel_size=(1, 1), bias=False), 4, activation=False)
        self.b2 = ConvBN(Conv2dSame1xK(4, 8, k=k_b, bias=False), 8, activation=True)
        self.c1 = ConvBN(nn.Conv2d(in_ch, 4, kernel_size=(1, 1), bias=False), 4, activation=False)
        self.c2 = ConvBN(Conv2dSame1xK(4, 8, k=k_c, bias=False), 8, activation=True)
        self.d1 = nn.AvgPool2d(kernel_size=(1, pool_k), stride=(1, 1), padding=(0, pool_k // 2))
        self.d2 = ConvBN(nn.Conv2d(in_ch, 8, kernel_size=(1, 1), bias=False), 8, activation=False)

    def forward(self, x):
        a = self.a1(x)
        b = self.b2(self.b1(x))
        c = self.c2(self.c1(x))
        d = self.d2(self.d1(x))
        return torch.cat([a, b, c, d], dim=1)


class ICNNStem(nn.Module):
    """ICNN stem: Input -> BasicConv -> Inception#1 -> Downsampling#1 -> Inception#2."""

    def __init__(self, in_channels: int, dropout_p: float = 0.2):
        super().__init__()
        self.temporal = ConvBN(Conv2dSame1xK(1, 8, k=64, bias=False), 8, activation=False)
        self.spatial = ConvBN(
            nn.Conv2d(8, 32, kernel_size=(in_channels, 1), stride=(1, 1), padding=(0, 0), groups=8, bias=False),
            32,
            activation=False,
        )
        self.incep1 = InceptionBlock(in_ch=32, k_b=16, k_c=64, pool_k=7)
        self.down1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 0)),
            nn.Dropout(p=dropout_p),
        )
        self.incep2 = InceptionBlock(in_ch=32, k_b=8, k_c=16, pool_k=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.incep1(x)
        x = self.down1(x)
        x = self.incep2(x)
        return x


class ICNNTail(nn.Module):
    """ICNN tail: Downsampling#2 -> Flatten -> FCs."""

    def __init__(self, feat_t: int, dropout_p: float = 0.2, n_classes: int = 2):
        super().__init__()
        self.down2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 4), padding=(0, 0)),
            nn.Dropout(p=dropout_p),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 32, 1, feat_t)
            feat_dim = self._forward_features(dummy).shape[1]
        self.fc1 = nn.Sequential(
            nn.Linear(feat_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        self.fc2 = nn.Linear(8, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down2(x)
        return x.flatten(1)

    def forward(self, feat_incep2: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(feat_incep2)
        x = self.fc1(x)
        return self.fc2(x)


class ICNN(nn.Module):
    """
    ICNN aligned with icnn.yaml (Table 1).
    Input: (B, 1, C, T). Output: logits (no softmax).
    """

    def __init__(self, in_channels: int, n_samples: int, dropout_p: float = 0.2, n_classes: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.n_samples = n_samples

        self.temporal = ConvBN(Conv2dSame1xK(1, 8, k=64, bias=False), 8, activation=False)
        self.spatial = ConvBN(
            nn.Conv2d(8, 32, kernel_size=(in_channels, 1), stride=(1, 1), padding=(0, 0), groups=8, bias=False),
            32,
            activation=False,
        )

        self.incep1 = InceptionBlock(in_ch=32, k_b=16, k_c=64, pool_k=7)
        self.down1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 0)),
            nn.Dropout(p=dropout_p),
        )

        self.incep2 = InceptionBlock(in_ch=32, k_b=8, k_c=16, pool_k=3)
        self.down2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 4), padding=(0, 0)),
            nn.Dropout(p=dropout_p),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, n_samples)
            feat = self._forward_features(dummy)
            feat_dim = feat.shape[1]

        self.fc1 = nn.Sequential(
            nn.Linear(feat_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        self.fc2 = nn.Linear(8, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.incep1(x)
        x = self.down1(x)
        x = self.incep2(x)
        x = self.down2(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = self.fc1(x)
        return self.fc2(x)
