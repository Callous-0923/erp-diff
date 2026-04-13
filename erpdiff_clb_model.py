"""
ERPDiff — CLB branch model (conventional learning branch, standard ICNN).
"""

from icnn import ICNN


class CLBPretrainICNN(ICNN):
    """ICNN model wrapper for CLB pretraining."""

    def __init__(self, in_channels: int, n_samples: int, dropout_p: float = 0.2, n_classes: int = 2):
        super().__init__(in_channels=in_channels, n_samples=n_samples, dropout_p=dropout_p, n_classes=n_classes)


__all__ = ["CLBPretrainICNN"]
