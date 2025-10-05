# classification_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU."""

    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k,
                      stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ClassificationNet(nn.Module):
    """
    Simple CNN for multi-label classification over 3 classes.
    - Input:  (N, 1, H, W)  (grayscale)
    - Output: (N, 3) logits (use BCEWithLogitsLoss)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 3, dropout_p: float = 0.2):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 32, k=3, s=1, p=1),
            ConvBNAct(32, 32, k=3, s=1, p=1),
            nn.MaxPool2d(2),                    # /2

            ConvBNAct(32, 64, k=3, s=1, p=1),
            ConvBNAct(64, 64, k=3, s=1, p=1),
            nn.MaxPool2d(2),                    # /4

            ConvBNAct(64, 128, k=3, s=1, p=1),
        )

        # (N, 128, 53/4, 53/4) -> (N, 128, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, num_classes, bias=True)  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)  # (N, C, 1, 1)
        logits = self.classifier(x)  # (N, num_classes)/
        return logits


def build_classification_model(in_channels: int = 1, num_classes: int = 3) -> nn.Module:
    """
    Factory used by main.py's _create_model('classification').
    """
    return ClassificationNet(in_channels=in_channels, num_classes=num_classes)


def build_classification_criterion() -> nn.Module:
    """
    Factory for the loss. Use BCEWithLogitsLoss for multi-label.

    Example:
        criterion = build_classification_criterion(pos_weight)
    """
    return nn.BCEWithLogitsLoss()
