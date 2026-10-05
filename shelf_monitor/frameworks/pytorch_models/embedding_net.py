"""EmbeddingNet: MobileNetV3-Large backbone → 256-d L2-normalised embedding."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):
    """MobileNetV3-Large feature extractor with a Linear projection and L2-norm.

    Args:
        embedding_dim: Output dimension (default 256).
        input_size:    Expected square input side length in pixels (default 224).
        pretrained:    Load ImageNet weights for the backbone (default True).
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        input_size: int = 224,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_size = input_size

        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)
        self.features = backbone.features  # output: (B, C, H, W)

        # Detect the feature channel count via a dry run
        self.pool = nn.AdaptiveAvgPool2d(1)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            feat = self.features(dummy)
            feat_dim = feat.shape[1]

        self.fc = nn.Linear(feat_dim, embedding_dim)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """(C, H, W) expected input."""
        return (3, self.input_size, self.input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                       # (B, C, H, W)
        x = self.pool(x).squeeze(-1).squeeze(-1)   # (B, C)
        x = self.fc(x)                             # (B, embedding_dim)
        x = F.normalize(x, p=2, dim=1)            # L2 unit sphere
        return x
