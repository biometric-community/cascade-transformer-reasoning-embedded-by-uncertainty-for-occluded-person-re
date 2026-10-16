"""CTU network: ResNet-50 + cascaded UT/CA layers."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from .ut import (
    CommonTransformer,
    MultiHeadUT,
    cascade_mask,
    head_diversity_loss,
    uncertainty_loss,
)


class Backbone(nn.Module):
    def __init__(self, last_stride: int = 1):
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # remove GAP + FC
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        if last_stride == 1:
            # paper: stride of conv4 (layer4) set to 1
            self.layer4[0].conv2.stride = (1, 1)
            self.layer4[0].downsample[0].stride = (1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # [B, 2048, H, W]


class CTU(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        n_samples: int = 20,
        beta: float = math.log(4.0),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.n_samples = n_samples
        self.beta = beta

        self.backbone = Backbone(last_stride=1)
        self.reduce = nn.Conv2d(2048, dim, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(dim)

        self.ut_layers = nn.ModuleList(
            [MultiHeadUT(dim, num_heads=num_heads, n_samples=n_samples) for _ in range(num_layers)]
        )
        self.ct_layers = nn.ModuleList(
            [CommonTransformer(dim, num_heads=num_heads) for _ in range(num_layers)]
        )

        # per-layer classifiers for L_cls / L_sample
        self.layer_bns = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(num_layers)])
        self.layer_cls = nn.ModuleList(
            [nn.Linear(dim, num_classes, bias=False) for _ in range(num_layers)]
        )
        self.sample_bns = nn.ModuleList(
            [nn.BatchNorm1d(dim) for _ in range(num_layers)]
        )
        self.sample_cls = nn.ModuleList(
            [nn.Linear(dim, num_classes, bias=False) for _ in range(num_layers)]
        )

        self.bn_neck = nn.BatchNorm1d(dim)
        nn.init.constant_(self.bn_neck.weight, 1.0)
        nn.init.constant_(self.bn_neck.bias, 0.0)
        self.classifier = nn.Linear(dim, num_classes, bias=False)

    def _tokens(self, feat: torch.Tensor) -> torch.Tensor:
        # [B, D, H, W] -> [B, L, D]
        b, d, h, w = feat.shape
        return feat.flatten(2).transpose(1, 2)

    def _gap(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)

    def forward(
        self, images: torch.Tensor, labels: Optional[torch.Tensor] = None, train: bool = True
    ) -> Dict[str, torch.Tensor]:
        fmap = self.backbone(images)
        fmap = self.bn_reduce(self.reduce(fmap))
        tokens = self._tokens(fmap)  # input F

        n_samples = self.n_samples if train else 1
        layer_losses: List[Dict[str, torch.Tensor]] = []
        cur = tokens
        for i in range(self.num_layers):
            ut = self.ut_layers[i](cur, n_samples=n_samples)
            # L_cls on input of this layer
            gap_in = self._gap(cur)
            logits_cls = self.layer_cls[i](self.layer_bns[i](gap_in))
            # L_sample on each Fatt
            sample_logits = []
            for fatt in ut["fatt_list"]:
                g = self._gap(fatt)
                sample_logits.append(self.sample_cls[i](self.sample_bns[i](g)))
            lhd = head_diversity_loss(ut["heads"])
            lun = uncertainty_loss(ut["sigma"], self.beta)

            layer_losses.append(
                {
                    "logits_cls": logits_cls,
                    "sample_logits": sample_logits,
                    "lhd": lhd,
                    "lun": lun,
                }
            )

            # Cascade attention → next input
            cm = cascade_mask(ut["sigma"], ut["k"], ut["v"])
            guided = cur * cm
            cur = self.ct_layers[i](guided)

        # Final embedding = GAP of feature feeding last UT = tokens before last UT
        # Paper: GAP(FP) where FP is feature feeding to the last UT layer.
        # We use GAP of last UT mean output as a practical embedding (also GAP of cur after last CA).
        feat = self._gap(ut["out"])
        feat_bn = self.bn_neck(feat)
        logits = self.classifier(feat_bn)
        return {
            "feat": feat,
            "feat_bn": feat_bn,
            "logits": logits,
            "layer_losses": layer_losses,
        }


def build_ctu(cfg: dict, num_classes: int) -> CTU:
    m = cfg.get("model", {})
    return CTU(
        num_classes=num_classes,
        dim=int(m.get("dim", 256)),
        num_layers=int(m.get("num_layers", 2)),
        num_heads=int(m.get("num_heads", 8)),
        n_samples=int(m.get("n_samples", 20)),
        beta=float(m.get("beta", math.log(4.0))),
    )
