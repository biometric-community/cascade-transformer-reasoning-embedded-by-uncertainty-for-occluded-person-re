"""CTU losses: CE, sample CE, uncertainty, head diversity, batch-hard triplet."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTriplet(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats = F.normalize(feats, dim=-1)
        dist = torch.cdist(feats, feats, p=2)
        labels = labels.view(-1)
        mask_pos = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        mask_neg = ~mask_pos
        eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
        mask_pos = mask_pos & ~eye
        dist_ap = dist.clone()
        dist_ap[~mask_pos] = -1.0
        hardest_pos, _ = dist_ap.max(dim=1)
        dist_an = dist.clone()
        dist_an[~mask_neg] = 1e6
        hardest_neg, _ = dist_an.min(dim=1)
        valid = (hardest_pos > -0.5) & (hardest_neg < 1e5)
        if valid.sum() == 0:
            return dist.new_zeros(())
        return F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin).mean()


def ctu_layer_loss(
    layer: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    ce: nn.Module,
) -> torch.Tensor:
    l_cls = ce(layer["logits_cls"], labels)
    l_sample = sum(ce(lg, labels) for lg in layer["sample_logits"]) / max(
        len(layer["sample_logits"]), 1
    )
    return l_sample + layer["lhd"] + l_cls + layer["lun"]


def ctu_total_loss(
    outputs: Dict,
    labels: torch.Tensor,
    ce: nn.Module,
    triplet: BatchHardTriplet,
) -> torch.Tensor:
    layers: List[Dict] = outputs["layer_losses"]
    p = len(layers)
    loss = sum(ctu_layer_loss(L, labels, ce) for L in layers) / max(p, 1)
    loss = loss + triplet(outputs["feat"], labels)
    # also CE on final classifier
    loss = loss + ce(outputs["logits"], labels)
    return loss
