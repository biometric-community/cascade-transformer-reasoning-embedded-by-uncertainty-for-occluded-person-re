"""Uncertainty Embedded Transformer (UT) + Cascade Attention (CA)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadUT(nn.Module):
    """Uncertainty-embedded multi-head self-attention (paper Sec. III-A)."""

    def __init__(self, dim: int, num_heads: int = 8, n_samples: int = 20):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.n_samples = n_samples
        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.w_mu = nn.Linear(self.head_dim, self.head_dim)
        self.w_sigma = nn.Linear(self.head_dim, self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, D] -> [B, H, L, d]
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(
        self, x: torch.Tensor, n_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, L, D] tokens
        Returns:
            out: [B, L, D] mean over samples (for cascade path)
            fatt_list: list of [B, L, D] per sample (for L_sample)
            head_feats: [B, T, N, L, d] for diversity loss (pooled later)
            sigma: [B, H, L, d] query std for CA / Lun
            k: [B, H, L, d]
            v: [B, H, L, d]
        """
        n_samples = int(n_samples if n_samples is not None else self.n_samples)
        b, l, d = x.shape
        residual = x
        x_n = self.norm1(x)
        q = self._reshape(self.wq(x_n))
        k = self._reshape(self.wk(x_n))
        v = self._reshape(self.wv(x_n))

        mu = self.w_mu(q)
        sigma = F.softplus(self.w_sigma(q)) + 1e-6

        fatt_list: List[torch.Tensor] = []
        head_list: List[torch.Tensor] = []
        for _ in range(n_samples):
            eps = torch.randn_like(sigma)
            q_s = mu + eps * sigma  # [B, H, L, d]
            attn = torch.matmul(q_s, k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            head = torch.matmul(attn, v)  # [B, H, L, d]
            head_list.append(head)
            out = head.permute(0, 2, 1, 3).contiguous().view(b, l, d)
            out = self.proj(out)
            out = residual + out
            out = out + self.ffn(self.norm2(out))
            fatt_list.append(out)

        stacked = torch.stack(fatt_list, dim=0).mean(0)
        heads = torch.stack(head_list, dim=2)  # [B, H, N, L, d]
        return {
            "out": stacked,
            "fatt_list": fatt_list,
            "heads": heads,
            "sigma": sigma,
            "k": k,
            "v": v,
        }


class CommonTransformer(nn.Module):
    """Deterministic transformer block (CT) for cascade attention."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


def cascade_mask(sigma: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """CM = 1 - minmax(Softmax(σ K^T / √d) V)  (Eq. 10–11).

    Args:
        sigma, k, v: [B, H, L, d]
    Returns:
        cm: [B, L, 1] broadcastable spatial mask (mean over heads/channels)
    """
    scale = k.size(-1) ** -0.5
    # σ K^T : treat sigma as query-like — paper uses σ K^T / √dk then softmax over spatial
    logits = torch.matmul(sigma, k.transpose(-2, -1)) * scale  # [B,H,L,L]
    w = logits.softmax(dim=-1)
    vvar = torch.matmul(w, v)  # [B,H,L,d]
    # pool heads/channels → [B, L]
    score = vvar.mean(dim=(1, 3))
    bmin = score.amin(dim=-1, keepdim=True)
    bmax = score.amax(dim=-1, keepdim=True)
    norm = (score - bmin) / (bmax - bmin + 1e-6)
    cm = 1.0 - norm
    return cm.unsqueeze(-1)  # [B, L, 1]


def head_diversity_loss(heads: torch.Tensor) -> torch.Tensor:
    """L_hd: push apart head features (Eq. 18).

    heads: [B, T, N, L, d] — average over L then cosine between heads.
    """
    # pool tokens
    h = heads.mean(dim=3)  # [B, T, N, d]
    b, t, n, d = h.shape
    h = F.normalize(h, dim=-1)
    # pairwise cosine between heads for each sample
    loss = h.new_zeros(())
    count = 0
    for i in range(t):
        for j in range(t):
            if i == j:
                continue
            # [B, N]
            cos = (h[:, i] * h[:, j]).sum(-1)
            loss = loss + cos.mean()
            count += 1
    if count == 0:
        return h.new_zeros(())
    return loss / count


def uncertainty_loss(sigma: torch.Tensor, beta: float) -> torch.Tensor:
    """L_un = mean(max(0, β - log(σ)))  (Eq. 15–16 with β absorbing constants)."""
    # sigma: [B, H, L, d]
    lun = F.relu(beta - torch.log(sigma.clamp_min(1e-6)))
    return lun.mean()
