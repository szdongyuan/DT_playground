"""Inference layers for the pinned MIT-licensed FISHER-small-0723 snapshot.

See LICENSE-FISHER for the upstream license. The layer names follow the
upstream checkpoint. This module implements the
unmasked, post-normalization inference path used by that snapshot.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BAND_WIDTH = 50
EMBED_DIM = 384
DEPTH = 12
NUM_HEADS = 6
PATCH_SIZE = 16


def _positions():
    freq, time = np.meshgrid(np.arange(50, dtype=np.float32), np.arange(128, dtype=np.float32))
    omega = np.arange(EMBED_DIM // 4, dtype=np.float32) / (EMBED_DIM / 4)
    omega = 1.0 / 10000 ** omega
    axes = []
    for positions in (freq, time):
        angles = np.einsum("m,d->md", positions.reshape(-1), omega)
        axes.extend((np.sin(angles), np.cos(angles)))
    return torch.from_numpy(np.concatenate(axes, axis=1)).unsqueeze(0)


class _PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_encoder = nn.Module()
        self.local_encoder.proj = nn.Conv2d(1, EMBED_DIM, PATCH_SIZE, stride=PATCH_SIZE)
        self.extra_tokens = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.context_encoder = nn.Module()
        self.context_encoder.norm = nn.LayerNorm(EMBED_DIM, eps=1e-6)
        self.register_buffer("positions", _positions(), persistent=False)

    def forward(self, bands):
        tokens = self.local_encoder.proj(bands).flatten(2).transpose(1, 2)
        tokens = tokens + self.positions[:, :tokens.shape[1]]
        tokens = torch.cat((self.extra_tokens.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        return self.context_encoder.norm(tokens)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)

    def forward(self, tokens):
        batch, length, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, NUM_HEADS, EMBED_DIM // NUM_HEADS)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(query, key, value)
        return self.proj(attended.transpose(1, 2).reshape(batch, length, EMBED_DIM))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM, eps=1e-6)
        self.norm2 = nn.LayerNorm(EMBED_DIM, eps=1e-6)
        self.attn = _Attention()
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(EMBED_DIM, 4 * EMBED_DIM)
        self.mlp.fc2 = nn.Linear(4 * EMBED_DIM, EMBED_DIM)

    def forward(self, tokens):
        tokens = self.norm1(tokens + self.attn(tokens))
        return self.norm2(tokens + self.mlp.fc2(F.gelu(self.mlp.fc1(tokens))))


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.modality_encoders = nn.ModuleDict({"IMAGE": _PatchEncoder()})
        self.blocks = nn.Sequential(*(_Block() for _ in range(DEPTH)))

    def forward(self, spectrograms):
        batch, _, frames, bins = spectrograms.shape
        band_count = bins // BAND_WIDTH
        bands = spectrograms[..., :band_count * BAND_WIDTH]
        bands = bands.reshape(batch, frames, band_count, BAND_WIDTH).transpose(1, 2)
        bands = bands.reshape(batch * band_count, 1, frames, BAND_WIDTH).contiguous()
        tokens = self.modality_encoders["IMAGE"](bands)
        return self.blocks(tokens)[:, 0].reshape(batch, band_count * EMBED_DIM)


class FisherSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Encoder()

    def forward(self, spectrograms):
        return self.model(spectrograms)
