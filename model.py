"""Small GPT-2-style causal decoder trained from scratch for Othello.

This is a custom board policy, not pretrained OpenAI GPT-2 weights.
Eight row tokens + a readout token -> one of 64 coordinate tokens.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    width: int = 96
    layers: int = 4
    heads: int = 4

    def __post_init__(self):
        if self.width < 8 or self.layers < 1 or self.heads < 1 or self.width % self.heads:
            raise ValueError('width must be >=8 and divisible by heads; layers >=1')


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.width
        self.heads = config.heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.out = nn.Linear(4 * d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = [z.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for z in self.qkv(self.ln1(x)).chunk(3, dim=-1)]
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        x = x + self.proj(att.transpose(1, 2).reshape(b, t, d))
        return x + self.out(F.gelu(self.fc(self.ln2(x)), approximate='tanh'))


class OthelloGPT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        d = self.config.width
        self.embedding = nn.Embedding(6562, d)
        self.position = nn.Parameter(torch.zeros(1, 9, d))
        self.blocks = nn.ModuleList([Block(self.config) for _ in range(self.config.layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 64)
        self.apply(self._initialize)
        nn.init.normal_(self.position, std=0.02)
        for block in self.blocks:
            for layer in (block.proj, block.out):
                nn.init.normal_(layer.weight, std=0.02 / math.sqrt(2 * self.config.layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def features(self, rows):
        query = torch.full((len(rows), 1), 6561, dtype=torch.long, device=rows.device)
        x = self.embedding(torch.cat((rows, query), dim=1)) + self.position
        for block in self.blocks:
            x = block(x)
        return self.norm(x[:, -1])

    def forward(self, rows):
        return self.head(self.features(rows))

    def description(self):
        return {**asdict(self.config), 'parameters': sum(p.numel() for p in self.parameters()),
                'input': '8 base-3 row tokens plus readout token',
                'attention': 'causal, pre-LayerNorm, GELU, no dropout',
                'pretrained': False, 'all_parameters_trainable': True}


class ComplexTaylor(nn.Module):
    """Differentiable version of the preceding degree-2 complex word model.

    Each of 6561 row words defines a normalized quadratic complex map.
    Compose eight maps, apply a next-word map, then measure 64 coordinates
    using the same symmetric-whitened POVM as the previous experiment.
    Complex parameters are stored as paired real scalars for Adam/checkpoints.
    """
    def __init__(self, dimension=4):
        super().__init__()
        self.dimension = dimension
        d = dimension
        for name, shape in {
            'observation': (64, d, 2),
            'a0': (6561, d, 2), 'a1': (6561, d, d, 2),
            'a2': (6561, d, d, d, 2),
            'next_a0': (d, 2), 'next_a1': (d, d, 2),
            'next_a2': (d, d, d, 2),
        }.items():
            values = torch.zeros(shape)
            if name.startswith('next_'):
                if name == 'next_a1':
                    values[..., 0] = torch.eye(d)
            else:
                scale = (1 if name == 'observation' else .1) / math.sqrt(2 * d)
                if name == 'a2':
                    scale /= math.sqrt(d)
                values.normal_(std=scale)
                if name == 'a1':
                    values[..., 0] += torch.eye(d)
            self.register_parameter(name, nn.Parameter(values))

    def forward(self, rows):
        d = self.dimension
        z0 = torch.full((len(rows), d), complex(1, 1) / math.sqrt(2 * d),
                        dtype=torch.complex64, device=rows.device)

        def normalize(z):
            norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
            return torch.where(norm > 1e-12, z / norm.clamp_min(1e-12), z0)

        def polynomial(z, a0, a1, a2):
            return normalize(a0 + torch.einsum('bij,bj->bi', a1, z)
                             + .5 * torch.einsum('bijk,bj,bk->bi', a2, z, z))

        z = z0
        coeffs = [torch.view_as_complex(getattr(self, name)) for name in ('a0', 'a1', 'a2')]
        for row in range(8):
            z = polynomial(z, *(coeff[rows[:, row]] for coeff in coeffs))
        nxt = [torch.view_as_complex(getattr(self, name)).expand(len(rows), *getattr(self, name).shape[:-1])
               for name in ('next_a0', 'next_a1', 'next_a2')]
        z = polynomial(z, *nxt)

        # Use double precision for the tiny 4x4 whitening eigensystem. The
        # observation frame stays inside autograd, including normalization.
        vectors = torch.view_as_complex(self.observation).to(torch.complex128)
        vectors = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(1e-12)
        gram = torch.einsum('ma,mc->ac', vectors, vectors.conj())
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        floor = (eigenvalues[-1] * 1e-10).clamp_min(1e-12)
        inverse_root = (eigenvectors / eigenvalues.clamp_min(floor).sqrt()[None, :]) @ eigenvectors.mH
        frame = (vectors.conj() @ inverse_root).to(torch.complex64)
        amplitude = torch.einsum('md,bd->bm', frame, z)
        probabilities = amplitude.abs().square()
        probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-12)
        return probabilities.clamp_min(1e-30).log()

    def description(self):
        return {'dimension': self.dimension, 'order': 2,
                'parameters': sum(p.numel() for p in self.parameters()),
                'input': '8 base-3 row tokens; ordered function composition',
                'readout': '64-coordinate symmetric-whitened POVM',
                'all_parameters_trainable': True, 'pretrained': False,
                'source': 'Differentiable port of prior local row_models.ComplexTaylor'}


def build_model(family, config=None, dimension=4):
    if family == 'transformer':
        return OthelloGPT(config)
    if family == 'taylor':
        return ComplexTaylor(dimension)
    raise ValueError(f'Unknown model family: {family}')


def masked_log_probs(logits, legal):
    if not bool(legal.any(dim=-1).all()):
        raise ValueError('Passes must be handled by the environment')
    return F.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=-1)
