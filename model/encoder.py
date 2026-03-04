"""
Bidirectional transformer encoder for the heterogeneous MoE.

Shallow (2 layers Tiny, 4 Small). Pre-norm with RoPE.
No causal mask — full bidirectional attention.
Shares embedding table with decoder.

Output is used as K/V for Type A expert cross-attention.
Only runs on samples where encoder_available=True.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Chosen for length generalization: the corpus spans Homer to Shannon,
    and inference may need to handle sequences longer than training.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embedding to input tensor.

    x: (batch, n_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim)
    """
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # Split into even/odd halves and apply rotation
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class EncoderSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE. Bidirectional (no causal mask)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(config.head_dim, config.context_len * 2)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x: (batch, seq_len, d_model)
        mask: (batch, seq_len) — True for valid positions, False for padding
        """
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(S)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Scaled dot-product attention (bidirectional)
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            # mask: (B, S) -> (B, 1, 1, S)
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class EncoderBlock(nn.Module):
    """Pre-norm encoder block: LayerNorm -> Self-Attn -> LayerNorm -> FFN."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = EncoderSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class Encoder(nn.Module):
    """
    Bidirectional transformer encoder stack.

    Shallow by design — the encoder's job is to provide K/V context for
    Type A expert cross-attention, not to do heavy processing.

    Does NOT include its own embedding layer — shares with the decoder.
    Expects pre-embedded input (after factored embedding projection).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(config) for _ in range(config.n_enc_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x: (batch, seq_len, d_model) — pre-embedded tokens
        mask: (batch, seq_len) — True for valid, False for pad

        Returns: (batch, seq_len, d_model) encoder hidden states
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)
