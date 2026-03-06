"""
Mamba-2 SSD (State Space Duality) block — pure PyTorch, MPS-compatible.

Simplified from tommyip/mamba2-minimal. No depthwise conv (removes MPS
friction point). Uses chunked parallel scan for efficient sequence processing.

All operations are matmul/einsum/cumsum — Metal safe.
Cast to float32 for the scan, cast back after.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


class Mamba2Block(nn.Module):
    """Mamba-2 block with chunked selective scan.

    Interface: (batch, seq_len, d_model) → (batch, seq_len, d_model)

    The selective scan uses input-dependent B (input selection), C (output
    selection), and dt (timescale). Within each chunk, the quadratic form
    (attention-like matmul) is used. Between chunks, the state recurrence
    is propagated sequentially.

    Hyperparams: d_state=16, n_heads=4, dt_rank=16, expand=1 (d_inner=d_model)
    ~160K params per instance at d_model=256.
    """

    def __init__(self, config: ModelConfig, chunk_size: int = 64):
        super().__init__()
        self.d_model = config.d_model
        self.d_inner = config.d_model  # expand=1
        self.d_state = config.ssm_d_state
        self.n_heads = config.n_heads
        self.dt_rank = config.ssm_dt_rank
        self.chunk_size = chunk_size

        assert self.d_inner % self.n_heads == 0
        self.head_dim = self.d_inner // self.n_heads

        # Single input projection for all components
        proj_size = (
            self.d_inner +                    # x
            self.d_inner +                    # z (gate)
            self.d_state * self.n_heads +     # B
            self.d_state * self.n_heads +     # C
            self.dt_rank                      # dt (low-rank)
        )
        self.in_proj = nn.Linear(self.d_model, proj_size, bias=False)

        # Per-head parameters
        self.A_log = nn.Parameter(torch.log(0.5 * torch.ones(self.n_heads)))
        self.D = nn.Parameter(torch.ones(self.n_heads))

        # Step size: low-rank projection to per-head scalar
        self.dt_proj = nn.Linear(self.dt_rank, self.n_heads, bias=True)

        # Output
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.norm = RMSNorm(self.d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        # Project and split
        proj = self.in_proj(x)
        x_proj, z, B_ssm, C_ssm, dt_low = proj.split([
            self.d_inner, self.d_inner,
            self.d_state * self.n_heads,
            self.d_state * self.n_heads,
            self.dt_rank,
        ], dim=-1)

        # Per-head step sizes
        dt = F.softplus(self.dt_proj(dt_low))  # (B, S, n_heads)

        # Reshape to multi-head: (B, S, H, D) where H=n_heads
        H, P, N = self.n_heads, self.head_dim, self.d_state
        x_proj = x_proj.view(B, S, H, P)
        B_ssm = B_ssm.view(B, S, H, N)
        C_ssm = C_ssm.view(B, S, H, N)

        A = -torch.exp(self.A_log.float())  # (H,)

        # Run scan in float32
        y = self._scan(x_proj.float(), dt.float(), A, B_ssm.float(), C_ssm.float())

        # Skip connection
        y = y + x_proj.float() * self.D.float()[None, None, :, None]

        # Norm + gate + project out
        y = y.reshape(B, S, self.d_inner)
        y = self.norm(y)
        y = (y * F.silu(z.float())).to(x.dtype)
        return self.out_proj(y)

    def _scan(self, x, dt, A, B_ssm, C_ssm):
        """Chunked selective scan.

        Notation:
            B=batch, L=seq_len, H=n_heads, P=head_dim, N=d_state, K=chunk_size

        x:     (B, L, H, P)  — input
        dt:    (B, L, H)     — step sizes
        A:     (H,)          — state decay (negative)
        B_ssm: (B, L, H, N)  — input selection
        C_ssm: (B, L, H, N)  — output selection

        Returns: (B, L, H, P)
        """
        batch, seq_len, H, P = x.shape
        N = B_ssm.shape[-1]
        K = self.chunk_size

        # Pad to chunk boundary
        pad = (K - seq_len % K) % K
        if pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
            dt = F.pad(dt, (0, 0, 0, pad))
            B_ssm = F.pad(B_ssm, (0, 0, 0, 0, 0, pad))
            C_ssm = F.pad(C_ssm, (0, 0, 0, 0, 0, pad))

        L = x.shape[1]
        nC = L // K  # number of chunks

        # Reshape to chunks: (B, nC, K, H, ...)
        x = x.contiguous().view(batch, nC, K, H, P)
        dt = dt.contiguous().view(batch, nC, K, H)
        B_ssm = B_ssm.contiguous().view(batch, nC, K, H, N)
        C_ssm = C_ssm.contiguous().view(batch, nC, K, H, N)

        # Log-decay per position: dt * A
        dA = dt * A[None, None, None, :]  # (B, nC, K, H) — negative values
        dA_cumsum = torch.cumsum(dA, dim=2)  # (B, nC, K, H)

        # ─── Intra-chunk (quadratic form) ───
        # Decay from j to i within a chunk: exp(dA_cumsum[i] - dA_cumsum[j])
        # L_matrix[i, j] = exp(cumsum[i] - cumsum[j]) * causal_mask[i >= j]
        L_matrix = torch.exp(
            dA_cumsum[:, :, :, None, :] - dA_cumsum[:, :, None, :, :]
        )  # (B, nC, K, K, H)
        causal = torch.tril(torch.ones(K, K, device=x.device, dtype=x.dtype))
        L_matrix = L_matrix * causal[None, None, :, :, None]

        # Attention-like: score[i,j,h] = C[i,h,:] · B[j,h,:]
        # C_ssm: (B, nC, K, H, N), B_ssm: (B, nC, K, H, N)
        # scores: (B, nC, K_i, K_j, H)
        scores = torch.einsum('bcihd, bcjhd -> bcijh', C_ssm, B_ssm)

        # Weight by decay and dt: dt[j] scales the input at position j
        scores = scores * L_matrix * dt[:, :, None, :, :]  # dt[j] broadcast

        # y_intra[i] = sum_j scores[i,j] * x[j]
        # scores: (B, nC, K_i, K_j, H), x: (B, nC, K_j, H, P)
        y_intra = torch.einsum('bcijh, bcjhd -> bcihd', scores, x)

        # ─── Inter-chunk (sequential recurrence) ───
        # State contribution from each chunk: sum_t decay_to_end[t] * dt[t] * outer(B[t], x[t])
        decay_to_end = torch.exp(
            dA_cumsum[:, :, -1:, :] - dA_cumsum
        )  # (B, nC, K, H)

        # Weighted x: dt * decay_to_end * x
        wx = (dt * decay_to_end)[..., None] * x  # (B, nC, K, H, P)

        # chunk_state = sum_t B[t] ⊗ wx[t] → (B, nC, H, N, P)
        chunk_state = torch.einsum('bckhn, bckhp -> bchpn', B_ssm, wx)
        # Swap to (B, nC, H, N, P)
        chunk_state = chunk_state.permute(0, 1, 2, 4, 3)

        # Decay across entire chunk: exp(dA_cumsum at last position)
        chunk_decay = torch.exp(dA_cumsum[:, :, -1, :])  # (B, nC, H)

        # Sequential pass: propagate hidden state
        h = x.new_zeros(batch, H, N, P)
        h_list = []
        for c in range(nC):
            h_list.append(h)
            h = chunk_decay[:, c, :, None, None] * h + chunk_state[:, c]

        h_states = torch.stack(h_list, dim=1)  # (B, nC, H, N, P)

        # Cross-chunk output: C[t] · (decay_from_start[t] * h_state)
        decay_from_start = torch.exp(dA_cumsum)  # (B, nC, K, H)

        # y_cross[b,c,t,h,p] = sum_n C[b,c,t,h,n] * h[b,c,h,n,p] * decay[b,c,t,h]
        y_cross = torch.einsum(
            'bcthn, bchnd, bcth -> bcthd',
            C_ssm, h_states, decay_from_start,
        )  # Hmm — need to verify this matches dimensions.
        # C_ssm: (B, nC, K, H, N) → indices b,c,t,h,n
        # h_states: (B, nC, H, N, P) → indices b,c,h,n,d
        # decay_from_start: (B, nC, K, H) → indices b,c,t,h
        # Want: (B, nC, K, H, P) → b,c,t,h,d
        # y[b,c,t,h,d] = sum_n C[b,c,t,h,n] * h[b,c,h,n,d] * decay[b,c,t,h]
        # einsum: 'bcthn, bchnd, bcth -> bcthd' ✓

        y = y_intra + y_cross  # (B, nC, K, H, P)

        # Reshape and trim padding
        y = y.contiguous().view(batch, L, H, P)
        if pad > 0:
            y = y[:, :seq_len]
        return y
