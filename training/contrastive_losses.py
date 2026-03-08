"""
Contrastive and similarity losses for expert specialization (v2.4).

Two complementary losses:
  - linear_cka / ExpertSimilarityLoss (SimSMoE): minimize CKA between expert outputs
  - Routing contrastive loss (ProMoE): InfoNCE on expert prototypes vs token centroids

Both are auxiliary losses computed during training only. They address different
failure modes:
  - CKA prevents representational collapse (experts producing same outputs)
  - RCL prevents routing incoherence (experts receiving semantically random tokens)

References:
  SimSMoE: Do et al., arXiv:2406.15883, NAACL 2025 Findings
  ProMoE: arXiv:2510.24711, ICLR 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Linear Centered Kernel Alignment between two representation matrices.

    X, Y: (n_samples, d) — expert outputs for the same set of tokens.
    Returns: scalar in [0, 1]. Higher = more similar representations.

    Efficient computation via d x d cross-covariance (not n x n Gram matrices).
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    XtY = X.T @ Y  # (d, d)
    hsic_xy = (XtY * XtY).sum()
    hsic_xx = (X.T @ X).pow(2).sum()
    hsic_yy = (Y.T @ Y).pow(2).sum()

    return hsic_xy / (hsic_xx * hsic_yy).sqrt().clamp(min=1e-8)


class ExpertSimilarityLoss(nn.Module):
    """CKA-based expert similarity loss (adapted from SimSMoE).

    Projects expert outputs through a shared bottleneck before computing
    linear CKA. Threshold-gated: only penalizes when CKA > threshold,
    meaning experts are too similar and need to be pushed apart.

    With 2 routed experts, this is a single CKA computation per layer.
    """

    def __init__(self, d_model: int, n_proj: int = 16,
                 threshold: float = 0.5, min_tokens: int = 8):
        super().__init__()
        self.proj = nn.Linear(d_model, n_proj, bias=False)
        self.threshold = threshold
        self.min_tokens = min_tokens

    def forward(self, expert_a_out: torch.Tensor,
                expert_b_out: torch.Tensor) -> torch.Tensor:
        """Compute CKA similarity between two expert outputs on co-activated tokens.

        expert_a_out: (n_co, d_model) — Type A output for co-active tokens
        expert_b_out: (n_co, d_model) — Type B output for same tokens

        Returns: CKA value (to be minimized) if above threshold, else 0.
        """
        if expert_a_out.shape[0] < self.min_tokens:
            return torch.tensor(0.0, device=expert_a_out.device, requires_grad=True)

        a_proj = self.proj(expert_a_out)
        b_proj = self.proj(expert_b_out)

        cka = linear_cka(a_proj, b_proj)

        # Only penalize if experts are too similar
        if cka.item() <= self.threshold:
            return torch.tensor(0.0, device=expert_a_out.device, requires_grad=True)

        return cka
