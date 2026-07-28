from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    logsumexp = torch.logsumexp(inputs, dim=-1)
    target_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return torch.mean(logsumexp - target_logits)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return

    total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(grad.detach(), 2) for grad in grads]), 2)
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    for grad in grads:
        grad.mul_(clip_coef.to(device=grad.device, dtype=grad.dtype))
