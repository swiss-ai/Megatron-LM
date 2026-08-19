# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Activation functions for Megatron-Core."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule


@jit_fuser
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU activation"""
    return torch.pow(F.relu(x), 2)


@jit_fuser
def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """Quick GELU activation"""
    return x * torch.sigmoid(1.702 * x)


@jit_fuser
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """Fast GELU activation"""
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


# Swiss AI XIELU family activations with learnable parameters

@jit_fuser
def compiled_xielu(x, alpha_p, alpha_n, beta=0.5, eps=-1e-6):
    return torch.where(x > 0,
                      alpha_p * x * x + beta * x,
                      alpha_n * torch.expm1(torch.min(x, eps)) - alpha_n * x + beta * x)


class XIELU(MegatronModule):
    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6, dtype=torch.float32):
        super().__init__(config=config)
        self.config = config
        self.alpha_p = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=dtype)) - 1.0).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_n_init - beta, dtype=dtype)) - 1.0).unsqueeze(0))
        self.beta = beta
        self.eps = torch.tensor(eps, dtype=dtype, device='cuda')

    def forward(self, x):
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = self.beta + F.softplus(self.alpha_n)
        return compiled_xielu(x, alpha_p, alpha_n, self.beta, self.eps)


@jit_fuser
def compiled_xssslur2(x, alpha_p, alpha_n, beta=0.5, eps=-1e-6):
    return torch.where(x > 0,
                      alpha_p * x * x + beta * x,
                      alpha_n * x * torch.nn.functional.softsign(x) + 0.5 * x)


class XSSSLUR2(MegatronModule):
    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6, dtype=torch.float32):
        super().__init__(config=config)
        self.config = config
        self.alpha_p = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=dtype)) - 1.0).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_n_init - beta, dtype=dtype)) - 1.0).unsqueeze(0))
        self.beta = beta
        self.eps = torch.tensor(eps, dtype=torch.bfloat16, device='cuda')

    def forward(self, x):
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = self.beta + F.softplus(self.alpha_n)
        return compiled_xssslur2(x, alpha_p, alpha_n, self.beta, self.eps)


@jit_fuser
def sss(x):
    return 0.5 * (torch.nn.functional.softsign(x) + 1)


class SSS(MegatronModule):
    def __init__(self, config=None):
        super().__init__(config=config)

    def forward(self, x):
        return sss(x)


@jit_fuser
def ssslu(x):
    return (0.5 * (torch.nn.functional.softsign(x) + 1)) * x


class SSSLU(MegatronModule):
    def __init__(self, config=None):
        super().__init__(config=config)

    def forward(self, x):
        return ssslu(x)


@jit_fuser
def xsss(x, alpha):
    return alpha * torch.nn.functional.softsign(x) + 0.5


class XSSS(MegatronModule):
    def __init__(self, config=None, alpha_init=0.8, dtype=torch.bfloat16):
        super().__init__(config=config)
        self.config = config
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=dtype).unsqueeze(0))

    def forward(self, x):
        return xsss(x, self.alpha)


@jit_fuser
def xssslu(x, alpha):
    return (alpha * torch.nn.functional.softsign(x) + 0.5) * x


class XSSSLU(MegatronModule):
    def __init__(self, config=None, alpha_init=0.8, dtype=torch.bfloat16):
        super().__init__(config=config)
        self.config = config
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=dtype).unsqueeze(0))

    def forward(self, x):
        return xssslu(x, self.alpha)
