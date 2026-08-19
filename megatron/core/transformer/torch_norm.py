# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Callable, Union

import torch
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_torch_min_version

_SEEDNORM_ACTIVATIONS = {
    "tanh": torch.tanh,
    "softsign": F.softsign,
}


class WrappedTorchNorm:
    """
    A conditional wrapper to initialize an instance of PyTorch's
    `LayerNorm` or `RMSNorm` based on input
    """

    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        # TODO: unused arguments.
        # See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/issues/223
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",
    ):
        if config.normalization != "SeeDNorm":
            assert (
                not config.layernorm_zero_centered_gamma
            ), "layernorm_zero_centered_gamma requires SeeDNorm with the torch backend"

        assert not config.persist_layer_norm, f"persist_layer_norm not supported by torch LayerNorm"

        assert not config.sequence_parallel, f"sequence parallel not supported by torch LayerNorm"

        assert (
            not config.memory_efficient_layer_norm
        ), f"memory_efficient_layer_norm not supported by torch LayerNorm"

        if config.normalization == "SeeDNorm":
            return SeeDNorm(
                hidden_size=hidden_size,
                eps=eps,
                init=getattr(config, "seednorm_init", 1.0),
                sequence_parallel=config.sequence_parallel,
                activation=getattr(config, "seednorm_activation", "tanh"),
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
            )

        if config.normalization == "LayerNorm":
            norm_cls = torch.nn.LayerNorm
        elif config.normalization == "RMSNorm":
            assert is_torch_min_version(
                "2.4.0a0"
            ), 'Torch RMSNorm requires PyTorch version >= 2.4.0'

            norm_cls = torch.nn.RMSNorm
        elif config.normalization == "L2Norm":
            norm_cls = torch.nn.L2Norm
        else:
            raise Exception("Only LayerNorm, RMSNorm and L2Norm are currently supported")

        return norm_cls(normalized_shape=hidden_size, eps=eps)


class L2Norm(torch.nn.Module):
    """
    Applies L2 normalization to the input tensor along the last dimension.

    This module normalizes the input tensor such that the mean of the squared values
    along the last dimension is 1 (within a small epsilon for numerical stability).

    Args:
        hidden_size (int): Expected input shape for normalization (not used internally).
        eps (float, optional): A small value added to the denominator for numerical stability.
            Default: 1e-6.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    @jit_fuser
    def _norm(self, x):
        """
        Performs the actual L2 normalization.

        Args:
            x (torch.Tensor): The input tensor to normalize.

        Returns:
            torch.Tensor: The L2-normalized tensor.
        """
        x_float = x.float()
        return (x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)

    def forward(self, x):
        """
        Forward pass of the L2Norm module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: L2-normalized tensor with the same dtype as input.
        """
        return self._norm(x)

class SeeDNorm(torch.nn.Module):
    """SeeDNorm implementation following the SeeDNorm pseudocode."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        init: float = 1.0,
        sequence_parallel: bool = False,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "tanh",
        zero_centered_gamma: bool = False,
        **kwargs,
    ):
        super().__init__()
        if isinstance(hidden_size, torch.Size):
            normalized_shape = hidden_size
        elif isinstance(hidden_size, (tuple, list)):
            normalized_shape = torch.Size(hidden_size)
        else:
            normalized_shape = torch.Size((hidden_size,))

        # Determine activation function
        if isinstance(activation, str):
            if activation not in _SEEDNORM_ACTIVATIONS:
                raise ValueError(
                    f"Unsupported SeeDNorm activation '{activation}'. "
                    f"Supported values: {list(_SEEDNORM_ACTIVATIONS.keys())}"
                )
            self._activation = _SEEDNORM_ACTIVATIONS[activation]
            self.activation_name = activation
        elif callable(activation):
            self._activation = activation
            self.activation_name = getattr(activation, "__name__", "custom_activation")
        else:
            raise TypeError("activation must be a string identifier or a callable.")

        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        self.alpha = torch.nn.Parameter(torch.ones(normalized_shape) * init)
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))
        gamma_init = (
            torch.zeros(normalized_shape)
            if zero_centered_gamma
            else torch.ones(normalized_shape)
        )
        self.gamma = torch.nn.Parameter(gamma_init)

        setattr(self.alpha, "sequence_parallel", sequence_parallel)
        setattr(self.alpha, "apply_weight_decay", True)
        setattr(self.beta, "sequence_parallel", sequence_parallel)
        if activation == "tanh":
            setattr(self.beta, "apply_weight_decay", True)
        else:
            setattr(self.beta, "skip_weight_decay", True)
        setattr(self.gamma, "sequence_parallel", sequence_parallel)

    @jit_fuser
    def _seed_norm(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        beta = self.beta.float()
        alpha = self.alpha.float()
        gamma = self.gamma.float()
        if self.zero_centered_gamma:
            gamma = gamma + 1.0

        rescale = self._activation(torch.matmul(x_float, beta))
        inv_rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x_float * inv_rms
        dynamic_scale = rescale.unsqueeze(-1) * alpha
        return ((dynamic_scale + gamma) * x_norm).type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._seed_norm(x)
