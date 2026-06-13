# Copyright (c) 2026, Swiss AI Institute
"""
This module implements utility classes and functions for Mixture of Experts (MoE) in the Megatron-LM framework, including:
1) ExpertsWgradScheduler: a utility class to manage the scheduling of weight gradient computations for MoE experts, allowing for delayed computation of weight gradients to enable better interleaving of GPU computation and CPU-GPU communication.
2) MergedSwiGLU: a custom autograd function that implements the forward and backward pass of the SwiGLU activation function, with optional probability scaling for the forward pass and corresponding adjustments in the backward pass.
3) GroupedSwiMLP: a custom autograd function that implements the forward and backward pass of a grouped linear layer followed by a SwiGLU activation and another grouped linear layer, with support for delayed weight gradient computation and optional FP8 activation quantization for memory efficiency.
"""
from __future__ import annotations
import torch
import collections
import queue
from typing import Optional

from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from transformer_engine.pytorch import (
        Float8BlockQuantizer,
    )
    from transformer_engine.pytorch.constants import TE_DType
    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None

class ExpertsWgradScheduler:
    def __init__(self, delay_wgrad_compute: bool = False):
        self.delay_wgrad_compute = delay_wgrad_compute
        self.queue = queue.Queue()

    def register(self, grad_func, *grad_parms):
        if self.delay_wgrad_compute:
            self.queue.put((grad_func, grad_parms))

    def pop_callback(self):
        if self.queue.qsize() > 0 and self.delay_wgrad_compute:
            grad_func, grad_parms = self.queue.get()
            return grad_func(*grad_parms)
        else:
            # If there is no token assigned to the expert in this MoE layer,
            # then there will be case that the wgrad compute is not registered
            return
        

class MergedSwiGLU(torch.autograd.Function):
    """Re-implementation of Silu
    """

    @classmethod
    @torch.compile()
    def call_forward(
        cls,
        input_tensor: torch.Tensor,
        probs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """forward with optional probability scaling for SwiGLU activation. 
        If `probs` is provided, it will be used to scale the output of the SwiGLU activation, 
        otherwise it will compute the standard SwiGLU activation without scaling.

        Args:
            input_tensor (torch.Tensor): input tensor to the activation function
            probs (torch.Tensor | None, optional): Defaults to None.

        Returns:
            torch.Tensor: activation output
        """
        if probs is not None:
            return MergedSwiGLU.call_forward_silu_probs(input_tensor, probs)
        else:
            return MergedSwiGLU.call_forward_silu(input_tensor)

    @classmethod
    @torch.compile()
    def call_forward_silu(
        cls,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """forward pass for SwiGLU activation without probability scaling.

        Args:
            input_tensor (torch.Tensor): input tensor to the activation function

        Returns:
            torch.Tensor: activation output
        """
        a, b = input_tensor.chunk(2, dim=-1)
        return (torch.nn.functional.silu(a) * b).to(input_tensor.dtype)
    
    @classmethod
    @torch.compile()
    def call_forward_silu_probs(
        cls,
        input_tensor: torch.Tensor,
        probs: torch.Tensor
    ) -> torch.Tensor:
        """actual forward function with probability.

        Args:
            input_tensor (torch.Tensor): input tensor to the activation function
            probs (torch.Tensor): probability derived from router

        Returns:
            torch.Tensor: activation output
        """
        a, b = input_tensor.chunk(2, dim=-1)
        return ((torch.nn.functional.silu(a) * b) * probs).to(input_tensor.dtype)
    
    @classmethod
    @torch.compile()
    def call_backward(
        cls,
        grad_output: torch.Tensor,
        input_tensor: torch.Tensor,
        probs: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor | None]:
        """backward function for SwiGLU activation with optional probability scaling.

        Args:
            grad_output (torch.Tensor): gradient of the output from the activation function
            input_tensor (torch.Tensor): input tensor to the activation function
            probs (torch.Tensor | None, optional): Defaults to None.
        """
        if probs is not None:
            return MergedSwiGLU.call_backward_silu_probs(grad_output, input_tensor, probs)
        else:
            return MergedSwiGLU.call_backward_silu(grad_output, input_tensor)
    
    @classmethod
    @torch.compile()
    def call_backward_silu(
        cls,
        grad_output: torch.Tensor,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """actual backward function without probability.

        Args:
            grad_output (torch.Tensor): gradient of the output from the activation function
            input_tensor (torch.Tensor): input tensor to the activation function

        Returns:
            torch.Tensor: gradient of the input tensor
        """
        a, b = input_tensor.chunk(2, dim=-1)
        sigmoid_a = torch.sigmoid(a)
        ones = torch.ones(sigmoid_a.shape, device=sigmoid_a.device, dtype=sigmoid_a.dtype)
        grad_a = grad_output * (sigmoid_a + a * sigmoid_a * (ones - sigmoid_a)) * b
        grad_b = grad_output * torch.nn.functional.silu(a)
        return torch.cat([grad_a, grad_b], dim=-1)
    
    @classmethod
    @torch.compile()
    def call_backward_silu_probs(
        cls,
        grad_output: torch.Tensor,
        input_tensor: torch.Tensor,
        probs: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """actual backward function with probability. 
        It computes the gradient of the input tensor and the probability.

        Args:
            grad_output (torch.Tensor): gradient of the output from the activation function
            input_tensor (torch.Tensor): input tensor to the activation function
            probs (torch.Tensor): probability derived from router
        """
        input_grad = MergedSwiGLU.call_backward_silu(
            grad_output * probs, 
            input_tensor
        )
        weights_grad = MergedSwiGLU.call_forward_silu(
            input_tensor
        ) * grad_output.to(probs.dtype)
        weights_grad = torch.sum(
            weights_grad, dim=-1
        )

        return input_grad.to(input_tensor.dtype) if input_grad is not None else None, \
        weights_grad.to(probs.dtype) if weights_grad is not None else None
    
    @staticmethod
    def forward(
        ctx,
        *args,
    ):
        args_q = collections.deque(args)
        input_tensor: torch.Tensor = args_q.popleft()
        probs: torch.Tensor | None = args_q.popleft()

        ctx.save_for_backward(input_tensor, probs)
        return MergedSwiGLU.call_forward(input_tensor, probs)
    
    @staticmethod
    def backward(
        ctx, 
        *grad_outputs
    ):
        grad_y: torch.Tensor = grad_outputs[0]
        (x, probs) = ctx.saved_tensors
        input_grad, prob_grad = MergedSwiGLU.call_backward(
            grad_y, x, probs
        )
        if prob_grad is not None:
            prob_grad = prob_grad.unsqueeze(-1)
        return input_grad, prob_grad
        

def release(t: torch.Tensor):
    """Helper function to release tensors that are no longer needed to save memory.
    """
    t.untyped_storage().resize_(0)


_dummy_wgrads = {}

def get_dummy_wgrad(
    shape: list,
    dtype: torch.dtype,
    device,
    zero=False
) -> torch.Tensor:
    """Returns a dummy tensor of given shape."""
    global _dummy_wgrads
    wgard_key = (*shape, dtype)
    if wgard_key not in _dummy_wgrads:
        _dummy_wgrads[wgard_key] = torch.empty(
            shape,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
    if zero:
        _dummy_wgrads[wgard_key].fill_(0)
    return _dummy_wgrads[wgard_key].detach()

# deprecated legacy MLP implementation
class GroupedSwiMLP(torch.autograd.Function):
    @classmethod
    def call_forward_a(
        cls,
        w1: list[torch.nn.Parameter],
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """First linear projection in forward pass.

        Args:
            w1 (torch.nn.Parameter): weight parameter for the first linear layer
            permuted_local_hidden_states (torch.Tensor): input hidden states
            tokens_per_expert (torch.Tensor): number of tokens assigned to each expert

        Returns:
            torch.Tensor: output of the first linear layer
        """
        fc1_output = grouped_gemm.grouped_gemm.backend.gmmfwd(
            permuted_local_hidden_states, 
            w1, 
            tokens_per_expert, 
            trans_a=False, 
            trans_b=False,
            compute_streams=[],
        )

        return fc1_output
    
    @classmethod
    def call_forward_y(
        cls,
        w2: list[torch.nn.Parameter],
        a: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Activation and second linear projection in forward pass.

        Args:
            w2 (torch.nn.Parameter): weight parameter for the second linear layer
            a (torch.Tensor): output of the first linear layer
            tokens_per_expert (torch.Tensor): number of tokens assigned to each expert
            permuted_probs (torch.Tensor): probability derived from router

        Returns:
            tuple[torch.Tensor, torch.Tensor]
        """
        s = MergedSwiGLU.call_forward(
            a, permuted_probs.unsqueeze(-1)
        )
        fc2_output = grouped_gemm.grouped_gemm.backend.gmmfwd(
            s, 
            w2, 
            tokens_per_expert, 
            trans_a=False,
            trans_b=False,
            compute_streams=[],
        )
        return fc2_output, s
        

    @classmethod
    def call_backward_grad_a(
        cls,
        grad_y: torch.Tensor,
        a: torch.Tensor,
        w2: list[torch.nn.Parameter],
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the gradient of the input to the activation function in the backward pass.

        Args:
            grad_y (torch.Tensor): gradient of the output
            a (torch.Tensor): input to the activation function
            w2 (torch.nn.Parameter): weight parameter for the second linear layer
            tokens_per_expert (torch.Tensor): number of tokens assigned to each expert
            permuted_probs (torch.Tensor): probability derived from router
        """
        grad_s = grouped_gemm.grouped_gemm.backend.gmmfwd(
            grad_y, 
            w2, 
            tokens_per_expert,
            trans_a=False,
            trans_b=True,
            compute_streams=[],
        )
        return MergedSwiGLU.call_backward(grad_s, a, permuted_probs.unsqueeze(-1))

    @classmethod
    def call_backward_grad_x(
        cls,
        grad_a: torch.Tensor,
        w1: list[torch.nn.Parameter],
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the gradient of the input to the first linear layer in the backward pass.

        Args:
            grad_a (torch.Tensor): gradient of the input to the activation function
            w1 (torch.nn.Parameter): weight parameter for the first linear layer
            tokens_per_expert (torch.Tensor): number of tokens assigned to each expert
        """
        grad_x = grouped_gemm.grouped_gemm.backend.gmmfwd(
            grad_a,
            w1,
            tokens_per_expert,
            trans_a=False,
            trans_b=True,
            compute_streams=[],
        )

        return grad_x
    
    @staticmethod
    def _wgrad_post_process(
        w: list[torch.nn.Parameter],
        wgrad_output: list[torch.Tensor],
        fuse_gradient_accumulation: bool,
    ):
        # handle ddp
        for i in range(len(w)):
            if fuse_gradient_accumulation:
                w[i].grad_added_to_main_grad = True
            else:
                w[i].grad_added_to_main_grad = False
                w[i].grad = wgrad_output[i].view(w[i].shape)

    @classmethod
    def call_backward_grad_w2(
        cls,
        grad_y: torch.Tensor,
        a: torch.Tensor,
        w2: list[torch.nn.Parameter],
        tokens_per_expert: torch.Tensor,
        num_local_experts: int,
        w2_slice_shape: tuple,
        permuted_probs: torch.Tensor,
        wgrad_scheduler: ExpertsWgradScheduler = None,
        delay_wgrad_compute: bool = False,
        fuse_gradient_accumulation: bool = False,
    ) -> list[torch.Tensor]:
        """Calculate the gradient of the weight parameter for the second linear layer in the backward pass.
        Note: For now fuse_gradient_accumulation is not supported.
        Args:
            grad_y (torch.Tensor): gradient of the output
            a (torch.Tensor): input to the activation function
            w2 (torch.nn.Parameter): weight parameter for the second linear layer
            tokens_per_expert (torch.Tensor): number of tokens assigned to each expert
            permuted_probs (torch.Tensor): probability derived from router
            fuse_gradient_accumulation (bool, optional): Fuse gradient accumulation in gemm. Defaults to False.

        Returns:
            torch.Tensor: gradient of the weight parameter for the second linear layer
        """
        s = MergedSwiGLU.call_forward(a, permuted_probs.unsqueeze(-1))
        
        wgrad_output = None
        alpha = 1.0
        beta = 0.0
        if fuse_gradient_accumulation:
            wgrad_output = [w.main_grad for w in w2]
            beta = 1.0
        else:
            wgrad_output = [torch.empty(
                w.shape, 
                device=w.device, 
                dtype=w.dtype
            ) for w in w2]
            beta = 0.0

        # If delay_wgrad_compute is True and wgrad_scheduler is provided, 
        # register the wgrad computation to be executed later.
        if delay_wgrad_compute and wgrad_scheduler is not None:
            def _compute_w2_grad(*args):
                grouped_gemm.grouped_gemm.backend.gmmbwd(*args)   

            wgrad_scheduler.register(
                _compute_w2_grad,
                s, grad_y, tokens_per_expert, True, False, wgrad_output, alpha, beta
            )

            # handle ddp
            GroupedSwiMLP._wgrad_post_process(w2, wgrad_output, fuse_gradient_accumulation)
            return wgrad_output
        else:
        
            # compute wgrad immediately if not delay_wgrad_compute or wgrad_scheduler is None
            grad_w2 = grouped_gemm.grouped_gemm.backend.gmmbwd(
                s, 
                grad_y,
                tokens_per_expert,
                trans_a=True,
                trans_b=False,
                c = wgrad_output,
                alpha = alpha,
                beta = beta,
                compute_streams=[],
            )

            # post process wgrad for ddp
            GroupedSwiMLP._wgrad_post_process(w2, grad_w2, fuse_gradient_accumulation)
            return grad_w2

    @classmethod
    def call_backward_grad_w1(
        cls,
        grad_a: torch.Tensor,
        x: torch.Tensor,
        w1: list[torch.nn.Parameter],
        tokens_per_expert: torch.Tensor,
        num_local_experts: int,
        w1_slice_shape: tuple,
        wgrad_scheduler: ExpertsWgradScheduler = None,
        delay_wgrad_compute: bool = False,
        fuse_gradient_accumulation: bool = False,
    ) -> list[torch.Tensor]:
        """Calculate the gradient of the weight parameter for the first linear layer in the backward pass.
        Note: For now fuse_gradient_accumulation is not supported.

        Args:
            grad_a (torch.Tensor): gradient of the input to the activation function
            x (torch.Tensor): input to the first linear layer
            w1 (torch.nn.Parameter): weight parameter for the first linear layer
            tokens_per_expert (torch.Tensor): number of tokens assigned to each expert
            fuse_gradient_accumulation (bool, optional): Fuse gradient accumulation in gemm. Defaults to False.

        Returns:
            torch.Tensor: gradient of the weight parameter for the first linear layer
        """
        wgrad_output = None
        alpha = 1.0
        beta = 0.0
        if fuse_gradient_accumulation:
            wgrad_output = [w.main_grad for w in w1]
            beta = 1.0
        else:
            wgrad_output = [torch.empty(
                w.shape, 
                device=w.device, 
                dtype=w.dtype
            ) for w in w1]
            beta = 0.0

        # If delay_wgrad_compute is True and wgrad_scheduler is provided, 
        # register the wgrad computation to be executed later.
        if delay_wgrad_compute and wgrad_scheduler is not None:
            def _compute_w1_grad(*args):
                grouped_gemm.grouped_gemm.backend.gmmbwd(*args)
            
            wgrad_scheduler.register(
                _compute_w1_grad,
                x, grad_a, tokens_per_expert, True, False, wgrad_output, alpha, beta
            )

            # post process wgrad for ddp
            GroupedSwiMLP._wgrad_post_process(w1, wgrad_output, fuse_gradient_accumulation)
            return wgrad_output
        else:
            # compute wgrad immediately if not delay_wgrad_compute or wgrad_scheduler is None
            grad_w1 = grouped_gemm.grouped_gemm.backend.gmmbwd(
                x,
                grad_a,
                tokens_per_expert,
                trans_a=True,
                trans_b=False,
                c = wgrad_output,
                alpha = alpha,
                beta = beta,
                compute_streams=[],
            )

            # post process wgrad for ddp
            GroupedSwiMLP._wgrad_post_process(w1, grad_w1, fuse_gradient_accumulation)
            return grad_w1

    @staticmethod
    def forward(
        ctx,
        *args, 
        **kwargs
    ):
        if len(args) < 7:
            raise ValueError(f"Insufficient arguments for forward pass of GroupedSwiMLP. Expected at least 6, got {len(args)}")
        
        weights: list[torch.nn.Parameter] = args[:-6]
        w1: list[torch.nn.Parameter] = weights[:len(weights)//2]
        w2: list[torch.nn.Parameter] = weights[len(weights)//2:]
        permuted_local_hidden_states: torch.Tensor = args[-6]
        tokens_per_expert: torch.Tensor = args[-5]
        num_local_experts: int = args[-4]
        permuted_probs: torch.Tensor = args[-3]
        expert_wgrad_scheduler: ExpertsWgradScheduler = args[-2]
        config: TransformerConfig = args[-1]

        input_size = config.hidden_size if config.moe_latent_size is None else config.moe_latent_size
        w1_slice_shape = (num_local_experts, input_size, -1)
        w2_slice_shape = (num_local_experts, -1, input_size)

        # mlp1
        a = GroupedSwiMLP.call_forward_a(
            w1, permuted_local_hidden_states, tokens_per_expert
        )

        # act + mlp2
        y, _ = GroupedSwiMLP.call_forward_y(
            w2, a, tokens_per_expert, permuted_probs
        )

        # context saving
        ctx.expert_wgrad_scheduler = expert_wgrad_scheduler
        ctx.w1 = w1
        ctx.w2 = w2
        ctx.tokens_per_expert = tokens_per_expert
        ctx.num_local_experts = num_local_experts
        ctx.config = config
        ctx.w1_slice_shape = w1_slice_shape
        ctx.w2_slice_shape = w2_slice_shape

        activation_recompute = (
            config.recompute_granularity == 'selective'
            and "moe_act" in config.recompute_modules
        )
        ctx.activation_recompute = activation_recompute
        if config.moe_use_fp8_activation:
            if HAVE_TE:
                quantizer = Float8BlockQuantizer(
                    fp8_dtype=TE_DType[torch.float8_e4m3fn],
                    rowwise=True,
                    columnwise=False,
                    amax_epsilon=0.0,
                    force_pow_2_scales=True,
                    block_scaling_dim=1,
                )
                qx = quantizer.make_empty(
                    permuted_local_hidden_states.shape, 
                    dtype=permuted_local_hidden_states.dtype, 
                    device=permuted_local_hidden_states.device, 
                    requires_grad=False
                )
                qx = quantizer.update_quantized(
                    permuted_local_hidden_states, qx
                )
                release(permuted_local_hidden_states)
                

                if activation_recompute:
                    ctx.qx = qx
                    ctx.qa = None
                    ctx.save_for_backward(
                        None, None, permuted_probs
                    )
                    release(a)
                else:
                    qa = quantizer.make_empty(
                        a.shape, 
                        dtype=a.dtype, 
                        device=a.device, 
                        requires_grad=False
                    )
                    qa = quantizer.update_quantized(
                        a, qa
                    )
                    ctx.qx = qx
                    ctx.qa = qa
                    ctx.save_for_backward(
                        None, None, permuted_probs
                    )
                    release(a)
        else:
            if activation_recompute:
                ctx.save_for_backward(
                    permuted_local_hidden_states, None, permuted_probs
                )
                release(a)
            else:
                ctx.save_for_backward(
                    permuted_local_hidden_states, a, permuted_probs
                )

        return y, None

    @staticmethod
    def backward(
        ctx, 
        *grad_outputs
    ):
        config: TransformerConfig = ctx.config
        tokens_per_expert: torch.Tensor = ctx.tokens_per_expert
        num_local_experts: int = ctx.num_local_experts
        w1: list[torch.nn.Parameter] = ctx.w1
        w2: list[torch.nn.Parameter] = ctx.w2
        (x, a, probs) = ctx.saved_tensors
        w1_slice_shape = ctx.w1_slice_shape
        w2_slice_shape = ctx.w2_slice_shape
        expert_wgrad_scheduler: ExpertsWgradScheduler = ctx.expert_wgrad_scheduler

        # rematerialize activation if needed
        # NOTE: fp8 tensors have to be manually released after dequantization
        if config.moe_use_fp8_activation:
            x = ctx.qx.dequantize()
            release(ctx.qx)
            if not ctx.activation_recompute:
                a = ctx.qa.dequantize()
                release(ctx.qa)
            else:
                a = GroupedSwiMLP.call_forward_a(
                    w1, x, tokens_per_expert
                )
        else:
            if ctx.activation_recompute:
                a = GroupedSwiMLP.call_forward_a(
                    w1, x, tokens_per_expert
                )

        grad_y = grad_outputs[0].contiguous()

        # backward computation
        grad_a, grad_probs = GroupedSwiMLP.call_backward_grad_a(
            grad_y, 
            a, 
            w2, 
            tokens_per_expert,
            probs,
        )

        grad_x = None if grad_a is None else GroupedSwiMLP.call_backward_grad_x(
            grad_a,
            w1,
            tokens_per_expert,
        )

        grad_w2 = GroupedSwiMLP.call_backward_grad_w2(
            grad_y, 
            a,
            w2,
            tokens_per_expert, 
            num_local_experts,
            w2_slice_shape,
            probs,
            expert_wgrad_scheduler,
            config.delay_wgrad_compute,
            config.gradient_accumulation_fusion,
        )

        grad_w1 = None if grad_a is None else GroupedSwiMLP.call_backward_grad_w1(
            grad_a, 
            x, 
            w1,
            tokens_per_expert,
            num_local_experts,
            w1_slice_shape,
            expert_wgrad_scheduler,
            config.delay_wgrad_compute,
            config.gradient_accumulation_fusion,
        )

        return *grad_w1, *grad_w2, grad_x, None, None, grad_probs, None, None
    
def grouped_swiglu_mlp(
    w1: list[torch.nn.Parameter],
    w2: list[torch.nn.Parameter],
    permuted_local_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_local_experts: int,
    permuted_probs: torch.Tensor,
    expert_wgrad_scheduler: ExpertsWgradScheduler,
    config: TransformerConfig,
) -> torch.Tensor:
    """Autograd function for Grouped SwiGLU MLP.

    Args:
        w1 (torch.nn.Parameter): weight parameter for the first linear layer
        w2 (torch.nn.Parameter): weight parameter for the second linear layer
        permuted_local_hidden_states (torch.Tensor): input hidden states
        tokens_per_expert (torch.Tensor): number of tokens assigned to each expert
        permuted_probs (torch.Tensor): probability derived from router
        config (TransformerConfig): transformer configuration

    Returns:
        torch.Tensor: output of the MLP
    """
    output, _ = GroupedSwiMLP.apply(
        *w1, 
        *w2, 
        permuted_local_hidden_states,
        tokens_per_expert,
        num_local_experts,
        permuted_probs,
        expert_wgrad_scheduler,
        config,
    )

    return output

def grouped_swiglu_mlp_torch_ref(
    w1,
    w2,
    permuted_local_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_local_experts: int,
    permuted_probs: torch.Tensor,
    expert_wgrad_scheduler: Optional[ExpertsWgradScheduler] = None,
    config: Optional["TransformerConfig"] = None,
) -> torch.Tensor:
    """Pure-PyTorch reference path for the grouped SwiGLU MoE experts.

    Drop-in replacement for ``grouped_swiglu_mlp`` used only to verify
    correctness. Each expert is evaluated with plain ``torch.matmul`` and the
    backward pass is handled entirely by autograd -- no grouped GEMM, no custom
    CUDA streams / weight-prefetch buffers, and no manual ``main_grad`` writes.
    Expert-weight gradients reach ``main_grad`` through the standard DDP backward
    hook, exactly like an ordinary linear layer. This isolates whether a failure
    lives in the grouped-GEMM / offloading machinery or in the surrounding
    dispatcher / VPP wiring.

    Per expert ``i`` (``x_i: [t_i, in]``, ``w1[i]: [in, 2H]``, ``w2[i]: [H, in]``):

        fc1 = x_i @ w1[i]                 # [t_i, 2H]
        gate, lin = fc1.chunk(2, dim=-1)
        s = (silu(gate) * lin) * probs_i  # [t_i, H]
        y_i = s @ w2[i]                   # [t_i, in]

    ``expert_wgrad_scheduler`` and ``config`` are accepted only for signature
    compatibility with ``grouped_swiglu_mlp`` and are unused.
    """
    # Normalize weights to a list of per-expert 2D tensors (supports both the
    # per-expert parameter list and a stacked [E, in, 2H] / [E, H, in] tensor).
    w1_list = list(torch.unbind(w1, dim=0)) if isinstance(w1, torch.Tensor) else list(w1)
    w2_list = list(torch.unbind(w2, dim=0)) if isinstance(w2, torch.Tensor) else list(w2)

    # torch.split needs python ints; .tolist() syncs if tokens_per_expert is on GPU.
    tokens = (
        tokens_per_expert.tolist()
        if isinstance(tokens_per_expert, torch.Tensor)
        else list(tokens_per_expert)
    )

    x_chunks = torch.split(permuted_local_hidden_states, tokens, dim=0)
    probs_chunks = torch.split(permuted_probs.reshape(-1), tokens, dim=0)

    outputs = []
    for i in range(num_local_experts):
        x_i = x_chunks[i]
        fc1 = torch.matmul(x_i, w1_list[i])                       # [t_i, 2H]
        gate, lin = fc1.chunk(2, dim=-1)
        s = F.silu(gate) * lin                                    # [t_i, H]
        s = (s * probs_chunks[i].unsqueeze(-1)).to(x_i.dtype)
        outputs.append(torch.matmul(s, w2_list[i]))               # [t_i, in]

    return torch.cat(outputs, dim=0)