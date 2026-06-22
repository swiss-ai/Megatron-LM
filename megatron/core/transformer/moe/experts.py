# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from math import ceil
from typing import Optional, Protocol, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from megatron.core import tensor_parallel
from megatron.core.activations import PolyNorm, squared_relu
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fusions.fused_bias_geglu import quick_gelu, weighted_bias_quick_geglu_impl
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.transformer.mlp import (
    MLP,
    MLPSubmodules,
    TEActivationFunctionBuilder,
    apply_swiglu_sharded_factory,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    ProcessGroupCollection,
    get_align_size_for_quantization,
)
from megatron.core.transformer.moe.experts_util import (
    grouped_swiglu_mlp_torch_ref,
    ExpertsWgradScheduler,
)
from megatron.core.transformer.moe.experts_fp8_util import (
    fp8_grouped_swiglu_mlp,
)
from megatron.core.transformer.moe.experts_offloading_util import (
    StreamManager, 
    offloading_grouped_swiglu_mlp,
)
from megatron.core.transformer.moe.experts_offloading_fp8_util import (
    OffloadingFP8Config,
    offloading_fp8_grouped_swiglu_mlp,
)

from megatron.core.transformer.moe.fp8_utils import (
    build_offloading_expert_sharded_tensor,
    make_fused_experts_sharded_factory,
)

from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    sharded_state_dict_default,
)
from megatron.core.typed_torch import apply_module, not_none

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import Fp8Padding, Fp8Unpadding
else:
    Fp8Padding, Fp8Unpadding = None, None

try:
    import flashinfer.fused_moe as fused_moe
    from flashinfer.fused_moe.core import ActivationType

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False

from megatron.core.inference.moe import ActivationType as McoreActivationType
from megatron.core.inference.moe import (
    InferenceGroupedGemmBackend,
    mcore_fused_moe,
    resolve_inference_grouped_gemm_backend,
)

logger = logging.getLogger(__name__)

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None

from megatron.core.tensor_parallel import (
    get_cuda_rng_tracker,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    set_tensor_model_parallel_attributes,
)


class GroupedLinearFc1Interface(Protocol):
    """Interface for linear_fc1 module in TEGroupedMLP."""

    def forward(
        self, permuted_local_hidden_states: torch.Tensor, tokens_per_expert: list[int], /
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc1 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc1 module."""
        ...


class GroupedLinearFc1Builder(Protocol):
    """Protocol describing how to build a linear_fc1 layer in TEGroupedMLP."""

    def __call__(
        self,
        num_local_experts: int,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        pg_collection: ProcessGroupCollection | None,
    ) -> GroupedLinearFc1Interface:
        """Builds a linear_fc1 layer for TEGroupedMLP."""
        ...


class GroupedLinearFc2Interface(Protocol):
    """Protocol for linear_fc2 module in TEGroupedMLP."""

    def forward(
        self, intermediate_parallel: torch.Tensor, tokens_per_expert: list[int], /
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc2 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc2 module."""
        ...


class GroupedLinearFc2Builder(Protocol):
    """Protocol describing how to build a linear_fc2 layer in TEGroupedMLP."""

    def __call__(
        self,
        num_local_experts: int,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        pg_collection: ProcessGroupCollection | None,
    ) -> GroupedLinearFc2Interface:
        """Builds a linear_fc2 layer for TEGroupedMLP."""
        ...


@dataclass
class GroupedMLPSubmodules:
    """
    The dataclass for ModuleSpecs of TEGroupedMLP submodules
    including  linear fc1, activation function, linear fc2.
    """

    linear_fc1: GroupedLinearFc1Builder

    linear_fc2: GroupedLinearFc2Builder

    activation_func: TEActivationFunctionBuilder | None = None
    """
    Builder for an activation function module; only used if config.use_te_activation_func is True.
    """


class TEGroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using TE's GroupedLinear.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    # TODO(M4): breaking api, switched from pass in tp_group to pass in pg_collection.
    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: GroupedMLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.input_size = self.config.hidden_size
        assert not (
            self.config.add_bias_linear and config.bias_dropout_fusion
        ), "bias_dropout_fusion is not supported in TEGroupedMLP when add_bias_linear=True"

        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp

        # Double the output width with gated linear unit, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = not_none(self.config.moe_ffn_hidden_size)
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = submodules.linear_fc1(
            self.num_local_experts,
            self.input_size if self.config.moe_latent_size is None else self.config.moe_latent_size,
            ffn_hidden_size,
            config=self.config,
            init_method=not_none(self.config.init_method),
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=True,
            tp_comm_buffer_name='fc1',
            pg_collection=pg_collection,
        )

        if self.config.use_te_activation_func and not (submodules.activation_func is None):
            self.activation_func = apply_module(submodules.activation_func(config=self.config))
        else:
            self.activation_func = self.config.activation_func

        # PolyNorm GLU replaces the GLU gate with a learnable module. The grouped path runs
        # all local experts in one call, so we hold one coefficient pair per local expert and
        # expand them per-token via tokens_per_expert (see PolyNorm). tp_group is the
        # expert-TP group, over which each expert's ffn is sharded when ETP > 1.
        if self.config.pnglu:
            self.polynorm_glu = PolyNorm(
                num_local_experts=self.num_local_experts,
                config=self.config,
                tp_group=self.tp_group,
            )

        self.linear_fc2 = submodules.linear_fc2(
            self.num_local_experts,
            not_none(self.config.moe_ffn_hidden_size),
            (
                self.config.hidden_size
                if self.config.moe_latent_size is None
                else self.config.moe_latent_size
            ),
            config=self.config,
            init_method=not_none(self.config.output_layer_init_method),
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name='fc2',
            pg_collection=pg_collection,
        )

        self.offload_expert_fc1 = (
            self.config.fine_grained_activation_offloading
            and "expert_fc1" in self.config.offload_modules
        )

        self.offload_moe_act = (
            self.config.fine_grained_activation_offloading
            and "moe_act" in self.config.offload_modules
        )

        self.activation_recompute = (
            self.config.recompute_granularity == 'selective'
            and "moe_act" in self.config.recompute_modules
        )
        if self.activation_recompute and (self.config.fp8 or self.config.fp4):
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.linear_fc2)

        # This is to avoid the CPU overhead of multiple d2h copies
        if self.offload_expert_fc1:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.linear_fc1)

        if self.config.fp8 or self.config.fp4:
            assert HAVE_TE, "FP8 and FP4 requires TE."
            self.quantization_padding = Fp8Padding(self.num_local_experts)
            self.quantization_unpadding = Fp8Unpadding(self.num_local_experts)

    @staticmethod
    def _apply_bias(intermediate_parallel, bias_parallel, tokens_per_expert, permuted_probs):
        if bias_parallel is None:
            return intermediate_parallel
        shape = intermediate_parallel.shape
        return (
            torch.cat(
                [
                    t + b * p
                    for t, b, p in zip(
                        torch.split(intermediate_parallel.view(-1, shape[-1]), tokens_per_expert),
                        bias_parallel,
                        torch.split(permuted_probs, tokens_per_expert),
                    )
                ]
            )
            .view(shape)
            .to(intermediate_parallel.dtype)
        )

    def bias_act_func(
        self, intermediate_parallel, bias_parallel, permuted_probs, tokens_per_expert=None
    ):
        """
        Applies bias and activation function to the output of linear_fc1.

        ``tokens_per_expert`` (the per-local-expert token counts of ``intermediate_parallel``)
        is only used when ``config.pnglu`` is set, to map each expert's PolyNorm
        coefficients onto its tokens.
        """
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if permuted_probs is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * permuted_probs
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
            if self.activation_func == F.silu and self.config.gated_linear_unit:
                # dtype is handled inside the fused kernel
                intermediate_parallel = weighted_bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    permuted_probs,
                    self.config.activation_func_fp8_input_store,
                )
            elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                intermediate_parallel = weighted_bias_quick_geglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    permuted_probs,
                    self.config.activation_func_fp8_input_store,
                    self.config.glu_linear_offset,
                    self.config.activation_func_clamp_value,
                )
            else:
                raise ValueError("Only support fusion of swiglu and quick_gelu in TEGroupedMLP.")
        elif self.activation_func == squared_relu and self.config.use_fused_weighted_squared_relu:
            assert bias_parallel is None, "Bias is not supported with fused weighted squared relu."
            intermediate_parallel = weighted_squared_relu_impl(
                intermediate_parallel, permuted_probs
            )
        else:
            # When PolyNorm fuses the permuted_probs multiply into its kernel we skip the
            # eager post-multiply below.
            probs_fused = False
            if self.config.gated_linear_unit and self.config.pnglu:
                x_glu, x_linear = torch.chunk(intermediate_parallel, 2, dim=-1)
                if (val := self.config.activation_func_clamp_value) is not None:
                    x_glu = x_glu.clamp(min=None, max=val)
                    x_linear = x_linear.clamp(min=-val, max=val)
                if self.config.glu_linear_offset != 0.0:
                    x_linear = x_linear + self.config.glu_linear_offset
                intermediate_parallel = self.polynorm_glu(
                    x_glu, x_linear, tokens_per_expert=tokens_per_expert, scores=permuted_probs
                )
                probs_fused = True
            elif self.config.gated_linear_unit:

                def glu(x):
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    gate = self.config.activation_func(x_glu)
                    return gate * (x_linear + self.config.glu_linear_offset)

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)
            if not probs_fused:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * permuted_probs
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        return intermediate_parallel

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of TEGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.
            permuted_probs (torch.Tensor): The permuted probs of each token produced by the router.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        tokens_per_expert: list[int] = tokens_per_expert.tolist()
        if self.config.fp8 or self.config.fp4:
            actual_tokens_per_expert = tokens_per_expert
            permuted_local_hidden_states, tokens_per_expert = self.quantization_padding(
                permuted_local_hidden_states, tokens_per_expert
            )
            permuted_probs, _ = self.quantization_padding(
                permuted_probs.unsqueeze(-1), actual_tokens_per_expert
            )
        else:
            permuted_probs = permuted_probs.unsqueeze(-1)

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        with off_interface(
            self.offload_expert_fc1, permuted_local_hidden_states, "expert_fc1"
        ) as permuted_local_hidden_states:
            fc1_output, bias_parallel = apply_module(self.linear_fc1)(
                permuted_local_hidden_states, tokens_per_expert
            )
        if self.offload_expert_fc1:
            fc1_output = off_interface.group_commit(
                fc1_output,
                name="expert_fc1",
                forced_released_tensors=[permuted_local_hidden_states],
            )

        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with off_interface(self.offload_moe_act, fc1_output, "moe_act") as fc1_output:
                # Bind tokens_per_expert (a non-tensor) via partial so the checkpointed
                # function still receives only tensor args.
                bias_act_output = self.activation_checkpoint.checkpoint(
                    partial(self.bias_act_func, tokens_per_expert=tokens_per_expert),
                    fc1_output,
                    bias_parallel,
                    permuted_probs,
                )
        else:
            with off_interface(self.offload_moe_act, fc1_output, "moe_act") as fc1_output:
                bias_act_output = self.bias_act_func(
                    fc1_output, bias_parallel, permuted_probs, tokens_per_expert
                )
        output, output_bias = apply_module(self.linear_fc2)(bias_act_output, tokens_per_expert)
        if self.activation_recompute:
            self.activation_checkpoint.discard_output_and_register_recompute(output)

        # Delay the offload of the moe act until after the linear_fc2 has been computed
        # to make sure the fc1_output is reloaded to GPU before recomputing moe_act.
        if self.offload_moe_act:
            output = off_interface.group_commit(
                output, name="moe_act", forced_released_tensors=[fc1_output]
            )
        output = self._apply_bias(output, output_bias, tokens_per_expert, permuted_probs)

        # upad and concat the output
        if self.config.fp8 or self.config.fp4:
            output = self.quantization_unpadding(output, actual_tokens_per_expert)

        output_bias = None

        return output, output_bias

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Maps local expert to global experts.
        The sharded state dict is interchangable with SequentialMLP's.
        """
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        sharded_state_dict = {}
        for name, module in self._modules.items():
            module_sharded_offsets = sharded_offsets
            if name == 'polynorm_glu' and not singleton_local_shards:
                # The PolyNorm coefficients are stored as a single (num_local_experts,)
                # tensor. Prepend an expert-parallel sharding axis so this rank's coefficients
                # occupy the [ep_rank * num_local_experts : ...] slice of the global tensor,
                # mirroring how the expert weights are mapped to global experts. Without this,
                # every EP rank would write the same key with identical offsets/replica_id and
                # silently overwrite each other. (Note: this maps local->global experts for a
                # fixed EP size; resharding the coefficients across a different EP size, or
                # cross-loading them between TEGroupedMLP and SequentialMLP, is not supported.)
                module_sharded_offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), self.ep_group.rank(), self.ep_group.size()),
                )
            sub_sd = sharded_state_dict_default(
                module, f'{name}.', module_sharded_offsets, metadata, tp_group=self.tp_group
            )
            if name == 'linear_fc1' and self.config.gated_linear_unit:
                num_global_experts = self.ep_group.size() * self.num_local_experts
                local_expert_indices_offset = self.ep_group.rank() * self.num_local_experts
                ep_axis = len(sharded_offsets)
                for i in range(self.num_local_experts):
                    if singleton_local_shards:
                        new_sharded_offsets = sharded_offsets
                    else:
                        new_sharded_offsets = (
                            *sharded_offsets,
                            (ep_axis, local_expert_indices_offset + i, num_global_experts),
                        )
                    for k in (f'{name}.weight{i}', f'{name}.bias{i}'):
                        if k in sub_sd:
                            sub_sd[k] = apply_swiglu_sharded_factory(
                                sub_sd[k], new_sharded_offsets, singleton_local_shards
                            )
            if singleton_local_shards:
                replace_prefix_for_sharding(sub_sd, '', f'{prefix}experts.')
            else:
                # Add prefix here to match sequential's keys
                replace_prefix_for_sharding(sub_sd, f'{name}.', f'{prefix}experts.{name}.')
            sharded_state_dict.update({f"{prefix}{k}": v for k, v in sub_sd.items()})
        return sharded_state_dict

    def backward_dw(self):
        """Performs backward pass for weight gradients in TEGroupedMLP.

        This method executes the backward pass for weight gradients by calling
        backward_dw() on the linear layers in reverse order (fc2 followed by fc1).
        If an error occurs during execution, it is caught and re-raised with a
        descriptive message.
        """
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()


class InferenceGroupedMLP(TEGroupedMLP):
    """Inference-optimized GroupedMLP with GPU-resident offsets.

    Inherits from TEGroupedMLP to reuse weight initialization and checkpoint compatibility.
    Supports three forward paths:
    - Training: delegates to parent TEGroupedMLP
    - Inference + CUDA graphed: FlashInfer cutlass_fused_moe (fused permute + GEMM)
    - Inference + eager: torch.nn.functional.grouped_mm with GPU-resident cumsum offsets
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: GroupedMLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        # Initialize parent TEGroupedMLP (creates linear_fc1, linear_fc2)
        super().__init__(
            num_local_experts=num_local_experts,
            config=config,
            submodules=submodules,
            pg_collection=pg_collection,
        )

        # Concatenated weights are built lazily on first forward to ensure
        # checkpoint loading has already populated the per-expert parameters.
        self._concatenated_weights_built = False

        self.is_inference_cuda_graphed_iteration = False

        if HAVE_FLASHINFER:
            self._flashinfer_activation_type = self._resolve_flashinfer_activation_type()

        self._mcore_activation_type = self._resolve_mcore_activation_type()
        self.inference_grouped_gemm_backend = config.inference_grouped_gemm_backend

    def _resolve_flashinfer_activation_type(self):
        """Map megatron activation config to FlashInfer ActivationType."""
        assert (
            HAVE_FLASHINFER
        ), "flashinfer-python is required to resolve FlashInfer activation type."
        func = self.config.activation_func
        if func == F.silu:
            return ActivationType.Silu
        elif func == F.gelu:
            return ActivationType.Gelu
        elif func == F.relu:
            return ActivationType.Relu
        elif func == squared_relu:
            return ActivationType.Relu2
        raise ValueError(f"No FlashInfer ActivationType mapping for activation_func={func}")

    def _resolve_mcore_activation_type(self):
        """Map megatron activation config to mcore_fused_moe ActivationType."""
        func = self.config.activation_func
        if func == squared_relu:
            return McoreActivationType.SQUARED_RELU
        raise ValueError(f"No mcore_fused_moe ActivationType mapping for activation_func={func}")

    def set_inference_cuda_graphed_iteration(self):
        """Enable CUDA-graphed iteration mode."""
        self.is_inference_cuda_graphed_iteration = True

    def unset_inference_cuda_graphed_iteration(self):
        """Disable CUDA-graphed iteration mode."""
        self.is_inference_cuda_graphed_iteration = False

    def _build_concatenated_mxfp8_weights(self):
        """Build stacked MXFP8 weight tensors from per-expert MXFP8Tensor attributes.

        After quantize_model_to_mxfp8, each per-expert weight (weight0, weight1, ...)
        has been replaced with an MXFP8Tensor. This method stacks their data and
        scales into _fc1_weight / _fc2_weight for scaled_grouped_mm.

        Note: this creates a contiguous copy since per-expert MXFP8Tensor attributes
        are not contiguous across experts. This is a one-time cost at first forward.

        Unlike _build_concatenated_weights, this does not create nn.Parameter views
        back into the buffer — MXFP8 weights are not nn.Parameters (they are plain
        MXFP8Tensor attributes set by quantize_model_to_mxfp8). This path is only
        intended for non-colocated inference.
        """

        for linear_name, buf_name in [('linear_fc1', '_fc1_weight'), ('linear_fc2', '_fc2_weight')]:
            linear = getattr(self, linear_name)
            q_list, s_list = [], []
            for i in range(self.num_local_experts):
                w = getattr(linear, f'weight{i}')
                if isinstance(w, MXFP8Tensor):
                    mxfp8 = w
                elif hasattr(w, 'data') and isinstance(w.data, MXFP8Tensor):
                    mxfp8 = w.data
                else:
                    raise RuntimeError(
                        f"Expected MXFP8Tensor for {linear_name}.weight{i}, "
                        f"got {type(w).__name__}. Was quantize_model_to_mxfp8 called?"
                    )
                q_list.append(mxfp8.data)
                s_list.append(mxfp8.scale)

            stacked_data = torch.stack(q_list, dim=0).contiguous()
            stacked_scale = torch.stack(s_list, dim=0).contiguous()

            setattr(self, buf_name, MXFP8Tensor(data=stacked_data, scale=stacked_scale))

            # Redirect per-expert weight .data to views into the stacked buffer,
            # mirroring _build_concatenated_weights. This frees the original
            # allocations while keeping the Parameter objects intact.
            for i in range(self.num_local_experts):
                w = getattr(linear, f'weight{i}')
                if isinstance(w, MXFP8Tensor):
                    w.data = stacked_data[i]
                    w.scale = stacked_scale[i]
                elif hasattr(w, 'data') and isinstance(w.data, MXFP8Tensor):
                    w.data.data = stacked_data[i]
                    w.data.scale = stacked_scale[i]

    @torch.inference_mode(False)  # needed for non-colocated inference.
    def _build_concatenated_weights(self):
        """Create big contiguous weight tensors that share storage with TE's per-expert parameters.

        Creates _fc1_weight and _fc2_weight as contiguous tensors of shape
        [num_experts, out_features, in_features]. Instead of replacing TE's parameters
        (which breaks TE's internal bookkeeping), we redirect each parameter's .data
        to be a view into the contiguous buffer. The nn.Parameter objects themselves
        remain untouched in TE's module, preserving FP8 scaling state, etc.

        This allows:
        - TE's forward to work correctly (same Parameter objects, same internal state)
        - Training updates to flow through (param.data is a view into the big tensor)
        - torch.nn.functional.grouped_mm / FlashInfer to use the big tensor directly
        """
        # Get device/dtype from existing TE weights
        device = self.linear_fc1.weight0.device
        dtype = self.linear_fc1.weight0.dtype

        fc1_shape = self.linear_fc1.weight0.shape  # [out_features, in_features]
        fc2_shape = self.linear_fc2.weight0.shape

        # Create big contiguous tensors
        _fc1_weight = torch.empty(self.num_local_experts, *fc1_shape, device=device, dtype=dtype)
        _fc2_weight = torch.empty(self.num_local_experts, *fc2_shape, device=device, dtype=dtype)

        # Copy existing TE weights into big tensors, then point param.data to the views
        for i in range(self.num_local_experts):
            fc1_param = getattr(self.linear_fc1, f'weight{i}')
            fc2_param = getattr(self.linear_fc2, f'weight{i}')

            # Copy initialized data into contiguous buffer
            _fc1_weight[i].copy_(fc1_param.data)
            _fc2_weight[i].copy_(fc2_param.data)

            # Redirect param.data to view into contiguous buffer.
            # The nn.Parameter object stays the same — TE's internal state is preserved.
            fc1_param.data = _fc1_weight[i]
            fc2_param.data = _fc2_weight[i]

        # Register big tensors as non-persistent buffers (for .to() device movement, not saved)
        self.register_buffer('_fc1_weight', _fc1_weight, persistent=False)
        self.register_buffer('_fc2_weight', _fc2_weight, persistent=False)

    def _flashinfer_forward(self, hidden_states, routing_map, probs):
        """FlashInfer fused MoE kernel for CUDA-graphed inference iterations."""
        assert HAVE_FLASHINFER, "flashinfer-python is required for FlashInfer forward path."
        assert probs.dtype == torch.float32, "FlashInfer forward path requires fp32 probabilities."
        output = fused_moe.cutlass_fused_moe(
            hidden_states,
            routing_map.int(),
            probs,
            self._fc1_weight,
            self._fc2_weight,
            hidden_states.dtype,
            quant_scales=None,
            activation_type=self._flashinfer_activation_type,
            ep_size=self.ep_group.size(),
            ep_rank=self.ep_group.rank(),
        )[0]
        return output, None

    def _mcore_fused_moe_forward(
        self, hidden_states, probs, routing_map=None, tokens_per_expert=None, skip_permute=False
    ):
        """Torch grouped_mm fused MoE forward via mcore_fused_moe."""
        local_expert_start = self.ep_group.rank() * self.num_local_experts
        output = mcore_fused_moe(
            hidden_states,
            probs,
            self._fc1_weight,
            self._fc2_weight,
            activation_type=self._mcore_activation_type,
            num_local_experts=self.num_local_experts,
            local_expert_start=local_expert_start,
            routing_map=routing_map,
            tokens_per_expert=tokens_per_expert,
            skip_permute=skip_permute,
            disable_fused_quant_kernels=self.config.inference_moe_disable_fused_quant_kernels,
        )
        return output, None

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: Optional[torch.Tensor],
        permuted_probs: torch.Tensor,
        routing_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with three modes:

        - Training: delegates to parent TEGroupedMLP.
        - Inference + CUDA graphed: FlashInfer cutlass_fused_moe. tokens_per_expert
          is not used in this path; the FlashInfer kernel operates directly on
          routing_map.
        - Inference + eager: torch.nn.functional.grouped_mm with GPU-resident cumsum offsets.

        Args:
            permuted_local_hidden_states: [num_tokens, hidden_size] input hidden states.
            tokens_per_expert: [num_experts] number of tokens routed to each expert.
                None when using the CUDA-graphed FlashInfer path.
            permuted_probs: [num_tokens, topk] routing probabilities.
            routing_map: [num_tokens, topk] token-to-expert assignment indices.
                Required for the FlashInfer CUDA-graphed path, None otherwise.
        """

        if self.training:
            assert (
                not self.config.fp8_recipe == "mxfp8"
            ), "MXFP8 inference optimized is not compatible with training / colocated RL."
            return super().forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)

        # Lazily build concatenated weights on first forward (after checkpoint load)
        if not self._concatenated_weights_built:
            w = self.linear_fc1.weight0
            if isinstance(w, MXFP8Tensor) or (
                hasattr(w, 'data') and isinstance(w.data, MXFP8Tensor)
            ):
                self._build_concatenated_mxfp8_weights()
            else:
                self._build_concatenated_weights()
            self._concatenated_weights_built = True

        resolved_backend = resolve_inference_grouped_gemm_backend(
            self.inference_grouped_gemm_backend,
            self.is_inference_cuda_graphed_iteration,
            is_mxfp8=self.config.fp8_recipe == "mxfp8",
        )

        if resolved_backend == InferenceGroupedGemmBackend.FLASHINFER:
            assert routing_map is not None, "routing_map is required for FlashInfer forward pass."
            assert (
                self.is_inference_cuda_graphed_iteration
            ), "FlashInfer forward path is only used in CUDA-graphed inference iterations."
            return self._flashinfer_forward(
                permuted_local_hidden_states, routing_map, permuted_probs
            )
        elif resolved_backend == InferenceGroupedGemmBackend.TORCH:
            return self._mcore_fused_moe_forward(
                permuted_local_hidden_states,
                permuted_probs,
                routing_map=routing_map,
                tokens_per_expert=tokens_per_expert,
                skip_permute=(not self.is_inference_cuda_graphed_iteration),
            )
        elif resolved_backend == InferenceGroupedGemmBackend.TE:
            return super().forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)


class SequentialMLP(MegatronModule):
    """An implementation of the Experts layer using a sequence of MLP layers.

    This class executes each expert sequentially.
    """

    # TODO(M4): breaking api, switched from pass in tp_group to pass in pg_collection.
    def __init__(
        self,
        num_local_experts,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):

        if config.moe_ffn_hidden_size == config.ffn_hidden_size:
            super().__init__(config=config)
        else:
            # Local SequentialMLP can still be used here by overriding the ffn_hidden_size
            # with a deepcopied config.
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.ffn_hidden_size = config.moe_ffn_hidden_size
            super().__init__(config=sequential_mlp_config)

        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()
        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp
        # use pg_collection.expt_dp_group as data parallel group in this module.
        # TODO (Hepteract): expt_dp wont be needed here once distributed checkpoint is refactored
        self.dp_group = pg_collection.expt_dp

        for _ in range(self.num_local_experts):
            expert = MLP(
                self.config,
                submodules,
                ffn_hidden_size=self.config.moe_ffn_hidden_size,
                is_expert=True,
                tp_group=pg_collection.expt_tp,
            )
            self.local_experts.append(expert)

    def _pad_tensor_for_quantization(self, hidden, probs):
        """Padding tensor shape to multiples of 16/32."""
        actual_num_tokens = hidden.shape[0]
        divisor = get_align_size_for_quantization(self.config)
        padded_num_tokens = ceil(actual_num_tokens / divisor) * divisor - actual_num_tokens
        if padded_num_tokens > 0:
            pad_tensor = torch.zeros(
                padded_num_tokens, hidden.shape[1], dtype=hidden.dtype, device=hidden.device
            )
            hidden = torch.cat((hidden, pad_tensor), dim=0)
            pad_probs = torch.zeros(padded_num_tokens, dtype=probs.dtype, device=probs.device)
            probs = torch.cat((probs, pad_probs), dim=0)
        return hidden, probs

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward step of the SequentialMLP."""

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if self.num_local_experts == 1:
            if self.config.fp8 or self.config.fp4:
                hidden, probs = self._pad_tensor_for_quantization(
                    permuted_local_hidden_states, permuted_probs
                )
                output, output_bias = self.local_experts[0](hidden, probs)
                output = output[: permuted_local_hidden_states.shape[0]]
            else:
                output, output_bias = self.local_experts[0](
                    permuted_local_hidden_states, permuted_probs
                )

            return output, output_bias
        else:
            tokens_per_expert = tokens_per_expert.tolist()
            tokens_list = torch.split(permuted_local_hidden_states, tokens_per_expert)
            probs_list = torch.split(permuted_probs, tokens_per_expert)

            output_local_list = []

            for expert, tokens, probs in zip(self.local_experts, tokens_list, probs_list):
                if self.config.fp8 or self.config.fp4:
                    hidden, probs = self._pad_tensor_for_quantization(tokens, probs)
                    output, output_bias = expert(hidden, probs)
                    output = output[: tokens.shape[0]]
                else:
                    output, output_bias = expert(tokens, probs)
                output_local_list.append(output)

            output_local = torch.cat(output_local_list, dim=0)
            output_bias_local = None
            # Note: if bias is enabled on experts, it is already added to the output at this point
            return output_local, output_bias_local

    def backward_dw(self):
        """Backward pass for weight gradients in SequentialMLP."""
        for expert in self.local_experts:
            expert.backward_dw()

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Maps local expert to global experts."""
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        num_global_experts = self.ep_group.size() * self.num_local_experts
        local_expert_indices_offset = self.ep_group.rank() * self.num_local_experts

        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)

        for expert_local_idx, expert in enumerate(self.local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_state_dict_prefix = f'{prefix}local_experts.{expert_local_idx}.'
            if singleton_local_shards:
                expert_sharded_prefix = f'{prefix}experts.{expert_global_idx}.'
                expert_sharded_offsets = sharded_offsets
            else:
                expert_sharded_prefix = f'{prefix}experts.'
                expert_sharded_offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), expert_global_idx, num_global_experts),
                )

            expert_state_dict = expert.sharded_state_dict(
                expert_state_dict_prefix, expert_sharded_offsets, metadata
            )
            # Remove expert layers indexing from sharded keys
            replace_prefix_for_sharding(
                expert_state_dict, expert_state_dict_prefix, expert_sharded_prefix
            )
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in expert_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                    len(replica_id) == 3
                ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'

                sh_ten.replica_id = (*replica_id[:2], self.dp_group.rank())

            sharded_state_dict.update(expert_state_dict)
        return sharded_state_dict
    


class OffloadingExpertsMLP(MegatronModule):
    """An implementation of the Experts layer with fine-grained experts offloading.

    This class executes each expert sequentially and offloads expert to CPU
    to save GPU memory.
    """

    def __init__(
        self, 
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config)
        self.num_local_experts = num_local_experts

        assert config.gradient_accumulation_fusion

        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.tp_group = pg_collection.expt_tp
        # use pg_collection.expt_dp_group as data parallel group in this module.
        self.dp_group = pg_collection.expt_dp
        # How many feature each rank holds for fc1 and fc2, respectively.
        etp_size = self.tp_group.size()
        etp_rank = self.tp_group.rank()

        assert etp_size == 1, "Expert-Tensor parallelism is not supported in OffloadingExpertsMLP"
        self.expert_parallel = config.expert_model_parallel_size > 1
        
        self.input_size = self.config.hidden_size \
            if self.config.moe_latent_size is None \
            else self.config.moe_latent_size
        
        fc1_output_size = self.config.moe_ffn_hidden_size * (2 if self.config.gated_linear_unit else 1)
        fc1_output_size_per_partition = fc1_output_size
        fc2_input_size = self.config.moe_ffn_hidden_size
        fc2_input_size_per_partition = fc2_input_size

        # Determine expert shape
        # NOTE: when FP8 is enabled, we use NT layout for GEMM computation
        # and the weight shape is initialized as (output_size, input_size)
        fc1_expert_weight_shape = (
            self.input_size,
            fc1_output_size,
        ) if not self.config.moe_use_inplace_fp8_param else (
            fc1_output_size,
            self.input_size,
        )

        fc2_expert_weight_shape = (
            fc2_input_size,
            self.input_size,
        ) if not self.config.moe_use_inplace_fp8_param else (
            self.input_size,
            fc2_input_size,
        )

        if self.config.pnglu:
            self.polynorm_glu = PolyNorm(
                num_local_experts=self.num_local_experts,
                config=self.config,
                tp_group=self.tp_group,
            )
        
        # For now all expert weights are offloaded in CPU.
        # NOTE: when FP8 is enabled, we allocate experts as a single tensor
        # to make sure we can access main_grad as a single tensor in the backward
        if self.config.moe_use_inplace_fp8_param:
            self.weight1 = Parameter(
                torch.empty(
                    self.num_local_experts,
                    *fc1_expert_weight_shape,
                    dtype=config.params_dtype,
                    device="cpu" if not self.config.moe_offloading_experts_debug_mode else torch.cuda.current_device(),
                    pin_memory= not self.config.moe_offloading_experts_debug_mode,
                    # device=torch.cuda.current_device(),
                )
            )
            self.weight1.skip_backward_post_hook = config.moe_offloading_experts_skip_post_backward_hook
            self.weight1.is_cpu_offloaded = not self.config.moe_offloading_experts_debug_mode

            self.weight2 = Parameter(
                torch.empty(
                    self.num_local_experts,
                    *fc2_expert_weight_shape,
                    dtype=config.params_dtype,
                    device="cpu" if not self.config.moe_offloading_experts_debug_mode else torch.cuda.current_device(),
                    pin_memory= not self.config.moe_offloading_experts_debug_mode,
                    # device=torch.cuda.current_device(),
                )
            )
            self.weight2.skip_backward_post_hook = config.moe_offloading_experts_skip_post_backward_hook
            self.weight2.is_cpu_offloaded = not self.config.moe_offloading_experts_debug_mode

            self.weight1_list = None
            self.weight2_list = None

            if config.perform_initialization:
                if config.moe_offloading_experts_te_style_init:
                    # Inplace-FP8 weights are already stored in TE's [out, in] layout.
                    self._init_expert_weights_like_te(
                        self.weight1,
                        self.weight2,
                        fc1_out_size=fc1_output_size,
                        fc1_in_size=self.input_size,
                        fc2_out_size=self.input_size,
                        fc2_in_size=fc2_input_size,
                        weights_in_te_layout=True,
                    )
                else:
                    for i in range(self.num_local_experts):
                        _initialize_affine_weight_cpu(
                            self.weight1[i],
                            *fc1_expert_weight_shape,
                            fc1_output_size_per_partition,
                            partition_dim=1,
                            init_method=config.init_method,
                            params_dtype=config.params_dtype,
                            rank=etp_rank,
                            world_size=etp_size,
                        )
                        _initialize_affine_weight_cpu(
                            self.weight2[i],
                            *fc2_expert_weight_shape,
                            fc2_input_size_per_partition,
                            partition_dim=0,
                            init_method=config.output_layer_init_method,
                            params_dtype=config.params_dtype,
                            rank=etp_rank,
                            world_size=etp_size,
                        )

            setattr(self.weight1, 'allreduce', not self.expert_parallel)
            setattr(self.weight2, 'allreduce', not self.expert_parallel)
        else:
            self.weight1 = []
            self.weight2 = []
            for i in range(self.num_local_experts):
                self.register_parameter(
                    f'weight1_expert_{i}',
                    Parameter(
                        torch.empty(
                            self.input_size,
                            self.config.moe_ffn_hidden_size * (2 if config.gated_linear_unit else 1),
                            dtype=config.params_dtype,
                            device="cpu" if not self.config.moe_offloading_experts_debug_mode else torch.cuda.current_device(),
                            pin_memory= not self.config.moe_offloading_experts_debug_mode,
                        )
                    ),
                )
                self.weight1.append(getattr(self, f'weight1_expert_{i}'))
                self.weight1[i].skip_backward_post_hook = config.moe_offloading_experts_skip_post_backward_hook and not config.moe_offloading_experts_debug_mode
                self.weight1[i].is_cpu_offloaded = not self.config.moe_offloading_experts_debug_mode

                self.register_parameter(
                    f'weight2_expert_{i}',
                    Parameter(
                        torch.empty(
                            self.config.moe_ffn_hidden_size,
                            self.input_size,
                            dtype=config.params_dtype,
                            device="cpu" if not self.config.moe_offloading_experts_debug_mode else torch.cuda.current_device(),
                            pin_memory= not self.config.moe_offloading_experts_debug_mode,
                        )
                    ),
                )
                self.weight2.append(getattr(self, f'weight2_expert_{i}'))
                self.weight2[i].skip_backward_post_hook = config.moe_offloading_experts_skip_post_backward_hook and not config.moe_offloading_experts_debug_mode
                self.weight2[i].is_cpu_offloaded = not self.config.moe_offloading_experts_debug_mode

                # TE-style init runs as a group after all params are registered
                # (see below); the original CPU init runs per-expert here.
                if config.perform_initialization and not config.moe_offloading_experts_te_style_init:
                    _initialize_affine_weight_cpu(
                        self.weight1[i],
                        self.input_size,
                        fc1_output_size,
                        fc1_output_size_per_partition,
                        partition_dim=1,
                        init_method=config.init_method,
                        params_dtype=config.params_dtype,
                        rank=etp_rank,
                        world_size=etp_size,
                    )
                    _initialize_affine_weight_cpu(
                        self.weight2[i],
                        fc2_input_size,
                        self.input_size,
                        fc2_input_size_per_partition,
                        partition_dim=0,
                        init_method=config.output_layer_init_method,
                        params_dtype=config.params_dtype,
                        rank=etp_rank,
                        world_size=etp_size,
                    )
                setattr(self.weight1[i], 'allreduce', not self.expert_parallel)
                setattr(self.weight2[i], 'allreduce', not self.expert_parallel)

            if config.perform_initialization and config.moe_offloading_experts_te_style_init:
                # Non-FP8 weights are stored transposed ([in, out]) relative to TE.
                self._init_expert_weights_like_te(
                    self.weight1,
                    self.weight2,
                    fc1_out_size=fc1_output_size,
                    fc1_in_size=self.input_size,
                    fc2_out_size=self.input_size,
                    fc2_in_size=fc2_input_size,
                    weights_in_te_layout=False,
                )

        # GPU buffer to prefetch CPU weights
        self.num_stages = self.config.moe_offloading_num_stages
        self.num_chunks = self.config.moe_offloading_num_chunks
        assert num_local_experts % self.num_chunks == 0, "num_local_experts should be divisible by num_steps."
        self.chunk_size = num_local_experts // self.num_chunks # one chunk contains num_local_experts // self.num_chunks experts
        self.config.moe_offloading_chunk_size = self.chunk_size

        # lightweight config for the FP8 offloading autograd function
        self.fp8_config = OffloadingFP8Config.from_transformer_config(self.config)

        # allocate tensors for gpu buffers
        buffer_dtype = config.params_dtype if not self.config.moe_use_inplace_fp8_param else torch.float8_e4m3fn
        experts1_gpu_buffers_storage = torch.empty(
            self.num_stages * self.chunk_size * fc1_expert_weight_shape[0] * fc1_expert_weight_shape[1],
            device=torch.cuda.current_device(),
            dtype=buffer_dtype,
        ).view(
            self.num_stages, self.chunk_size, fc1_expert_weight_shape[0], fc1_expert_weight_shape[1]
        )
        experts2_gpu_buffers_storage = torch.empty(
            self.num_stages * self.chunk_size * fc2_expert_weight_shape[0] * fc2_expert_weight_shape[1],
            device=torch.cuda.current_device(),
            dtype=buffer_dtype,
        ).view(
            self.num_stages, self.chunk_size, fc2_expert_weight_shape[0], fc2_expert_weight_shape[1]
        )

        # organize as [num_stages, chunk_size, (in, out)]
        self.experts1_gpu_buffers = [
            [experts1_gpu_buffers_storage[s, c] for c in range(self.chunk_size)]
            for s in range(self.num_stages)
        ]
        self.experts2_gpu_buffers = [
            [experts2_gpu_buffers_storage[s, c] for c in range(self.chunk_size)]
            for s in range(self.num_stages)
        ]

        # organize as [num_stages, (chunk_size, in, out)]
        self.experts1_gpu_chunks = [
            experts1_gpu_buffers_storage[s] for s in range(self.num_stages)
        ]
        self.experts2_gpu_chunks = [
            experts2_gpu_buffers_storage[s] for s in range(self.num_stages)
        ]

        # cuda stream manager for h2d transfer and computation
        self.stream_manager = StreamManager.get_instance(num_compute_streams=1 if self.config.moe_use_inplace_fp8_param else 4)

        # scheduler to determine when to trigger wgrad compute
        self.expert_wgrad_scheduler = ExpertsWgradScheduler(config.delay_wgrad_compute)

        # store hooks after wgrad reduce
        self.wgrad_accumulation_and_reduce_hooks = []

        # padding function
        if self.config.moe_use_inplace_fp8_param:
            self.quantization_padding = Fp8Padding(self.num_local_experts, 128)
            self.quantization_unpadding = Fp8Unpadding(self.num_local_experts, 128)

    def _init_expert_weights_like_te(
        self,
        weights1,
        weights2,
        fc1_out_size: int,
        fc1_in_size: int,
        fc2_out_size: int,
        fc2_in_size: int,
        weights_in_te_layout: bool,
    ):
        """Initialize expert weights to match the TEGroupedMLP path.

        TEGroupedMLP delegates to TE's ``GroupedLinear``, which builds each expert weight on
        GPU in ``[out_features, in_features]`` layout and applies ``init_method`` under the
        expert-parallel CUDA RNG tracker, looping over experts within a single fork so that
        consecutive experts draw distinct values (and different EP ranks differ). The default
        CPU path (``_initialize_affine_weight_cpu``) instead builds an fp32 master weight with
        the un-forked global CPU RNG, so it produces different values. This method reproduces
        the TE behaviour:

        - ``linear_fc1`` is initialized before ``linear_fc2`` so the tracker's RNG stream
          advances in the same order as TE (which constructs fc1 before fc2).
        - each expert weight is created on GPU in ``params_dtype`` and ``[out, in]`` layout,
          matching TE's weight tensors element-for-element.
        - the result is copied into our (CPU-resident, possibly transposed) parameter, and the
          tensor-model-parallel attributes that ``_initialize_affine_weight_cpu`` would have
          stamped are preserved.

        ``weights_in_te_layout`` is True when the parameters are already stored as
        ``[out, in]`` (the inplace-FP8 case) and False when stored transposed as ``[in, out]``.
        """
        device = torch.cuda.current_device()
        # TE only forks the tracker once it has been seeded (e.g. via
        # model_parallel_cuda_manual_seed); otherwise it falls back to the default RNG.
        use_tracker = get_cuda_rng_tracker().is_initialized()

        def _init_group(weights, out_size, in_size, init_method, partition_dim):
            def _do_init():
                for i in range(self.num_local_experts):
                    # TE orientation: [out_features, in_features], params_dtype, on GPU.
                    weight = torch.empty(
                        out_size, in_size, dtype=self.config.params_dtype, device=device
                    )
                    init_method(weight)
                    if not weights_in_te_layout:
                        # Our parameter stores [in_features, out_features]; transpose to match.
                        weight = weight.t()
                    with torch.no_grad():
                        weights[i].data.copy_(weight)
                    # Preserve the TP attributes set by _initialize_affine_weight_cpu, which are
                    # read by sharded_state_dict and gradient bookkeeping.
                    set_tensor_model_parallel_attributes(
                        tensor=weights[i], is_parallel=True, dim=partition_dim, stride=1
                    )

            if use_tracker:
                with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
                    _do_init()
            else:
                _do_init()

        # Order matters: TE constructs linear_fc1 before linear_fc2.
        _init_group(weights1, fc1_out_size, fc1_in_size, self.config.init_method, partition_dim=1)
        _init_group(
            weights2, fc2_out_size, fc2_in_size, self.config.output_layer_init_method, partition_dim=0
        )

    def _apply(self, fn, recurse=True):
        saved = {}
        for name, p in list(self._parameters.items()):
            if p is not None and getattr(p, 'is_cpu_offloaded', False):
                saved[name] = self._parameters.pop(name)
        out = super()._apply(fn, recurse=recurse)
        for name, p in saved.items():
            self._parameters[name] = p
        return out

    def _forward_debug(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Run a GPU-resident reference path without expert weight offloading."""
        if not self.config.moe_use_inplace_fp8_param:
            return grouped_swiglu_mlp_torch_ref(
                self.weight1,
                self.weight2,
                permuted_local_hidden_states,
                tokens_per_expert,
                self.num_local_experts,
                permuted_probs,
                self.expert_wgrad_scheduler,
                self.config,
            )

        tokens_per_expert_list = tokens_per_expert.tolist()
        has_tokens = permuted_local_hidden_states.nelement() != 0
        if has_tokens:
            permuted_local_hidden_states, tokens_per_expert_padded = self.quantization_padding(
                permuted_local_hidden_states, tokens_per_expert_list
            )
            permuted_probs, _ = self.quantization_padding(
                permuted_probs.unsqueeze(-1), tokens_per_expert_list
            )
            tokens_per_expert = torch.tensor(
                tokens_per_expert_padded, dtype=torch.int32, device='cpu'
            )

        output = fp8_grouped_swiglu_mlp(
            self.weight1,
            self.weight2,
            permuted_local_hidden_states,
            tokens_per_expert,
            self.num_local_experts,
            permuted_probs,
            self.expert_wgrad_scheduler,
            self.config,
            self.wgrad_accumulation_and_reduce_hooks,
        )

        if has_tokens:
            output = self.quantization_unpadding(output, tokens_per_expert_list)
        return output
    
    def _polynorm_glu_coeffs(self, tokens_per_expert):
        """Per-token positive PolyNorm-GLU coefficients.

        abs and repeat_interleave run outside the offloading autograd Function so the 
        gradients of alpha parameters flow back via standard autograd. 
        """
        if not self.config.pnglu:
            return None, None
        assert (
            self.config.activation_func_clamp_value is None
            and self.config.glu_linear_offset == 0.0
        ), (
            "PolyNorm GLU on the FP8 offloading path does not yet support "
            "activation_func_clamp_value / glu_linear_offset."
        )
        device = self.polynorm_glu.alpha_1.device
        if isinstance(tokens_per_expert, torch.Tensor):
            tpe_tensor = tokens_per_expert.to(device=device)
        else:
            tpe_tensor = torch.tensor(tokens_per_expert, device=device)
        a1 = torch.repeat_interleave(torch.abs(self.polynorm_glu.alpha_1), tpe_tensor)
        a2 = torch.repeat_interleave(torch.abs(self.polynorm_glu.alpha_2), tpe_tensor)
        return a1, a2

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        if permuted_local_hidden_states.nelement() != 0:
            if self.config.moe_offloading_experts_debug_mode:
                return self._forward_debug(
                    permuted_local_hidden_states, tokens_per_expert, permuted_probs
                ), None
            
            if self.config.moe_use_inplace_fp8_param:
                # create weight list
                # NOTE: we cannot create the unbind list in __init__ 
                # because the weight tensor will be replaced by DDP
                if self.weight1_list is None:
                    self.weight1_list = list(torch.unbind(self.weight1, dim=0))
                if self.weight2_list is None:
                    self.weight2_list = list(torch.unbind(self.weight2, dim=0))
                
                tokens_per_expert_list = tokens_per_expert.tolist()
                permuted_local_hidden_states, tokens_per_expert_padded = self.quantization_padding(
                    permuted_local_hidden_states, tokens_per_expert_list
                )
                permuted_probs, _ = self.quantization_padding(
                    permuted_probs.unsqueeze(-1), tokens_per_expert_list
                )
                tokens_per_expert_padded = torch.tensor(tokens_per_expert_padded, dtype=torch.int32, device='cpu')

                # Per-token PolyNorm coefficients computation
                a1, a2 = self._polynorm_glu_coeffs(tokens_per_expert_padded)

                output = offloading_fp8_grouped_swiglu_mlp(
                    self.weight1,
                    self.weight2,
                    self.weight1_list,
                    self.weight2_list,
                    self.experts1_gpu_buffers,
                    self.experts2_gpu_buffers,
                    self.experts1_gpu_chunks,
                    self.experts2_gpu_chunks,
                    permuted_local_hidden_states,
                    tokens_per_expert_padded,
                    self.num_local_experts,
                    permuted_probs,
                    self.expert_wgrad_scheduler,
                    self.stream_manager,
                    self.fp8_config,
                    self.wgrad_accumulation_and_reduce_hooks,
                    a1,
                    a2,
                )

                output = self.quantization_unpadding(output, tokens_per_expert_list)

                return output, None
            
            output = offloading_grouped_swiglu_mlp(
                self.weight1,
                self.weight2,
                self.experts1_gpu_buffers,
                self.experts2_gpu_buffers,
                permuted_local_hidden_states,
                tokens_per_expert,
                self.num_local_experts,
                permuted_probs,
                self.expert_wgrad_scheduler,
                self.stream_manager,
                self.config,
                self.wgrad_accumulation_and_reduce_hooks,
            )

            return output, None
        else:
            if self.config.moe_offloading_experts_debug_mode:
                return self._forward_debug(
                    permuted_local_hidden_states, tokens_per_expert, permuted_probs
                ), None

            if self.config.moe_use_inplace_fp8_param:
                # create weight list
                # NOTE: we cannot create the unbind list in __init__ 
                # because the weight tensor will be replaced by DDP
                if self.weight1_list is None:
                    self.weight1_list = list(torch.unbind(self.weight1, dim=0))
                if self.weight2_list is None:
                    self.weight2_list = list(torch.unbind(self.weight2, dim=0))

                # Empty input: counts sum to 0, so a1/a2 are length-0 (None on the SwiGLU path).
                a1, a2 = self._polynorm_glu_coeffs(tokens_per_expert)

                output = offloading_fp8_grouped_swiglu_mlp(
                    self.weight1,
                    self.weight2,
                    self.weight1_list,
                    self.weight2_list,
                    self.experts1_gpu_buffers,
                    self.experts2_gpu_buffers,
                    self.experts1_gpu_chunks,
                    self.experts2_gpu_chunks,
                    permuted_local_hidden_states,
                    tokens_per_expert,
                    self.num_local_experts,
                    permuted_probs,
                    self.expert_wgrad_scheduler,
                    self.stream_manager,
                    self.fp8_config,
                    self.wgrad_accumulation_and_reduce_hooks,
                    a1,
                    a2,
                )

                return output, None

            # NOTE: it should be safe to pass empty tensor to the custom function,
            # but it will introduce meanless h2d transfer.
            # TODO: add cost free path for empty input
            output = offloading_grouped_swiglu_mlp(
                self.weight1,
                self.weight2,
                self.experts1_gpu_buffers,
                self.experts2_gpu_buffers,
                permuted_local_hidden_states,
                tokens_per_expert,
                self.num_local_experts,
                permuted_probs,
                self.expert_wgrad_scheduler,
                self.stream_manager,
                self.config,
                self.wgrad_accumulation_and_reduce_hooks,
            )
            return output, None
        
    def backward_dw(self):
        # Debug paths compute weight gradients during the regular backward pass.
        if self.config.delay_wgrad_compute and not self.config.moe_offloading_experts_debug_mode:
            self.expert_wgrad_scheduler.pop_callback()
            self.expert_wgrad_scheduler.pop_callback()

            # trigger grad reduce hook
            for hook_fn in self.wgrad_accumulation_and_reduce_hooks:
                hook_fn()
    
    def register_wgrad_accumulation_and_reduce_hooks(self, hook_fn):
        self.wgrad_accumulation_and_reduce_hooks.append(hook_fn)
    
    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Maps local experts to global experts (interchangeable across variants).

        Both OffloadingExpertsMLP variants emit the same per-expert, expert-parallel
        sharded layout under keys ``{prefix}experts.weight{1,2}`` in ``(in, out)``
        orientation, so a checkpoint saved by one is loadable by the other:

        - bf16 variant: per-expert params already ``(in, out)`` -> saved directly.
        - inplace-fp8 variant: a single fused ``(num_local, out, in)`` master is
          transposed per expert via a ``ShardedTensorFactory`` (one per fused
          param, keyed by the real param name so ``load_state_dict`` maps it back).
        """
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        assert self.tp_group.size() == 1, "OffloadingExpertsMLP assumes ETP size == 1"

        num_global_experts = self.ep_group.size() * self.num_local_experts
        local_expert_indices_offset = self.ep_group.rank() * self.num_local_experts
        replica_id = (0, 0, self.dp_group.rank())

        if self.config.moe_use_inplace_fp8_param:
            # The fused bf16 master is only valid when fp8 lives in extra storage;
            # otherwise self.weight1/2 are overwritten in place with packed fp8 bytes
            # (see _quantize_weight in experts_offloading_fp8_util.py).
            assert self.config.moe_use_extra_fp8_param_storage, (
                "Checkpointing inplace-fp8 OffloadingExpertsMLP requires "
                "moe_use_extra_fp8_param_storage=True so the bf16 weights are "
                "preserved (otherwise self.weight1/2 hold packed fp8 bytes)."
            )
            sharded_state_dict = {}
            for wname, fused_weight in (('weight1', self.weight1), ('weight2', self.weight2)):
                sharded_state_dict[f'{prefix}{wname}'] = make_fused_experts_sharded_factory(
                    fused_weight,
                    prefix,
                    wname,
                    num_local_experts=self.num_local_experts,
                    local_expert_indices_offset=local_expert_indices_offset,
                    num_global_experts=num_global_experts,
                    sharded_offsets=sharded_offsets,
                    replica_id=replica_id,
                    singleton_local_shards=singleton_local_shards,
                )
            return sharded_state_dict

        sharded_state_dict = {}
        for i in range(self.num_local_experts):
            g_idx = local_expert_indices_offset + i
            w1 = getattr(self, f'weight1_expert_{i}')
            w2 = getattr(self, f'weight2_expert_{i}')

            sharded_state_dict[f'{prefix}weight1_expert_{i}'] = (
                build_offloading_expert_sharded_tensor(
                    w1,
                    prefix,
                    'weight1',
                    g_idx,
                    sharded_offsets=sharded_offsets,
                    num_global_experts=num_global_experts,
                    replica_id=replica_id,
                    singleton_local_shards=singleton_local_shards,
                    transpose=False,
                )
            )
            sharded_state_dict[f'{prefix}weight2_expert_{i}'] = (
                build_offloading_expert_sharded_tensor(
                    w2,
                    prefix,
                    'weight2',
                    g_idx,
                    sharded_offsets=sharded_offsets,
                    num_global_experts=num_global_experts,
                    replica_id=replica_id,
                    singleton_local_shards=singleton_local_shards,
                    transpose=False,
                )
            )
        return sharded_state_dict
