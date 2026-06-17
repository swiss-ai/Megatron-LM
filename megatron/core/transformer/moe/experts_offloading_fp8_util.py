# Copyright (c) 2026, Swiss AI Institute
"""
Utilities for MoE experts offloading with FP8 support, including:
1) FP8ExpertsParameterManager: a singleton class to manage the FP8 quantization of expert weights
2) OffloadingExpertsFP8GroupedSwiMLP: an autograd function for the forward and backward pass of the grouped SwiGLU MLP in offloading experts with FP8 support
"""
from __future__ import annotations
import torch
import itertools

from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None

from megatron.core.transformer.moe.experts_offloading_util import StreamManager
from megatron.core.transformer.moe.experts_util import ExpertsWgradScheduler, MergedSwiGLU
from megatron.core.transformer.moe.fp8_utils import (
    m_grouped_fp8_gemm_nt_contiguous, 
    k_grouped_fp8_gemm_nt_contiguous,
    release,
)
from megatron.core.transformer.moe.fp8_jit import (
    per_block_cast_to_fp8_gpu,
    per_token_cast_to_fp8,
    per_token_cast_to_fp8_chunked_fused,
    per_channel_cast_to_fp8_pack_kmajor,
    per_token_dequant_from_fp8,
    per_token_dequant_from_fp8_chunked,
)
from megatron.core.transformer.moe.swiglu_jit import (
    swiglu_forward,
    swiglu_backward,
)

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

def _dequant_packed_kmajor(
    fp8_packed: torch.Tensor,
    sf_t: torch.Tensor,
    ks: list[int],
    n: int,
    gran_k: int = 128,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Inverse of ``per_channel_cast_to_fp8_pack_kmajor`` (debug/guard only).

    Reconstructs the ``(sum(ks), n)`` operand from the per-expert k-major packed
    fp8 buffer and its transposed scale view ``sf_t`` of shape
    ``(n, sum(ks)//gran_k)``. Used to measure how much energy the quantization
    actually preserved.
    """
    device = fp8_packed.device
    m_total = sum(ks)
    out = torch.empty(m_total, n, dtype=out_dtype, device=device)
    sf = sf_t.float()
    off_row = off_elem = off_grp = 0
    for k_g in ks:
        if k_g == 0:
            continue
        # expert slab is (n, k_g): slab[channel, token_local]
        slab = fp8_packed[off_elem:off_elem + k_g * n].view(n, k_g).float()
        g = k_g // gran_k
        sc = sf[:, off_grp:off_grp + g]                 # (n, g) per-channel/group scale
        sc_full = sc.repeat_interleave(gran_k, dim=1)   # (n, k_g)
        out[off_row:off_row + k_g] = (slab * sc_full).t().to(out_dtype)
        off_row += k_g
        off_elem += k_g * n
        off_grp += g
    return out

def guarded_per_channel_cast_to_fp8_pack_kmajor(
    x: torch.Tensor,
    ks: list[int],
    ks_tensor: torch.Tensor,
    prefixes: torch.Tensor,
    *,
    name: str,
    config: TransformerConfig,
    gran_k: int = 128,
    free_input: bool = False,
):
    """``per_channel_cast_to_fp8_pack_kmajor`` + optional wgrad guard.

    Drop-in replacement for the raw cast at the w2-wgrad operand sites. When the
    guard is disabled this is behaviorally identical to calling the cast with the
    same ``free_input``. When enabled, the input is validated *before* being
    freed, so the guard can compare against the original bf16 operand.
    """
    def _fp8_wgrad_guard_mode(config) -> str:
        """Return 'off' | 'warn' | 'raise'.

        Enabled via env ``MOE_FP8_WGRAD_GUARD`` (``1``/``warn`` -> warn,
        ``raise``/``strict``/``2`` -> raise) or ``config.moe_fp8_wgrad_guard``
        (with ``config.moe_fp8_wgrad_guard_strict`` selecting raise).
        """
        # env = os.environ.get("MOE_FP8_WGRAD_GUARD", "")
        # if env:
        #     return "raise" if env.lower() in ("raise", "strict", "2") else "warn"
        # if getattr(config, "moe_fp8_wgrad_guard", False):
        #     return "raise" if getattr(config, "moe_fp8_wgrad_guard_strict", False) else "warn"
        return "off"
    
    def guard_fp8_wgrad_operand(
        name: str,
        x: torch.Tensor,
        fp8_packed: torch.Tensor,
        sf_t: torch.Tensor,
        ks: list[int],
        mode: str,
        gran_k: int = 128,
        rel_tol: float = 0.03,
    ) -> None:
        """Validate one FP8-quantized w2-wgrad operand.

        Checks, in increasing cost:
        1. ``sf`` and the packed fp8 buffer are finite (catches the denormal-scale
            reciprocal overflow / ``0*inf`` NaN path);
        2. the dequantized operand is within ``rel_tol`` of the bf16 input
            (catches zeroing / saturation / a too-coarse amax floor).

        ``x`` must still be alive (call before any ``free_input``). On failure the
        message includes the operand name and the stats needed to localize it.
        """
        if mode == "off":
            # print(
            #     f"{name}: x_mean={x.float().mean().item():.3e} x_std={x.float().std().item():.3e} x_amax={x.float().abs().amax().item():.3e} x_norm={x.float().norm().item():.3e}"
            # )
            return

        problems = []
        sf_f = sf_t.float()
        fp8_f = fp8_packed.float()
        sf_finite = torch.isfinite(sf_f).all().item()
        fp8_finite = torch.isfinite(fp8_f).all().item()
        if not sf_finite:
            problems.append("scale factors contain NaN/Inf")
        if not fp8_finite:
            problems.append("packed fp8 contains NaN/Inf")

        x_norm = x.float().norm()
        rel = float("nan")
        if sf_finite and fp8_finite and x_norm > 0:
            x_deq = _dequant_packed_kmajor(fp8_packed, sf_t, ks, x.shape[1], gran_k)
            rel = ((x_deq - x.float()).norm() / x_norm).item()
            if not (rel == rel):  # NaN
                problems.append("dequant produced NaN")
            elif rel > rel_tol:
                problems.append(
                    f"dequant rel-error {rel:.3f} > {rel_tol} "
                    f"(energy lost -- zeroing / saturation / coarse amax floor)"
                )

        if not problems:
            return

        detail = (
            f"[wgrad2 fp8 guard] operand '{name}': {'; '.join(problems)} | "
            f"x_amax={x.float().abs().amax().item():.3e} x_norm={x_norm.item():.3e} "
            f"sf_min={sf_f.min().item():.3e} sf_max={sf_f.max().item():.3e} "
            f"rel_err={rel:.3f} shape={tuple(x.shape)} ks_sum={sum(ks)}"
        )
        if mode == "raise":
            print(detail)
            raise FloatingPointError(detail)
    
    
    mode = _fp8_wgrad_guard_mode(config)
    # Defer freeing until after the guard has inspected the bf16 input.
    fp8 = per_channel_cast_to_fp8_pack_kmajor(
        x, ks, ks_tensor, prefixes,
        use_ue8m0=False, gran_k=gran_k,
        free_input=(free_input and mode == "off"),
    )

    guard_fp8_wgrad_operand(name, x, fp8[0], fp8[1], ks, mode, gran_k=gran_k)
    if free_input:
        x.data.untyped_storage().resize_(0)
    return fp8


class FP8ExpertsParameterManager:
    _instance = None

    @classmethod
    def create_instance(
        cls,
        config: TransformerConfig,
    ):
        if cls._instance is None:
            cls._instance = cls(config=config)

    @classmethod
    def get_instance(
        cls,
    ) -> FP8ExpertsParameterManager:
        assert cls._instance is not None, "FP8ExpertsParameterManager instance is not created yet."
        return cls._instance
    
    @classmethod
    def refresh(
        cls,
        bf16_weight: torch.Tensor,
    ):
        if cls._instance is not None:
            cls._instance.refresh_fp8_weights(bf16_weight)

    @classmethod
    def reset_instance(
        cls,
    ):
        if cls._instance is not None:
            cls._instance.reset()
    
    @classmethod
    def mark_first_microbatch(
        cls,
    ):
        if cls._instance is not None:
            cls._instance.set_is_first_microbatch()

    def __init__(
        self,
        fp8_recipe: int = 128,
        config: TransformerConfig = None,
    ):
        # wid is the data pointer of each expert weight
        self._wid_to_bf16_weight_map: dict[int, torch.nn.Parameter] = {}
        self._wid_to_sliced_bf16_weight_map: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._wid_to_fp8_weight_map: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        
        # uid is the data pointer of the first local expert weight
        self._uid_to_fp8_weights_map = {}

        # flag to indicate whether it is the first microbatch
        # which is used to determine whether to refresh the fp8 weights
        self._uid_to_is_first_microbatch: dict[int, bool] = {}

        # pre-allocate quantization buffer for expert weights
        self.expert_quantization_buffer: dict[int, torch.Tensor] = {}
        self.expert_quantization_buffer[
            config.moe_ffn_hidden_size * (2 if config.gated_linear_unit else 1) * config.hidden_size
        ] = (
            torch.empty(
                config.moe_ffn_hidden_size * (2 if config.gated_linear_unit else 1) * config.hidden_size, 
                device=torch.cuda.current_device(), 
                dtype=torch.bfloat16
            ), 
            torch.empty(
                config.moe_ffn_hidden_size * (2 if config.gated_linear_unit else 1) * config.hidden_size,
                device=torch.cuda.current_device(),
                dtype=torch.float8_e4m3fn
            )
        )
        self.expert_quantization_buffer[
            config.moe_ffn_hidden_size * config.hidden_size
        ] = (
            torch.empty(
                config.moe_ffn_hidden_size * config.hidden_size,
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16
            ),
            torch.empty(
                config.moe_ffn_hidden_size * config.hidden_size,
                device=torch.cuda.current_device(),
                dtype=torch.float8_e4m3fn
            )
        )
        self.config = config

        if self.config.moe_use_extra_fp8_param_storage:
            # extra storage for fp8 weights
            self._wid_to_extra_fp8_weight_storage: dict[int, torch.Tensor] = {}
            self._wid_to_extra_fp8_weight_storage_slices: dict[int, tuple[torch.Tensor]] = {}

    def get_fp8_weights(
        self,
        bf16_weights: list[torch.nn.Parameter] | list[torch.Tensor],
        transposed: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]: # (fp8_weights, fp8_weight_scales)
        uid = bf16_weights[0].data_ptr()

        # when the weights are first accessed
        if uid not in self._uid_to_fp8_weights_map:
            wids = [w.data_ptr() for w in bf16_weights]

            # build wid maps
            for wid, w in zip(wids, bf16_weights):
                if wid not in self._wid_to_bf16_weight_map:
                    self._wid_to_bf16_weight_map[wid] = w
                    sliced_w = list(torch.unbind(w.data.view(2, w.shape[0] // 2, *w.shape[1:]), dim=0))
                    self._wid_to_sliced_bf16_weight_map[wid] = \
                        (sliced_w[0].view(torch.float8_e4m3fn).view(w.shape[0], w.shape[1]), 
                         sliced_w[1].view(torch.float8_e4m3fn).view(w.shape[1], w.shape[0]))
                    self._wid_to_fp8_weight_map[wid] = self._quantize_weight(wid)
            
            # build uid map
            fp8_weights = []
            fp8_weights_scales = []
            fp8_weights_t = []
            fp8_weights_t_scales = []
            for wid in wids:
                (fp8_weight, fp8_weight_scale, fp8_weight_t, fp8_weight_t_scale) = self._wid_to_fp8_weight_map[wid]
                fp8_weights.append(fp8_weight)
                fp8_weights_scales.append(fp8_weight_scale)
                fp8_weights_t.append(fp8_weight_t)
                fp8_weights_t_scales.append(fp8_weight_t_scale)
            
            # create new tensor for the stacked scale tensors
            fp8_weights_scales_stack = torch.stack(fp8_weights_scales)
            fp8_weights_t_scales_stack = torch.stack(fp8_weights_t_scales)
            fp8_weights_scales_list = list(torch.unbind(fp8_weights_scales_stack))
            fp8_weights_t_scales_list = list(torch.unbind(fp8_weights_t_scales_stack))
            
            # replace the scale tensors in the fp8 weight maps with the new stacked scale tensors
            for eid, wid in enumerate(wids):
                fp8_weight, _, fp8_weight_t, _ = self._wid_to_fp8_weight_map[wid]
                self._wid_to_fp8_weight_map[wid] = (fp8_weight, fp8_weights_scales_list[eid], fp8_weight_t, fp8_weights_t_scales_list[eid])
            
            fp8_weights_scales_stack_per_chunk = list(torch.split(fp8_weights_scales_stack, self.config.moe_offloading_chunk_size))
            fp8_weights_t_scales_stack_per_chunk = list(torch.split(fp8_weights_t_scales_stack, self.config.moe_offloading_chunk_size))
            self._uid_to_fp8_weights_map[uid] = \
                (fp8_weights, fp8_weights_scales_stack_per_chunk, fp8_weights_t, fp8_weights_t_scales_stack_per_chunk)
            self._uid_to_is_first_microbatch[uid] = False
        
        # when the weights have been accessed before, 
        # refresh the fp8 weights at the first microbatch 
        elif uid in self._uid_to_fp8_weights_map and self._uid_to_is_first_microbatch[uid]:
            for wid in [w.data_ptr() for w in bf16_weights]:
                self._wid_to_fp8_weight_map[wid] = self._quantize_weight(wid)
            self._uid_to_is_first_microbatch[uid] = False

        # return the fp8 weights and scales based on the transposed flag
        fp8_weights, fp8_weights_scales, fp8_weights_t, fp8_weights_t_scales = self._uid_to_fp8_weights_map[uid]
        if transposed:
            return fp8_weights_t, fp8_weights_t_scales
        else:
            return fp8_weights, fp8_weights_scales
    
    def refresh_fp8_weights(
        self,
        bf16_weight: torch.Tensor,
    ):
        wid = bf16_weight.data_ptr()

        if wid in self._wid_to_bf16_weight_map:
            # re-quantize the weight to fp8 and update the fp8 weight maps
            fp8_weight, fp8_weight_scale, fp8_weight_t, fp8_weight_t_scale = self._quantize_weight(wid)
            self._wid_to_fp8_weight_map[wid] = (fp8_weight, fp8_weight_scale, fp8_weight_t, fp8_weight_t_scale)
        
        # do nothing if the weight is not in the map yet, 
        # as it will be quantized and added to the map when it is first accessed

    def reset(self):
        self._wid_to_bf16_weight_map.clear()
        self._wid_to_fp8_weight_map.clear()
        self._uid_to_fp8_weights_map.clear()

    def set_is_first_microbatch(self):
        for uid in self._uid_to_is_first_microbatch:
            self._uid_to_is_first_microbatch[uid] = True
    
    def _quantize_weight(
        self, 
        wid: int,
    ):
        bf16_param = self._wid_to_bf16_weight_map[wid]
        bf16_param_slices = self._wid_to_sliced_bf16_weight_map[wid]

        # fetch tensor storage
        (
            _, 
            fp8_weight_scale_tensor, 
            _, 
            fp8_weight_t_scale_tensor
        ) = self._wid_to_fp8_weight_map.get(wid, (None, None, None, None))

        # block-wise quantization to fp8 weights
        # As the parameter tensor is on CPU, we do the following:
        # H2D -> per_block_cast_to_fp8 -> D2H
        param_numel = bf16_param.numel()

        # H2D
        device_buffer = self.expert_quantization_buffer[param_numel][0].view(bf16_param.shape).to(torch.cuda.current_device())
        device_buffer.copy_(
            bf16_param.data,
            non_blocking=False,
        )

        # per_block_cast_to_fp8
        # reuse quantization buffer and scale tensor
        # NOTE: it is IMPORTANT to reuse the scale tensors, because we replace the scale tensors
        # during the first access to the weights, and we use uid instead of wid to access the 
        # quantized weights. Scale tensors must be the same between wid and uid maps.
        fp8_weight, fp8_weight_scale = per_block_cast_to_fp8_gpu(
            device_buffer,
            gran_k=128,
            sf=fp8_weight_scale_tensor,
        )

        # NOTE: transposed version might not need separate quantization
        # fp8_weight_t, fp8_weight_t_scale = fp8_weight.T, fp8_weight_scale.T
        fp8_weight_t, fp8_weight_t_scale = per_block_cast_to_fp8_gpu(
            device_buffer.T.contiguous(),
            gran_k=128,
            sf=fp8_weight_t_scale_tensor,
        )

        # D2H
        with torch.no_grad():
            if not self.config.moe_use_extra_fp8_param_storage:
                # in-place update 
                fp8_weight = (
                    bf16_param_slices[0]
                    .copy_(fp8_weight.data, non_blocking=False)
                )
                fp8_weight_t = (
                    bf16_param_slices[1]
                    .copy_(fp8_weight_t.data, non_blocking=False)
                )
            else:
                extra_fp8_storage = self._wid_to_extra_fp8_weight_storage.get(wid, None)
                extra_fp8_storage_slices = self._wid_to_extra_fp8_weight_storage_slices.get(wid, None)
                if extra_fp8_storage is None:
                    # allocate new storage if not exist
                    extra_fp8_storage = torch.empty_like(bf16_param.data, device=bf16_param.data.device, pin_memory=bf16_param.data.is_pinned())
                    self._wid_to_extra_fp8_weight_storage[wid] = extra_fp8_storage
                    extra_fp8_storage_slices = list(torch.unbind(extra_fp8_storage.view(2, bf16_param.shape[0] // 2, *bf16_param.shape[1:]), dim=0))
                    self._wid_to_extra_fp8_weight_storage_slices[wid] = (
                        extra_fp8_storage_slices[0].view(torch.float8_e4m3fn).view(bf16_param.shape[0], bf16_param.shape[1]),
                        extra_fp8_storage_slices[1].view(torch.float8_e4m3fn).view(bf16_param.shape[1], bf16_param.shape[0])
                    )
                    
                extra_fp8_storage = self._wid_to_extra_fp8_weight_storage[wid]
                extra_fp8_storage_slices = self._wid_to_extra_fp8_weight_storage_slices[wid]
                
                # use extra storage
                fp8_weight = (
                    extra_fp8_storage_slices[0]
                    .copy_(fp8_weight.data, non_blocking=False)
                )
                fp8_weight_t = (
                    extra_fp8_storage_slices[1]
                    .copy_(fp8_weight_t.data, non_blocking=False)
                )
        
        
        # if fp8_weight_scale_tensor is not None:
        #     fp8_weight_scale = fp8_weight_scale_tensor.data.copy_(fp8_weight_scale, non_blocking=False)
        # if fp8_weight_t_scale_tensor is not None:
        #     fp8_weight_t_scale = fp8_weight_t_scale_tensor.data.copy_(fp8_weight_t_scale, non_blocking=False)
        
        # update the fp8 weight maps
        self._wid_to_fp8_weight_map[wid] = (fp8_weight, fp8_weight_scale, fp8_weight_t, fp8_weight_t_scale)
        return fp8_weight, fp8_weight_scale, fp8_weight_t, fp8_weight_t_scale

                

class OffloadingExpertsFP8GroupedSwiMLP(torch.autograd.Function):
    """Autograd function for Offloading Experts Grouped SwiGLU MLP with FP8 support. """
    
    @classmethod
    def call_forward_a(
        cls,
        cpu_w1: list[torch.Tensor],
        gpu_w1_buffers: list[list[torch.Tensor]],
        gpu_w1_chunks: list[torch.Tensor],
        permuted_local_hidden_states: tuple[torch.Tensor, torch.Tensor],
        hidden_state_per_chunk: tuple[list[torch.Tensor], list[torch.Tensor]],
        total_token_num_per_chunk: list[int],
        tokens_per_expert_chunks_psum: list[torch.Tensor],
        stream_manager: StreamManager,
        fp8_parameter_manager: FP8ExpertsParameterManager,
        config: TransformerConfig,
    ):
        # allocate output buffer for the first linear layer
        fc1_output = torch.empty(
            permuted_local_hidden_states[0].shape[0],
            config.moe_ffn_hidden_size * (2 if config.gated_linear_unit else 1),
            device=permuted_local_hidden_states[0].device,
            dtype=torch.bfloat16,
        )
        fc1_output_per_chunk = list(torch.split(fc1_output, total_token_num_per_chunk))

        # prefetch the first chunk of expert weights to GPU
        curr_buffer_metadata = cls.prefetch_expert_weights(0, cpu_w1, gpu_w1_buffers, stream_manager, fp8_parameter_manager, config)

        # fc1 chunk-level interleaving computation
        for chunk_idx in range(config.moe_offloading_num_chunks):
            # prefetch the next chunk of expert weights to GPU buffer
            if chunk_idx + 1 < config.moe_offloading_num_chunks:
                next_buffer_metadata = cls.prefetch_expert_weights(chunk_idx + 1, cpu_w1, gpu_w1_buffers, stream_manager, fp8_parameter_manager, config)
            
            # computation on the current GPU buffer
            fp8_experts_chunk = gpu_w1_chunks[curr_buffer_metadata[0]]
            fp8_experts_chunk_scales = curr_buffer_metadata[2]
            hidden_states_chunk = hidden_state_per_chunk[0][chunk_idx]
            hidden_states_chunk_scales = hidden_state_per_chunk[1][chunk_idx]
            fc1_output_chunk = fc1_output_per_chunk[chunk_idx]

            # wait for the current chunk of weights to be ready on GPU
            stream_manager.compute_streams_wait_launch_streams()
            stream_manager.consumer_streams_wait_event(curr_buffer_metadata[-1])
            m_grouped_fp8_gemm_nt_contiguous(
                tokens_per_expert_chunks_psum[chunk_idx],
                (hidden_states_chunk, hidden_states_chunk_scales),
                (fp8_experts_chunk, fp8_experts_chunk_scales),
                output=fc1_output_chunk,
                compute_stream=stream_manager.compute_streams[0],
            )
            stream_manager.h2d_stream_wait_consumer_streams(curr_buffer_metadata[1])
            stream_manager.launch_streams_wait_compute_streams()

            # update current buffer metadata
            curr_buffer_metadata = next_buffer_metadata if chunk_idx + 1 < config.moe_offloading_num_chunks else None

        return fc1_output

    @classmethod
    def call_forward_y(
        cls,
        cpu_w2: list[torch.nn.Parameter],
        gpu_w2_buffers: list[list[torch.Tensor]],
        gpu_w2_chunks: list[torch.Tensor],
        permuted_local_hidden_states: tuple[torch.Tensor, torch.Tensor],
        fc1_output: torch.Tensor,
        total_token_num_per_chunk: list[int],
        tokens_per_expert_chunks_psum: list[torch.Tensor],
        permuted_probs: torch.Tensor,
        stream_manager: StreamManager,
        fp8_parameter_manager: FP8ExpertsParameterManager,
        config: TransformerConfig,
    ):
        # prefetch the first chunk of expert weights to GPU
        curr_buffer_metadata = cls.prefetch_expert_weights(0, cpu_w2, gpu_w2_buffers, stream_manager, fp8_parameter_manager, config)

        s = swiglu_forward(
            fc1_output,
            permuted_probs.unsqueeze(-1)
        )

        # quantize the swiglu output to FP8
        # fp8_s = per_token_cast_to_fp8(s, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
        # fp8_s_per_chunk = (
        #     list(torch.split(fp8_s[0], total_token_num_per_chunk)),
        #     list(torch.split(fp8_s[1], total_token_num_per_chunk)),
        # )
        fp8_s, fp8_s_chunks, fp8_s_scales_chunks = per_token_cast_to_fp8_chunked_fused(
            s, total_token_num_per_chunk, gran_k=128,
        )
        fp8_s_per_chunk = (fp8_s_chunks, fp8_s_scales_chunks)

        # fc2 chunk-level interleaving computation
        fc2_output = torch.empty_like(permuted_local_hidden_states[0], dtype=torch.bfloat16)
        fc2_output_per_chunk = list(torch.split(fc2_output, total_token_num_per_chunk))

        for chunk_idx in range(config.moe_offloading_num_chunks):
            # prefetch the next chunk of expert weights to GPU buffer
            if chunk_idx + 1 < config.moe_offloading_num_chunks:
                next_buffer_metadata = cls.prefetch_expert_weights(chunk_idx + 1, cpu_w2, gpu_w2_buffers, stream_manager, fp8_parameter_manager, config)

            # computation on the current GPU buffer
            fp8_experts_chunk = gpu_w2_chunks[curr_buffer_metadata[0]]
            fp8_experts_chunk_scales = curr_buffer_metadata[2]
            s_chunk = fp8_s_per_chunk[0][chunk_idx]
            s_chunk_scales = fp8_s_per_chunk[1][chunk_idx]
            fc2_output_chunk = fc2_output_per_chunk[chunk_idx]

            stream_manager.compute_streams_wait_launch_streams()
            stream_manager.consumer_streams_wait_event(curr_buffer_metadata[-1])
            m_grouped_fp8_gemm_nt_contiguous(
                tokens_per_expert_chunks_psum[chunk_idx],
                (s_chunk, s_chunk_scales),
                (fp8_experts_chunk, fp8_experts_chunk_scales),
                output=fc2_output_chunk,
                compute_stream=stream_manager.compute_streams[0],
            )
            stream_manager.h2d_stream_wait_consumer_streams(curr_buffer_metadata[1])
            stream_manager.launch_streams_wait_compute_streams()

            # update current buffer metadata
            curr_buffer_metadata = next_buffer_metadata if chunk_idx + 1 < config.moe_offloading_num_chunks else None
        
        return fc2_output, fp8_s


    @classmethod
    def call_backward_grad_a(
        cls,
        grad_y: tuple,  # (fp8_full, fp8_chunks_list, sf_chunks_list)
        a: torch.Tensor,
        cpu_w2: list[torch.Tensor],
        gpu_w2_buffers: list[list[torch.Tensor]],
        gpu_w2_chunks: list[torch.Tensor],
        total_token_num_per_chunk: list[int],
        tokens_per_expert_chunks_psum: list[torch.Tensor],
        permuted_probs: torch.Tensor,
        stream_manager: StreamManager,
        fp8_parameter_manager: FP8ExpertsParameterManager,
        config: TransformerConfig,
    ):
        """
        ds [m, H] = grad_y [m, h] @ w2.T [H, h]

        da [m, 2*H] = backward_swiglu(da, a, permuted_probs)
        """
        fp8_grad_y_per_chunk = grad_y[1]
        fp8_grad_y_scales_per_chunk = grad_y[2]
        grad_s = torch.empty(
            grad_y[0].shape[0],
            config.moe_ffn_hidden_size,
            device=grad_y[0].device,
            dtype=torch.bfloat16,
        )
        grad_s_per_chunk = list(torch.split(grad_s, total_token_num_per_chunk))
        
        # prefetch the first chunk of expert weights to GPU
        curr_buffer_metadata = cls.prefetch_expert_weights(0, cpu_w2, gpu_w2_buffers, stream_manager, fp8_parameter_manager, config, True)

        for chunk_idx in range(config.moe_offloading_num_chunks):
            # prefetch the next chunk of expert weights to GPU buffer
            if chunk_idx + 1 < config.moe_offloading_num_chunks:
                next_buffer_metadata = cls.prefetch_expert_weights(chunk_idx + 1, cpu_w2, gpu_w2_buffers, stream_manager, fp8_parameter_manager, config, True)

            # computation on the current GPU buffer
            # fp8_experts_chunk = gpu_w2_buffers[curr_buffer_metadata[0]]
            fp8_experts_chunk = gpu_w2_chunks[curr_buffer_metadata[0]].view(
                config.moe_offloading_chunk_size,
                config.moe_ffn_hidden_size,
                config.hidden_size,
            )
            fp8_experts_chunk_scales = curr_buffer_metadata[2]
            fp8_grad_y_chunk = fp8_grad_y_per_chunk[chunk_idx]
            fp8_grad_y_chunk_scales = fp8_grad_y_scales_per_chunk[chunk_idx]

            stream_manager.compute_streams_wait_launch_streams()
            stream_manager.consumer_streams_wait_event(curr_buffer_metadata[-1])
            m_grouped_fp8_gemm_nt_contiguous(
                tokens_per_expert_chunks_psum[chunk_idx],
                (fp8_grad_y_chunk, fp8_grad_y_chunk_scales),
                (fp8_experts_chunk, fp8_experts_chunk_scales),
                output=grad_s_per_chunk[chunk_idx],
                compute_stream=stream_manager.compute_streams[0],
            )
            stream_manager.h2d_stream_wait_consumer_streams(curr_buffer_metadata[1])
            stream_manager.launch_streams_wait_compute_streams()

            # update current buffer metadata
            curr_buffer_metadata = next_buffer_metadata if chunk_idx + 1 < config.moe_offloading_num_chunks else None
        
        return swiglu_backward(grad_s, a, permuted_probs.unsqueeze(-1))

    @classmethod
    def call_backward_grad_x(
        cls,
        grad_a: tuple,  # (fp8_full, fp8_chunks_list, sf_chunks_list)
        cpu_w1: list[torch.Tensor],
        gpu_w1_buffers: list[list[torch.Tensor]],
        gpu_w1_chunks: list[torch.Tensor],
        total_token_num_per_chunk: list[int],
        tokens_per_expert_chunks_psum: list[torch.Tensor],
        stream_manager: StreamManager,
        fp8_parameter_manager: FP8ExpertsParameterManager,
        config: TransformerConfig,
    ):
        """
        dx [m, h] = a [m, 2*H] @ w1.T [h, 2*H]
        """
        fp8_grad_a_per_chunk = grad_a[1]
        fp8_grad_a_scales_per_chunk = grad_a[2]
        grad_x = torch.empty(
            grad_a[0].shape[0],
            config.hidden_size,
            device=grad_a[0].device, 
            dtype=torch.bfloat16
        )
        grad_x_per_chunk = list(torch.split(grad_x, total_token_num_per_chunk))

        # prefetch the first chunk of expert weights to GPU
        curr_buffer_metadata = cls.prefetch_expert_weights(0, cpu_w1, gpu_w1_buffers, stream_manager, fp8_parameter_manager, config, True)
        
        for chunk_idx in range(config.moe_offloading_num_chunks):            
            # prefetch the next chunk of expert weights to GPU buffer
            if chunk_idx + 1 < config.moe_offloading_num_chunks:
                next_buffer_metadata = cls.prefetch_expert_weights(chunk_idx + 1, cpu_w1, gpu_w1_buffers, stream_manager, fp8_parameter_manager, config, True)

            # computation on the current GPU buffer
            fp8_experts_chunk = gpu_w1_chunks[curr_buffer_metadata[0]].view(
                config.moe_offloading_chunk_size,
                config.hidden_size,
                config.moe_ffn_hidden_size * (2 if config.gated_linear_unit else 1),
            )
            fp8_experts_chunk_scales = curr_buffer_metadata[2]
            fp8_grad_a_chunk = fp8_grad_a_per_chunk[chunk_idx]
            fp8_grad_a_chunk_scales = fp8_grad_a_scales_per_chunk[chunk_idx]

            stream_manager.compute_streams_wait_launch_streams()
            stream_manager.consumer_streams_wait_event(curr_buffer_metadata[-1])
            m_grouped_fp8_gemm_nt_contiguous(
                tokens_per_expert_chunks_psum[chunk_idx],
                (fp8_grad_a_chunk, fp8_grad_a_chunk_scales),
                (fp8_experts_chunk, fp8_experts_chunk_scales),
                output=grad_x_per_chunk[chunk_idx],
                compute_stream=stream_manager.compute_streams[0],
            )
            stream_manager.h2d_stream_wait_consumer_streams(curr_buffer_metadata[1])
            stream_manager.launch_streams_wait_compute_streams()

            # update current buffer metadata
            curr_buffer_metadata = next_buffer_metadata if chunk_idx + 1 < config.moe_offloading_num_chunks else None

        return grad_x

    @staticmethod
    def _wgrad_post_process(
        w: list[torch.nn.Parameter],
        wgrad_output: list[torch.Tensor],
        fuse_gradient_accumulation: bool,
    ):
        # handle ddp
        assert fuse_gradient_accumulation, \
            "Only support fuse_gradient_accumulation for offloading experts."
        for i in range(len(w)):
            if fuse_gradient_accumulation:
                w[i].grad_added_to_main_grad = True
                w[i].grad = get_dummy_wgrad(w[i].shape, w[i].dtype, w[i].device)

    @classmethod
    def call_backward_grad_w2(
        cls,
        grad_y: tuple[torch.Tensor, torch.Tensor],
        a: torch.Tensor,
        cpu_w2: torch.nn.Parameter,
        tokens_per_expert_list: list[int],
        tokens_per_expert_cuda: torch.Tensor,
        tokens_per_expert_cumsum: torch.Tensor,
        permuted_probs: torch.Tensor,
        stream_manager: StreamManager,
        num_local_experts: int,
        config: TransformerConfig,
        wgrad_scheduler: ExpertsWgradScheduler = None,
        delay_wgrad_compute: bool = False,
        fuse_gradient_accumulation: bool = False,
    ):
        """
        dw2 [h, H] = grad_y.T [h, m] @ s.T [H, m]
        with k_grouped_fp8_gemm: grad_y [m, h] @ s [m, H]
        """
        s = swiglu_forward(a, permuted_probs.unsqueeze(-1))
        fp8_s = guarded_per_channel_cast_to_fp8_pack_kmajor(
            s, tokens_per_expert_list, tokens_per_expert_cuda, tokens_per_expert_cumsum,
            name="w2_s", config=config, gran_k=128, free_input=True,
        )

        assert cpu_w2.main_grad is not None
        wgrad_output = cpu_w2.main_grad

        # If delay_wgrad_compute is True and wgrad_scheduler is provided, 
        # register the wgrad computation to be executed later.
        if delay_wgrad_compute and wgrad_scheduler is not None:
            wgrad_scheduler.register(
                k_grouped_fp8_gemm_nt_contiguous,
                tokens_per_expert_list, tokens_per_expert_cuda, grad_y, fp8_s, num_local_experts, torch.cuda.current_stream(), wgrad_output
            )
            cls._wgrad_post_process([cpu_w2], wgrad_output, fuse_gradient_accumulation)
            return
        
        stream_manager.compute_streams_wait_launch_streams()
        k_grouped_fp8_gemm_nt_contiguous(
            tokens_per_expert_list,
            tokens_per_expert_cuda,
            grad_y,
            fp8_s,
            num_local_experts,
            stream_manager.compute_streams[0],
            output=wgrad_output,
        )
        stream_manager.launch_streams_wait_compute_streams()
        cls._wgrad_post_process([cpu_w2], wgrad_output, fuse_gradient_accumulation)


    @classmethod
    def call_backward_grad_w1(
        cls,
        grad_a: tuple[torch.Tensor, torch.Tensor],
        x: tuple[torch.Tensor, torch.Tensor],
        cpu_w1: list[torch.nn.Parameter],
        tokens_per_expert_list: list[int],
        tokens_per_expert_cuda: torch.Tensor,
        tokens_per_expert_cumsum: torch.Tensor,
        num_local_experts: int,
        stream_manager: StreamManager,
        wgrad_scheduler: ExpertsWgradScheduler = None,
        delay_wgrad_compute: bool = False,
        fuse_gradient_accumulation: bool = False,
    ):
        """
        dw1 [2*H, h] = grad_a.T [2*H, m] @ x.T [h, m]
        with k_grouped_fp8_gemm: grad_a [m, 2*H] @ x [m, h]
        """
        assert cpu_w1.main_grad is not None
        wgrad_output = cpu_w1.main_grad

        # If delay_wgrad_compute is True and wgrad_scheduler is provided, 
        # register the wgrad computation to be executed later.
        if delay_wgrad_compute and wgrad_scheduler is not None:
            wgrad_scheduler.register(
                k_grouped_fp8_gemm_nt_contiguous,
                tokens_per_expert_list, tokens_per_expert_cuda, grad_a, x, num_local_experts, torch.cuda.current_stream(), wgrad_output
            )
            cls._wgrad_post_process([cpu_w1], wgrad_output, fuse_gradient_accumulation)
            return
        
        stream_manager.compute_streams_wait_launch_streams()
        k_grouped_fp8_gemm_nt_contiguous(
            tokens_per_expert_list,
            tokens_per_expert_cuda,
            grad_a,
            x,
            num_local_experts,
            stream_manager.compute_streams[0],
            output=wgrad_output,
        )
        stream_manager.launch_streams_wait_compute_streams()
        cls._wgrad_post_process([cpu_w1], wgrad_output, fuse_gradient_accumulation)

    @classmethod
    def prefetch_expert_weights(
        cls,
        chunk_idx: int,
        cpu_weights: list[torch.nn.Parameter] | list[torch.Tensor],
        gpu_buffers: list[list[torch.Tensor]],
        stream_manager: StreamManager,
        fp8_parameter_manager: FP8ExpertsParameterManager,
        config: TransformerConfig,
        transposed: bool = False,
    ) -> tuple:
        h2d_stream_idx = chunk_idx % config.moe_offloading_num_stages
        gpu_buffer_idx = h2d_stream_idx
        h2d_stream = stream_manager.get_h2d_stream(h2d_stream_idx)
        fp8_cpu_weights, fp8_weight_scales = \
            fp8_parameter_manager.get_fp8_weights(cpu_weights, transposed)
        
        with torch.cuda.stream(h2d_stream):
            experts_idx_start = chunk_idx * config.moe_offloading_chunk_size
            experts_idx_end = (chunk_idx + 1) * config.moe_offloading_chunk_size
            fp8_cpu_weights_slice = fp8_cpu_weights[experts_idx_start:experts_idx_end]
            buf = gpu_buffers[gpu_buffer_idx]

            # NOTE: the view logic for transposed buffer might be unnecessary
            # if transposed:
            #     buf = [b.view(b.shape[1], b.shape[0]) for b in buf]

            assert len(fp8_cpu_weights_slice) == len(buf), \
                f"Number of weights in CPU slice {len(fp8_cpu_weights_slice)} does not match number of GPU buffers {len(buf)}"
            # NOTE: batched H2D copy is used to reduce cpu overhead
            if not config.moe_offloading_experts_debug_mode:
                grouped_gemm.grouped_gemm.backend.batched_h2d_async(
                    fp8_cpu_weights_slice,
                    buf,
                    h2d_stream.cuda_stream
                )
            else: # NOTE: fallback to non-batched copy if not pinned
                if transposed:
                    buf = [b.view(b.shape[1], b.shape[0]) for b in buf]
                for idx in range(experts_idx_start, experts_idx_end):
                    buf[idx - experts_idx_start].copy_(fp8_cpu_weights[idx].data, non_blocking=True)

        copy_done_event = torch.cuda.Event()
        copy_done_event.record(h2d_stream)

        return (gpu_buffer_idx, h2d_stream_idx, fp8_weight_scales[chunk_idx], copy_done_event)
    
    @staticmethod
    def forward(
        ctx,
        *args, 
        **kwargs
    ):
        if len(args) < 9:
            raise ValueError(f"Insufficient arguments for forward pass of GroupedSwiMLP. Expected at least 9, got {len(args)}")
        
        cpu_w1: torch.nn.Parameter =  args[-16]
        cpu_w2: torch.nn.Parameter =  args[-15]
        cpu_w1_list: list[torch.Tensor] = args[-14]
        cpu_w2_list: list[torch.Tensor] = args[-13]
        gpu_w1_buffers: list[torch.Tensor] = args[-12]
        gpu_w2_buffers: list[torch.Tensor] = args[-11]
        gpu_w1_chunks: list[torch.Tensor] = args[-10]
        gpu_w2_chunks: list[torch.Tensor] = args[-9]
        permuted_local_hidden_states: torch.Tensor = args[-8]
        tokens_per_expert: torch.Tensor = args[-7]
        num_local_experts: int = args[-6]
        permuted_probs: torch.Tensor = args[-5]
        expert_wgrad_scheduler: ExpertsWgradScheduler = args[-4]
        stream_manager: StreamManager = args[-3]
        config: TransformerConfig = args[-2]
        wgrad_accumulation_and_reduce_hooks: list = args[-1]
        fp8_parameter_manager: FP8ExpertsParameterManager = FP8ExpertsParameterManager.get_instance()
        fp8_parameter_manager.config = config

        assert cpu_w1.shape[0] == num_local_experts, f"Expected cpu_w1 to have {num_local_experts} experts, but got {cpu_w1.shape[0]}"
        assert cpu_w2.shape[0] == num_local_experts, f"Expected cpu_w2 to have {num_local_experts} experts, but got {cpu_w2.shape[0]}"
    
        # split hidden states, outputs and token_per_experts into chunks for each expert chunks
        # NOTE: this part of logic is CPU-bound and presents ovehead. 
        # tokens_per_expert_chunks = torch.split(tokens_per_expert, config.moe_offloading_chunk_size)
        # tokens_per_expert_chunks_psum = [torch.cumsum(t, dim=0).to(torch.int32).to(permuted_local_hidden_states.device) for t in tokens_per_expert_chunks]
        # total_token_num_per_chunk = [chunk.sum().item() for chunk in tokens_per_expert_chunks]
        tokens_per_expert_chunks_psum, total_token_num_per_chunk = \
            grouped_gemm.grouped_gemm.backend.tokens_per_expert_chunk_sum(tokens_per_expert, config.moe_offloading_chunk_size, permuted_local_hidden_states.device)

        # per-chunk FP8 quantize: produces sf buffers in MN-major
        # TMA-aligned layout so DeepGEMM's transform_sf_into_required_layout
        # early-returns without launching transpose kernel for sf
        fp8_x_full, fp8_x_chunks, fp8_x_sf_chunks = per_token_cast_to_fp8_chunked_fused(
            permuted_local_hidden_states, total_token_num_per_chunk, gran_k=128,
        )

        # build a full sf tensor view for downstream sites that expect a single
        # (M, num_groups) tensor (e.g. saved_for_backward, dequant). The SF
        # storage is non-contiguous across chunks, so this concatenation copies;
        # we therefore only materialize it lazily when needed.
        fp8_permuted_local_hidden_states = (fp8_x_full, fp8_x_sf_chunks)
        fp8_hidden_state_per_chunk = (fp8_x_chunks, fp8_x_sf_chunks)

        # forward for the first linear layer
        fc1_output = OffloadingExpertsFP8GroupedSwiMLP.call_forward_a(
            cpu_w1_list,
            gpu_w1_buffers,
            gpu_w1_chunks,
            fp8_permuted_local_hidden_states,
            fp8_hidden_state_per_chunk,
            total_token_num_per_chunk,
            tokens_per_expert_chunks_psum,
            stream_manager,
            fp8_parameter_manager,
            config,
        )

        # activation and forward for the second linear layer
        y, _ = OffloadingExpertsFP8GroupedSwiMLP.call_forward_y(
            cpu_w2_list,
            gpu_w2_buffers,
            gpu_w2_chunks,
            fp8_permuted_local_hidden_states,
            fc1_output,
            total_token_num_per_chunk,
            tokens_per_expert_chunks_psum,
            permuted_probs,
            stream_manager,
            fp8_parameter_manager,
            config,
        )

        # context saving
        ctx.fp8_parameter_manager = fp8_parameter_manager
        ctx.fp8_hidden_state_per_chunk = fp8_hidden_state_per_chunk
        ctx.wgrad_accumulation_and_reduce_hooks = wgrad_accumulation_and_reduce_hooks
        ctx.num_local_experts = num_local_experts
        ctx.tokens_per_expert = tokens_per_expert
        ctx.tokens_per_expert_chunks_psum = tokens_per_expert_chunks_psum
        ctx.total_token_num_per_chunk = total_token_num_per_chunk
        ctx.expert_wgrad_scheduler = expert_wgrad_scheduler
        ctx.cpu_w1 = cpu_w1
        ctx.cpu_w2 = cpu_w2
        ctx.cpu_w1_list = cpu_w1_list
        ctx.cpu_w2_list = cpu_w2_list
        ctx.gpu_w1_buffers = gpu_w1_buffers
        ctx.gpu_w2_buffers = gpu_w2_buffers
        ctx.gpu_w1_chunks = gpu_w1_chunks
        ctx.gpu_w2_chunks = gpu_w2_chunks
        ctx.stream_manager = stream_manager
        ctx.config = config

        activation_recompute = (
            config.recompute_granularity == 'selective'
            and "moe_act" in config.recompute_modules
        )
        ctx.activation_recompute = activation_recompute

        # NOTE: fp8 sf chunks live on ctx (list of tensors); save_for_backward
        # only stores tensors. The full sf can be reconstructed by concatenation
        # if needed, but the chunks are already in the layout we want.
        ctx.fp8_permuted_local_hidden_states_sf_chunks = fp8_x_sf_chunks
        if activation_recompute:
            release(fc1_output)
            ctx.save_for_backward(
                fp8_permuted_local_hidden_states[0],
                None, None,
                permuted_probs
            )
        else:
            fp8_fc1_output = per_token_cast_to_fp8(fc1_output, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
            ctx.save_for_backward(
                fp8_permuted_local_hidden_states[0],
                fp8_fc1_output[0],
                fp8_fc1_output[1],
                permuted_probs
            )

        return y, None



    @staticmethod
    def backward(
        ctx, 
        *grad_outputs
    ):
        config: TransformerConfig = ctx.config
        cpu_w1: torch.nn.Parameter = ctx.cpu_w1
        cpu_w2: torch.nn.Parameter = ctx.cpu_w2
        cpu_w1_list: list[torch.Tensor] = ctx.cpu_w1_list
        cpu_w2_list: list[torch.Tensor] = ctx.cpu_w2_list
        gpu_w1_buffers: list[list[torch.Tensor]] = ctx.gpu_w1_buffers
        gpu_w2_buffers: list[list[torch.Tensor]] = ctx.gpu_w2_buffers
        gpu_w1_chunks: list[torch.Tensor] = ctx.gpu_w1_chunks
        gpu_w2_chunks: list[torch.Tensor] = ctx.gpu_w2_chunks
        stream_manager: StreamManager = ctx.stream_manager
        expert_wgrad_scheduler: ExpertsWgradScheduler = ctx.expert_wgrad_scheduler
        total_token_num_per_chunk: list[int] = ctx.total_token_num_per_chunk
        tokens_per_expert_chunks_psum: list[torch.Tensor] = ctx.tokens_per_expert_chunks_psum
        tokens_per_expert: torch.Tensor = ctx.tokens_per_expert
        fp8_parameter_manager: FP8ExpertsParameterManager = ctx.fp8_parameter_manager
        fp8_hidden_state_per_chunk: tuple[list[torch.Tensor], list[torch.Tensor]] = ctx.fp8_hidden_state_per_chunk
        fp8_permuted_local_hidden_states_sf_chunks: list = ctx.fp8_permuted_local_hidden_states_sf_chunks
        (
            fp8_permuted_local_hidden_states,
            fp8_fc1_output, fp8_fc1_output_scales,
            permuted_probs
        ) = ctx.saved_tensors

        if ctx.activation_recompute:
            # recompute the activation for backward
            fc1_output = OffloadingExpertsFP8GroupedSwiMLP.call_forward_a(
                cpu_w1_list,
                gpu_w1_buffers,
                gpu_w1_chunks,
                (fp8_permuted_local_hidden_states, fp8_permuted_local_hidden_states_sf_chunks),
                fp8_hidden_state_per_chunk,
                total_token_num_per_chunk,
                tokens_per_expert_chunks_psum,
                stream_manager,
                fp8_parameter_manager,
                config,
            )
        else:
            fc1_output = per_token_dequant_from_fp8(fp8_fc1_output, fp8_fc1_output_scales)

        grad_y = grad_outputs[0].contiguous()

        # prepare tokens_per_expert for packing the grad_y and grad_a
        tokens_per_expert_list = tokens_per_expert.tolist()
        tokens_per_expert_cuda = tokens_per_expert.to(torch.int32).to(grad_y.device, non_blocking=True)
        tokens_per_expert_cumsum = torch.tensor(
            [0, *itertools.accumulate(tokens_per_expert_list[:-1])],
            dtype=torch.int32
        ).to(grad_y.device, non_blocking=True)

        # dequantize the input from FP8 to BF16 (chunked SF layout)
        bf16_x = per_token_dequant_from_fp8_chunked(
            fp8_permuted_local_hidden_states,
            fp8_permuted_local_hidden_states_sf_chunks,
            total_token_num_per_chunk,
            gran_k=128,
        )

        # backward computation (per-chunk quantize -> avoids transpose in DeepGEMM)
        fp8_grad_y_full, fp8_grad_y_chunks, fp8_grad_y_sf_chunks = per_token_cast_to_fp8_chunked_fused(
            grad_y, total_token_num_per_chunk, gran_k=128,
        )
        grad_a, grad_probs = OffloadingExpertsFP8GroupedSwiMLP.call_backward_grad_a(
            (fp8_grad_y_full, fp8_grad_y_chunks, fp8_grad_y_sf_chunks),
            fc1_output,
            cpu_w2_list,
            gpu_w2_buffers,
            gpu_w2_chunks,
            total_token_num_per_chunk,
            tokens_per_expert_chunks_psum,
            permuted_probs,
            stream_manager,
            fp8_parameter_manager,
            config,
        )

        # backward grad_x computation
        fp8_grad_a_full, fp8_grad_a_chunks, fp8_grad_a_sf_chunks = per_token_cast_to_fp8_chunked_fused(
            grad_a, total_token_num_per_chunk, gran_k=128,
        )
        grad_x = OffloadingExpertsFP8GroupedSwiMLP.call_backward_grad_x(
            (fp8_grad_a_full, fp8_grad_a_chunks, fp8_grad_a_sf_chunks),
            cpu_w1_list,
            gpu_w1_buffers,
            gpu_w1_chunks,
            total_token_num_per_chunk,
            tokens_per_expert_chunks_psum,
            stream_manager,
            fp8_parameter_manager,
            config,
        )

        # backward grad_w2 computation
        fp8_grad_y_t = guarded_per_channel_cast_to_fp8_pack_kmajor(
            grad_y, tokens_per_expert_list, tokens_per_expert_cuda, tokens_per_expert_cumsum,
            name="w2_grad_y", config=config, gran_k=128, free_input=True,
        )
        OffloadingExpertsFP8GroupedSwiMLP.call_backward_grad_w2(
            fp8_grad_y_t,
            fc1_output,
            cpu_w2,
            tokens_per_expert_list,
            tokens_per_expert_cuda,
            tokens_per_expert_cumsum,
            permuted_probs,
            stream_manager,
            ctx.num_local_experts,
            config,
            expert_wgrad_scheduler,
            config.delay_wgrad_compute,
            config.gradient_accumulation_fusion,
        )

        # backward grad_w1 computation
        fp8_grad_a_t = per_channel_cast_to_fp8_pack_kmajor(
            grad_a, tokens_per_expert_list, tokens_per_expert_cuda, tokens_per_expert_cumsum,
            use_ue8m0=False, gran_k=128, free_input=True,
        )
        fp8_x_t = per_channel_cast_to_fp8_pack_kmajor(
            bf16_x, tokens_per_expert_list, tokens_per_expert_cuda, tokens_per_expert_cumsum,
            use_ue8m0=False, gran_k=128, free_input=True,
        )
        OffloadingExpertsFP8GroupedSwiMLP.call_backward_grad_w1(
            fp8_grad_a_t,
            fp8_x_t,
            cpu_w1,
            tokens_per_expert_list,
            tokens_per_expert_cuda,
            tokens_per_expert_cumsum,
            ctx.num_local_experts,
            stream_manager,
            expert_wgrad_scheduler,
            config.delay_wgrad_compute,
            config.gradient_accumulation_fusion,
        )

        # NOTE: gradients have been attached in _wgrad_post_process, 
        # so we can return None for grad_w1 and grad_w2
        grad_w1_ret = None
        grad_w2_ret = None

        # NOTE: manually trigger wgrad accumulation hook
        # this is needed as the hook may fail to be triggered if 
        # the parameter is on CPU, and hence cause hanging when
        # overlap_grad_reduce is enabled
        if not config.delay_wgrad_compute:
            for hook_fn in ctx.wgrad_accumulation_and_reduce_hooks:
                hook_fn()

        return grad_w1_ret, grad_w2_ret, None, None, None, None, None, None, grad_x, None, None, grad_probs, None, None, None, None





def offloading_fp8_grouped_swiglu_mlp(
    cpu_w1: torch.nn.Parameter,
    cpu_w2: torch.nn.Parameter,
    cpu_w1_list: list[torch.Tensor],
    cpu_w2_list: list[torch.Tensor],
    gpu_w1_buffers: list[list[torch.Tensor]],
    gpu_w2_buffers: list[list[torch.Tensor]],
    gpu_w1_chunks: list[torch.Tensor],
    gpu_w2_chunks: list[torch.Tensor],
    permuted_local_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_local_experts: int,
    permuted_probs: torch.Tensor,
    expert_wgrad_scheduler: ExpertsWgradScheduler,
    stream_manager: StreamManager,
    config: TransformerConfig,
    wgrad_accumulation_and_reduce_hooks: list,
) -> torch.Tensor:
    """Autograd function for Offloading Experts Grouped SwiGLU MLP.

    Args:
        cpu_w1 (list[torch.nn.Parameter]): CPU weight parameters for the first linear layer
        cpu_w2 (list[torch.nn.Parameter]): CPU weight parameters for the second linear layer
        gpu_w1_buffers (list[torch.Tensor]): GPU buffers for w1 weights
        gpu_w2_buffers (list[torch.Tensor]): GPU buffers for w2 weights
        permuted_local_hidden_states (torch.Tensor): input hidden states
        tokens_per_expert (torch.Tensor): number of tokens assigned to each expert
        num_local_experts (int): number of local experts
        permuted_probs (torch.Tensor): probability derived from router
        expert_wgrad_scheduler (ExpertsWgradScheduler): scheduler for expert weight gradients
        stream_manager (StreamManager): manager for CUDA streams
        config (TransformerConfig): transformer configuration

    Returns:
        torch.Tensor: output of the MLP
    """
    output, _ = OffloadingExpertsFP8GroupedSwiMLP.apply(
        cpu_w1,
        cpu_w2,
        cpu_w1_list,
        cpu_w2_list,
        gpu_w1_buffers,
        gpu_w2_buffers,
        gpu_w1_chunks,
        gpu_w2_chunks,
        permuted_local_hidden_states,
        tokens_per_expert,
        num_local_experts,
        permuted_probs,
        expert_wgrad_scheduler,
        stream_manager,
        config,
        wgrad_accumulation_and_reduce_hooks
    )

    return output
