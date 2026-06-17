from __future__ import annotations
import torch
import itertools

from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.core.transformer.moe.experts_util import ExpertsWgradScheduler
from megatron.core.transformer.moe.fp8_utils import (
    m_grouped_fp8_gemm_nt_contiguous,
    k_grouped_fp8_gemm_nt_contiguous,
    diff_tensor_norm,
)
from megatron.core.transformer.moe.fp8_jit import (
    per_token_dequant_from_fp8,
    per_block_cast_to_fp8_gpu,
    per_token_cast_to_fp8,
    per_channel_cast_to_fp8,
    pack_to_kmajor,
)
from megatron.core.transformer.moe.swiglu_jit import (
    swiglu_forward,
    swiglu_backward,
)

class FP8GPUExpertsParameterManager:
    """Caches FP8-quantized expert weights for GPU-resident bf16 parameters.

    Mirrors the role of FP8ExpertsParameterManager in the offloading path,
    but skips H2D staging — bf16 params already live on GPU, so quantization
    is a single per_block_cast_to_fp8_gpu call.
    """

    _instance = None

    @classmethod
    def create_instance(cls, config: TransformerConfig):
        if cls._instance is None:
            cls._instance = cls(config=config)

    @classmethod
    def get_instance(cls, config: TransformerConfig = None) -> "FP8GPUExpertsParameterManager":
        assert cls._instance is not None, "FP8GPUExpertsParameterManager instance is not created yet."
        return cls._instance

    @classmethod
    def reset_instance(cls):
        if cls._instance is not None:
            cls._instance.reset()

    @classmethod
    def mark_first_microbatch(cls):
        if cls._instance is not None:
            cls._instance.set_is_first_microbatch()

    def __init__(self, config: TransformerConfig = None):
        # uid -> (fp8_w, fp8_w_scale, fp8_w_t, fp8_w_t_scale) where each is a stacked (E, ...) tensor
        self._uid_to_fp8_weights_map: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # uid -> bf16 source tensor (the stacked Parameter), used on refresh
        self._uid_to_bf16_param: dict[int, torch.Tensor] = {}
        # whether the next access should re-quantize
        self._uid_to_is_first_microbatch: dict[int, bool] = {}
        self.config = config

    def reset(self):
        self._uid_to_fp8_weights_map.clear()
        self._uid_to_bf16_param.clear()
        self._uid_to_is_first_microbatch.clear()

    def set_is_first_microbatch(self):
        for uid in self._uid_to_is_first_microbatch:
            self._uid_to_is_first_microbatch[uid] = True

    def get_fp8_weights(
        self,
        bf16_weight: torch.Tensor,
        transposed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (fp8_weight_stack, fp8_weight_scale_stack) for a stacked
        expert weight tensor of shape (E, M, K).

        If transposed is True, return the per-expert transposed view + matching scale.
        """
        uid = bf16_weight.data_ptr()

        if uid not in self._uid_to_fp8_weights_map:
            self._uid_to_bf16_param[uid] = bf16_weight
            self._uid_to_fp8_weights_map[uid] = self._quantize(bf16_weight)
            self._uid_to_is_first_microbatch[uid] = False
        elif self._uid_to_is_first_microbatch[uid]:
            self._uid_to_fp8_weights_map[uid] = self._quantize(bf16_weight)
            self._uid_to_is_first_microbatch[uid] = False

        fp8_w, fp8_w_scale, fp8_w_t, fp8_w_t_scale = self._uid_to_fp8_weights_map[uid]
        if transposed:
            return fp8_w_t, fp8_w_t_scale
        return fp8_w, fp8_w_scale

    def _quantize(self, bf16_weight: torch.Tensor):
        """Per-expert block-cast to FP8. bf16_weight has shape (E, M, K)."""
        assert bf16_weight.dim() == 3, f"Expected stacked expert weight of shape (E, M, K), got {bf16_weight.shape}"
        E = bf16_weight.shape[0]

        fp8_per_expert = []
        scale_per_expert = []
        for e in range(E):
            fp8_w_e, fp8_w_scale_e = per_block_cast_to_fp8_gpu(
                bf16_weight[e].contiguous(),
                gran_k=128,
            )
            fp8_per_expert.append(fp8_w_e)
            scale_per_expert.append(fp8_w_scale_e)

        fp8_w = torch.stack(fp8_per_expert, dim=0).contiguous()
        fp8_w_scale = torch.stack(scale_per_expert, dim=0).contiguous()

        # transposed view per expert, then stacked
        fp8_w_t = torch.stack([fp8_per_expert[e].t().contiguous() for e in range(E)], dim=0)
        # block-cast scales are symmetric for a square block layout, so reuse;
        # if your kernel needs a transposed scale layout, swap to a recomputed one here.
        fp8_w_t_scale = torch.stack([scale_per_expert[e].t().contiguous() for e in range(E)], dim=0)

        return fp8_w, fp8_w_scale, fp8_w_t, fp8_w_t_scale


class ExpertsFP8GroupedSwiMLP(torch.autograd.Function):
    """Autograd function for Experts Grouped SwiGLU MLP with FP8 support.

    Non-offloading variant: weights live on GPU, no chunking, no H2D prefetch,
    single grouped-GEMM call per linear. Reuses the same FP8 kernels and
    wgrad path as the offloading variant.
    """

    @classmethod
    def call_forward_a(
        cls,
        w1: torch.nn.Parameter,
        fp8_permuted_local_hidden_states: tuple[torch.Tensor, torch.Tensor],
        tokens_per_expert_psum: torch.Tensor,
        fp8_parameter_manager: FP8GPUExpertsParameterManager,
        config: TransformerConfig,
    ) -> torch.Tensor:
        fp8_w1, fp8_w1_scale = fp8_parameter_manager.get_fp8_weights(w1, transposed=False)

        fc1_output = torch.empty(
            fp8_permuted_local_hidden_states[0].shape[0],
            config.moe_ffn_hidden_size * (2 if config.gated_linear_unit else 1),
            device=fp8_permuted_local_hidden_states[0].device,
            dtype=torch.bfloat16,
        )
        m_grouped_fp8_gemm_nt_contiguous(
            tokens_per_expert_psum,
            fp8_permuted_local_hidden_states,
            (fp8_w1, fp8_w1_scale),
            compute_stream=torch.cuda.current_stream(),
            output=fc1_output,
        )
        return fc1_output

    @classmethod
    def call_forward_y(
        cls,
        w2: torch.nn.Parameter,
        fc1_output: torch.Tensor,
        permuted_probs: torch.Tensor,
        tokens_per_expert_psum: torch.Tensor,
        fp8_parameter_manager: FP8GPUExpertsParameterManager,
        config: TransformerConfig,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        fp8_w2, fp8_w2_scale = fp8_parameter_manager.get_fp8_weights(w2, transposed=False)

        s = swiglu_forward(fc1_output, permuted_probs.unsqueeze(-1))
        # s = MergedSwiGLU.call_forward(fc1_output, permuted_probs.unsqueeze(-1))
        fp8_s = per_token_cast_to_fp8(s, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)

        fc2_output = torch.empty(
            fp8_s[0].shape[0],
            config.hidden_size,
            device=fp8_s[0].device,
            dtype=torch.bfloat16,
        )
        m_grouped_fp8_gemm_nt_contiguous(
            tokens_per_expert_psum,
            fp8_s,
            (fp8_w2, fp8_w2_scale),
            compute_stream=torch.cuda.current_stream(),
            output=fc2_output,
        )
        return fc2_output, fp8_s

    @classmethod
    def call_backward_grad_a(
        cls,
        fp8_grad_y: tuple[torch.Tensor, torch.Tensor],
        a: torch.Tensor,
        w2: torch.nn.Parameter,
        permuted_probs: torch.Tensor,
        tokens_per_expert_psum: torch.Tensor,
        fp8_parameter_manager: FP8GPUExpertsParameterManager,
        config: TransformerConfig,
    ):
        """ds [m, H] = grad_y [m, h] @ w2.T [H, h];  da = swiglu_bwd(ds, a, probs)."""
        fp8_w2_t, fp8_w2_t_scale = fp8_parameter_manager.get_fp8_weights(w2, transposed=True)

        grad_s = torch.empty(
            fp8_grad_y[0].shape[0],
            config.moe_ffn_hidden_size,
            device=fp8_grad_y[0].device,
            dtype=torch.bfloat16,
        )
        m_grouped_fp8_gemm_nt_contiguous(
            tokens_per_expert_psum,
            fp8_grad_y,
            (fp8_w2_t, fp8_w2_t_scale),
            compute_stream=torch.cuda.current_stream(),
            output=grad_s,
        )
        # return MergedSwiGLU.call_backward(grad_s, a, permuted_probs.unsqueeze(-1))
        return swiglu_backward(grad_s, a, permuted_probs.unsqueeze(-1))

    @classmethod
    def call_backward_grad_x(
        cls,
        fp8_grad_a: tuple[torch.Tensor, torch.Tensor],
        w1: torch.nn.Parameter,
        tokens_per_expert_psum: torch.Tensor,
        fp8_parameter_manager: FP8GPUExpertsParameterManager,
        config: TransformerConfig,
    ) -> torch.Tensor:
        """dx [m, h] = grad_a [m, 2H] @ w1.T [2H, h]."""
        fp8_w1_t, fp8_w1_t_scale = fp8_parameter_manager.get_fp8_weights(w1, transposed=True)

        grad_x = torch.empty(
            fp8_grad_a[0].shape[0],
            config.hidden_size,
            device=fp8_grad_a[0].device,
            dtype=torch.bfloat16,
        )
        m_grouped_fp8_gemm_nt_contiguous(
            tokens_per_expert_psum,
            fp8_grad_a,
            (fp8_w1_t, fp8_w1_t_scale),
            compute_stream=torch.cuda.current_stream(),
            output=grad_x,
        )
        return grad_x

    @staticmethod
    def _wgrad_post_process(w: torch.nn.Parameter, fuse_gradient_accumulation: bool):
        if fuse_gradient_accumulation:
            w.grad_added_to_main_grad = True
            w.grad = torch.empty_like(w)

    @classmethod
    def call_backward_grad_w2(
        cls,
        fp8_grad_y_t: tuple[torch.Tensor, torch.Tensor],
        a: torch.Tensor,
        w2: torch.nn.Parameter,
        permuted_probs: torch.Tensor,
        tokens_per_expert_list: list[int],
        tokens_per_expert_cuda: torch.Tensor,
        tokens_per_expert_cumsum: torch.Tensor,
        num_local_experts: int,
        fuse_gradient_accumulation: bool = False,
    ):
        """dw2 [h, H] = grad_y.T [h, m] @ s.T [H, m]
        With k-grouped fp8: grad_y [m, h] @ s [m, H]."""
        assert fuse_gradient_accumulation, \
            "ExpertsFP8GroupedSwiMLP currently only supports fuse_gradient_accumulation."

        s = swiglu_forward(a, permuted_probs.unsqueeze(-1))
        # s = MergedSwiGLU.call_forward(a, permuted_probs.unsqueeze(-1))
        fp8_s = per_channel_cast_to_fp8(s, use_ue8m0=False, gran_k=128, transpose=False)

        wgrad_output = w2.main_grad

        grad_y_packed = (
            pack_to_kmajor(fp8_grad_y_t[0], tokens_per_expert_list, tokens_per_expert_cuda, tokens_per_expert_cumsum),
            fp8_grad_y_t[1].T,
        )
        s_packed = (
            pack_to_kmajor(fp8_s[0], tokens_per_expert_list, tokens_per_expert_cuda, tokens_per_expert_cumsum),
            fp8_s[1].T,
        )

        k_grouped_fp8_gemm_nt_contiguous(
            tokens_per_expert_list,
            tokens_per_expert_cuda,
            grad_y_packed,
            s_packed,
            num_local_experts,
            compute_stream=torch.cuda.current_stream(),
            output=wgrad_output,
        )
        cls._wgrad_post_process(w2, fuse_gradient_accumulation)

    @classmethod
    def call_backward_grad_w1(
        cls,
        fp8_grad_a_t: tuple[torch.Tensor, torch.Tensor],
        fp8_x_t: tuple[torch.Tensor, torch.Tensor],
        w1: torch.nn.Parameter,
        tokens_per_expert_list: list[int],
        tokens_per_expert_cuda: torch.Tensor,
        tokens_per_expert_cumsum: torch.Tensor,
        num_local_experts: int,
        fuse_gradient_accumulation: bool = False,
    ):
        """dw1 [2H, h] = grad_a.T [2H, m] @ x.T [h, m]
        With k-grouped fp8: grad_a [m, 2H] @ x [m, h]."""
        assert fuse_gradient_accumulation, \
            "ExpertsFP8GroupedSwiMLP currently only supports fuse_gradient_accumulation."

        wgrad_output = w1.main_grad

        grad_a_packed = (
            pack_to_kmajor(fp8_grad_a_t[0], tokens_per_expert_list, tokens_per_expert_cuda, tokens_per_expert_cumsum),
            fp8_grad_a_t[1].T,
        )
        x_packed = (
            pack_to_kmajor(fp8_x_t[0], tokens_per_expert_list, tokens_per_expert_cuda, tokens_per_expert_cumsum),
            fp8_x_t[1].T,
        )

        k_grouped_fp8_gemm_nt_contiguous(
            tokens_per_expert_list,
            tokens_per_expert_cuda,
            grad_a_packed,
            x_packed,
            num_local_experts,
            compute_stream=torch.cuda.current_stream(),
            output=wgrad_output,
        )
        cls._wgrad_post_process(w1, fuse_gradient_accumulation)

    # ------------------------------------------------------------------
    # BF16 reference implementations (no FP8, no grouped-gemm kernels).
    # Each `*_ref` mirrors its FP8 counterpart but takes/returns BF16
    # tensors directly and uses per-expert torch.matmul. Useful as a
    # numerical reference for correctness checks.
    #
    # Weight layouts (matching the FP8 path, stacked over experts):
    #   w1: (E, 2H, h)   so out = x @ w1[e].T -> (m_e, 2H)
    #   w2: (E, h,  H)   so out = s @ w2[e].T -> (m_e, h)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_by_expert(x: torch.Tensor, tokens_per_expert_list: list[int]) -> list[torch.Tensor]:
        out, off = [], 0
        for n in tokens_per_expert_list:
            out.append(x[off:off + n])
            off += n
        return out

    @classmethod
    def call_forward_a_ref(
        cls,
        w1: torch.nn.Parameter,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        config: TransformerConfig,
    ) -> torch.Tensor:
        """BF16 reference for ``call_forward_a``: fc1 = x @ w1[e].T per expert."""
        E = w1.shape[0]
        out_dim = config.moe_ffn_hidden_size * (2 if config.gated_linear_unit else 1)
        fc1_output = torch.empty(
            permuted_local_hidden_states.shape[0], out_dim,
            device=permuted_local_hidden_states.device,
            dtype=permuted_local_hidden_states.dtype,
        )
        tpe = tokens_per_expert.tolist()
        x_chunks = cls._split_by_expert(permuted_local_hidden_states, tpe)
        off = 0
        for e in range(E):
            n = tpe[e]
            if n == 0:
                continue
            fc1_output[off:off + n] = x_chunks[e] @ w1[e].t()
            off += n
        return fc1_output

    @classmethod
    def call_forward_y_ref(
        cls,
        w2: torch.nn.Parameter,
        fc1_output: torch.Tensor,
        permuted_probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        config: TransformerConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """BF16 reference for ``call_forward_y``: swiglu then fc2 = s @ w2[e].T."""
        s = swiglu_forward(fc1_output, permuted_probs.unsqueeze(-1))
        # s = MergedSwiGLU.call_forward(fc1_output, permuted_probs.unsqueeze(-1))
        E = w2.shape[0]
        fc2_output = torch.empty(
            s.shape[0], config.hidden_size,
            device=s.device, dtype=s.dtype,
        )
        tpe = tokens_per_expert.tolist()
        s_chunks = cls._split_by_expert(s, tpe)
        off = 0
        for e in range(E):
            n = tpe[e]
            if n == 0:
                continue
            fc2_output[off:off + n] = s_chunks[e] @ w2[e].t()
            off += n
        return fc2_output, s

    @classmethod
    def call_backward_grad_a_ref(
        cls,
        grad_y: torch.Tensor,
        a: torch.Tensor,
        w2: torch.nn.Parameter,
        permuted_probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        config: TransformerConfig,
    ):
        """BF16 reference for ``call_backward_grad_a``.

        ds = grad_y @ w2[e] per expert; (grad_a, grad_probs) = swiglu_bwd(ds, a, probs).
        """
        E = w2.shape[0]
        grad_s = torch.empty(
            grad_y.shape[0], config.moe_ffn_hidden_size,
            device=grad_y.device, dtype=grad_y.dtype,
        )
        tpe = tokens_per_expert.tolist()
        gy_chunks = cls._split_by_expert(grad_y, tpe)
        off = 0
        for e in range(E):
            n = tpe[e]
            if n == 0:
                continue
            grad_s[off:off + n] = gy_chunks[e] @ w2[e]
            off += n
        return swiglu_backward(grad_s, a, permuted_probs.unsqueeze(-1))
        # return MergedSwiGLU.call_backward(grad_s, a, permuted_probs.unsqueeze(-1))

    @classmethod
    def call_backward_grad_x_ref(
        cls,
        grad_a: torch.Tensor,
        w1: torch.nn.Parameter,
        tokens_per_expert: torch.Tensor,
        config: TransformerConfig,
    ) -> torch.Tensor:
        """BF16 reference for ``call_backward_grad_x``: dx = grad_a @ w1[e] per expert."""
        E = w1.shape[0]
        grad_x = torch.empty(
            grad_a.shape[0], config.hidden_size,
            device=grad_a.device, dtype=grad_a.dtype,
        )
        tpe = tokens_per_expert.tolist()
        ga_chunks = cls._split_by_expert(grad_a, tpe)
        off = 0
        for e in range(E):
            n = tpe[e]
            if n == 0:
                continue
            grad_x[off:off + n] = ga_chunks[e] @ w1[e]
            off += n
        return grad_x

    @classmethod
    def call_backward_grad_w2_ref(
        cls,
        grad_y: torch.Tensor,
        a: torch.Tensor,
        w2: torch.nn.Parameter,
        permuted_probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        fuse_gradient_accumulation: bool = False,
    ) -> torch.Tensor:
        """BF16 reference for ``call_backward_grad_w2``.

        dw2[e] = grad_y[e].T @ s[e], shape (h, H). If fuse_gradient_accumulation,
        accumulate into w2.main_grad in-place (matching the FP8 path).
        """
        s = swiglu_forward(a, permuted_probs.unsqueeze(-1))
        # s = MergedSwiGLU.call_forward(a, permuted_probs.unsqueeze(-1))
        E = w2.shape[0]
        tpe = tokens_per_expert.tolist()
        gy_chunks = cls._split_by_expert(grad_y, tpe)
        s_chunks = cls._split_by_expert(s, tpe)

        if fuse_gradient_accumulation:
            wgrad = w2.main_grad
            for e in range(E):
                if tpe[e] == 0:
                    continue
                wgrad[e].add_((gy_chunks[e].t() @ s_chunks[e]))
            cls._wgrad_post_process(w2, fuse_gradient_accumulation)
            return wgrad

        wgrad = torch.zeros_like(w2)
        for e in range(E):
            if tpe[e] == 0:
                continue
            wgrad[e] = (gy_chunks[e].t() @ s_chunks[e]).to(wgrad.dtype)
        return wgrad

    @classmethod
    def call_backward_grad_w1_ref(
        cls,
        grad_a: torch.Tensor,
        x: torch.Tensor,
        w1: torch.nn.Parameter,
        tokens_per_expert: torch.Tensor,
        fuse_gradient_accumulation: bool = False,
    ) -> torch.Tensor:
        """BF16 reference for ``call_backward_grad_w1``.

        dw1[e] = grad_a[e].T @ x[e], shape (2H, h).
        """
        E = w1.shape[0]
        tpe = tokens_per_expert.tolist()
        ga_chunks = cls._split_by_expert(grad_a, tpe)
        x_chunks = cls._split_by_expert(x, tpe)

        if fuse_gradient_accumulation:
            wgrad = w1.main_grad
            for e in range(E):
                if tpe[e] == 0:
                    continue
                wgrad[e].add_((ga_chunks[e].t() @ x_chunks[e]))
            cls._wgrad_post_process(w1, fuse_gradient_accumulation)
            return wgrad

        wgrad = torch.zeros_like(w1)
        for e in range(E):
            if tpe[e] == 0:
                continue
            wgrad[e] = (ga_chunks[e].t() @ x_chunks[e]).to(wgrad.dtype)
        return wgrad

    @staticmethod
    def forward(ctx, *args):
        if len(args) not in (8, 9):
            raise ValueError(
                "Invalid arguments for ExpertsFP8GroupedSwiMLP forward pass. "
                f"Expected 8 or 9, got {len(args)}"
            )

        w1: torch.nn.Parameter = args[0]
        w2: torch.nn.Parameter = args[1]
        permuted_local_hidden_states: torch.Tensor = args[2]
        tokens_per_expert: torch.Tensor = args[3]
        num_local_experts: int = args[4]
        permuted_probs: torch.Tensor = args[5]
        expert_wgrad_scheduler: ExpertsWgradScheduler = args[6]
        config: TransformerConfig = args[7]
        wgrad_accumulation_and_reduce_hooks: list = args[8] if len(args) == 9 else []
        fp8_parameter_manager = FP8GPUExpertsParameterManager.get_instance(config)

        assert w1.shape[0] == num_local_experts, f"Expected w1 to have {num_local_experts} experts, but got {w1.shape[0]}"
        assert w2.shape[0] == num_local_experts, f"Expected w2 to have {num_local_experts} experts, but got {w2.shape[0]}"

        # quantize hidden states to FP8 (per-token rowwise)
        fp8_permuted_local_hidden_states = per_token_cast_to_fp8(
            permuted_local_hidden_states, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False,
        )

        # tokens_per_expert prefix-sum on device, used as group offsets by the m-grouped kernel
        tokens_per_expert_psum = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32).to(permuted_local_hidden_states.device)

        # mlp1
        fc1_output = ExpertsFP8GroupedSwiMLP.call_forward_a(
            w1, fp8_permuted_local_hidden_states, tokens_per_expert_psum,
            fp8_parameter_manager, config,
        )

        # act + mlp2
        y, _ = ExpertsFP8GroupedSwiMLP.call_forward_y(
            w2, fc1_output, permuted_probs, tokens_per_expert_psum,
            fp8_parameter_manager, config,
        )

        # context saving
        ctx.fp8_parameter_manager = fp8_parameter_manager
        ctx.num_local_experts = num_local_experts
        ctx.tokens_per_expert = tokens_per_expert
        ctx.tokens_per_expert_psum = tokens_per_expert_psum
        ctx.expert_wgrad_scheduler = expert_wgrad_scheduler
        ctx.wgrad_accumulation_and_reduce_hooks = wgrad_accumulation_and_reduce_hooks
        ctx.has_wgrad_hooks_input = len(args) == 9
        ctx.w1 = w1
        ctx.w2 = w2
        ctx.config = config
        ctx.save_for_backward(
            fp8_permuted_local_hidden_states[0],
            fp8_permuted_local_hidden_states[1],
            fc1_output,
            permuted_probs,
        )
        return y, None

    @staticmethod
    def backward(ctx, *grad_outputs):
        config: TransformerConfig = ctx.config
        w1: torch.nn.Parameter = ctx.w1
        w2: torch.nn.Parameter = ctx.w2
        tokens_per_expert: torch.Tensor = ctx.tokens_per_expert
        tokens_per_expert_psum: torch.Tensor = ctx.tokens_per_expert_psum
        fp8_parameter_manager: FP8GPUExpertsParameterManager = ctx.fp8_parameter_manager
        (fp8_x, fp8_x_scales, fc1_output, permuted_probs) = ctx.saved_tensors

        grad_y = grad_outputs[0].contiguous()

        # tokens_per_expert in the layouts the wgrad kernels need
        tokens_per_expert_list = tokens_per_expert.tolist()
        tokens_per_expert_cuda = tokens_per_expert.to(torch.int32).to(grad_y.device)
        tokens_per_expert_cumsum = torch.tensor(
            [0, *itertools.accumulate(tokens_per_expert_list[:-1])],
            device=grad_y.device, dtype=torch.int32,
        )

        # quantize grad_y in both layouts
        fp8_grad_y = per_token_cast_to_fp8(grad_y, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
        fp8_grad_y_t = per_channel_cast_to_fp8(grad_y, use_ue8m0=False, gran_k=128, transpose=False)

        # grad_a, grad_probs
        grad_a, grad_probs = ExpertsFP8GroupedSwiMLP.call_backward_grad_a(
            fp8_grad_y, fc1_output, w2, permuted_probs,
            tokens_per_expert_psum, fp8_parameter_manager, config,
        )
        grad_a_ref, grad_probs = ExpertsFP8GroupedSwiMLP.call_backward_grad_a_ref(
            grad_y, fc1_output, w2, permuted_probs,
            tokens_per_expert, config,
        )
        d = diff_tensor_norm(grad_a, grad_a_ref)
        assert d < 0.05, f"grad_a diff norm {d:.2e} exceeds threshold"


        # quantize grad_a in both layouts
        fp8_grad_a = per_token_cast_to_fp8(grad_a, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
        fp8_grad_a_t = per_channel_cast_to_fp8(grad_a, use_ue8m0=False, gran_k=128, transpose=False)

        # grad_x
        grad_x = ExpertsFP8GroupedSwiMLP.call_backward_grad_x(
            fp8_grad_a, w1, tokens_per_expert_psum, fp8_parameter_manager, config,
        )
        grad_x_ref = ExpertsFP8GroupedSwiMLP.call_backward_grad_x_ref(
            grad_a, w1, tokens_per_expert, config,
        )
        d = diff_tensor_norm(grad_x, grad_x_ref)
        assert d < 0.05, f"grad_x diff norm {d:.2e} exceeds threshold"

        # grad_w2
        ExpertsFP8GroupedSwiMLP.call_backward_grad_w2(
            fp8_grad_y_t, fc1_output, w2, permuted_probs,
            tokens_per_expert_list, tokens_per_expert_cuda, tokens_per_expert_cumsum,
            ctx.num_local_experts, config.gradient_accumulation_fusion,
        )
        # ExpertsFP8GroupedSwiMLP.call_backward_grad_w2_ref(
        #     grad_y, fc1_output, w2, permuted_probs, tokens_per_expert, config.gradient_accumulation_fusion,
        # )

        # grad_w1: x is already FP8 rowwise; we need a colwise cast for the k-grouped kernel.
        bf16_x = per_token_dequant_from_fp8(fp8_x, fp8_x_scales)
        fp8_x_t = per_channel_cast_to_fp8(bf16_x, use_ue8m0=False, gran_k=128, transpose=False)
        ExpertsFP8GroupedSwiMLP.call_backward_grad_w1(
            fp8_grad_a_t, fp8_x_t, w1,
            tokens_per_expert_list, tokens_per_expert_cuda, tokens_per_expert_cumsum,
            ctx.num_local_experts, config.gradient_accumulation_fusion,
        )
        # ExpertsFP8GroupedSwiMLP.call_backward_grad_w1_ref(
        #     grad_a, bf16_x, w1, tokens_per_expert, config.gradient_accumulation_fusion,
        # )

        for hook_fn in ctx.wgrad_accumulation_and_reduce_hooks:
            hook_fn()

        # gradients for w1, w2 are written into .main_grad inside _wgrad_post_process.
        gradients = (None, None, grad_x, None, None, grad_probs, None, None)
        if ctx.has_wgrad_hooks_input:
            gradients = (*gradients, None)
        return gradients


def fp8_grouped_swiglu_mlp(
    w1: torch.nn.Parameter,
    w2: torch.nn.Parameter,
    permuted_local_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_local_experts: int,
    permuted_probs: torch.Tensor,
    expert_wgrad_scheduler: ExpertsWgradScheduler,
    config: TransformerConfig,
    wgrad_accumulation_and_reduce_hooks: list | None = None,
) -> torch.Tensor:
    """Autograd entry point for the GPU-resident FP8 grouped SwiGLU MLP.

    Args:
        w1: stacked first-linear weight of shape (E, 2*ffn, hidden) on GPU.
        w2: stacked second-linear weight of shape (E, hidden, ffn) on GPU.
        permuted_local_hidden_states: token-permuted activations.
        tokens_per_expert: 1-D tensor of length E.
        num_local_experts: E.
        permuted_probs: per-token routing probs aligned to permutation.
        expert_wgrad_scheduler: kept for API parity; not consumed in the FP8 path.
        config: transformer configuration.
        wgrad_accumulation_and_reduce_hooks: optional hooks to invoke after fused
            weight-gradient accumulation.

    Returns:
        torch.Tensor: MLP output, same shape as permuted_local_hidden_states.
    """
    args = [
        w1,
        w2,
        permuted_local_hidden_states,
        tokens_per_expert,
        num_local_experts,
        permuted_probs,
        expert_wgrad_scheduler,
        config,
    ]
    if wgrad_accumulation_and_reduce_hooks is not None:
        args.append(wgrad_accumulation_and_reduce_hooks)
    output, _ = ExpertsFP8GroupedSwiMLP.apply(*args)
    return output
