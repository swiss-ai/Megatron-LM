import math
import sys
import types
from typing import Optional, Union

import pytest
import torch
import torch.nn as nn
import triton
import triton.language as tl

from flash_attn import flash_attn_func
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

def _get_differential_self_attn_submodules():
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

    return get_gpt_layer_with_transformer_engine_spec(differential=True).submodules.self_attention.submodules

@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    nheads,
    rotary_dim,
    seqlen_ro,
    CACHE_KEY_SEQLEN,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not INTERLEAVED:
        # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(
            COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0
        ).to(tl.float32)
        sin = tl.load(
            SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x0 = tl.load(
            X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            X + rotary_dim_half * stride_x_headdim,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        # write back result
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)
        tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
        tl.store(
            OUT + rotary_dim_half * stride_out_headdim,
            o1,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        )
    else:
        # We don't want to load X[0, 2, 4, ...] and X[1, 3, 5, ...] separately since both are slow.
        # Instead, we load x0 = X[0, 1, 2, 3, ...] and x1 = X[1, 0, 3, 2, ...].
        # Loading x0 will be fast but x1 will be slow.
        # Then we load cos = COS[0, 0, 1, 1, ...] and sin = SIN[0, 0, 1, 1, ...].
        # Then we do the calculation and use tl.where to pick put the right outputs for the even
        # and for the odd indices.
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1  # 1, 0, 3, 2, 5, 4, ...
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0).to(
            tl.float32
        )
        x1 = tl.load(
            X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    assert (
        cos.dtype == sin.dtype
    ), f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert (
        x.dtype == cos.dtype
    ), f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = (
        32
        if rotary_dim <= 32
        else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    )
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)  # noqa
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](
            output,  # data ptrs
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,  # shapes
            nheads,
            rotary_dim,
            seqlen_ro,
            seqlen // 128,  # key for triton cache (limit number of compilations)
            output.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            output.stride(-3),  # seqlen_stride or total_seqlen_stride
            output.stride(-2),  # nheads_stride
            output.stride(-1),  # headdim_stride
            x.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            x.stride(-3),  # seqlen stride or total_seqlen_stride
            x.stride(-2),  # nheads stride
            x.stride(-1),  # headdim stride
            BLOCK_K,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M,
        )
    return output


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            # Can't save int with save_for_backward
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
        # "[CUDA]: invalid device context", and cloning makes it work. Idk why. Triton 2.1.0 works.
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = apply_rotary(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmb.apply(
        x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
    )

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
    

class MultiheadFlashDiff2(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, num_kv_heads=None, rotary_interleaved=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.rotary_interleaved = rotary_interleaved

        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(0, 0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(0, 0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(0, 0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(0, 0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x, rel_pos, attn_mask=None):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, 2 * self.head_dim)
        k = k.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2, self.head_dim)

        cos, sin = rel_pos
        cos = cos.to(dtype=q.dtype, device=q.device)
        sin = sin.to(dtype=q.dtype, device=q.device)

        q = apply_rotary_emb(q, cos, sin, interleaved=self.rotary_interleaved)
        k = apply_rotary_emb(k, cos, sin, interleaved=self.rotary_interleaved)

        q1, q2 = torch.chunk(q, 2, dim=-1)
        k1, k2 = torch.chunk(k, 2, dim=-1)
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        attn11 = flash_attn_func(q1, k1, v1, causal=True)
        attn12 = flash_attn_func(q1, k1, v2, causal=True)
        attn1 = torch.cat([attn11, attn12], dim=-1)

        attn21 = flash_attn_func(q2, k2, v1, causal=True)
        attn22 = flash_attn_func(q2, k2, v2, causal=True)
        attn2 = torch.cat([attn21, attn22], dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn

class TestDifferentialAttention:
    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
        if not torch.cuda.is_available():
            pytest.skip("Differential attention unit test requires CUDA.")
        Utils.initialize_model_parallel(1, 1)
        yield
        Utils.destroy_model_parallel()

    def test_correctness_vs_reference(self, monkeypatch):
        torch.manual_seed(0)
        device = torch.device("cuda")

        batch, seqlen, embed_dim, num_heads = 2, 8, 64, 4

        config = TransformerConfig(
            num_layers=1,
            hidden_size=embed_dim,
            num_attention_heads=num_heads,
            num_query_groups=num_heads,
            use_cpu_initialization=True,
            differential_attention=True,
            transformer_impl="transformer_engine",
            normalization="RMSNorm",
            add_bias_linear=False,
            rotary_interleaved=True,
            apply_rope_fusion=False,
        )

        subs = _get_differential_self_attn_submodules()
        # Replace the default LayNorm fused linear with TE ColumnParallelLinear
        subs.linear_qkv = TEColumnParallelLinear
        attention = SelfAttention(
            config=config,
            submodules=subs,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).to(device)

        reference = MultiheadFlashDiff2(
            embed_dim=embed_dim, depth=1, num_heads=num_heads, rotary_interleaved=True
        ).to(device)

        # prevent dropout
        attention.eval()
        reference.eval()

        # Ensure models are in bfloat16 to match input data type (required for FlashAttention)
        reference.to(dtype=torch.bfloat16)
        attention.to(dtype=torch.bfloat16)

        # Copy weights (Ref -> Megatron)
        with torch.no_grad():
            head_dim_full = embed_dim // num_heads
            q_w = reference.q_proj.weight.view(num_heads, head_dim_full, embed_dim)
            k_w = reference.k_proj.weight.view(num_heads, head_dim_full, embed_dim)
            v_w = reference.v_proj.weight.view(num_heads, head_dim_full, embed_dim)
            qkv_blocks = []
            for h in range(num_heads):
                qkv_blocks.extend([q_w[h], k_w[h], v_w[h]])
            qkv_weight = torch.cat(qkv_blocks, dim=0)
            attention.linear_qkv.weight.copy_(qkv_weight)
            attention.linear_proj.weight.copy_(reference.out_proj.weight)
            attention.lambda_q1.copy_(reference.lambda_q1)
            attention.lambda_k1.copy_(reference.lambda_k1)
            attention.lambda_q2.copy_(reference.lambda_q2)
            attention.lambda_k2.copy_(reference.lambda_k2)
            attention.lambda_init = reference.lambda_init
            if attention.subln.weight is not None:
                attention.subln.weight.copy_(reference.subln.weight)

        # Build rotary embeddings
        rotary_emb = RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_interleaved=config.rotary_interleaved,
            rotary_percent=0.5,
            use_cpu_initialization=True,
        )
        rotary_pos_emb = rotary_emb(seqlen).to(device)
        cos, sin = rotary_emb.get_cos_sin(seqlen)
        hidden_states = torch.randn(batch, seqlen, embed_dim, device=device, dtype=torch.bfloat16)
        attention_mask = torch.zeros(1, 1, seqlen, seqlen, device=device, dtype=torch.bool)
        cos = cos.to(device=device, dtype=hidden_states.dtype)
        sin = sin.to(device=device, dtype=hidden_states.dtype)

        ref_out = reference(hidden_states, rel_pos=(cos, sin))
        megatron_out, _ = attention(
            hidden_states.transpose(0, 1), attention_mask, rotary_pos_emb=(rotary_pos_emb, rotary_pos_emb)
        )

        torch.testing.assert_close(ref_out, megatron_out.transpose(0, 1), atol=1e-2, rtol=1e-3)
