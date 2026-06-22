# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Fused Triton kernels for the 2nd-order PolyNorm GLU activation.

For a token row ``x`` (the GLU gate half) and ``m`` (the GLU linear half ``x_linear``) this computes,
in a single pass over the ffn feature dimension ``D``::

    norm(z)_i = z_i * rsqrt( mean_j(z_j**2) + eps )          # reduce over the feature dim, fp32
    gate_i    = a1 * norm(x)_i + a2 * norm(x**2)_i
    out_i     = gate_i * m_i * score                         # score optional (per token)

It fuses the gate (a feature-wise RMS-style reduction), the GLU multiply by ``x_linear`` and the
optional per-token ``score`` (MoE router probs / per-token scale) multiply into one kernel, mirroring
the SwiGLU fusion in ``fused_bias_swiglu.py`` so PolyNorm GLU runs close to SwiGLU throughput. The
grid is over token rows, so the kernel is shape-agnostic in the token count (no torch.compile-style
recompiles on the variable-token MoE path).

The coefficients ``a1/a2`` are passed *per token* (shape ``(M,)``, contiguous); the owning module
(:class:`~megatron.core.activations.PolyNorm`) expands its per-expert coefficients to per-token
before calling, so this kernel serves both the dense and the grouped-expert paths with one code path.
``abs()`` on the coefficients is applied by the module *outside* the autograd Function, so the
gradients returned here are w.r.t. the already-positive coefficients (torch's ``abs`` backward then
supplies the sign, and ``repeat_interleave``/``expand`` backward reduces the per-token coefficient
gradients back to the per-expert parameter).
"""
from typing import Optional

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    # Triton requires a CUDA device; on CPU-only boxes we keep the module importable but route all
    # real work to the torch fallback in PolyNorm.
    HAVE_TRITON = torch.cuda.is_available()
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    from unittest.mock import MagicMock

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


# Largest per-shard feature dim the single-block kernel handles. Above this the module routes to the
# torch fallback (keeps register/SRAM pressure bounded). Per-shard ffn sizes are well within this.
MAX_FUSED_FEATURE_DIM = 8192


def _num_warps_for(block_size: int) -> int:
    """Heuristic warp count for a single-row reduction over ``block_size`` features."""
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    if block_size >= 512:
        return 4
    return 2


@triton.jit
def _polynorm_glu_fwd_kernel(
    out_ptr,  # (M, D) output
    inv_ptr,  # (M, 2) fp32, saved inverse-RMS per order for backward
    x_ptr,  # (M, D) gate half (x_glu)
    m_ptr,  # (M, D) linear half (x_linear)
    a1_ptr,  # (M,) fp32 per-token coefficient for norm(x)
    a2_ptr,  # (M,) fp32 per-token coefficient for norm(x**2)
    score_ptr,  # (M,) per-token multiplier (only read when HAS_SCORE)
    stride_x_row,
    stride_x_col,
    stride_m_row,
    stride_m_col,
    stride_out_row,
    stride_out_col,
    D,
    rD,  # 1.0 / D, precomputed in fp32
    eps,
    HAS_SCORE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    x = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + row * stride_m_row + cols * stride_m_col, mask=mask, other=0.0).to(tl.float32)

    x2 = x * x

    # S_k = sum_j (x_j**k)**2 ; masked elements are 0 so they don't contribute.
    s1 = tl.sum(x2, axis=0)
    s2 = tl.sum(x2 * x2, axis=0)
    inv1 = tl.rsqrt(s1 * rD + eps)
    inv2 = tl.rsqrt(s2 * rD + eps)

    a1 = tl.load(a1_ptr + row)
    a2 = tl.load(a2_ptr + row)

    gate = a1 * x * inv1 + a2 * x2 * inv2
    out = gate * m
    if HAS_SCORE:
        out = out * tl.load(score_ptr + row)

    tl.store(out_ptr + row * stride_out_row + cols * stride_out_col, out, mask=mask)
    tl.store(inv_ptr + row * 2 + 0, inv1)
    tl.store(inv_ptr + row * 2 + 1, inv2)


@triton.jit
def _polynorm_glu_bwd_kernel(
    dx_ptr,  # (M, D) grad w.r.t. x_glu
    dm_ptr,  # (M, D) grad w.r.t. x_linear
    da1_ptr,  # (M,) fp32 per-token grad for a1
    da2_ptr,  # (M,) fp32 per-token grad for a2
    dscore_ptr,  # (M,) per-token grad for score (only written when HAS_SCORE)
    dout_ptr,  # (M, D) incoming grad
    x_ptr,  # (M, D) gate half (saved)
    m_ptr,  # (M, D) linear half (saved)
    inv_ptr,  # (M, 2) fp32 saved inverse-RMS
    a1_ptr,
    a2_ptr,
    score_ptr,
    stride_dout_row,
    stride_dout_col,
    stride_x_row,
    stride_x_col,
    stride_m_row,
    stride_m_col,
    stride_dx_row,
    stride_dx_col,
    stride_dm_row,
    stride_dm_col,
    D,
    rD,
    HAS_SCORE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    x = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + row * stride_m_row + cols * stride_m_col, mask=mask, other=0.0).to(tl.float32)
    dout = tl.load(
        dout_ptr + row * stride_dout_row + cols * stride_dout_col, mask=mask, other=0.0
    ).to(tl.float32)

    inv1 = tl.load(inv_ptr + row * 2 + 0)
    inv2 = tl.load(inv_ptr + row * 2 + 1)
    a1 = tl.load(a1_ptr + row)
    a2 = tl.load(a2_ptr + row)

    x2 = x * x
    x3 = x2 * x  # used by the k=2 coupling term below

    gate = a1 * x * inv1 + a2 * x2 * inv2

    if HAS_SCORE:
        score = tl.load(score_ptr + row)
    else:
        score = 1.0

    dgate = dout * m * score
    # grad w.r.t. x_linear (the GLU "mul" operand)
    dm = dout * gate * score
    tl.store(dm_ptr + row * stride_dm_row + cols * stride_dm_col, dm, mask=mask)

    # Per-row reductions A_k = sum_i dgate_i * x_i**k (masked elements contribute 0).
    A1 = tl.sum(dgate * x, axis=0)
    A2 = tl.sum(dgate * x2, axis=0)

    # dX_j = dgate_j * (a1*inv1 + 2*a2*inv2*x_j)
    #        - (1/D) * ( a1*inv1**3*x_j*A1 + 2*a2*inv2**3*x_j**3*A2 )
    direct = a1 * inv1 + 2.0 * a2 * inv2 * x
    coupling = (
        a1 * inv1 * inv1 * inv1 * x * A1
        + 2.0 * a2 * inv2 * inv2 * inv2 * x3 * A2
    )
    dx = dgate * direct - rD * coupling
    tl.store(dx_ptr + row * stride_dx_row + cols * stride_dx_col, dx, mask=mask)

    # da_k = inv_k * A_k (per token; autograd reduces these back to the per-expert coefficient).
    tl.store(da1_ptr + row, inv1 * A1)
    tl.store(da2_ptr + row, inv2 * A2)

    if HAS_SCORE:
        # dscore_r = sum_j dout_j * gate_j * m_j
        tl.store(dscore_ptr + row, tl.sum(dout * gate * m, axis=0))


class FusedPolyNormGLUFunction(torch.autograd.Function):
    """Autograd wrapper around the fused 2nd-order PolyNorm GLU Triton kernels.

    Inputs are 2D ``(M, D)`` (the caller flattens leading dims). ``a1/a2`` are positive per-token
    coefficients of shape ``(M,)``; ``score`` is an optional per-token multiplier of shape ``(M,)``.
    """

    @classmethod
    def call_forward(
        cls, x_glu, x_linear, a1, a2, eps, score
    ):
        M, D = x_glu.shape
        out = torch.empty((M, D), dtype=x_glu.dtype, device=x_glu.device)
        inv = torch.empty((M, 2), dtype=torch.float32, device=x_glu.device)
        has_score = score is not None
        score_arg = score if has_score else x_glu  # dummy pointer when unused

        block = triton.next_power_of_2(D)
        _polynorm_glu_fwd_kernel[(M,)](
            out,
            inv,
            x_glu,
            x_linear,
            a1,
            a2,
            score_arg,
            x_glu.stride(0),
            x_glu.stride(1),
            x_linear.stride(0),
            x_linear.stride(1),
            out.stride(0),
            out.stride(1),
            D,
            1.0 / D,
            eps,
            HAS_SCORE=has_score,
            BLOCK_SIZE=block,
            num_warps=_num_warps_for(block),
        )
        return out, inv

    @staticmethod
    def forward(ctx, x_glu, x_linear, a1, a2, eps, score):
        M, D = x_glu.shape
        out = torch.empty((M, D), dtype=x_glu.dtype, device=x_glu.device)
        inv = torch.empty((M, 2), dtype=torch.float32, device=x_glu.device)
        has_score = score is not None
        score_arg = score if has_score else x_glu  # dummy pointer when unused

        block = triton.next_power_of_2(D)
        _polynorm_glu_fwd_kernel[(M,)](
            out,
            inv,
            x_glu,
            x_linear,
            a1,
            a2,
            score_arg,
            x_glu.stride(0),
            x_glu.stride(1),
            x_linear.stride(0),
            x_linear.stride(1),
            out.stride(0),
            out.stride(1),
            D,
            1.0 / D,
            eps,
            HAS_SCORE=has_score,
            BLOCK_SIZE=block,
            num_warps=_num_warps_for(block),
        )

        saved = [x_glu, x_linear, a1, a2, inv]
        if has_score:
            saved.append(score)
        ctx.save_for_backward(*saved)
        ctx.has_score = has_score
        ctx.eps = eps
        return out
    
    @classmethod
    def call_backward(
        cls, 
        grad_output, 
        saved
    ):
        x_glu, x_linear, a1, a2, inv = saved[:5]
        score = saved[5] if len(saved) > 5 else None
        M, D = x_glu.shape

        grad_output = grad_output.contiguous()
        dx = torch.empty((M, D), dtype=x_glu.dtype, device=x_glu.device)
        dm = torch.empty((M, D), dtype=x_linear.dtype, device=x_linear.device)
        da1 = torch.empty((M,), dtype=a1.dtype, device=x_glu.device)
        da2 = torch.empty((M,), dtype=a2.dtype, device=x_glu.device)
        has_score = score is not None
        if has_score:
            dscore = torch.empty((M,), dtype=score.dtype, device=score.device)
            score_arg = score
        else:
            dscore = None
            score_arg = x_glu  # dummy pointer

        block = triton.next_power_of_2(D)
        _polynorm_glu_bwd_kernel[(M,)](
            dx,
            dm,
            da1,
            da2,
            dscore if has_score else x_glu,
            grad_output,
            x_glu,
            x_linear,
            inv,
            a1,
            a2,
            score_arg,
            grad_output.stride(0),
            grad_output.stride(1),
            x_glu.stride(0),
            x_glu.stride(1),
            x_linear.stride(0),
            x_linear.stride(1),
            dx.stride(0),
            dx.stride(1),
            dm.stride(0),
            dm.stride(1),
            D,
            1.0 / D,
            HAS_SCORE=has_score,
            BLOCK_SIZE=block,
            num_warps=_num_warps_for(block),
        )

        return dx, dm, da1, da2, None, (dscore if has_score else None)

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        x_glu, x_linear, a1, a2, inv = saved[:5]
        score = saved[5] if ctx.has_score else None
        M, D = x_glu.shape

        grad_output = grad_output.contiguous()
        dx = torch.empty((M, D), dtype=x_glu.dtype, device=x_glu.device)
        dm = torch.empty((M, D), dtype=x_linear.dtype, device=x_linear.device)
        # Match each coefficient's dtype so autograd receives a same-dtype grad (alpha may be
        # fp32 master params or bf16 under mixed precision); the kernel casts fp32->dtype on store.
        da1 = torch.empty((M,), dtype=a1.dtype, device=x_glu.device)
        da2 = torch.empty((M,), dtype=a2.dtype, device=x_glu.device)
        has_score = ctx.has_score
        if has_score:
            dscore = torch.empty((M,), dtype=score.dtype, device=score.device)
            score_arg = score
        else:
            dscore = None
            score_arg = x_glu  # dummy pointer

        block = triton.next_power_of_2(D)
        _polynorm_glu_bwd_kernel[(M,)](
            dx,
            dm,
            da1,
            da2,
            dscore if has_score else x_glu,
            grad_output,
            x_glu,
            x_linear,
            inv,
            a1,
            a2,
            score_arg,
            grad_output.stride(0),
            grad_output.stride(1),
            x_glu.stride(0),
            x_glu.stride(1),
            x_linear.stride(0),
            x_linear.stride(1),
            dx.stride(0),
            dx.stride(1),
            dm.stride(0),
            dm.stride(1),
            D,
            1.0 / D,
            HAS_SCORE=has_score,
            BLOCK_SIZE=block,
            num_warps=_num_warps_for(block),
        )

        # eps (position 4) gets no gradient; score grad is None when score was None.
        return dx, dm, da1, da2, None, (dscore if has_score else None)


def fused_polynorm_glu_impl(
    x_glu: torch.Tensor,
    x_linear: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    eps: float,
    score: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused 2nd-order PolyNorm GLU: ``gate(x_glu) * x_linear * [score]``.

    Args:
        x_glu, x_linear: ``(..., D)`` gate / linear halves of the fc1 output (same shape & dtype).
        a1, a2: positive per-token coefficients, broadcastable to ``(M,)`` where ``M`` is the
            number of tokens (``prod(x_glu.shape[:-1])``). The owning module passes either a single
            shared coefficient or one per token (grouped experts).
        eps: RMS epsilon.
        score: optional per-token multiplier ``(..., 1)`` / ``(...)`` (MoE router probs / scale).

    Returns:
        Tensor of ``x_glu``'s shape and dtype.
    """
    ori_shape = x_glu.shape
    D = ori_shape[-1]
    x2 = x_glu.reshape(-1, D)
    m2 = x_linear.reshape(-1, D)
    M = x2.shape[0]

    a1 = a1.reshape(-1)
    a2 = a2.reshape(-1)
    if a1.numel() == 1:
        # Shared (dense) coefficient: materialize one per token so the kernel can index by row.
        # The expand→contiguous keeps the autograd link to the (1,) parameter (expand-backward sums).
        a1 = a1.expand(M).contiguous()
        a2 = a2.expand(M).contiguous()

    score2 = None
    if score is not None:
        score2 = score.reshape(-1)

    out = FusedPolyNormGLUFunction.apply(x2, m2, a1, a2, eps, score2)
    return out.reshape(ori_shape)


def _split_glu_halves(fc1_output: torch.Tensor):
    """Split a fused ``(..., 2D)`` fc1 output into its gate / linear ``(M, D)`` halves.
    """
    ori_shape = fc1_output.shape
    two_d = ori_shape[-1]
    assert two_d % 2 == 0, "fc1 output last dim must be 2*D for a gated linear unit"
    d = two_d // 2
    flat = fc1_output.reshape(-1, two_d)
    x_glu, x_linear = torch.split(flat, d, dim=-1)
    return x_glu, x_linear, ori_shape[:-1], d


def _per_token_coeffs(a1: torch.Tensor, a2: torch.Tensor, num_tokens: int):
    """Normalize ``a1``/``a2`` to per-token ``(M,)`` coefficients for the row-indexed kernels.

    Mirrors :func:`fused_polynorm_glu_impl`: a single shared coefficient is expanded to one per
    token (the ``expand`` keeps the autograd link), and already-per-token coefficients pass
    through unchanged.
    """
    a1 = a1.reshape(-1)
    a2 = a2.reshape(-1)
    if a1.numel() == 1:
        a1 = a1.expand(num_tokens).contiguous()
        a2 = a2.expand(num_tokens).contiguous()
    return a1, a2


def fused_polynorm_glu_forward(
    fc1_output: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    eps: float,
    score: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2nd-order PolyNorm GLU forward.
    """
    x_glu, x_linear, lead_shape, d = _split_glu_halves(fc1_output)
    a1, a2 = _per_token_coeffs(a1, a2, x_glu.shape[0])
    score2 = score.reshape(-1) if score is not None else None
    out, inv = FusedPolyNormGLUFunction.call_forward(x_glu, x_linear, a1, a2, eps, score2)
    return out.reshape(lead_shape + (d,)), inv


def fused_polynorm_glu_backward(
    grad_output: torch.Tensor,
    fc1_output: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    inv: torch.Tensor,
    eps: float,
    score: Optional[torch.Tensor] = None,
):
    """2nd-order PolyNorm GLU backward.

    Recomputes the inverse-RMS state from fc1_output then runs the fused backward kernel.
    """
    x_glu, x_linear, lead_shape, d = _split_glu_halves(fc1_output)
    # A single shared coefficient receives the summed per-token gradient (mirrors the
    # expand-backward in ``fused_polynorm_glu_impl``); a per-token coefficient keeps its shape.
    shared_coeff = a1.numel() == 1
    a1_c, a2_c = _per_token_coeffs(a1, a2, x_glu.shape[0])
    score2 = score.reshape(-1) if score is not None else None

    # Recompute inverse-RMS.
    # _, to_save = FusedPolyNormGLUFunction.call_forward(x_glu, x_linear, a1_c, a2_c, eps, None)
    # inv = to_save[4]
    
    # reconstruct the tensors
    assert inv is not None
    saved = [x_glu, x_linear, a1_c, a2_c, inv]
    if score2 is not None:
        saved.append(score2)

    grad_flat = grad_output.reshape(-1, d)
    dx, dm, da1, da2, _, dscore = FusedPolyNormGLUFunction.call_backward(grad_flat, saved)
    grad_fc1 = torch.cat([dx, dm], dim=-1).reshape(lead_shape + (2 * d,))
    if shared_coeff:
        da1 = da1.sum(0, keepdim=True).reshape(a1.shape)
        da2 = da2.sum(0, keepdim=True).reshape(a2.shape)
    else:
        da1 = da1.reshape(a1.shape)
        da2 = da2.reshape(a2.shape)
    grad_score = dscore.reshape(score.shape[:-1]) if score is not None else None
    return grad_fc1, da1, da2, grad_score
