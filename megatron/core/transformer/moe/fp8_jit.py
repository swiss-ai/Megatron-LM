"""Triton kernels for FP8 quantization and dequantization with per-block and per-channel scaling."""
import torch
import triton
import triton.language as tl
from typing import Tuple

from deep_gemm.utils.math import align, pack_ue8m0_to_int

# referred from DeepGEMM utils
def per_block_cast_to_fp8_cpu(x: torch.Tensor, gran_k: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    assert m % gran_k == 0 and n % gran_k == 0, \
        f"per_block_cast_to_fp8 requires m and n to be multiples of gran_k"
    # x_padded = torch.zeros((align(m, gran_k), align(n, gran_k)), dtype=x.dtype, device=x.device)
    # x_padded[:m, :n] = x
    x_view = x.view(-1, gran_k, x.size(1) // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x)[:m, :n].contiguous(), sf.view(x_view.size(0), x_view.size(2))


@triton.jit
def _per_block_cast_to_fp8_kernel(
    x_ptr,
    x_fp8_ptr,
    sf_ptr,
    m,
    n,
    stride_m,
    stride_fp8_m,
    stride_sf_m,
    gran_k: tl.constexpr,
):
    """Each program handles one gran_k x gran_k block; one scale factor per block."""
    block_row = tl.program_id(0)
    block_col = tl.program_id(1)

    rows = block_row * gran_k + tl.arange(0, gran_k)
    cols = block_col * gran_k + tl.arange(0, gran_k)

    offsets = rows[:, None] * stride_m + cols[None, :]
    mask = (rows[:, None] < m) & (cols[None, :] < n)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    x_amax = tl.max(tl.abs(x))
    # x_amax = tl.maximum(x_amax, 1e-4)

    sf = x_amax / 448.0
    sf = tl.where(sf == 0.0, 1.0, sf)
    inv_sf = 1.0 / sf
    x_scaled = x * inv_sf
    # x_scaled = tl.clamp(x_scaled, -448.0, 448.0).to(tl.float8e4nv)

    fp8_offsets = rows[:, None] * stride_fp8_m + cols[None, :]
    tl.store(x_fp8_ptr + fp8_offsets, x_scaled.to(tl.float8e4nv), mask=mask)

    tl.store(sf_ptr + block_row * stride_sf_m + block_col, sf)


@torch._dynamo.allow_in_graph
def per_block_cast_to_fp8_gpu(
    x: torch.Tensor, 
    gran_k: int = 128,
    x_fp8: torch.Tensor = None,
    sf: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton implementation of per_block_cast_to_fp8 (GPU only).

    Each gran_k x gran_k block of the input gets a single fp32 scale factor.
    Matches the reference CPU implementation `per_block_cast_to_fp8`.

    Args:
        x: 2D tensor on a CUDA device. Both dims must be multiples of gran_k.
        gran_k: Block size.

    Returns:
        x_fp8: (m, n) tensor in torch.float8_e4m3fn.
        sf: (m // gran_k, n // gran_k) fp32 tensor.
    """
    assert x.dim() == 2
    assert x.is_cuda, "per_block_cast_to_fp8_triton requires a CUDA tensor"
    m, n = x.shape
    assert m % gran_k == 0 and n % gran_k == 0, \
        f"per_block_cast_to_fp8_triton requires m and n to be multiples of gran_k"

    if x.stride(-1) != 1:
        x = x.contiguous()
    if x_fp8 is None:
        x_fp8 = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    if sf is None:
        sf = torch.empty((m // gran_k, n // gran_k), dtype=torch.float32, device=x.device)

    grid = (m // gran_k, n // gran_k)
    _per_block_cast_to_fp8_kernel[grid](
        x, x_fp8, sf,
        m, n,
        x.stride(0), x_fp8.stride(0), sf.stride(0),
        gran_k=gran_k,
    )
    return x_fp8, sf


@triton.jit
def _ceil_to_ue8m0(sf):
    """Ceil a float32 value to the nearest representable UE8M0 value (exponent-only)."""
    bits = sf.to(tl.int32, bitcast=True)
    bits = bits & 0x7FFFFFFF  # abs: clear sign bit
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exp = exp + tl.where(mantissa > 0, 1, 0)
    # tl.clamp only accepts floating-point tensors in current Triton.
    exp = tl.where(exp < 1, 1, tl.where(exp > 254, 254, exp))
    return (exp << 23).to(tl.float32, bitcast=True)


@triton.jit
def _per_token_cast_to_fp8_kernel(
    x_ptr,
    x_fp8_ptr,
    sf_ptr,
    m,
    n,
    padded_n,
    stride_m,
    stride_fp8_m,
    stride_sf_m,
    stride_sf_k,
    BLOCK_M: tl.constexpr,
    gran_k: tl.constexpr,
    use_ue8m0: tl.constexpr,
):
    """Each program handles one group (gran_k columns) for BLOCK_M rows."""
    group_idx = tl.program_id(0)
    row_start = tl.program_id(1) * BLOCK_M

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = group_idx * gran_k + tl.arange(0, gran_k)

    # Load input tile; padding elements (cols >= n) are filled with 0
    offsets = rows[:, None] * stride_m + cols[None, :]
    load_mask = (rows[:, None] < m) & (cols[None, :] < n)
    x = tl.load(x_ptr + offsets, mask=load_mask, other=0.0).to(tl.float32)

    # Per-row absolute max within the group
    x_amax = tl.max(tl.abs(x), axis=1)
    # x_amax = tl.maximum(x_amax, 1e-4)

    # Scale factor
    sf = x_amax / 448.0
    if use_ue8m0:
        sf = _ceil_to_ue8m0(sf)

    # Scale and store FP8 output
    sf = tl.where(sf == 0.0, 1.0, sf)
    # Floor sf so inv_sf cannot overflow when a token's group amax is tiny but
    # nonzero (denormal sf -> inv_sf = +inf -> 0*inf = NaN). See
    # _pack_kmajor_per_channel_fp8_kernel for the full rationale.
    sf = tl.maximum(sf, 1e-30)
    inv_sf = 1.0 / sf[:, None]
    x_scaled = x * inv_sf
    # x_scaled = tl.clamp(x_scaled, -448.0, 448.0).to(tl.float8e4nv)
    fp8_offsets = rows[:, None] * stride_fp8_m + cols[None, :]
    fp8_mask = (rows[:, None] < m) & (cols[None, :] < n)
    tl.store(x_fp8_ptr + fp8_offsets, x_scaled.to(tl.float8e4nv), mask=fp8_mask)

    # Store scale factors
    tl.store(sf_ptr + rows * stride_sf_m + group_idx * stride_sf_k, sf, mask=(rows < m))

@torch._dynamo.allow_in_graph
def per_token_cast_to_fp8(
    x: torch.Tensor,
    use_ue8m0: bool,
    gran_k: int = 128,
    use_packed_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton implementation of per_token_cast_to_fp8.

    Quantizes a 2D tensor to FP8 with per-group scale factors (one scale per
    `gran_k` elements along the token dimension). Fuses padding, reduction,
    scaling, and type conversion into a single kernel.

    Args:
        x: Input tensor of shape (m, n), any float dtype.
        use_ue8m0: Encode scale factors as UE8M0 (ceil to nearest exponent).
        gran_k: Group size for scale factor computation.
        use_packed_ue8m0: Pack UE8M0 scale factors into int32.

    Returns:
        x_fp8: Quantized tensor of shape (m, n) in torch.float8_e4m3fn.
        sf: Scale factors of shape (m, num_groups) in fp32, or packed int32.
    """
    assert x.dim() == 2, f"Expected 2D input, got shape {x.shape}"
    m, n = x.shape
    padded_n = align(n, gran_k)
    num_groups = padded_n // gran_k

    x_fp8 = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m, num_groups), dtype=torch.float32, device=x.device)

    BLOCK_M = 4
    grid = (num_groups, triton.cdiv(m, BLOCK_M))

    _per_token_cast_to_fp8_kernel[grid](
        x, x_fp8, sf,
        m, n, padded_n,
        x.stride(0), x_fp8.stride(0),
        sf.stride(0), sf.stride(1),
        BLOCK_M=BLOCK_M,
        gran_k=gran_k,
        use_ue8m0=use_ue8m0,
    )

    if use_packed_ue8m0:
        sf = pack_ue8m0_to_int(sf)

    return x_fp8, sf


@torch._dynamo.allow_in_graph
def per_token_cast_to_fp8_chunked(
    x: torch.Tensor,
    chunk_sizes: list,
    gran_k: int = 128,
    use_ue8m0: bool = False,
) -> Tuple[torch.Tensor, list, list]:
    """Per-token FP8 quantization that writes each chunk's scale factor into
    its own MN-major, TMA-aligned buffer.

    Returns (fp8_full, fp8_chunks, sf_chunks) where each sf_chunks[i] has
    shape (chunk_sizes[i], num_groups) with strides (1, align(chunk_sizes[i], 4))
    — already in the layout DeepGEMM's m_grouped_fp8_gemm_nt_contiguous
    early-returns on, so no transpose kernel is launched downstream.
    """
    assert x.dim() == 2, f"Expected 2D input, got shape {x.shape}"
    assert not use_ue8m0, "ue8m0 path not supported for chunked variant"
    m, n = x.shape
    padded_n = align(n, gran_k)
    num_groups = padded_n // gran_k

    fp8_full = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    fp8_chunks = list(torch.split(fp8_full, chunk_sizes, dim=0))
    x_chunks = list(torch.split(x, chunk_sizes, dim=0))

    sf_chunks = []
    BLOCK_M = 4
    for cm, xc, fp8c in zip(chunk_sizes, x_chunks, fp8_chunks):
        tma_aligned_cm = align(cm, 4)  # 16B / sizeof(fp32) = 4 elements
        sf = torch.empty_strided(
            (cm, num_groups),
            (1, tma_aligned_cm),
            dtype=torch.float32,
            device=x.device,
        )
        grid = (num_groups, triton.cdiv(cm, BLOCK_M))
        _per_token_cast_to_fp8_kernel[grid](
            xc, fp8c, sf,
            cm, n, padded_n,
            xc.stride(0), fp8c.stride(0),
            sf.stride(0), sf.stride(1),
            BLOCK_M=BLOCK_M,
            gran_k=gran_k,
            use_ue8m0=use_ue8m0,
        )
        sf_chunks.append(sf)

    return fp8_full, fp8_chunks, sf_chunks


@triton.jit
def _per_token_cast_to_fp8_chunked_fused_kernel(
    x_ptr,
    x_fp8_ptr,
    sf_flat_ptr,
    chunk_sizes_ptr,        # int32[num_chunks]
    chunk_row_starts_ptr,   # int32[num_chunks]
    sf_offsets_ptr,         # int32[num_chunks]
    sf_strides_ptr,         # int32[num_chunks]  (== align(cm, 4))
    n,
    padded_n,
    stride_m,
    stride_fp8_m,
    BLOCK_M: tl.constexpr,
    gran_k: tl.constexpr,
    use_ue8m0: tl.constexpr,
):
    """Fused per-chunk quantize. Grid = (num_groups, num_chunks, ceil(max_cm/BLOCK_M)).

    Each program handles one (chunk, row-tile, column-group). Programs whose
    local_rows fall past the chunk's cm are masked out and become no-ops.
    Scale factors are written into a flat fp32 buffer at per-chunk offsets with
    strides ``(1, align(cm, 4))`` — MN-major TMA-aligned for DeepGEMM.
    """
    group_idx = tl.program_id(0)
    chunk_id  = tl.program_id(1)
    row_tile  = tl.program_id(2)

    cm        = tl.load(chunk_sizes_ptr      + chunk_id)
    row_start = tl.load(chunk_row_starts_ptr + chunk_id)
    sf_off    = tl.load(sf_offsets_ptr       + chunk_id)
    sf_str_k  = tl.load(sf_strides_ptr       + chunk_id)

    local_rows  = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask    = local_rows < cm
    global_rows = row_start + local_rows

    cols = group_idx * gran_k + tl.arange(0, gran_k)

    offsets   = global_rows[:, None] * stride_m + cols[None, :]
    load_mask = row_mask[:, None] & (cols[None, :] < n)
    x = tl.load(x_ptr + offsets, mask=load_mask, other=0.0).to(tl.float32)

    x_amax = tl.max(tl.abs(x), axis=1)

    sf = x_amax / 448.0
    if use_ue8m0:
        sf = _ceil_to_ue8m0(sf)
    sf = tl.where(sf == 0.0, 1.0, sf)
    # Floor sf so inv_sf cannot overflow when a token's group amax is tiny but
    # nonzero (denormal sf -> inv_sf = +inf -> 0*inf = NaN). See
    # _pack_kmajor_per_channel_fp8_kernel for the full rationale.
    sf = tl.maximum(sf, 1e-30)
    inv_sf = 1.0 / sf[:, None]
    x_scaled = x * inv_sf

    fp8_offsets = global_rows[:, None] * stride_fp8_m + cols[None, :]
    fp8_mask    = row_mask[:, None] & (cols[None, :] < n)
    tl.store(x_fp8_ptr + fp8_offsets, x_scaled.to(tl.float8e4nv), mask=fp8_mask)

    sf_addr = sf_off + local_rows + group_idx * sf_str_k
    tl.store(sf_flat_ptr + sf_addr, sf, mask=row_mask)


@torch._dynamo.allow_in_graph
def per_token_cast_to_fp8_chunked_fused(
    x: torch.Tensor,
    chunk_sizes: list,
    gran_k: int = 128,
    use_ue8m0: bool = False,
) -> Tuple[torch.Tensor, list, list]:
    """Single-launch fused variant of :func:`per_token_cast_to_fp8_chunked`.

    Allocates one flat fp32 SF buffer; per-chunk views are constructed via
    ``torch.as_strided`` with strides ``(1, align(cm, 4))`` so DeepGEMM still
    early-returns from ``get_mn_major_tma_aligned_tensor`` (no transpose).
    """
    assert x.dim() == 2, f"Expected 2D input, got shape {x.shape}"
    assert not use_ue8m0, "ue8m0 path not supported for fused variant"
    m, n = x.shape
    padded_n = align(n, gran_k)
    num_groups = padded_n // gran_k
    num_chunks = len(chunk_sizes)
    assert sum(chunk_sizes) == m, \
        f"sum(chunk_sizes)={sum(chunk_sizes)} != m={m}"

    # Per-chunk metadata (small int32 arrays, packed into one H2D)
    sf_strides_list = [align(cm, 4) for cm in chunk_sizes]
    sf_sizes_list   = [s * num_groups for s in sf_strides_list]
    sf_offsets_list = [0]
    for s in sf_sizes_list[:-1]:
        sf_offsets_list.append(sf_offsets_list[-1] + s)
    chunk_row_starts_list = [0]
    for cm in chunk_sizes[:-1]:
        chunk_row_starts_list.append(chunk_row_starts_list[-1] + cm)

    total_sf_size = sum(sf_sizes_list)
    max_cm = max(chunk_sizes)

    # Pack the four small int32 arrays into a single CPU tensor and copy once
    meta_cpu = torch.tensor(
        list(chunk_sizes) + chunk_row_starts_list + sf_offsets_list + sf_strides_list,
        dtype=torch.int32,
    )
    meta = meta_cpu.to(x.device, non_blocking=True)
    chunk_sizes_cuda      = meta[0:num_chunks]
    chunk_row_starts_cuda = meta[num_chunks:2 * num_chunks]
    sf_offsets_cuda       = meta[2 * num_chunks:3 * num_chunks]
    sf_strides_cuda       = meta[3 * num_chunks:4 * num_chunks]

    fp8_full = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    sf_flat  = torch.empty(total_sf_size, dtype=torch.float32, device=x.device)

    BLOCK_M = 4
    grid = (num_groups, num_chunks, triton.cdiv(max_cm, BLOCK_M))
    _per_token_cast_to_fp8_chunked_fused_kernel[grid](
        x, fp8_full, sf_flat,
        chunk_sizes_cuda, chunk_row_starts_cuda,
        sf_offsets_cuda, sf_strides_cuda,
        n, padded_n,
        x.stride(0), fp8_full.stride(0),
        BLOCK_M=BLOCK_M,
        gran_k=gran_k,
        use_ue8m0=use_ue8m0,
    )

    fp8_chunks = list(torch.split(fp8_full, chunk_sizes, dim=0))
    sf_chunks = [
        torch.as_strided(sf_flat, (cm, num_groups), (1, st), storage_offset=off)
        for cm, st, off in zip(chunk_sizes, sf_strides_list, sf_offsets_list)
    ]

    return fp8_full, fp8_chunks, sf_chunks


@triton.jit
def _per_token_dequant_from_fp8_kernel(
    x_fp8_ptr,
    sf_ptr,
    out_ptr,
    m,
    n,
    stride_fp8_m,
    stride_sf_m,
    stride_sf_k,
    stride_out_m,
    BLOCK_M: tl.constexpr,
    gran_k: tl.constexpr,
):
    """Inverse of _per_token_cast_to_fp8_kernel.

    Each program handles one group (gran_k columns) for BLOCK_M rows.
    out[i, j] = x_fp8[i, j].to(fp32) * sf[i, j // gran_k]
    """
    group_idx = tl.program_id(0)
    row_start = tl.program_id(1) * BLOCK_M

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = group_idx * gran_k + tl.arange(0, gran_k)

    fp8_offsets = rows[:, None] * stride_fp8_m + cols[None, :]
    mask = (rows[:, None] < m) & (cols[None, :] < n)
    x_fp8 = tl.load(x_fp8_ptr + fp8_offsets, mask=mask, other=0.0).to(tl.float32)

    sf = tl.load(sf_ptr + rows * stride_sf_m + group_idx * stride_sf_k, mask=(rows < m), other=0.0)

    out = x_fp8 * sf[:, None]

    out_offsets = rows[:, None] * stride_out_m + cols[None, :]
    tl.store(out_ptr + out_offsets, out, mask=mask)

@torch._dynamo.allow_in_graph
def per_token_dequant_from_fp8(
    x_fp8: torch.Tensor,
    sf: torch.Tensor,
    gran_k: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Triton dequantization: inverse of ``per_token_cast_to_fp8``.

    Args:
        x_fp8: FP8 tensor of shape ``(m, n)`` in ``torch.float8_e4m3fn``.
        sf: Scale factors of shape ``(m, num_groups)`` in fp32, where
            ``num_groups == ceil(n / gran_k)``. Must NOT be packed (callers
            using ``use_packed_ue8m0=True`` need to unpack first).
        gran_k: Group size used at quantization time.
        out_dtype: Output dtype (default bfloat16).

    Returns:
        Dequantized tensor of shape ``(m, n)`` in ``out_dtype``.
    """
    assert x_fp8.dim() == 2, f"Expected 2D input, got shape {x_fp8.shape}"
    assert sf.dtype == torch.float32, f"sf must be fp32 (unpacked), got {sf.dtype}"
    m, n = x_fp8.shape
    padded_n = align(n, gran_k)
    num_groups = padded_n // gran_k
    assert sf.shape == (m, num_groups), \
        f"Expected sf shape ({m}, {num_groups}), got {sf.shape}"

    out = torch.empty((m, n), dtype=out_dtype, device=x_fp8.device)

    BLOCK_M = 4
    grid = (num_groups, triton.cdiv(m, BLOCK_M))

    _per_token_dequant_from_fp8_kernel[grid](
        x_fp8, sf, out,
        m, n,
        x_fp8.stride(0), sf.stride(0), sf.stride(1), out.stride(0),
        BLOCK_M=BLOCK_M,
        gran_k=gran_k,
    )

    return out


@torch._dynamo.allow_in_graph
def per_token_dequant_from_fp8_chunked(
    x_fp8: torch.Tensor,
    sf_chunks: list,
    chunk_sizes: list,
    gran_k: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Per-chunk dequant. Each ``sf_chunks[i]`` is shape
    ``(chunk_sizes[i], num_groups)`` with arbitrary 2D strides.
    """
    assert x_fp8.dim() == 2
    m, n = x_fp8.shape
    padded_n = align(n, gran_k)
    num_groups = padded_n // gran_k
    assert sum(chunk_sizes) == m

    out = torch.empty((m, n), dtype=out_dtype, device=x_fp8.device)
    fp8_chunks = list(torch.split(x_fp8, chunk_sizes, dim=0))
    out_chunks = list(torch.split(out, chunk_sizes, dim=0))

    BLOCK_M = 4
    for cm, fp8c, sfc, outc in zip(chunk_sizes, fp8_chunks, sf_chunks, out_chunks):
        assert sfc.shape == (cm, num_groups)
        grid = (num_groups, triton.cdiv(cm, BLOCK_M))
        _per_token_dequant_from_fp8_kernel[grid](
            fp8c, sfc, outc,
            cm, n,
            fp8c.stride(0), sfc.stride(0), sfc.stride(1), outc.stride(0),
            BLOCK_M=BLOCK_M,
            gran_k=gran_k,
        )
    return out


@triton.jit
def _per_channel_cast_to_fp8_kernel(
    x_ptr,
    x_fp8_ptr,
    sf_ptr,
    m,
    n,
    stride_m,
    stride_fp8_outer,
    stride_sf_outer,
    num_row_groups,
    BLOCK_N: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    gran_k: tl.constexpr,
    use_ue8m0: tl.constexpr,
    TRANSPOSE: tl.constexpr,
):
    row_group = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    col_mask = cols < n

    # Phase 1: compute per-column abs-max across the group of gran_k rows.
    # Init to 0 (not a 1e-4 floor): wgrad operands are routinely ~1e-9 and a
    # coarse amax floor would collapse the per-channel scale. The scale is
    # floored just above the fp32 reciprocal cliff after the division below,
    # matching _pack_kmajor_per_channel_fp8_kernel so the fused and unfused
    # paths stay numerically identical.
    row_amax = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for r in range(0, gran_k, BLOCK_ROWS):
        rows = row_group * gran_k + r + tl.arange(0, BLOCK_ROWS)
        offsets = rows[:, None] * stride_m + cols[None, :]
        mask = (rows[:, None] < m) & col_mask[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_amax = tl.maximum(row_amax, tl.max(tl.abs(x), axis=0))

    # Compute scale factor. Floor sf so inv_sf = 1/sf cannot overflow to +inf
    # (a denormal sf otherwise saturates the column to 448 and turns exact zeros
    # into 0*inf = NaN). See _pack_kmajor_per_channel_fp8_kernel for details.
    sf = row_amax / 448.0
    sf = tl.where(sf == 0.0, 1.0, sf)
    sf = tl.maximum(sf, 1e-30)
    if use_ue8m0:
        sf = _ceil_to_ue8m0(sf)

    # Phase 2: scale and store FP8 output
    inv_sf = 1.0 / sf[None, :]
    for r in range(0, gran_k, BLOCK_ROWS):
        rows = row_group * gran_k + r + tl.arange(0, BLOCK_ROWS)
        offsets = rows[:, None] * stride_m + cols[None, :]
        mask = (rows[:, None] < m) & col_mask[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x_scaled = x * inv_sf
        # x_scaled = tl.clamp(x_scaled, -448.0, 448.0).to(tl.float8e4nv)
        if TRANSPOSE:
            # out shape (n, m): out[col, row] = x_scaled[row, col]
            fp8_offsets = cols[None, :] * stride_fp8_outer + rows[:, None]
        else:
            # out shape (m, n): out[row, col] = x_scaled[row, col]
            fp8_offsets = rows[:, None] * stride_fp8_outer + cols[None, :]
        tl.store(x_fp8_ptr + fp8_offsets, x_scaled.to(tl.float8e4nv), mask=mask)

    # Store scale factors
    if TRANSPOSE:
        # sf shape (n, num_row_groups): sf_T[col, row_group] = sf[col]
        tl.store(sf_ptr + cols * stride_sf_outer + row_group, sf, mask=col_mask)
    else:
        # sf shape (num_row_groups, n): sf[row_group, col] = sf[col]
        tl.store(sf_ptr + row_group * stride_sf_outer + cols, sf, mask=col_mask)

@torch._dynamo.allow_in_graph
def per_channel_cast_to_fp8(
    x: torch.Tensor,
    use_ue8m0: bool,
    gran_k: int = 128,
    transpose: bool = False,
    pack_kmajor: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton implementation of per_channel_cast_to_fp8.

    Quantizes a 2D tensor to FP8.  Rows are partitioned into groups of
    ``gran_k``; within each group every column gets its own scale factor,
    shared across all rows in the group.

    Args:
        x: Input tensor of shape ``(m, n)`` where ``m % gran_k == 0``.
        use_ue8m0: Encode scale factors as UE8M0.
        gran_k: Number of rows per group.
        transpose: If True, both the FP8 output and the scale factors are
            written in transposed layout: x_fp8 shape ``(n, m)`` and sf shape
            ``(n, m // gran_k)``. Otherwise output shape is ``(m, n)`` and sf
            shape is ``(m // gran_k, n)``.

    Returns:
        x_fp8: Quantized tensor in ``torch.float8_e4m3fn``. Shape is
            ``(n, m)`` if ``transpose`` else ``(m, n)``.
        sf: Scale factors in fp32. Shape is ``(n, m // gran_k)`` if
            ``transpose`` else ``(m // gran_k, n)``.
    """
    assert x.dim() == 2 and x.size(0) % gran_k == 0, \
        f"Expected 2D input with m % {gran_k} == 0, got shape {x.shape}"
    m, n = x.shape
    num_row_groups = m // gran_k

    if transpose:
        x_fp8 = torch.empty((n, m), dtype=torch.float8_e4m3fn, device=x.device)
        sf = torch.empty((n, num_row_groups), dtype=torch.float32, device=x.device)
    else:
        x_fp8 = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
        sf = torch.empty((num_row_groups, n), dtype=torch.float32, device=x.device)

    BLOCK_N = 64
    BLOCK_ROWS = 16
    grid = (num_row_groups, triton.cdiv(n, BLOCK_N))

    _per_channel_cast_to_fp8_kernel[grid](
        x, x_fp8, sf,
        m, n,
        x.stride(0), x_fp8.stride(0), sf.stride(0),
        num_row_groups,
        BLOCK_N=BLOCK_N,
        BLOCK_ROWS=BLOCK_ROWS,
        gran_k=gran_k,
        use_ue8m0=use_ue8m0,
        TRANSPOSE=transpose,
    )

    return x_fp8, sf


@triton.jit
def _pack_kmajor_kernel(
    a_ptr, out_ptr,
    prefixes_ptr, ks_ptr,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    g = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    prefix = tl.load(prefixes_ptr + g)
    K_g = tl.load(ks_ptr + g)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K_g

    a_ptrs = a_ptr + (prefix + offs_k)[:, None] * M + offs_m[None, :]
    tile = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :])

    out_base = out_ptr + prefix * M
    out_ptrs = out_base + offs_m[:, None] * K_g + offs_k[None, :]
    tl.store(out_ptrs, tl.trans(tile),
             mask=mask_m[:, None] & mask_k[None, :])

@torch._dynamo.allow_in_graph
def pack_to_kmajor(
    a: torch.Tensor, 
    ks: list[int], 
    ks_tensor: torch.Tensor,
    prefixes: torch.Tensor
):
    K_total, M = a.shape                                                                                                               
    assert a.is_contiguous(), f"pack_to_kmajor: a not contiguous, strides={a.stride()}, shape={a.shape}"                               
    assert a.dim() == 2                                                                                                                
    assert K_total == sum(ks), f"K_total={K_total} but sum(ks)={sum(ks)}"                                                              
    assert ks_tensor.numel() == len(ks)                                                                                              
    assert ks_tensor.dtype in (torch.int32, torch.int64)
    out = torch.empty(K_total * M, dtype=a.dtype, device=a.device)
    
    BLOCK_M=64
    BLOCK_K=64
    num_warps=4
    grid = (len(ks), triton.cdiv(M, BLOCK_M), triton.cdiv(max(ks), BLOCK_K))
    _pack_kmajor_kernel[grid](a, out, prefixes, ks_tensor, M,
                              BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
                              num_warps=num_warps)
    a.data.untyped_storage().resize_(0)
    return out


@triton.jit
def _pack_kmajor_per_channel_fp8_kernel(
    x_ptr,            # bf16/fp16/fp32 input, shape (M_total, N), contiguous
    out_ptr,          # fp8 flat output buffer, size M_total * N
    sf_ptr,           # fp32 scale factors, shape (M_total // gran_k, N), contiguous
    prefixes_ptr,     # int (num_experts,): cumulative row offsets
    ks_ptr,           # int (num_experts,): rows per expert
    N,
    stride_xm,
    BLOCK_N: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    gran_k: tl.constexpr,
    use_ue8m0: tl.constexpr,
):
    """Fused per_channel_cast_to_fp8 + pack_to_kmajor.

    Per expert g (rows [prefix[g] : prefix[g]+K_g)), within each gran_k-row group:
      - phase 1: per-column abs-max over gran_k rows → scale factor
      - phase 2: scale, cast to fp8, store transposed into expert's packed slab
                 (layout (N, K_g) at byte offset prefix[g]*N elements).

    Caller-side preconditions:
      - K_g % gran_k == 0 for every expert.
      - x is contiguous, M_total % gran_k == 0.
    """
    g = tl.program_id(0)
    pid_n = tl.program_id(1)
    rg = tl.program_id(2)  # row-group index within the expert

    K_g = tl.load(ks_ptr + g)
    # Skip programs past this expert's row-group count.
    if rg * gran_k >= K_g:
        return

    prefix = tl.load(prefixes_ptr + g)

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    base_row = prefix + rg * gran_k

    # Phase 1: per-column abs-max across the gran_k rows of this group.
    # Do NOT floor amax at a coarse value like 1e-4: the offloading wgrad
    # operands (grads / swiglu activations) are routinely ~1e-9, with near-dead
    # channels far below that, and a coarse floor collapses the per-channel scale
    # and discards most of the FP8 dynamic range. The scale is instead floored
    # just above the fp32 reciprocal cliff after the division below.
    row_amax = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for r in range(0, gran_k, BLOCK_ROWS):
        rows = base_row + r + tl.arange(0, BLOCK_ROWS)
        offs = rows[:, None] * stride_xm + cols[None, :]
        x = tl.load(x_ptr + offs, mask=col_mask[None, :], other=0.0).to(tl.float32)
        row_amax = tl.maximum(row_amax, tl.max(tl.abs(x), axis=0))

    sf = row_amax / 448.0
    sf = tl.where(sf == 0.0, 1.0, sf)
    # Guard the fp32 reciprocal. If a channel's amax is tiny but nonzero, sf
    # underflows to a denormal and inv_sf = 1/sf overflows to +inf. After that
    # the amax element saturates to 448 (the whole per-channel spread collapses
    # to the max) and any exact zero in the column -- e.g. a padded token --
    # becomes 0*inf = NaN. Both silently corrupt the k-grouped wgrad GEMM and
    # surface as an exploding grad norm. The 1e-30 floor keeps inv_sf finite and
    # normal while only touching channels whose amax < ~4.5e-28, i.e. ~18 orders
    # of magnitude below the training activation scale (negligible).
    sf = tl.maximum(sf, 1e-7)
    if use_ue8m0:
        sf = _ceil_to_ue8m0(sf)
    inv_sf = 1.0 / sf

    # Phase 2: rescale, cast to fp8, store as transposed tile into the
    # expert's packed (N, K_g) slab.
    out_base = out_ptr + prefix * N
    for r in range(0, gran_k, BLOCK_ROWS):
        rows = base_row + r + tl.arange(0, BLOCK_ROWS)
        rows_local = rg * gran_k + r + tl.arange(0, BLOCK_ROWS)
        offs = rows[:, None] * stride_xm + cols[None, :]
        x = tl.load(x_ptr + offs, mask=col_mask[None, :], other=0.0).to(tl.float32)
        x_scaled = x * inv_sf[None, :]
        x_fp8 = x_scaled.to(tl.float8e4nv)                  # (BLOCK_ROWS, BLOCK_N)
        x_fp8_t = tl.trans(x_fp8)                           # (BLOCK_N, BLOCK_ROWS)
        out_offs = cols[:, None] * K_g + rows_local[None, :]
        tl.store(out_base + out_offs, x_fp8_t,
                 mask=col_mask[:, None])

    # Scale factors: write in (M_total // gran_k, N) layout; caller returns .T view.
    global_rg = base_row // gran_k
    tl.store(sf_ptr + global_rg * N + cols, sf, mask=col_mask)


@torch._dynamo.allow_in_graph
def per_channel_cast_to_fp8_pack_kmajor(
    x: torch.Tensor,
    ks: list[int],
    ks_tensor: torch.Tensor,
    prefixes: torch.Tensor,
    use_ue8m0: bool = False,
    gran_k: int = 128,
    free_input: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused ``per_channel_cast_to_fp8(transpose=False)`` + ``pack_to_kmajor``.

    Equivalent to::

        fp8, sf = per_channel_cast_to_fp8(x, use_ue8m0, gran_k, transpose=False)
        fp8_packed = pack_to_kmajor(fp8, ks, ks_tensor, prefixes)
        return fp8_packed, sf.T

    but eliminates the intermediate ``(M_total, N)`` fp8 staging buffer and the
    second HBM round-trip.

    Args:
        x: Input tensor of shape ``(M_total, N)``. Must be contiguous.
            ``M_total == sum(ks)`` and ``M_total % gran_k == 0``.
        ks: Per-expert row counts. Each ``ks[i]`` MUST be a multiple of
            ``gran_k`` so that scale row-groups never straddle expert
            boundaries.
        ks_tensor: ``ks`` as an int32/int64 CUDA tensor.
        prefixes: int32/int64 CUDA tensor of cumulative row offsets
            (``[0, ks[0], ks[0]+ks[1], ...]``).
        use_ue8m0: Encode scale factors as UE8M0.
        gran_k: Number of rows per scale-factor group.
        free_input: If True, resize ``x``'s storage to 0 after launch (matches
            the eager-free behavior of the legacy ``pack_to_kmajor``).

    Returns:
        out: ``(M_total * N,)`` flat fp8 buffer, per-expert k-major packed.
        sf: ``(N, M_total // gran_k)`` fp32 view (transposed).
    """
    assert x.dim() == 2 and x.is_cuda, f"expected 2D CUDA tensor, got {x.shape}"
    assert x.is_contiguous(), f"x must be contiguous, strides={x.stride()}"
    M_total, N = x.shape
    assert M_total % gran_k == 0, f"M_total={M_total} not a multiple of gran_k={gran_k}"
    assert ks_tensor.numel() == len(ks)
    assert ks_tensor.dtype in (torch.int32, torch.int64)
    assert prefixes.dtype in (torch.int32, torch.int64)
    assert sum(ks) == M_total, f"sum(ks)={sum(ks)} != M_total={M_total}"
    for i, k in enumerate(ks):
        assert k % gran_k == 0, (
            f"ks[{i}]={k} must be a multiple of gran_k={gran_k} "
            f"(scale groups cannot straddle expert boundaries)"
        )

    num_experts = len(ks)
    num_row_groups_total = M_total // gran_k
    max_ks = max(ks) if num_experts > 0 else 0
    max_row_groups = max_ks // gran_k

    out = torch.empty(M_total * N, dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((num_row_groups_total, N), dtype=torch.float32, device=x.device)

    if max_row_groups == 0:
        return out, sf.T

    # NOTE on tuning: the transposed fp8 store writes BLOCK_ROWS contiguous
    # bytes per row of the output tile (stride K_g separates rows). Small
    # BLOCK_ROWS leaves stores well below a memory transaction and tanks
    # throughput. Empirically BLOCK_ROWS=64 matches pack_to_kmajor's tile
    # size and saturates HBM. Phase-1 amax is unaffected by BLOCK_ROWS.
    BLOCK_N = 64
    BLOCK_ROWS = 64
    num_warps = 4
    grid = (num_experts, triton.cdiv(N, BLOCK_N), max_row_groups)
    _pack_kmajor_per_channel_fp8_kernel[grid](
        x, out, sf,
        prefixes, ks_tensor,
        N,
        x.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_ROWS=BLOCK_ROWS,
        gran_k=gran_k,
        use_ue8m0=use_ue8m0,
        num_warps=num_warps,
    )

    if free_input:
        x.data.untyped_storage().resize_(0)

    return out, sf.T