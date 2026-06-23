import torch
import deep_gemm

_KS = [256, 128, 128, 256]
_HIDDEN = 2048   # n: grad_y channels / wgrad rows
_FFN = 2048      # H: activation channels / wgrad cols
_AMAX_FLOOR = 0

def _per_channel_cast_to_fp8(x, gran_k=128, sf_val=1e-30):
    m, n = x.shape
    x_view = x.view(-1, gran_k, n)
    amax = x_view.abs().float().amax(dim=1).view(-1, n)
    if _AMAX_FLOOR > 0:
        amax = amax.clamp(_AMAX_FLOOR)
    sf = amax / 448.0

    # NOTE: handle cases when the channel contains all zeros
    sf = torch.where(sf==0, torch.full_like(sf, sf_val), sf)
    fp8 = (x_view * (1.0 / sf.unsqueeze(1))).to(torch.float8_e4m3fn).view(m, n)
    return fp8, sf

def _pack_kmajor(fp8, ks, mcols):
    """Per-expert k-major pack, mirroring generators.py::generate_k_grouped_contiguous."""
    out = torch.empty(sum(ks) * mcols, dtype=fp8.dtype, device=fp8.device)
    prefix = 0
    for k in ks:
        out[prefix * mcols:(prefix + k) * mcols] = fp8[prefix:prefix + k].T.flatten()
        prefix += k
    return out

def _make_operand(x, ks, sf_val=1e-30):
    fp8, sf = _per_channel_cast_to_fp8(x, 128, sf_val)
    return _pack_kmajor(fp8, ks, x.shape[1]), sf.T


def _build_operands(device, seed=0):
    """Synthetic grad_y / activation with the minimal trigger structure."""
    g = torch.Generator(device=device).manual_seed(seed)
    m = sum(_KS)
    grad_y = torch.randn(m, _HIDDEN, device=device, dtype=torch.bfloat16, generator=g) * 1e-7
    s = torch.randn(m, _FFN, device=device, dtype=torch.bfloat16, generator=g) * 0.4

    offsets, off = [], 0
    for k in _KS:
        offsets.append(off)
        off += k
    
    # NOTE: setup one all-zero channel for 128-token expert
    for e, k in enumerate(_KS):
        if k == 128:
            grad_y[offsets[e]:offsets[e] + k, 0] = 0
    return grad_y, s, offsets

def diff_tensor_norm(t1, t2):
    t1f = t1.to(torch.float)
    t2f = t2.to(torch.float)
    return torch.norm(t1f - t2f).item() / torch.norm(t2f).item()

def test_k_grouped_fp8_gemm_is_deterministic(sf_val):
    """The grouped FP8 wgrad GEMM must return the same correct result every call."""
    device = torch.cuda.current_device()
    iters = 10000
    num_experts = len(_KS)

    grad_y, s, offsets = _build_operands(device)
    grouped_layout = torch.tensor(_KS, dtype=torch.int32, device=device)

    # per-channel cast
    fp8_grad_y = _make_operand(grad_y, _KS, sf_val)
    fp8_s = _make_operand(s, _KS, sf_val)
    torch.cuda.synchronize()

    # reference wgrad
    ref = torch.zeros(num_experts, _HIDDEN, _FFN, device=device, dtype=torch.float32)
    for e, k in enumerate(_KS):
        if k == 0:
            continue
        gy_e = grad_y[offsets[e]:offsets[e] + k].float()
        s_e = s[offsets[e]:offsets[e] + k].float()
        ref[e] = gy_e.t() @ s_e
    ref_norms = torch.linalg.vector_norm(ref.reshape(num_experts, -1), dim=1)
    thresholds = (100.0 * ref_norms).clamp_min(1.0)  # blow-up = >=100x the true grad

    failures = []
    for it in range(iters):
        out = torch.zeros(num_experts, _HIDDEN, _FFN, device=device, dtype=torch.float32)
        torch.cuda.synchronize()
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            fp8_grad_y, fp8_s, out, _KS, grouped_layout, out, recipe=(1, 1, 128),
            use_psum_layout=False,
        )
        torch.cuda.synchronize()
        out_norms = torch.linalg.vector_norm(out.reshape(num_experts, -1), dim=1)

        diff = diff_tensor_norm(out, ref) 
        bad = torch.nonzero(out_norms > thresholds).flatten().tolist()
        if diff > 0.06:
            print(f"iter {it}: diff={diff:.3e} > 0.06")
            failures.append((it, bad, out_norms.max().item(), diff))

    if failures:
        first_it, experts, worst = failures[0]
        print(
            f"\nk_grouped_fp8_gemm_nt_contiguous is nondeterministic: "
            f"{len(failures)}/{iters} calls blew up (first at iter {first_it}, "
            f"experts {experts}); worst output norm {worst:.3e} vs reference max "
            f"{ref_norms.max().item():.3e}. Operands were cast once and are identical "
            f"across calls (single stream, fully synced); amax_floor={_AMAX_FLOOR} sf_val={sf_val}."
        )

if __name__ == "__main__":
    for sf_val in [1e-4]:
        test_k_grouped_fp8_gemm_is_deterministic(sf_val)