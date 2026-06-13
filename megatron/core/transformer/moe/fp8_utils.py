"""
FP8 functions interface for checkpointing and DeepGEMM calls in MoE layers.
"""
import torch
try:
    import deep_gemm
except ImportError:
    deep_gemm = None
import matplotlib.pyplot as plt

from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.transformer.moe.fp8_jit import (
    per_token_cast_to_fp8,
    per_token_dequant_from_fp8,
)

def diff_tensor_norm(t1, t2):
    t1f = t1.to(torch.float)
    t2f = t2.to(torch.float)
    return torch.norm(t1f - t2f).item() / torch.norm(t2f).item()

def plot_tensor_hist(tensor: torch.Tensor, name, bins=100, title="Tensor Value Distribution"):
    values = tensor.flatten().detach().cpu().float().numpy()
    mean = values.mean()
    var = values.var()
    std = values.std()

    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.75)

    # vertical lines for mean ± std
    plt.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.4e}")
    plt.axvline(mean + std, color="orange", linestyle=":", label=f"+1 Std: {mean+std:.4e}")
    plt.axvline(mean - std, color="orange", linestyle=":", label=f"−1 Std: {mean-std:.4e}")

    # stats box in the corner
    stats_text = f"Var: {var:.4e}\nStd: {std:.4e}\nMean: {mean:.4e}"
    plt.text(0.97, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc="upper left")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./{name}_histogram.png")

def build_offloading_expert_sharded_tensor(
    weight_slice: torch.Tensor,
    prefix: str,
    weight_name: str,
    global_expert_idx: int,
    *,
    sharded_offsets: tuple,
    num_global_experts: int,
    replica_id,
    singleton_local_shards: bool,
    transpose: bool,
):
    """Build the ShardedTensor for a single offloaded expert weight.

    Produces the *same* on-disk representation for both OffloadingExpertsMLP
    variants so their checkpoints are interchangeable:

    - bf16 variant: per-expert params stored as ``(in, out)`` -> ``transpose=False``
    - inplace-fp8 variant: fused master stored as ``(out, in)`` (NT layout)
      -> ``transpose=True`` to land on the same ``(in, out)`` checkpoint layout.

    The expert is expressed as a *prepended* axis fragment exactly like
    ``SequentialMLP``/``TEGroupedMLP`` (see ``apply_swiglu_sharded_factory``),
    i.e. ``prepend_axis_num == len(offsets)``, which is what makes the expert
    dimension compose cleanly with any pipeline/tensor ``sharded_offsets``.

    Args:
        weight_slice (torch.Tensor): one expert's 2D weight. ``(in, out)`` when
            ``transpose=False`` else ``(out, in)``.
        prefix (str): module prefix for the on-disk key.
        weight_name (str): ``'weight1'`` or ``'weight2'``.
        global_expert_idx (int): this expert's index in the global expert range.
        sharded_offsets (tuple): offsets inherited from parent modules (PP, ...).
        num_global_experts (int): total experts across the expert-parallel group.
        replica_id: ShardedTensor replica id (PP, TP, DP).
        singleton_local_shards (bool): when True each expert is saved under its own
            global key with no expert sharding axis.
        transpose (bool): transpose ``(out, in) -> (in, out)`` before saving.
    """
    data = weight_slice.transpose(0, 1).contiguous() if transpose else weight_slice
    if singleton_local_shards:
        key = f'{prefix}experts.{global_expert_idx}.{weight_name}'
        offsets = sharded_offsets
    else:
        key = f'{prefix}experts.{weight_name}'
        offsets = (
            *sharded_offsets,
            (len(sharded_offsets), global_expert_idx, num_global_experts),
        )
    return ShardedTensor.from_rank_offsets(
        key,
        data,
        *offsets,
        replica_id=replica_id,
        prepend_axis_num=len(offsets),
    )


def make_fused_experts_sharded_factory(
    fused_weight: torch.Tensor,
    prefix: str,
    weight_name: str,
    *,
    num_local_experts: int,
    local_expert_indices_offset: int,
    num_global_experts: int,
    sharded_offsets: tuple,
    replica_id,
    singleton_local_shards: bool,
):
    """ShardedTensorFactory for one fused inplace-fp8 expert weight.

    ``fused_weight`` is the single ``(num_local_experts, out, in)`` bf16 master
    parameter used by the inplace-fp8 ``OffloadingExpertsMLP``. The factory:

    - ``build_fn``: emits one per-expert ShardedTensor (transposed to the
      ``(in, out)`` checkpoint layout, sharded over the expert axis), so the
      on-disk tensor matches the per-expert bf16 checkpoint exactly.
    - ``merge_fn``: re-stacks the loaded per-expert ``(in, out)`` tensors back
      into the fused ``(num_local_experts, out, in)`` master on load.

    The factory is keyed by the real parameter name (``f'{prefix}{weight_name}'``)
    so ``module.load_state_dict`` maps the merged tensor back to the fused param.

    ``build_fn`` honors the ``key`` it is handed (``ShardedTensorFactory.build``
    passes ``self.key``) rather than rebuilding keys solely from the captured
    ``prefix``/``weight_name``. This is required because callers can re-key the
    factory in two ways:

    - VPP replaces the local layer prefix with the global checkpoint layer prefix.
    - A mixed-precision optimizer prepends its fp32 master-copy namespace.

    Deriving the emitted shard prefix from the runtime key handles both cases.
    """

    @torch.no_grad()
    def build_fn(key, t, rep_id, flattened_range):
        assert flattened_range is None, "flattening unsupported for offloaded experts"
        assert key.endswith(weight_name), (key, weight_name)
        runtime_prefix = key[: -len(weight_name)]
        shards = []
        for i in range(num_local_experts):
            global_expert_idx = local_expert_indices_offset + i
            sh_ten = build_offloading_expert_sharded_tensor(
                t[i],
                runtime_prefix,
                weight_name,
                global_expert_idx,
                sharded_offsets=sharded_offsets,
                num_global_experts=num_global_experts,
                replica_id=rep_id,
                singleton_local_shards=singleton_local_shards,
                transpose=True,
            )
            shards.append(sh_ten)
        return shards

    @torch.no_grad()
    def merge_fn(loaded_shards):
        # list of (in, out) -> fused (num_local_experts, out, in)
        return torch.stack([s.transpose(0, 1) for s in loaded_shards], dim=0).contiguous()

    return ShardedTensorFactory(
        f'{prefix}{weight_name}',
        fused_weight,
        build_fn,
        merge_fn,
        replica_id,
    )


def m_grouped_fp8_gemm_nt_contiguous(
    tokens_per_expert: torch.Tensor,
    fp8_a: tuple[torch.Tensor, torch.Tensor],
    fp8_b: tuple[torch.Tensor, torch.Tensor],
    compute_stream: torch.cuda.Stream,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """M-grouped contiguous FP8 GEMM for one MoE chunk.

    Computes Y[i] = A[i] @ B[expert_of(i)].T in a single kernel launch,
    replacing the per-expert multi-stream loop in multi_stream_fp8_gemm_nt.

    Args:
        tokens_per_expert: int tensor [G], rows assigned to each local expert.
            Each entry must be a multiple of
            deep_gemm.get_m_alignment_for_contiguous_layout() (128 on SM90).
            Permuted activations must be padded accordingly upstream.
        fp8_a: (a, sfa) where
            a:   [M, K]      fp8_e4m3, K-major contiguous
            sfa: [M, K//128] float32  (recipe (1, 128), per-token block-K scales)
        fp8_b: (b, sfb) where
            b:   [G, N, K]      fp8_e4m3, K-major contiguous (stacked weights)
            sfb: [G, N//128, K//128] float32  (recipe (128, 128) block scales)
        output: optional [M, N] bfloat16 destination. Allocated if None.
        m_indices: optional int32 [M] expert-id-per-row. Built from
            tokens_per_expert if not provided. Pass it in to avoid rebuilding
            across fc1/fc2 of the same chunk.

    Returns:
        output tensor [M, N] bfloat16.
    """
    assert deep_gemm is not None
    a, sfa = fp8_a
    b, sfb = fp8_b
    assert b.dim() == 3, f"b must be stacked [G, N, K], got shape {tuple(b.shape)}"
    assert sfb.dim() == 3, f"sfb must be stacked [G, N//128, K//128], got {tuple(sfb.shape)}"

    g, n, k = b.shape
    m = a.shape[0]
    assert a.shape[1] == k, f"K mismatch: a {a.shape[1]} vs b {k}"
    assert tokens_per_expert.numel() == g, \
        f"tokens_per_expert length {tokens_per_expert.numel()} != num_groups {g}"

    # align = deep_gemm.get_m_alignment_for_contiguous_layout()
    # assert (tokens_per_expert % align == 0).all(), \
    #     f"each tokens_per_expert entry must be a multiple of {align}"

    if output is None:
        output = torch.empty(m, n, dtype=torch.bfloat16, device=a.device)
    else:
        assert output.shape == (m, n) and output.dtype == torch.bfloat16

    with torch.cuda.stream(compute_stream):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a, sfa),
            (b, sfb),
            output,
            tokens_per_expert,
            recipe_a=(1, 128),
            recipe_b=(128, 128),
            disable_ue8m0_cast=True,
            use_psum_layout=True,
        )
    return output

def k_grouped_fp8_gemm_nt_contiguous(
    ks: list[int],
    ks_tensor: torch.Tensor,
    fp8_a: tuple[torch.Tensor, torch.Tensor],
    fp8_b: tuple[torch.Tensor, torch.Tensor],
    num_local_experts: int,
    compute_stream: torch.cuda.Stream,
    output: torch.Tensor,
):
    with torch.cuda.stream(compute_stream):
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            fp8_a,
            fp8_b,
            output,
            ks,
            ks_tensor,
            output,
            recipe=(1, 1, 128),
        )

    return output

def multi_stream_fp8_gemm_nt_1d1d(
    batch_sizes: torch.Tensor,
    fp8_a: tuple[torch.Tensor, torch.Tensor],
    fp8_b: tuple[torch.Tensor, torch.Tensor],
    compute_streams: list[torch.cuda.Stream],
    output: list[torch.Tensor],
    accumulate: bool = False,
):
    assert deep_gemm is not None
    batch_sizes_list = batch_sizes.tolist()
    
    num_gemms = len(batch_sizes_list)
    a, a_scales = fp8_a
    b, b_scales = fp8_b

    sliced_a = torch.split(a, batch_sizes_list, dim=1)
    sliced_b = torch.split(b, batch_sizes_list, dim=1)

    slice_sizes = (batch_sizes//128).tolist()
    sliced_a_scales = torch.split(a_scales, slice_sizes, dim=1)
    sliced_b_scales = torch.split(b_scales, slice_sizes, dim=1)

    for i in range(num_gemms):
        stream = compute_streams[i%len(compute_streams)]
        with torch.cuda.stream(stream):
            deep_gemm.fp8_gemm_nt(
                (sliced_a[i].contiguous(), sliced_a_scales[i].contiguous()),
                (sliced_b[i].contiguous(), sliced_b_scales[i].contiguous()),
                output[i],
                c=None if not accumulate else output[i],
                recipe=None,
                recipe_a=(1, 128),
                recipe_b=(128, 128) if not accumulate else (1, 128),
                disable_ue8m0_cast=True,
            )
    return output

class FP8CastFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        input,
    ):
        fp8_input, sf = per_token_cast_to_fp8(input, use_ue8m0=False)
        return fp8_input, sf

    @staticmethod
    def backward(
        ctx, 
        fp8_input, 
        sf,
    ):
        grad_input = per_token_dequant_from_fp8(fp8_input, sf)
        return grad_input
    
def fp8_cast(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return FP8CastFunction.apply(input)
    
class FP8DeCastFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        fp8_input, 
        sf,
    ):
        input = per_token_dequant_from_fp8(fp8_input, sf)
        return input

    @staticmethod
    def backward(
        ctx, 
        grad_input,
    ):
        fp8_grad_input, sf = per_token_cast_to_fp8(grad_input, use_ue8m0=False)
        return fp8_grad_input, sf
    
def fp8_decast(fp8_input: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    return FP8DeCastFunction.apply(fp8_input, sf)
