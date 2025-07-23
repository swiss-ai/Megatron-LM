import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32 * 128}),
        triton.Config({"BLOCK_SIZE": 64 * 128}),
        triton.Config({"BLOCK_SIZE": 128 * 128}),
    ],
    key=[],
)
@triton.jit
def binary_forward_kernel(
    x_ptr,
    hadamard_matrix_ptr,
    output_ptr,
    clip_mask_ptr,
    n_elements: tl.constexpr,
    hadamard_dim: tl.constexpr,
    gaussian_scale: tl.constexpr,
    clip_multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):    
    offsets_hadamard = tl.arange(0, hadamard_dim * hadamard_dim)
    hadamard_matrix = tl.load(hadamard_matrix_ptr + offsets_hadamard).reshape(hadamard_dim, hadamard_dim)
    
    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask)
    
    # hadamard transform
    x = tl.reshape(x_flat, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    x_had = tl.dot(x, hadamard_matrix)
    
    # group
    x_had_grouped = tl.reshape(x_had, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    
    # scale

    mean_squared = tl.sum(x_had_grouped * x_had_grouped, axis=-1, keep_dims=True) / hadamard_dim
    mean = tl.sum(x_had_grouped, axis=-1, keep_dims=True) / hadamard_dim
    std = tl.sqrt(mean_squared - mean * mean)
    scales = gaussian_scale * std + 1e-8

    
    # mask
    quest_mask = tl.reshape(x_had_grouped.abs() < clip_multiplier * scales, (BLOCK_SIZE,))
    tl.store(clip_mask_ptr + offsets, quest_mask, mask=mask)

    # dequantize
    x_dequantized = tl.where(x_had_grouped > 0, scales, -scales)
    
    # Reshape back to flat form for storage
    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))
    
    # store
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def binary_forward_kernel_wrapper(
    x,
    hadamard_matrix,
    gaussian_scale,
    clip_multiplier,
):    
    # Make sure inputs are contiguous
    x = x.contiguous()
    
    # Create output tensor
    output = torch.empty_like(x)
    clip_mask = torch.empty_like(x, dtype=torch.bool)
    
    # Get total number of elements and calculate grid for launching the kernel
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Launch optimized kernel
    binary_forward_kernel[grid](
        x_ptr=x,
        hadamard_matrix_ptr=hadamard_matrix,
        output_ptr=output,
        clip_mask_ptr=clip_mask,
        n_elements=n_elements,
        hadamard_dim=hadamard_matrix.shape[-1],
        gaussian_scale=gaussian_scale,
        clip_multiplier=clip_multiplier,
    )
    
    return output, clip_mask


class QuestBinaryQuantizerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hadamard_matrix):
        x_dequantized, mask = binary_forward_kernel_wrapper(
            x,
            hadamard_matrix,
            gaussian_scale=0.7978845587140913,
            clip_multiplier=1.5,
        )
        ctx.save_for_backward(hadamard_matrix, mask)
        ctx.x_shape = x.shape
        return x_dequantized
    
    @staticmethod
    def backward(ctx, grad_output):
        hadamard_matrix, mask = ctx.saved_tensors
        grad_input = (grad_output * mask.to(grad_output.dtype)).view(-1, hadamard_matrix.shape[0]) @ hadamard_matrix.T
        return grad_input.view(ctx.x_shape), None
