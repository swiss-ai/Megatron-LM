from typing import Tuple

import torch
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


class _VocabParallelMaxZ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, z_loss_weight):
        """
        Forward pass for parallel max-z loss calculation.
        
        Args:
            vocab_parallel_logits: logits split across tensor parallel ranks
                shape: [sequence_length, batch_size, vocab_size/num_parallel_ranks]
        """
        # Maximum value along vocab dimension across all GPUs
        global_logits_values = torch.max(vocab_parallel_logits, dim=-1)[0]
        
        # All-reduce to get max across all tensor parallel ranks
        torch.distributed.all_reduce(
            global_logits_values,
            op=torch.distributed.ReduceOp.MAX,
            group=get_tensor_model_parallel_group()
        )
                
        # Compute loss: Lmax-z = weight * z^2
        loss = z_loss_weight * (global_logits_values ** 2)
        
        # Save tensors needed for backward pass
        ctx.save_for_backward(vocab_parallel_logits, global_logits_values)
        ctx.z_loss_weight = z_loss_weight
        
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for parallel max-z loss calculation.
        
        Args:
            grad_output: gradient of the loss with respect to the output
        """
        vocab_parallel_logits, global_logits_values = ctx.saved_tensors
        z_loss_weight = ctx.z_loss_weight
        
        # Reshape global values for broadcasting
        resize_global_values = global_logits_values.unsqueeze(-1)
        
        # Compute gradient: Only elements that achieved the maximum contribute to gradient
        grad_input = (vocab_parallel_logits == resize_global_values) * 2 * z_loss_weight * resize_global_values
        
        # Scale gradient by incoming gradient
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        
        return grad_input, None


def vocab_parallel_max_z(vocab_parallel_logits, z_loss_weight: float) -> torch.Tensor:
    """
    Performs max-z loss when logits are split across tensor parallel ranks
    
    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
            dimension is [sequence_length, batch_size, hidden_size]
        z_loss_weight: weight for the max-z loss
            
    Returns:
        loss: computed max-z loss value of size [sequence_length, batch_size]
    """
    return _VocabParallelMaxZ.apply(vocab_parallel_logits, z_loss_weight)
