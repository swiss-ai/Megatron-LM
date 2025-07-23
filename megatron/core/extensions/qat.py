import torch

from .quantization_fns.mxfp4_quest import QuestMXFP4QuantizerFn
from .quantization_fns.binary_quest import QuestBinaryQuantizerFn


class NoQuantizerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hadamard_matrix):
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


QUANTIZE_AUTOGRAD_FNS = {
    "none": NoQuantizerFn,
    "quest_mxfp4": QuestMXFP4QuantizerFn,
    "quest_binary": QuestBinaryQuantizerFn,
}