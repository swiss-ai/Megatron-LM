# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.experts_offloading_util import (
    StreamManager,
    offloading_grouped_swiglu_mlp,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class TestOffloadingMoELayerBF16:
    """Test MoE layer with bf16 precision and CPU weight offloading."""

    @pytest.mark.parametrize("num_moe_experts", [32])
    def test_offloading_moe_forward_backward(
        self, num_moe_experts, profile=False
    ):
        """Test MoE layer forward and backward pass with bf16 params and inputs."""
        hidden_size = 7168
        moe_ffn_hidden_size = 2048
        moe_offloading_num_chunks = 4
        moe_offloading_num_stages = 2
        moe_offloading_chunk_size = num_moe_experts // moe_offloading_num_chunks

        tokens_per_expert = torch.randint(
            511, 512, (num_moe_experts,), device="cpu"
        )
        tokens_per_expert_ref = tokens_per_expert.detach().clone()

        hidden_states = torch.randn(
            tokens_per_expert.sum().item(),
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        hidden_states_ref = hidden_states.detach().clone()

        permuted_probs = torch.rand(
            tokens_per_expert.sum().item(),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        permuted_probs_ref = permuted_probs.detach().clone()

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_moe_experts=num_moe_experts,
            num_attention_heads=16,
            use_cpu_initialization=True,
            perform_initialization=False,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            add_bias_linear=False,
            fp16=False,
            params_dtype=torch.bfloat16,
            gated_linear_unit=True,
            moe_offloading_num_chunks=moe_offloading_num_chunks,
            moe_offloading_num_stages=moe_offloading_num_stages,
            moe_offloading_chunk_size=moe_offloading_chunk_size,
            gradient_accumulation_fusion=True,
        )
        transformer_config.moe_use_fp8_activation = False

        cpu_w1 = [
            torch.nn.Parameter(
                torch.randn(
                    hidden_size, moe_ffn_hidden_size * 2,
                    device="cpu", dtype=torch.bfloat16,
                    requires_grad=True, pin_memory=True,
                )
            )
            for _ in range(num_moe_experts)
        ]
        cpu_w2 = [
            torch.nn.Parameter(
                torch.randn(
                    moe_ffn_hidden_size, hidden_size,
                    device="cpu", dtype=torch.bfloat16,
                    requires_grad=True, pin_memory=True,
                )
            )
            for _ in range(num_moe_experts)
        ]

        for i in range(num_moe_experts):
            cpu_w1[i].main_grad = torch.zeros_like(cpu_w1[i], device="cuda", dtype=torch.float32)
            cpu_w2[i].main_grad = torch.zeros_like(cpu_w2[i], device="cuda", dtype=torch.float32)

        gpu_w1_buffer = [
            [torch.empty(
                hidden_size, moe_ffn_hidden_size * 2,
                device="cuda", dtype=torch.bfloat16, requires_grad=False,
            ) for _ in range(moe_offloading_chunk_size)]
            for _ in range(moe_offloading_num_stages)
        ]
        gpu_w2_buffer = [
            [torch.empty(
                moe_ffn_hidden_size, hidden_size,
                device="cuda", dtype=torch.bfloat16, requires_grad=False,
            ) for _ in range(moe_offloading_chunk_size)]
            for _ in range(moe_offloading_num_stages)
        ]

        # Reference weights on GPU (bf16, no offloading)
        gpu_w1 = [
            torch.nn.Parameter(cpu_w1[i].detach().clone().cuda(), requires_grad=True)
            for i in range(num_moe_experts)
        ]
        gpu_w2 = [
            torch.nn.Parameter(cpu_w2[i].detach().clone().cuda(), requires_grad=True)
            for i in range(num_moe_experts)
        ]
        for i in range(num_moe_experts):
            gpu_w1[i].main_grad = torch.zeros_like(gpu_w1[i], device="cuda", dtype=torch.float32)
            gpu_w2[i].main_grad = torch.zeros_like(gpu_w2[i], device="cuda", dtype=torch.float32)

        torch.cuda.synchronize()

        stream_manager = StreamManager(moe_offloading_num_stages)

        # --- Profiling path (manual use only) ---
        if profile:
            wait, warmup, active = 1, 5, 2
            num_steps = wait + warmup + active
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=wait, warmup=warmup, active=active, repeat=1, skip_first=1
                ),
            ) as prof:
                for _ in range(num_steps):
                    output = offloading_grouped_swiglu_mlp(
                        cpu_w1, cpu_w2, gpu_w1_buffer, gpu_w2_buffer,
                        hidden_states, tokens_per_expert, num_moe_experts,
                        permuted_probs, None, stream_manager, transformer_config, [],
                    )
                    torch.cuda.synchronize()
                    output.sum().backward()
                    torch.cuda.synchronize()
                    prof.step()
            prof.export_chrome_trace("e2e.json")
            return

        # --- Reference path (bf16 matmuls on GPU) ---
        hidden_states_ref_list = list(
            torch.split(hidden_states_ref, tokens_per_expert_ref.tolist(), dim=0)
        )
        fc1_outputs = [
            torch.mm(hidden_states_ref_list[i], gpu_w1[i])
            for i in range(num_moe_experts)
        ]
        act_output = MergedSwiGLU.apply(
            torch.cat(fc1_outputs, dim=0), permuted_probs_ref.unsqueeze(-1)
        )
        act_list = list(torch.split(act_output, tokens_per_expert_ref.tolist(), dim=0))
        output_ref = torch.cat([
            torch.mm(act_list[i], gpu_w2[i])
            for i in range(num_moe_experts)
        ], dim=0)
        output_ref.sum().backward()
        torch.cuda.synchronize()

        # --- Offloading path (bf16 with CPU offload) ---
        output = offloading_grouped_swiglu_mlp(
            cpu_w1, cpu_w2, gpu_w1_buffer, gpu_w2_buffer,
            hidden_states, tokens_per_expert, num_moe_experts,
            permuted_probs, None, stream_manager, transformer_config, [],
        )
        output.sum().backward()
        torch.cuda.synchronize()

        # --- Comparison ---
        def diff_tensor_norm(tensor1, tensor2):
            return (
                torch.norm(tensor1.to(torch.float) - tensor2.to(torch.float)).item()
                / torch.norm(tensor2.to(torch.float)).item()
            )

        assert output.shape == output_ref.shape, (
            f"Output shape mismatch: {output.shape} vs {output_ref.shape}"
        )
        assert torch.allclose(output, output_ref, atol=1e-3), (
            f"Output mismatch: norm diff = {diff_tensor_norm(output, output_ref):.6f}"
        )

        for i in range(num_moe_experts):
            w1_diff = diff_tensor_norm(cpu_w1[i].main_grad, gpu_w1[i].grad)
            w2_diff = diff_tensor_norm(cpu_w2[i].main_grad, gpu_w2[i].grad)
            assert w1_diff < 0.01, (
                f"Expert {i}: W1 grad norm diff {w1_diff:.6f} exceeds threshold"
            )
            assert w2_diff < 0.01, (
                f"Expert {i}: W2 grad norm diff {w2_diff:.6f} exceeds threshold"
            )


if __name__ == "__main__":
    TestOffloadingMoELayerBF16().test_offloading_moe_forward_backward(
        num_moe_experts=64, profile=True
    )
