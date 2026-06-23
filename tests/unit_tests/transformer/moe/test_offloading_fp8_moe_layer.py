# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import glob
import os

import pytest
import torch

from megatron.core.transformer.moe.experts_util import MergedSwiGLU
from megatron.core.transformer.moe.experts_offloading_util import StreamManager
from megatron.core.transformer.moe.experts_fp8_util import ExpertsFP8GroupedSwiMLP
from megatron.core.transformer.moe.experts_offloading_fp8_util import (
    FP8ExpertsParameterManager,
    OffloadingExpertsFP8GroupedSwiMLP,
    OffloadingFP8Config,
    guarded_per_channel_cast_to_fp8_pack_kmajor,
    offloading_fp8_grouped_swiglu_mlp,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class TestOffloadingMoELayerFP8:
    """Test MoE layer with FP8 precision and CPU weight offloading."""

    @staticmethod
    def _tensor_with_mean_std_and_max(shape, std, max_value, device, dtype):
        """Population mean/std controlled tensor with a fixed positive maximum."""
        values = torch.randn(shape, device=device, dtype=torch.float32).flatten()
        n = values.numel()
        values[0] = max_value
        rest = values[1:]
        rest = rest - rest.mean()
        rest = rest / rest.square().mean().sqrt()
        rest_mean = -max_value / (n - 1)
        rest_var = (n * (std**2) - (max_value**2) - (n - 1) * (rest_mean**2)) / (n - 1)
        assert rest_var > 0
        values[1:] = rest_mean + rest_var**0.5 * rest
        return values.reshape(shape).to(dtype)

    def _run_offloading_moe_w2_grad_rel_l2_monitor(self):
        """Run the real offloading FP8 path and return the relative-L2 monitor."""
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)

        num_moe_experts = 4
        hidden_size = 256
        moe_ffn_hidden_size = 256
        moe_offloading_num_chunks = 2
        moe_offloading_num_stages = 2
        moe_offloading_chunk_size = num_moe_experts // moe_offloading_num_chunks
        tokens_per_expert = torch.tensor([128] * num_moe_experts, dtype=torch.int32)
        num_tokens = tokens_per_expert.sum().item()
        device = torch.cuda.current_device()

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_moe_experts=num_moe_experts,
            num_attention_heads=8,
            use_cpu_initialization=True,
            perform_initialization=False,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            add_bias_linear=False,
            fp16=False,
            params_dtype=torch.bfloat16,
            gated_linear_unit=True,
            moe_use_offloading_experts=True,
            moe_use_inplace_fp8_param=True,
            monitor_moe_activation_max=True,
            moe_offloading_num_chunks=moe_offloading_num_chunks,
            moe_offloading_num_stages=moe_offloading_num_stages,
            moe_offloading_chunk_size=moe_offloading_chunk_size,
            gradient_accumulation_fusion=True,
        )
        fp8_config = OffloadingFP8Config.from_transformer_config(transformer_config)

        hidden_states = torch.randn(
            num_tokens, hidden_size, device=device, dtype=torch.bfloat16
        ).requires_grad_(True)
        permuted_probs = torch.rand(num_tokens, device=device, dtype=torch.bfloat16)
        target = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)

        w1_init = (
            torch.randn(
                num_moe_experts,
                moe_ffn_hidden_size * 2,
                hidden_size,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.02
        )
        w2_init = (
            torch.randn(
                num_moe_experts,
                hidden_size,
                moe_ffn_hidden_size,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.02
        )
        cpu_w1 = torch.nn.Parameter(
            torch.empty_like(w1_init, device="cpu", pin_memory=True).copy_(w1_init),
            requires_grad=True,
        )
        cpu_w2 = torch.nn.Parameter(
            torch.empty_like(w2_init, device="cpu", pin_memory=True).copy_(w2_init),
            requires_grad=True,
        )
        cpu_w1_list = list(torch.unbind(cpu_w1, dim=0))
        cpu_w2_list = list(torch.unbind(cpu_w2, dim=0))
        cpu_w1.main_grad = torch.zeros_like(cpu_w1, device=device, dtype=torch.float32)
        cpu_w2.main_grad = torch.zeros_like(cpu_w2, device=device, dtype=torch.float32)

        experts1_gpu_buffers_storage = torch.empty(
            moe_offloading_num_stages,
            moe_offloading_chunk_size,
            moe_ffn_hidden_size * 2,
            hidden_size,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        experts2_gpu_buffers_storage = torch.empty(
            moe_offloading_num_stages,
            moe_offloading_chunk_size,
            hidden_size,
            moe_ffn_hidden_size,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        gpu_w1_buffer = [
            [experts1_gpu_buffers_storage[s, c] for c in range(moe_offloading_chunk_size)]
            for s in range(moe_offloading_num_stages)
        ]
        gpu_w2_buffer = [
            [experts2_gpu_buffers_storage[s, c] for c in range(moe_offloading_chunk_size)]
            for s in range(moe_offloading_num_stages)
        ]
        gpu_w1_chunks = [
            experts1_gpu_buffers_storage[s] for s in range(moe_offloading_num_stages)
        ]
        gpu_w2_chunks = [
            experts2_gpu_buffers_storage[s] for s in range(moe_offloading_num_stages)
        ]

        FP8ExpertsParameterManager.reset_instance()
        FP8ExpertsParameterManager.create_instance(config=fp8_config)
        stream_manager = StreamManager(moe_offloading_num_stages, 1)
        activation_monitor = {}

        output = offloading_fp8_grouped_swiglu_mlp(
            cpu_w1,
            cpu_w2,
            cpu_w1_list,
            cpu_w2_list,
            gpu_w1_buffer,
            gpu_w2_buffer,
            gpu_w1_chunks,
            gpu_w2_chunks,
            hidden_states,
            tokens_per_expert,
            num_moe_experts,
            permuted_probs,
            None,
            stream_manager,
            fp8_config,
            [],
            activation_monitor=activation_monitor,
        )
        ((output - target).float() ** 2).mean().backward()
        torch.cuda.synchronize()

        return activation_monitor

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_offloading_moe_w2_grad_rel_l2_monitor(self):
        """Run the real offloading FP8 path and check the w2 grad relative-L2 monitor."""
        activation_monitor = self._run_offloading_moe_w2_grad_rel_l2_monitor()
        rel_l2 = activation_monitor["max_w2_grad_ref_fp8_rel_l2"]
        print(f"max_w2_grad_ref_fp8_rel_l2={rel_l2.item():.8f}")
        assert torch.isfinite(rel_l2)
        assert rel_l2 < 0.08

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_offloading_moe_w2_grad_rel_l2_dump_on_threshold(self, tmp_path, monkeypatch):
        """Force the abort path and load the saved grad_y/fc1_output tensors."""
        monkeypatch.setenv("MEGATRON_MOE_W2_GRAD_REL_L2_ABORT_THRESHOLD", "1e-6")
        monkeypatch.setenv("MEGATRON_MOE_W2_GRAD_REL_L2_DUMP_DIR", str(tmp_path))

        with pytest.raises(RuntimeError, match="dumped grad_y/fc1_output"):
            self._run_offloading_moe_w2_grad_rel_l2_monitor()
        torch.cuda.synchronize()

        dumps = list(tmp_path.glob("w2_grad_rel_l2_rank*.pt"))
        assert len(dumps) == 1

        payload = torch.load(dumps[0], map_location="cpu")
        assert payload["rel_l2"].item() > 1e-6
        assert payload["threshold"] == 1e-6
        assert payload["grad_y"].shape == (512, 256)
        assert payload["grad_y"].dtype == torch.bfloat16
        assert payload["fc1_output"].shape == (512, 512)
        assert payload["fc1_output"].dtype == torch.bfloat16
        assert payload["tokens_per_expert"].tolist() == [128, 128, 128, 128]
        assert payload["permuted_probs"].shape == (512,)
        # The training-time grads are dumped so a replay can localize corruption.
        assert payload["fp8_w2_grad"].shape == (4, 256, 256)
        assert payload["ref_w2_grad"].shape == (4, 256, 256)
        # The dumped grads must reproduce the recorded relative-L2.
        dumped_fp8 = payload["fp8_w2_grad"].float()
        dumped_ref = payload["ref_w2_grad"].float()
        recomputed = (
            torch.linalg.vector_norm(dumped_fp8 - dumped_ref)
            / torch.linalg.vector_norm(dumped_ref)
        )
        assert torch.isclose(recomputed, payload["rel_l2"], rtol=1e-3)
        # PolyNorm-GLU coefficients are always present as keys; None for SwiGLU.
        assert "a1" in payload and "a2" in payload
        assert payload["a1"] is None and payload["a2"] is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_offloading_moe_w2_grad_rel_l2_realistic_distribution(self):
        """Compare real FP8 and reference w2 grads under a controlled data distribution."""
        torch.manual_seed(321)
        torch.cuda.manual_seed(321)

        num_moe_experts = 4
        hidden_size = 256
        moe_ffn_hidden_size = 256
        tokens_per_expert = torch.tensor([128] * num_moe_experts, dtype=torch.int32)
        tokens_per_expert_list = tokens_per_expert.tolist()
        num_tokens = tokens_per_expert.sum().item()
        device = torch.cuda.current_device()

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_moe_experts=num_moe_experts,
            num_attention_heads=8,
            use_cpu_initialization=True,
            perform_initialization=False,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            add_bias_linear=False,
            fp16=False,
            params_dtype=torch.bfloat16,
            gated_linear_unit=True,
            moe_use_offloading_experts=True,
            moe_use_inplace_fp8_param=True,
            monitor_moe_activation_max=True,
            moe_offloading_num_chunks=2,
            moe_offloading_num_stages=2,
            moe_offloading_chunk_size=2,
            gradient_accumulation_fusion=True,
        )
        fp8_config = OffloadingFP8Config.from_transformer_config(transformer_config)

        grad_y = self._tensor_with_mean_std_and_max(
            (num_tokens, hidden_size),
            std=1e-7,
            max_value=1e-6,
            device=device,
            dtype=torch.bfloat16,
        )
        activation_output = self._tensor_with_mean_std_and_max(
            (num_tokens, moe_ffn_hidden_size),
            std=3e-2,
            max_value=1.1,
            device=device,
            dtype=torch.bfloat16,
        )

        permuted_probs = torch.ones(num_tokens, device=device, dtype=torch.bfloat16)
        x_glu = torch.ones_like(activation_output)
        silu_one = torch.nn.functional.silu(torch.ones((), device=device, dtype=torch.float32))
        x_linear = (activation_output.float() / silu_one).to(torch.bfloat16)
        fc1_output = torch.cat([x_glu, x_linear], dim=-1).contiguous()

        cpu_w2_fp8 = torch.nn.Parameter(
            torch.empty(num_moe_experts, hidden_size, moe_ffn_hidden_size, dtype=torch.bfloat16)
        )
        cpu_w2_ref = torch.nn.Parameter(torch.empty_like(cpu_w2_fp8))
        cpu_w2_fp8.main_grad = torch.zeros(
            num_moe_experts, hidden_size, moe_ffn_hidden_size, device=device, dtype=torch.float32
        )
        cpu_w2_ref.main_grad = torch.zeros_like(cpu_w2_fp8.main_grad)

        tokens_per_expert_cuda = tokens_per_expert.to(device=device)
        tokens_per_expert_cumsum = torch.tensor(
            [0, 128, 256, 384], device=device, dtype=torch.int32
        )
        fp8_grad_y_t = guarded_per_channel_cast_to_fp8_pack_kmajor(
            grad_y,
            tokens_per_expert_list,
            tokens_per_expert_cuda,
            tokens_per_expert_cumsum,
            name="w2_grad_y",
            config=fp8_config,
            gran_k=128,
            free_input=False,
        )
        stream_manager = StreamManager(2, 1)
        activation_monitor = {}

        OffloadingExpertsFP8GroupedSwiMLP.call_backward_grad_w2(
            fp8_grad_y_t,
            fc1_output,
            cpu_w2_fp8,
            tokens_per_expert_list,
            tokens_per_expert_cuda,
            tokens_per_expert_cumsum,
            permuted_probs,
            stream_manager,
            num_moe_experts,
            fp8_config,
            None,
            False,
            True,
            activation_monitor=activation_monitor,
        )
        ExpertsFP8GroupedSwiMLP.call_backward_grad_w2_ref(
            grad_y,
            fc1_output,
            cpu_w2_ref,
            permuted_probs,
            tokens_per_expert,
            True,
            fp8_config,
        )
        torch.cuda.synchronize()

        rel_l2 = torch.linalg.vector_norm(
            (cpu_w2_fp8.main_grad - cpu_w2_ref.main_grad).float()
        ) / torch.linalg.vector_norm(cpu_w2_ref.main_grad.float())
        observed_activation_mean = (
            activation_monitor["sum_activation_before_grad_w2"]
            / activation_monitor["count_activation_before_grad_w2"]
        )
        observed_activation_var = (
            activation_monitor["sum_sq_activation_before_grad_w2"]
            / activation_monitor["count_activation_before_grad_w2"]
            - observed_activation_mean.square()
        )
        observed_grad_y = grad_y.float()

        print(f"grad_y_mean={observed_grad_y.mean().item():.8e}")
        print(f"grad_y_std={observed_grad_y.std(unbiased=False).item():.8e}")
        print(f"grad_y_max={observed_grad_y.max().item():.8e}")
        print(f"activation_output_mean={observed_activation_mean.item():.8e}")
        print(f"activation_output_std={observed_activation_var.clamp_min(0).sqrt().item():.8e}")
        print(f"activation_output_max={activation_monitor['max_activation_before_grad_w2'].item():.8e}")
        print(f"max_w2_grad_ref_fp8_rel_l2={rel_l2.item():.8f}")

        assert torch.isfinite(rel_l2)
        assert rel_l2 < 0.08

    @staticmethod
    def _replay_grad_w2_from_payload(payload, *, force_swiglu):
        """Replay a dumped grad_w2 case offline.

        Returns ``(rel_l2, fp8_grad, ref_grad)``: the FP8-vs-reference relative-L2
        and both freshly recomputed w2 grads (float32). ``force_swiglu`` runs the
        SwiGLU path (no PolyNorm coefficients needed); otherwise the PolyNorm-GLU
        path is used and ``a1``/``a2`` must be present in the payload.
        ``permuted_probs`` is taken from the dump when present (older dumps fall
        back to ones); it scales ``s`` identically in both the FP8 and reference
        paths.
        """
        device = torch.cuda.current_device()
        md = payload["metadata"]
        grad_y = payload["grad_y"].to(device)
        fc1_output = payload["fc1_output"].to(device).contiguous()
        tokens_per_expert = payload["tokens_per_expert"].to(torch.int32)
        tokens_per_expert_list = tokens_per_expert.tolist()
        num_tokens = int(tokens_per_expert.sum().item())
        num_moe_experts = tokens_per_expert.numel()
        hidden_size = md["hidden_size"]
        moe_ffn_hidden_size = md["moe_ffn_hidden_size"]

        pnglu = not force_swiglu
        a1 = a2 = None
        if pnglu:
            a1 = payload.get("a1")
            a2 = payload.get("a2")
            assert a1 is not None and a2 is not None, (
                "PolyNorm-GLU replay needs a1/a2 in the dump"
            )
            a1 = a1.to(device)
            a2 = a2.to(device)

        fp8_config = OffloadingFP8Config(
            hidden_size=hidden_size,
            moe_latent_size=None,
            input_hidden_size=hidden_size,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            gated_linear_unit=True,
            gated_polynorm_linear_unit=pnglu,
            monitor_moe_activation_max=True,
            moe_offloading_num_chunks=2,
            moe_offloading_num_stages=2,
            moe_offloading_chunk_size=max(num_moe_experts // 2, 1),
            gradient_accumulation_fusion=True,
        )
        assert fp8_config.fc1_out_size == fc1_output.shape[1]

        dumped_probs = payload.get("permuted_probs")
        if dumped_probs is not None:
            # Dumped as (num_tokens,) or (num_tokens, 1); grad_w2 unsqueezes(-1).
            permuted_probs = dumped_probs.to(device=device, dtype=torch.bfloat16).reshape(-1)
        else:
            permuted_probs = torch.ones(num_tokens, device=device, dtype=torch.bfloat16)
        tokens_per_expert_cuda = tokens_per_expert.to(device=device)
        tokens_per_expert_cumsum = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(tokens_per_expert_cuda, 0).to(torch.int32)[:-1],
        ])

        cpu_w2_fp8 = torch.nn.Parameter(
            torch.empty(num_moe_experts, hidden_size, moe_ffn_hidden_size, dtype=torch.bfloat16)
        )
        cpu_w2_ref = torch.nn.Parameter(torch.empty_like(cpu_w2_fp8))
        cpu_w2_fp8.main_grad = torch.zeros(
            num_moe_experts, hidden_size, moe_ffn_hidden_size, device=device, dtype=torch.float32
        )
        cpu_w2_ref.main_grad = torch.zeros_like(cpu_w2_fp8.main_grad)

        fp8_grad_y_t = guarded_per_channel_cast_to_fp8_pack_kmajor(
            grad_y, tokens_per_expert_list, tokens_per_expert_cuda, tokens_per_expert_cumsum,
            name="w2_grad_y", config=fp8_config, gran_k=128, free_input=False,
        )
        stream_manager = StreamManager(2, 1)

        OffloadingExpertsFP8GroupedSwiMLP.call_backward_grad_w2(
            fp8_grad_y_t, fc1_output, cpu_w2_fp8,
            tokens_per_expert_list, tokens_per_expert_cuda, tokens_per_expert_cumsum,
            permuted_probs, stream_manager, num_moe_experts, fp8_config,
            None, False, True, a1, a2,
            activation_monitor={},
        )
        ExpertsFP8GroupedSwiMLP.call_backward_grad_w2_ref(
            grad_y, fc1_output, cpu_w2_ref, permuted_probs, tokens_per_expert,
            True, fp8_config, a1, a2,
        )
        torch.cuda.synchronize()

        fp8_grad = cpu_w2_fp8.main_grad.float()
        ref_grad = cpu_w2_ref.main_grad.float()
        rel_l2 = (
            torch.linalg.vector_norm(fp8_grad - ref_grad)
            / torch.linalg.vector_norm(ref_grad)
        )
        return rel_l2, fp8_grad, ref_grad

    @staticmethod
    def _rel_l2(actual, reference):
        return (
            torch.linalg.vector_norm((actual - reference).float())
            / torch.linalg.vector_norm(reference.float())
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_offloading_moe_w2_grad_rel_l2_from_dump(self):
        """Replay a saved abort dump and report the SwiGLU FP8-vs-reference rel-L2.

        Locates the dump via ``MEGATRON_MOE_W2_GRAD_REL_L2_DUMP_FILE`` (a .pt
        path) or ``MEGATRON_MOE_W2_GRAD_REL_L2_DUMP_DIR`` (newest matching .pt),
        falling back to the default ``moe_w2_grad_rel_l2_dumps`` directory. The
        test is skipped when no dump exists so it stays CI-safe. Runs with SwiGLU
        (always available) and additionally with PolyNorm-GLU when the dump
        carries the ``a1``/``a2`` coefficients.
        """
        dump_file = os.environ.get("MEGATRON_MOE_W2_GRAD_REL_L2_DUMP_FILE")
        if not dump_file:
            dump_dir = os.environ.get(
                "MEGATRON_MOE_W2_GRAD_REL_L2_DUMP_DIR", "moe_w2_grad_rel_l2_dumps"
            )
            matches = sorted(glob.glob(os.path.join(dump_dir, "w2_grad_rel_l2_rank*.pt")))
            dump_file = matches[-1] if matches else None
        if not dump_file or not os.path.exists(dump_file):
            pytest.skip("no w2_grad_rel_l2 dump available")

        payload = torch.load(dump_file, map_location="cpu")
        recorded = payload["rel_l2"].item()
        print(f"\ndump={dump_file}")
        print(f"recorded rel_l2={recorded:.6f} (threshold {payload['threshold']})")

        swiglu_rel_l2, _, _ = self._replay_grad_w2_from_payload(payload, force_swiglu=True)
        print(f"replay SwiGLU rel_l2={swiglu_rel_l2.item():.8f}")
        assert torch.isfinite(swiglu_rel_l2)

        has_coeffs = payload.get("a1") is not None and payload.get("a2") is not None
        pnglu_rel_l2 = fp8_grad = ref_grad = None
        if has_coeffs:
            pnglu_rel_l2, fp8_grad, ref_grad = self._replay_grad_w2_from_payload(
                payload, force_swiglu=False
            )
            print(f"replay PolyNorm-GLU rel_l2={pnglu_rel_l2.item():.8f}")
            assert torch.isfinite(pnglu_rel_l2)

        # If the training-time grads were dumped, localize the corruption: a clean
        # offline replay should reproduce them. The reference (bf16) path is
        # deterministic, so a large ref mismatch means the saved operands differ
        # from what training actually fed the GEMM; a large FP8-only mismatch
        # points at the FP8 wgrad GEMM / its operand (stream race / buffer reuse).
        dumped_fp8 = payload.get("fp8_w2_grad")
        dumped_ref = payload.get("ref_w2_grad")
        if dumped_fp8 is not None and dumped_ref is not None:
            device = torch.cuda.current_device()
            dumped_fp8 = dumped_fp8.to(device).float()
            dumped_ref = dumped_ref.to(device).float()
            training_rel_l2 = self._rel_l2(dumped_fp8, dumped_ref)
            print(f"dumped-grad rel_l2 (recomputed)={training_rel_l2.item():.6f}")
            # The dumped grads must encode the same catastrophe the guard recorded.
            assert torch.isclose(
                training_rel_l2, torch.as_tensor(recorded, device=device), rtol=1e-3
            ), (training_rel_l2.item(), recorded)
            if fp8_grad is not None:
                ref_match = self._rel_l2(ref_grad, dumped_ref)
                fp8_match = self._rel_l2(fp8_grad, dumped_fp8)
                print(f"replay-vs-dumped ref_grad rel_l2={ref_match.item():.8f}")
                print(f"replay-vs-dumped fp8_grad rel_l2={fp8_match.item():.8f}")

    @pytest.mark.parametrize("num_moe_experts", [64])
    def test_offloading_moe_forward_backward(
        self, num_moe_experts, profile=False, num_repeats=10
    ):
        """Test MoE layer forward and backward pass with fp16 params and inputs."""
        hidden_size = 7168
        moe_ffn_hidden_size = 2048
        moe_offloading_num_chunks = 4
        moe_offloading_num_stages = 2
        moe_offloading_chunk_size = num_moe_experts // moe_offloading_num_chunks

        tokens_per_expert = torch.randint(
            1024, 1025, (num_moe_experts,), device="cpu"
        )
        tokens_per_expert_ref = tokens_per_expert.detach().clone()

        hidden_states = (
            torch.randn(
                tokens_per_expert.sum().item(),
                hidden_size,
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16,
                requires_grad=False,
            )
        ).detach().requires_grad_(True)
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

        # Draw weights on CUDA so the CUDA RNG stream advances identically to
        # the non-offloading test, then mirror to pinned CPU for the offload path.
        w1_init = torch.randn(
            num_moe_experts, moe_ffn_hidden_size * 2, hidden_size,
            device=torch.cuda.current_device(), dtype=torch.bfloat16,
        )
        w2_init = torch.randn(
            num_moe_experts, hidden_size, moe_ffn_hidden_size,
            device=torch.cuda.current_device(), dtype=torch.bfloat16,
        )
        cpu_w1 = torch.nn.Parameter(
            torch.empty_like(w1_init, device="cpu", pin_memory=True).copy_(w1_init),
            requires_grad=True,
        )
        cpu_w2 = torch.nn.Parameter(
            torch.empty_like(w2_init, device="cpu", pin_memory=True).copy_(w2_init),
            requires_grad=True,
        )

        cpu_w1_list = list(torch.unbind(cpu_w1, dim=0))
        cpu_w2_list = list(torch.unbind(cpu_w2, dim=0))

        cpu_w1.main_grad = torch.zeros_like(cpu_w1, device="cuda", dtype=torch.float32)
        cpu_w2.main_grad = torch.zeros_like(cpu_w2, device="cuda", dtype=torch.float32)

        # Allocate contiguous GPU buffers for FP8 weights: [num_stages, chunk_size, ...]
        experts1_gpu_buffers_storage = torch.empty(
            moe_offloading_num_stages * moe_offloading_chunk_size * moe_ffn_hidden_size * 2 * hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.float8_e4m3fn,
        ).view(
            moe_offloading_num_stages, moe_offloading_chunk_size, moe_ffn_hidden_size * 2, hidden_size
        )
        experts2_gpu_buffers_storage = torch.empty(
            moe_offloading_num_stages * moe_offloading_chunk_size * moe_ffn_hidden_size * hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.float8_e4m3fn,
        ).view(
            moe_offloading_num_stages, moe_offloading_chunk_size, hidden_size, moe_ffn_hidden_size
        )

        # Per-chunk views: [num_stages, chunk_size, ...]
        gpu_w1_buffer = [
            [experts1_gpu_buffers_storage[s, c] for c in range(moe_offloading_chunk_size)]
            for s in range(moe_offloading_num_stages)
        ]
        gpu_w2_buffer = [
            [experts2_gpu_buffers_storage[s, c] for c in range(moe_offloading_chunk_size)]
            for s in range(moe_offloading_num_stages)
        ]

        # Per-stage views: [num_stages, chunk_size, ...]
        gpu_w1_chunks = [
            experts1_gpu_buffers_storage[s] for s in range(moe_offloading_num_stages)
        ]
        gpu_w2_chunks = [
            experts2_gpu_buffers_storage[s] for s in range(moe_offloading_num_stages)
        ]

        # Reference weights on GPU (bf16, no offloading)
        gpu_w1 = torch.nn.Parameter(
            cpu_w1.detach().clone().cuda(), requires_grad=True,
        )
        gpu_w2 = torch.nn.Parameter(
            cpu_w2.detach().clone().cuda(), requires_grad=True,
        )
        gpu_w1.main_grad = torch.zeros_like(gpu_w1, device="cuda", dtype=torch.float32)
        gpu_w2.main_grad = torch.zeros_like(gpu_w2, device="cuda", dtype=torch.float32)

        torch.cuda.synchronize()

        stream_manager = StreamManager(moe_offloading_num_stages, 4)
        FP8ExpertsParameterManager.create_instance(config=transformer_config)

        # Realistic upstream gradient: simulate an MSE loss against a random
        # target, so grad_y has the same per-token magnitude structure as in
        # training.  Using .sum().backward() (grad_y = 1) makes grad_w1/grad_w2
        # errors depend on bf16-vs-fp32 accumulation order rather than on the
        # quantization paths we care about.
        target = torch.randn(
            tokens_per_expert.sum().item(), hidden_size,
            device=torch.cuda.current_device(), dtype=torch.bfloat16,
        )

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
                    output = offloading_fp8_grouped_swiglu_mlp(
                        cpu_w1, cpu_w2, cpu_w1_list, cpu_w2_list,
                        gpu_w1_buffer, gpu_w2_buffer,
                        gpu_w1_chunks, gpu_w2_chunks,
                        hidden_states, tokens_per_expert, num_moe_experts,
                        permuted_probs, None, stream_manager,
                        transformer_config, [],
                    )
                    torch.cuda.synchronize()
                    output.sum().backward()
                    torch.cuda.synchronize()
                    prof.step()
            prof.export_chrome_trace("fp8_e2e.json")
            return

        # --- Reference path (bf16 matmuls on GPU) ---
        hidden_states_ref_list = list(
            torch.split(hidden_states_ref, tokens_per_expert_ref.tolist(), dim=0)
        )
        fc1_outputs = [
            torch.mm(hidden_states_ref_list[i], gpu_w1[i].t())
            for i in range(num_moe_experts)
        ]
        act_output = MergedSwiGLU.apply(
            torch.cat(fc1_outputs, dim=0), permuted_probs_ref.unsqueeze(-1)
        )
        act_list = list(torch.split(act_output, tokens_per_expert_ref.tolist(), dim=0))
        output_ref = torch.cat([
            torch.mm(act_list[i], gpu_w2[i].t())
            for i in range(num_moe_experts)
        ], dim=0)
        ((output_ref - target).float() ** 2).sum().backward()
        torch.cuda.synchronize()

        # --- Offloading path (FP8 with CPU offload) ---
        outputs = []
        for _ in range(num_repeats):
            output = offloading_fp8_grouped_swiglu_mlp(
                cpu_w1, cpu_w2, cpu_w1_list, cpu_w2_list,
                gpu_w1_buffer, gpu_w2_buffer,
                gpu_w1_chunks, gpu_w2_chunks,
                hidden_states, tokens_per_expert, num_moe_experts,
                permuted_probs, None, stream_manager,
                transformer_config, [],
            )
            cpu_w1.main_grad.zero_()
            cpu_w2.main_grad.zero_()
            ((output - target).float() ** 2).sum().backward()
            outputs.append(output)
            torch.cuda.synchronize()

        # --- Comparison ---
        def diff_tensor_norm(tensor1, tensor2):
            return (
                torch.norm(tensor1.to(torch.float) - tensor2.to(torch.float)).item()
                / torch.norm(tensor2.to(torch.float)).item()
            )

        for i, output in enumerate(outputs):
            assert output.shape == output_ref.shape, (
                f"Output shape mismatch: {output.shape} vs {output_ref.shape}"
            )
            diff = diff_tensor_norm(output, output_ref)
            if diff > 0.07:
                half = tokens_per_expert.sum().item() // 2
                diff_half_1 = diff_tensor_norm(output[:half], output_ref[:half])
                diff_half_2 = diff_tensor_norm(output[half:], output_ref[half:])
                raise AssertionError(
                    f"[repeat {i}] Output norm diff {diff:.4f} exceeds threshold "
                    f"(half1={diff_half_1:.4f}, half2={diff_half_2:.4f})"
                )

        w1_grad_diff = diff_tensor_norm(cpu_w1.main_grad, gpu_w1.grad)
        w2_grad_diff = diff_tensor_norm(cpu_w2.main_grad, gpu_w2.grad)
        assert w1_grad_diff < 0.10, (
            f"W1 grad norm diff {w1_grad_diff:.4f} exceeds threshold"
        )
        assert w2_grad_diff < 0.10, (
            f"W2 grad norm diff {w2_grad_diff:.4f} exceeds threshold"
        )


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    TestOffloadingMoELayerFP8().test_offloading_moe_forward_backward(
        num_moe_experts=64, profile=False, num_repeats=10
    )
