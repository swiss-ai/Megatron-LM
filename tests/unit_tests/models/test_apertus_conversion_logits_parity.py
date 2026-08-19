#!/usr/bin/env python3
"""Test logits parity between HF model and Megatron checkpoint converted with native Megatron converter.

Usage (requires 2 GPUs for TP=2 checkpoint):
    torchrun --nproc-per-node=2 tests/unit_tests/models/test_apertus_conversion_logits_parity.py
"""

import os
import sys

# Use Swiss AI Megatron-LM fork (has XIELU)
sys.path.insert(0, "/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM")

import torch

# Get local rank for multi-GPU setup
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    is_rank_0 = local_rank == 0

    # 1. Load original HF model (only on rank 0 to save memory)
    hf_logits = None
    tokenizer = AutoTokenizer.from_pretrained("swiss-ai/Apertus-8B-2509")
    text = "The capital of France is"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    if is_rank_0:
        print("=" * 60)
        print("Loading original HuggingFace model...")
        print("=" * 60)
        hf_model = AutoModelForCausalLM.from_pretrained(
            "swiss-ai/Apertus-8B-2509",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        hf_model.eval()

        print(f"Test input: '{text}'")
        print(f"Input IDs: {inputs['input_ids']}")

        # Get HF logits
        print("\nRunning HF forward pass...")
        with torch.no_grad():
            hf_outputs = hf_model(**inputs)
            hf_logits = hf_outputs.logits

        print(f"HF logits shape: {hf_logits.shape}")
        print(f"HF last token logits (first 10): {hf_logits[0, -1, :10]}")

        # Free HF model memory
        del hf_model
        torch.cuda.empty_cache()

    # 2. Load Megatron model using megatron-core directly
    if is_rank_0:
        print("\n" + "=" * 60)
        print("Loading Megatron model from checkpoint using megatron-core...")
        print("=" * 60)

    from megatron.core import dist_checkpointing
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core import parallel_state

    # Initialize distributed (TP=2 for native converter checkpoint)
    world_size = int(os.environ.get("WORLD_SIZE", 2))
    rank = int(os.environ.get("RANK", local_rank))
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
        )
    # TP=2, PP=1 to match the converted checkpoint
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)

    # Initialize RNG tracker for model parallel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(42)

    # Apertus-8B config (Swiss AI Megatron-LM)
    from megatron.core.activations import XIELU
    config = TransformerConfig(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=21504,  # HF intermediate_size (not 14336 like Llama)
        num_attention_heads=32,
        num_query_groups=8,  # GQA
        hidden_dropout=0.0,
        attention_dropout=0.0,
        normalization="RMSNorm",
        add_bias_linear=False,
        gated_linear_unit=False,  # Apertus uses XIELU, not gated
        activation_func=XIELU,  # XIELU activation
        qk_layernorm=True,
        bf16=True,
        params_dtype=torch.bfloat16,  # This sets actual parameter dtype
    )

    # Create model
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
        vocab_size=131072,  # HF vocab_size
        max_sequence_length=131072,
        pre_process=True,
        post_process=True,
        position_embedding_type='rope',
        rotary_base=12000000,  # HF rope_theta (was incorrectly 500000)
    )
    model.to(dtype=torch.bfloat16, device="cuda")  # Explicit cast needed for Swiss AI fork

    # Load checkpoint (native Megatron converter output with TP=2, legacy format)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    ckpt_path = f"/iopsstor/scratch/cscs/xyixuan/apertus_megatron_native/iter_0000001/mp_rank_{tp_rank:02d}/model_optim_rng.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 3. Run Megatron forward pass
    if is_rank_0:
        print("\nRunning Megatron forward pass...")
    with torch.no_grad():
        input_ids = inputs["input_ids"]
        position_ids = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)

        megatron_logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
        )

    # Gather logits across TP ranks (vocab is split with TP=2)
    from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
    megatron_logits = gather_from_tensor_model_parallel_region(megatron_logits)

    # 4. Compare (only on rank 0)
    if is_rank_0:
        print(f"Megatron logits shape: {megatron_logits.shape}")
        print(f"Megatron last token logits (first 10): {megatron_logits[0, -1, :10]}")

        print("\n" + "=" * 60)
        print("Comparing logits...")
        print("=" * 60)

        hf_logits_cpu = hf_logits.cpu().float()
        megatron_logits_cpu = megatron_logits.cpu().float()

        # Logit statistics (informational)
        abs_diff = torch.abs(hf_logits_cpu - megatron_logits_cpu)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        print(f"Max absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")

        # Top-k agreement check (primary pass/fail criterion)
        print("\n" + "=" * 60)
        print("Top-k predictions comparison...")
        print("=" * 60)

        for k in [1, 5, 10]:
            hf_topk = torch.topk(hf_logits_cpu[0, -1], k)
            megatron_topk = torch.topk(megatron_logits_cpu[0, -1], k)

            # Exact position match (not just set)
            exact_matches = (hf_topk.indices == megatron_topk.indices).sum().item()
            print(f"Top-{k} exact match: {exact_matches}/{k} tokens match in exact order")

        # Top-5 details
        hf_top5 = torch.topk(hf_logits_cpu[0, -1], 5)
        megatron_top5 = torch.topk(megatron_logits_cpu[0, -1], 5)

        print("\nHF top-5 predictions:")
        for i, (idx, score) in enumerate(zip(hf_top5.indices, hf_top5.values)):
            token = tokenizer.decode([idx])
            print(f"  {i+1}. '{token}' (id={idx.item()}, score={score.item():.4f})")

        print("\nMegatron top-5 predictions:")
        for i, (idx, score) in enumerate(zip(megatron_top5.indices, megatron_top5.values)):
            token = tokenizer.decode([idx])
            print(f"  {i+1}. '{token}' (id={idx.item()}, score={score.item():.4f})")

        # Pass/fail based on exact top-k match
        top1_exact = hf_top5.indices[0].item() == megatron_top5.indices[0].item()
        top5_exact = (hf_top5.indices == megatron_top5.indices).all().item()

        if top5_exact:
            print("\nPASS: Top-5 predictions match exactly in order")
        elif top1_exact:
            print("\nPASS: Top-1 matches, Top-5 order differs slightly")
        else:
            print("\nFAIL: Top-1 prediction does not match")

    # Cleanup
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
