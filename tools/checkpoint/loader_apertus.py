# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Swiss AI. All rights reserved.

"""
Apertus HuggingFace checkpoint loader for Megatron conversion.

Apertus is a Swiss AI model based on Llama architecture with:
- XIELU activation (learnable alpha_p, alpha_n parameters)
- QK normalization
- Non-gated MLP (no SwiGLU)
- Llama3-style RoPE scaling

Usage:
    python tools/checkpoint/convert.py \
        --model-type GPT \
        --loader apertus \
        --saver core \
        --load-dir /path/to/Apertus-8B-2509 \
        --save-dir /path/to/output \
        --tokenizer-model /path/to/Apertus-8B-2509 \
        --target-tensor-parallel-size 2 \
        --bf16
"""

import json
import os
import sys
import types

import torch

try:
    import transformers
except ImportError:
    raise ImportError("The 'transformers' package is required. Install with: pip install transformers")


def add_arguments(parser):
    """Add Apertus-specific arguments to the parser."""
    group = parser.add_argument_group(title='Apertus loader')

    group.add_argument('--bf16', action='store_true', help='Load weights in bf16.')
    group.add_argument('--fp16', action='store_true', help='Load weights in fp16.')
    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='Original vocab size, if specified will trim padding from embedding table.')
    group.add_argument('--tokenizer-model', required=True,
                       help='Path to HuggingFace tokenizer/model directory.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Make vocab size divisible by this value')
    group.add_argument('--extend-vocab-to', type=int, default=None,
                       help='If set, pad the input/output embedding tables to this vocab size '
                            '(used for adding multimodal / image tokens before training). '
                            'New rows are initialized from the mean of the existing embeddings '
                            'plus small Gaussian jitter.')


def load_args_from_checkpoint(args, load_dir):
    """Load model configuration from HuggingFace config.json."""
    config_path = os.path.join(load_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Core architecture
    args.num_layers = config["num_hidden_layers"]
    args.hidden_size = config["hidden_size"]
    args.ffn_hidden_size = config["intermediate_size"]
    args.num_attention_heads = config["num_attention_heads"]
    args.num_query_groups = config.get("num_key_value_heads", args.num_attention_heads)
    args.kv_channels = config.get("head_dim", args.hidden_size // args.num_attention_heads)
    args.max_position_embeddings = config["max_position_embeddings"]
    args.seq_length = min(config["max_position_embeddings"], 8192)  # Default seq length
    args.vocab_size = config["vocab_size"]
    args.padded_vocab_size = config["vocab_size"]
    args.norm_epsilon = config.get("rms_norm_eps", 1e-5)
    args.rotary_base = config.get("rope_theta", 12000000)

    # Apertus-specific settings
    args.normalization = "RMSNorm"
    args.position_embedding_type = "rope"
    args.add_bias_linear = False
    args.add_qkv_bias = False
    args.swiglu = False  # Apertus uses XIELU, NOT SwiGLU
    args.xielu = True    # Apertus-specific
    args.qk_layernorm = config.get("qk_norm", True)  # Apertus has QK norm
    args.untie_embeddings_and_output_weights = not config.get("tie_word_embeddings", False)
    args.group_query_attention = args.num_query_groups != args.num_attention_heads

    # Iteration
    args.iteration = 1  # '0' doesn't work with some savers

    return args


def extend_embedding_table(weight: torch.Tensor, new_vocab_size: int) -> torch.Tensor:
    """Append rows to an embedding / lm_head weight table up to ``new_vocab_size``.

    The new rows are sampled from a Gaussian whose per-dimension mean and std
    match the empirical distribution of the existing rows. This matches the
    statistics of the pretrained tokens and converges faster than fixed-std
    or zero init.

    The first ``weight.shape[0]`` rows of the returned tensor are exactly the
    input rows (in the same dtype), so any forward pass that only indexes
    token IDs ``< weight.shape[0]`` is bit-identical to the original.
    """
    old_vocab_size = weight.shape[0]
    if new_vocab_size < old_vocab_size:
        raise ValueError(
            f"new_vocab_size={new_vocab_size} is smaller than current vocab size "
            f"({old_vocab_size}); shrinking is not supported."
        )
    if new_vocab_size == old_vocab_size:
        return weight

    n_new = new_vocab_size - old_vocab_size
    w_f32 = weight.to(torch.float32)
    mean = w_f32.mean(dim=0, keepdim=True)
    std = w_f32.std(dim=0, keepdim=True)
    new_rows = mean + std * torch.randn(n_new, w_f32.shape[1], dtype=torch.float32)
    return torch.cat([weight, new_rows.to(weight.dtype)], dim=0)


def _load_checkpoint(queue, args):
    """Load Apertus checkpoint and send weights through queue."""

    # Setup paths
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_global_variables
        from megatron.core import mpu
        from megatron.core.enums import ModelType
    except ModuleNotFoundError:
        print("Unable to import Megatron. Please specify --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # Build minimal args for Megatron
    sys.argv = [
        'script.py',
        '--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion',
        '--no-bias-dropout-fusion',
        '--no-async-tensor-model-parallel-allreduce',
        '--use-cpu-initialization',
        '--micro-batch-size', '1',
        '--no-load-optim',
        '--no-load-rng',
        '--no-save-optim',
        '--no-save-rng',
        '--mock-data',
        '--no-initialization',
        '--load', args.load_dir,
        '--no-one-logger',
    ]

    if args.make_vocab_size_divisible_by is not None:
        sys.argv.extend(['--make-vocab-size-divisible-by', str(args.make_vocab_size_divisible_by)])

    margs = parse_args()
    load_args_from_checkpoint(margs, args.load_dir)

    # Set tokenizer type
    margs.tokenizer_type = "HuggingFaceTokenizer"
    margs.tokenizer_model = args.tokenizer_model

    # Set dtype
    if args.bf16:
        margs.params_dtype = torch.bfloat16
    elif args.fp16:
        margs.params_dtype = torch.float16
    else:
        margs.params_dtype = torch.float32

    # Validate and set global variables
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size
    margs = validate_args(margs)
    margs.model_type = ModelType.encoder_or_decoder

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)

    # Load HuggingFace model
    print(f"Loading Apertus model from {args.load_dir}...")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.load_dir,
        torch_dtype=margs.params_dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Get true vocab size from tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)
    true_vocab_size = len(tokenizer)

    # Optionally extend the embedding / lm_head tables (e.g. for adding image tokens).
    # We do the padding here, on the source HF tensors, so that the saver only sees a
    # checkpoint that already has the new vocab size. This avoids any runtime module
    # surgery, plays nicely with TP/PP/FSDP, and correctly handles tied embeddings.
    # It was previously done in megatron/training/checkpointing.py but that
    # caused various complications and edge cases especially for TP/PP/FSDP.
    word_embed = hf_model.model.embed_tokens.weight.data
    lm_head_weight = hf_model.lm_head.weight.data if margs.untie_embeddings_and_output_weights else None

    if args.extend_vocab_to is not None:
        old_vocab_size = word_embed.shape[0]
        new_vocab_size = args.extend_vocab_to
        if new_vocab_size > old_vocab_size:
            print(f"Extending vocab from {old_vocab_size} to {new_vocab_size} "
                  f"(+{new_vocab_size - old_vocab_size} rows)")
            word_embed = extend_embedding_table(word_embed, new_vocab_size)
            if lm_head_weight is not None:
                lm_head_weight = extend_embedding_table(lm_head_weight, new_vocab_size)

            # Reflect the new vocab size in args/metadata so the saver builds a model
            # of the right size and chunks the embedding correctly across TP.
            true_vocab_size = new_vocab_size
            margs.vocab_size = new_vocab_size
            margs.padded_vocab_size = new_vocab_size
        elif new_vocab_size < old_vocab_size:
            raise ValueError(
                f"--extend-vocab-to={new_vocab_size} is smaller than the current vocab size "
                f"({old_vocab_size}); shrinking is not supported."
            )

    # Build metadata
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = False
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.qkv_bias = margs.add_qkv_bias
    md.norm_has_bias = False
    md.swiglu = False  # Apertus does NOT use SwiGLU
    md.xielu = True    # Apertus uses XIELU
    md.qk_layernorm = margs.qk_layernorm
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    md.true_vocab_size = true_vocab_size

    # Send metadata
    queue.put(md)

    def queue_put(name, msg):
        print(f"Sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings
    message = {
        "word embeddings": word_embed,
    }
    queue_put("embeddings", message)

    # Send transformer layers
    for layer_idx in range(margs.num_layers):
        hf_layer = hf_model.model.layers[layer_idx]
        message = {}

        # Layer norms (Apertus uses different names)
        # HF: attention_layernorm, feedforward_layernorm
        message["input norm weight"] = hf_layer.attention_layernorm.weight.data
        message["post norm weight"] = hf_layer.feedforward_layernorm.weight.data

        # QKV weights - combine Q, K, V into single tensor
        # Apertus uses GQA, so we need to handle num_query_groups
        q_weight = hf_layer.self_attn.q_proj.weight.data  # [num_heads * head_dim, hidden]
        k_weight = hf_layer.self_attn.k_proj.weight.data  # [num_kv_heads * head_dim, hidden]
        v_weight = hf_layer.self_attn.v_proj.weight.data  # [num_kv_heads * head_dim, hidden]

        # Reshape for interleaved QKV format expected by Megatron
        num_heads = margs.num_attention_heads
        num_kv_heads = margs.num_query_groups
        head_dim = margs.kv_channels

        # Reshape Q: [num_heads, head_dim, hidden]
        q_weight = q_weight.view(num_heads, head_dim, margs.hidden_size)
        # Reshape K, V: [num_kv_heads, head_dim, hidden]
        k_weight = k_weight.view(num_kv_heads, head_dim, margs.hidden_size)
        v_weight = v_weight.view(num_kv_heads, head_dim, margs.hidden_size)

        # Interleave: for each query group, we have (num_heads/num_kv_heads) Q heads, 1 K head, 1 V head
        heads_per_group = num_heads // num_kv_heads
        qkv_weight_list = []
        for group_idx in range(num_kv_heads):
            # Q heads for this group
            q_start = group_idx * heads_per_group
            q_end = q_start + heads_per_group
            qkv_weight_list.append(q_weight[q_start:q_end].reshape(-1, margs.hidden_size))
            # K head for this group
            qkv_weight_list.append(k_weight[group_idx:group_idx+1].reshape(-1, margs.hidden_size))
            # V head for this group
            qkv_weight_list.append(v_weight[group_idx:group_idx+1].reshape(-1, margs.hidden_size))

        qkv_weight = torch.cat(qkv_weight_list, dim=0)
        message["qkv weight"] = qkv_weight

        # Dense (output projection)
        message["dense weight"] = hf_layer.self_attn.o_proj.weight.data

        # QK normalization weights (Apertus-specific)
        if md.qk_layernorm:
            message["q norm weight"] = hf_layer.self_attn.q_norm.weight.data
            message["k norm weight"] = hf_layer.self_attn.k_norm.weight.data

        # MLP weights (NOT gated - Apertus uses XIELU)
        # HF: up_proj, down_proj (no gate_proj)
        message["mlp l0 weight"] = hf_layer.mlp.up_proj.weight.data
        message["mlp l1 weight"] = hf_layer.mlp.down_proj.weight.data

        # XIELU activation parameters (Apertus-specific)
        if md.xielu:
            message["mlp alpha_p"] = hf_layer.mlp.act_fn.alpha_p.data
            message["mlp alpha_n"] = hf_layer.mlp.act_fn.alpha_n.data

        queue_put(f"transformer layer {layer_idx}", message)

    # Send final layer norm
    message = {
        "weight": hf_model.model.norm.weight.data,
    }
    queue_put("final norm", message)

    # Send output layer (lm_head)
    if md.output_layer:
        message = {
            "weight": lm_head_weight,
        }
        queue_put("output layer", message)

    queue.put("done")
    print("Apertus checkpoint loading complete!")


def load_checkpoint(queue, args):
    """Entry point for checkpoint loading."""
    try:
        _load_checkpoint(queue, args)
    except Exception as e:
        queue.put("exit")
        raise
