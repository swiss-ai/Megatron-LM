# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import json
import os
import sys
import types
import torch

from utils import (
    init_converter_fake_groups,
    print_memory_usage,
    set_converter_fake_group_ranks,
)

class MegatronCheckpointLoaderBase:
    """Orchestrates loading a Megatron checkpoint and sending
    model parameters over a given multiprocessing queue.

    Args:
        args: argparse Namespace with Megatron checkpoint configurations.
        queue: A multiprocessing.Queue (or similar) used to send out loaded tensors.
    """

    def __init__(self, args, queue, build_tokenizer=False):
        self.args = args
        self.queue = queue
        self.build_tokenizer = build_tokenizer
        self.margs = None            # Will hold Megatron's main args
        self.checkpoint_args = None  # Will hold additional checkpoint args
        self.all_models = None       # Model sharded over different parallelism
        self.md = None               # Metadata sent to the saver
        self.consumed_train_samples = None
        self.consumed_valid_samples = None
        self._converter_fake_groups = {}
        self._use_converter_fake_groups = False
        self._plain_tensors = None   # Cache for --use-plain-tensor-loading

    def _maybe_parse_additional_megatron_args(self, margs, checkpoint_args):
        """
        Method used to optionally add arguments from the checkpoint to the main args.
        For instance, using margs.some_arg = checkpoint_args.some_arg
        """
        return margs

    def parse_megatron_args(self):
        """
        Parse Megatron arguments by forcibly overwriting sys.argv.
        Populates self.margs and self.checkpoint_args.
        """
        # Ensure we can import Megatron
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
        if self.args.megatron_path is not None:
            sys.path.insert(0, self.args.megatron_path)

        try:
            from megatron.training.arguments import parse_args, validate_args
            from megatron.training.checkpointing import load_args_from_checkpoint
        except ModuleNotFoundError:
            print("Unable to import Megatron. Please specify --megatron-path. Exiting.")
            self.queue.put("exit")
            sys.exit(1)

        # Overwrite sys.argv
        sys.argv = self.build_sys_argv()

        margs = parse_args()
        margs, checkpoint_args = load_args_from_checkpoint(margs)

        # Checkpoints saved with --ckpt-fully-parallel-save stripe each tensor across the
        # full data-parallel group used at training time (not just TP/PP), to speed up
        # checkpoint I/O. This converter runs as a single real process, so it can never
        # perform the real-rank gather that reconstructs those DP-striped pieces
        # (FullyParallelLoadStrategyWrapper requires >1 real ranks and no-ops otherwise).
        # Instead, when this mode is requested, we sidestep the problem entirely: request
        # each tensor fully unsharded (see _load_plain_model_tensors) and treat the
        # checkpoint as tensor_model_parallel_size=1 for every downstream TP-shard loop.
        # The model architecture itself (hidden size, heads, layers, ...) is
        # TP-independent, so this only changes how we load, not what gets loaded.
        self.real_tensor_model_parallel_size = margs.tensor_model_parallel_size
        if getattr(self.args, 'use_plain_tensor_loading', False):
            margs.tensor_model_parallel_size = 1

        # Adjust world size so validation doesn't fail
        margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

        # Copy data types from checkpoint
        margs.fp16 = checkpoint_args.fp16
        margs.bf16 = checkpoint_args.bf16

        # Expert parallelism requires sequence parallelism
        if margs.expert_model_parallel_size > 1:
            margs.sequence_parallel = True
        
        margs = self._maybe_parse_additional_megatron_args(margs, checkpoint_args)

        # Validate final arguments
        try:
            from megatron.training.arguments import validate_args
            margs = validate_args(margs)
        except Exception as e:
            print(f"Error validating Megatron arguments: {e}")
            self.queue.put("exit")
            sys.exit(1)

        margs.use_legacy_models = False
        margs.transformer_impl = self.args.loader_transformer_impl
        if self.args.loader_transformer_impl == "local" and margs.normalization == "RMSNorm":
            margs.no_persist_layer_norm = True

        self.margs = margs
        self.checkpoint_args = checkpoint_args

    def _maybe_ensure_additional_required_arguments(self):
        """
        Can be used to ensure some expected args are present.
        For instance, use self.check_for_arg('some_arg')
        """
        pass

    def check_for_arg(self, arg_name, default=None):
        if getattr(self.margs, arg_name, None) is None:
            if default is not None:
                setattr(self.margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify argument {arg_name}. Exiting.")
                print(f"Arguments: {self.margs}")
                self.queue.put("exit")
                sys.exit(1)

    def ensure_required_arguments(self):
        """
        Ensure that certain Megatron arguments (from checkpoint) are present.
        If missing, either set defaults or exit.
        """

        self.check_for_arg('tensor_model_parallel_size')
        self.check_for_arg('pipeline_model_parallel_size')
        self.check_for_arg('num_layers')
        self.check_for_arg('hidden_size')
        self.check_for_arg('seq_length')
        self.check_for_arg('num_attention_heads')
        self.check_for_arg('max_position_embeddings')
        self.check_for_arg('position_embedding_type')
        self.check_for_arg('tokenizer_type')
        self.check_for_arg('iteration')
        self.check_for_arg('bert_binary_head')
        self.check_for_arg('disable_bias_linear', False)
        self.check_for_arg('params_dtype')
        self.check_for_arg('swiglu', False)

        self._maybe_ensure_additional_required_arguments()

    def _initialize_converter_fake_groups(self, mpu):
        """Initialize fake process groups used by checkpoint converter-only runs."""
        self._converter_fake_groups = init_converter_fake_groups(
            mpu,
            tp_size=self.margs.tensor_model_parallel_size,
            pp_size=self.margs.pipeline_model_parallel_size,
            ep_size=self.margs.expert_model_parallel_size,
            cp_size=getattr(self.margs, 'context_parallel_size', 1),
        )

    def _set_converter_fake_group_ranks(self, mpu, tp_rank, pp_rank):
        """Update fake process-group ranks for the TP/PP shard being loaded."""
        set_converter_fake_group_ranks(
            mpu,
            self._converter_fake_groups,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            tp_size=self.margs.tensor_model_parallel_size,
            ep_size=self.margs.expert_model_parallel_size,
        )

    def initialize_megatron_env(self):
        """
        Initialize Megatron global variables and fused kernels.
        """
        try:
            from megatron.training.global_vars import set_global_variables
            from megatron.core import mpu
            from megatron.legacy import fused_kernels
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            self.queue.put("exit")
            sys.exit(1)

        set_global_variables(self.margs, build_tokenizer=self.build_tokenizer)
        # megatron.core.dist_checkpointing.load() calls torch.distributed.get_world_size()
        # directly (determine_global_metadata) regardless of the mpu fake-group shim below,
        # so a real (if trivial, single-rank) default process group must exist even for a
        # single-process converter run.
        if not torch.distributed.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12355")
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)

        # dist_checkpointing.load()'s validate_access_integrity (default True) checks that
        # all *real* ranks collectively cover a sharded tensor exactly once. This converter
        # loads TP/PP shards one at a time in a single real process (looping fake mpu ranks
        # around each dist_checkpointing.load() call, per get_models_for_pipeline_stage
        # below), so any call here only ever sees its own shard against a real world_size of
        # 1 -- that per-call view is always "incomplete" relative to the tensor's full
        # sharding, independent of whether the overall conversion is actually correct.
        # Coverage across the whole tensor is instead guaranteed by the outer per-rank loop,
        # so this cross-rank check is inapplicable to this execution model; disable it here
        # rather than in the shared training checkpoint-loading path.
        import megatron.core.dist_checkpointing as _dist_checkpointing

        if not getattr(_dist_checkpointing, "_converter_validate_access_integrity_patched", False):
            _orig_dist_ckpt_load = _dist_checkpointing.load

            def _dist_ckpt_load_no_validate(*load_args, **load_kwargs):
                load_kwargs.setdefault("validate_access_integrity", False)
                return _orig_dist_ckpt_load(*load_args, **load_kwargs)

            _dist_checkpointing.load = _dist_ckpt_load_no_validate
            _dist_checkpointing._converter_validate_access_integrity_patched = True

        mpu.set_tensor_model_parallel_world_size(self.margs.tensor_model_parallel_size)
        mpu.set_pipeline_model_parallel_world_size(self.margs.pipeline_model_parallel_size)
        mpu.set_virtual_pipeline_model_parallel_world_size(self.margs.virtual_pipeline_model_parallel_size)
        mpu.set_expert_model_parallel_world_size(self.margs.expert_model_parallel_size)
        mpu.set_expert_tensor_parallel_world_size(self.margs.tensor_model_parallel_size)

        # Converter fallback: only install fake groups if model-parallel groups
        # are missing. This avoids overriding real distributed initialization.
        has_tp_group = mpu.get_tensor_model_parallel_group(check_initialized=False) is not None
        has_pp_group = mpu.get_pipeline_model_parallel_group(check_initialized=False) is not None
        self._use_converter_fake_groups = not (has_tp_group and has_pp_group)
        if self._use_converter_fake_groups:
            self._initialize_converter_fake_groups(mpu)
            self._set_converter_fake_group_ranks(mpu, tp_rank=0, pp_rank=0)
        fused_kernels.load(self.margs)

    def compute_true_vocab_size(self):
        """Determine the 'true' (non-padded) vocab size."""
        if self.args.true_vocab_size is not None:
            return self.args.true_vocab_size
        elif self.args.vocab_file is not None:
            vocab = json.load(open(self.args.vocab_file))
            return len(vocab)
        else:
            return None

    def verify_vocabs_match(self, true_vocab_size):
        """
        If both --true-vocab-size and --vocab-file are specified, verify they match.
        Return False (and exit) if they don't match; True otherwise.
        """
        if self.args.true_vocab_size is not None and self.args.vocab_file is not None:
            vocab = json.load(open(self.args.vocab_file))
            if len(vocab) != self.args.true_vocab_size:
                print("Both --true-vocab-size and --vocab-file specified but vocab sizes do not match. Aborting.")
                return False
        return True

    def _load_plain_model_tensors(self):
        """
        Load all *model* tensors (no optimizer/RNG state) fully unsharded, i.e.
        without reconstructing however the checkpoint happened to be split across
        ranks at save time (TP/PP *and*, for --ckpt-fully-parallel-save checkpoints,
        an additional data-parallel split of each tensor). Used by --use-plain-tensor-
        loading as a drop-in replacement for the regular load_checkpoint() call, which
        can't reconstruct that extra DP-level split from a single real process (see
        parse_megatron_args for the full explanation).

        Cached on self, since it's expensive and the result is identical across every
        pipeline/virtual-pipeline stage that calls this during a single conversion run.
        """
        if self._plain_tensors is None:
            from megatron.core import dist_checkpointing
            from megatron.training.checkpointing import get_checkpoint_name

            # self.args.load_dir may be a bare tracker directory (a
            # latest_checkpointed_iteration.txt + iter_NNNNNNN layout) rather than the
            # checkpoint itself -- load_tensors_metadata()/dist_checkpointing.load() need
            # the actual checkpoint root (containing .metadata directly), same as what
            # the regular load_checkpoint() path resolves to internally.
            checkpoint_dir = get_checkpoint_name(
                self.args.load_dir, self.margs.iteration, release=False, return_base_dir=True
            )

            sharded_metadata = dist_checkpointing.load_tensors_metadata(checkpoint_dir)
            model_metadata = {
                k: v for k, v in sharded_metadata.items() if not k.startswith('optimizer.')
            }
            print(f"Loading {len(model_metadata)} model tensors fully unsharded "
                  f"(--use-plain-tensor-loading)...")
            plain_tensors = dist_checkpointing.load(
                model_metadata, checkpoint_dir, validate_access_integrity=False
            )
            self._plain_tensors = self._expand_stacked_layer_tensors(plain_tensors)
        return self._plain_tensors

    def _expand_stacked_layer_tensors(self, plain_tensors):
        """
        The checkpoint's *unsharded* representation stores each per-layer parameter as
        one tensor stacked across all layers (e.g. 'decoder.layers.mlp.linear_fc1.weight'
        with shape (num_layers, ...)), rather than the per-layer, numerically-indexed
        keys a real model's state_dict() uses (e.g. 'decoder.layers.0.mlp.linear_fc1.
        weight', '...layers.1...', ...). Slice each stacked tensor back into per-layer
        entries under those keys so model.load_state_dict(..., strict=False) can match
        them; embeddings/output_layer/final_norm keys have no layer axis and pass through
        unchanged. Slicing returns views (no extra memory copy).
        """
        block_key = {"GPT": "decoder", "BERT": "encoder"}[self.args.model_type]
        layer_prefix = f"{block_key}.layers."
        num_layers = self.margs.num_layers

        expanded = {}
        for key, tensor in plain_tensors.items():
            if key.startswith(layer_prefix):
                suffix = key[len(layer_prefix):]
                assert tensor.shape[0] == num_layers, (
                    f"Expected {key} to have a leading layer axis of size {num_layers} "
                    f"(margs.num_layers), got shape {tuple(tensor.shape)}"
                )
                for layer_idx in range(num_layers):
                    expanded[f"{layer_prefix}{layer_idx}.{suffix}"] = tensor[layer_idx]
            else:
                expanded[key] = tensor
        return expanded

    def load_model_shards(self, model_provider, dtype):
        """
        Build and load model shards for each tensor-parallel rank, returning:
          - A nested list of loaded models by [pipeline_rank][virtual_pipeline_rank].
          - consumed_train_samples, consumed_valid_samples
        """
        from megatron.core import mpu
        from megatron.training.checkpointing import load_checkpoint

        consumed_train_samples = None
        consumed_valid_samples = None
        tp_size = self.margs.tensor_model_parallel_size
        pp_size = self.margs.pipeline_model_parallel_size
        vp_size = self.margs.virtual_pipeline_model_parallel_size or 1

        all_models = []  # all_models[pp_rank][vp_rank] = [list of models across TP ranks]

        def get_models_for_pipeline_stage(count, dtype, pp_rank):
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            local_models_for_stage = [[] for _ in range(vp_size)]
            for tp_rank in range(count):
                if self._use_converter_fake_groups:
                    self._set_converter_fake_group_ranks(mpu, tp_rank=tp_rank, pp_rank=pp_rank)
                else:
                    mpu.set_tensor_model_parallel_rank(tp_rank)
                    mpu.set_pipeline_model_parallel_rank(pp_rank)
                model_list = []

                for i in range(vp_size):
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    pre_process = mpu.is_pipeline_first_stage()
                    post_process = mpu.is_pipeline_last_stage()
                    model_kwargs = {
                        "pre_process": pre_process,
                        "post_process": post_process,
                    }
                    if vp_size > 1:
                        model_kwargs["vp_stage"] = i
                    this_model = model_provider(**model_kwargs).to(dtype)
                    model_list.append(this_model)

                # Each time we load, we set counters to 0, pass None for optimizer/ LR
                self.margs.consumed_train_samples = 0
                self.margs.consumed_valid_samples = 0
                self.margs.exit_on_missing_checkpoint = True
                if getattr(self.args, 'use_plain_tensor_loading', False):
                    plain_tensors = self._load_plain_model_tensors()
                    for m in model_list:
                        missing, _unexpected = m.load_state_dict(plain_tensors, strict=False)
                        # '_extra_state' entries hold TE's FP8 amax-history/scaling
                        # metadata, not real weights; they're absent from the checkpoint
                        # (and fine to leave at their module-initialized default) whenever
                        # training didn't use FP8, as here.
                        missing = [k for k in missing if not k.endswith('_extra_state')]
                        assert not missing, (
                            f"Missing keys when populating model from plain (unsharded) "
                            f"checkpoint tensors: {missing}"
                        )
                else:
                    load_checkpoint(model_list, None, None)

                # Validate that train/valid samples match across ranks
                nonlocal consumed_train_samples, consumed_valid_samples
                if consumed_train_samples is not None:
                    assert self.margs.consumed_train_samples == consumed_train_samples
                else:
                    consumed_train_samples = self.margs.consumed_train_samples

                if consumed_valid_samples is not None:
                    assert self.margs.consumed_valid_samples == consumed_valid_samples
                else:
                    consumed_valid_samples = self.margs.consumed_valid_samples

                for vp_rank in range(vp_size):
                    local_models_for_stage[vp_rank].append(model_list[vp_rank])

                # Print memory usage
                print_memory_usage("loader", tp_rank, count)

            return local_models_for_stage

        # Load shards for each pipeline rank
        mpu.set_virtual_pipeline_model_parallel_rank(0)
        for pp_rank in range(pp_size):
            all_models.append(get_models_for_pipeline_stage(tp_size, dtype, pp_rank))

        return all_models, consumed_train_samples, consumed_valid_samples
    
    def send_metadata_over_queue(self):
        # Let the consumer know the overall metadata:
        self.md.consumed_train_samples = self.consumed_train_samples
        self.md.consumed_valid_samples = self.consumed_valid_samples
        self.queue.put(self.md)

    def queue_put(self, name, msg):
        print(f"sending {name}")
        msg["name"] = name
        self.queue.put(msg)

    def send_llm_over_queue(self, schema):
        """
        Using self.all_models, extract model parameters and send them over the queue.
        """
        # 2) Transformer layers
        tp_size = self.margs.tensor_model_parallel_size
        pp_size = self.margs.pipeline_model_parallel_size
        vp_size = self.margs.virtual_pipeline_model_parallel_size or 1

        # all_models[pp_rank][vp_rank] is a list across TP ranks
        # We'll start with pipeline=0, vp=0 for embeddings/final norm
        first_pipeline_models = self.all_models[0][0]

        # 1) Embeddings
        embeddings = [schema.get("embeddings", m) for m in first_pipeline_models]
        message = {
            "word embeddings": torch.cat([e["word"] for e in embeddings], dim=0)
        }
        if self.md.position_embedding_type == 'learned_absolute':
            # Only send one set from rank 0
            message["position embeddings"] = embeddings[0]["pos"]
        else:
            assert embeddings[0]["pos"] is None
        self.queue_put("embeddings", message)

        total_layer_num = 0
        for vp_rank in range(vp_size):
            for pp_rank in range(pp_size):
                models = self.all_models[pp_rank][vp_rank]
                num_layers = schema.get_num_layers(models[0])
                for layer_idx in range(num_layers):
                    message = {}
                    layer = schema.get_layer(models[0], layer_idx)

                    # Non-parallel params
                    message["input norm weight"] = layer["self_attn_norm_weight"]
                    message["post norm weight"] = layer["mlp_norm_weight"]
                    if self.md.norm_has_bias:
                        message["input norm bias"] = layer["self_attn_norm_bias"]
                        message["post norm bias"] = layer["mlp_norm_bias"]
                    if self.md.linear_bias:
                        message["dense bias"] = layer["self_attn_proj_bias"]
                        message["mlp l1 bias"] = layer["mlp_fc2_bias"]

                    # Collect parallel parameters
                    qkv_weight, qkv_bias = [], []
                    dense_weight = []
                    mlp_l0_weight, mlp_l0_bias = [], []
                    mlp_l1_weight = []

                    for model_tp in models:
                        layer_p = schema.get_layer(model_tp, layer_idx)
                        qkv_weight.append(layer_p["self_attn_qkv_weight"])
                        dense_weight.append(layer_p["self_attn_proj_weight"])
                        mlp_l0_weight.append(layer_p["mlp_fc1_weight"])
                        mlp_l1_weight.append(layer_p["mlp_fc2_weight"])
                        if self.md.qkv_bias:
                            qkv_bias.append(layer_p["self_attn_qkv_bias"])
                        if self.md.linear_bias:
                            mlp_l0_bias.append(layer_p["mlp_fc1_bias"])

                    # If we are using SwiGLU, chunk each mlp_l0_weight
                    if self.md.swiglu:
                        for i in range(tp_size):
                            mlp_l0_weight[i] = torch.chunk(mlp_l0_weight[i], 2, dim=0)
                        message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
                        message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
                    else:
                        message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

                    # Standard concatenations
                    message["qkv weight"] = torch.cat(qkv_weight, dim=0)
                    message["dense weight"] = torch.cat(dense_weight, dim=1)
                    message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)

                    if self.md.qkv_bias:
                        message["qkv bias"] = torch.cat(qkv_bias, dim=0)
                    if self.md.linear_bias:
                        if self.md.swiglu:
                            for i in range(tp_size):
                                mlp_l0_bias[i] = torch.chunk(mlp_l0_bias[i], 2, dim=0)
                            message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias], dim=0)
                            message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias], dim=0)
                        else:
                            message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)

                    if self.md.qk_layernorm:
                        message["q norm weight"] = layer["self_attn_q_layernorm_weight"]
                        message["k norm weight"] = layer["self_attn_k_layernorm_weight"]

                    if self.md.xielu:
                        # there is also mlp alpha_p and alpha_n?
                        message["mlp xielu alpha p"] = layer["mlp_xielu_alpha_p"]
                        message["mlp xielu alpha n"] = layer["mlp_xielu_alpha_n"]

                        if "mlp_xielu_beta" in layer and layer["mlp_xielu_beta"] is not None:
                            beta = layer["mlp_xielu_beta"]
                            message["mlp xielu beta"] = (
                                beta if isinstance(beta, torch.Tensor) else torch.tensor(beta)
                            )
                        if "mlp_xielu_eps" in layer and layer["mlp_xielu_eps"] is not None:
                            message["mlp xielu eps"] = layer["mlp_xielu_eps"]

                    self.queue_put(f"transformer layer {total_layer_num}", message)
                    total_layer_num += 1

        # 3) Final norm
        final_norm = schema.get("final_norm", models[0])
        message = {"weight": final_norm["weight"]}
        if self.md.norm_has_bias:
            message["bias"] = final_norm["bias"]
        self.queue_put("final norm", message)

        # 4) Output layer
        if self.md.output_layer:
            output_layers = [schema.get("output_layer", m) for m in models]
            message = {
                "weight": torch.cat([layer["weight"] for layer in output_layers], dim=0),
            }
            self.queue_put("output layer", message)

        # 5) BERT-specific parameters
        if self.md.model_type == 'BERT':
            # Pooler
            pooler = schema.get("pooler", models[0])
            message = {
                "weight": pooler["weight"],
                "bias": pooler["bias"],
            }
            self.queue_put("pooler", message)

            # LM head
            lm_head = schema.get("lm_head", models[0])
            message = {
                "dense weight": lm_head["dense_weight"],
                "dense bias": lm_head["dense_bias"],
                "norm weight": lm_head["norm_weight"],
            }
            if self.md.norm_has_bias:
                message["norm bias"] = lm_head["norm_bias"]
            self.queue_put("lm head", message)

            # Binary head
            if self.md.bert_binary_head:
                binary_head = schema.get("binary_head", models[0])
                message = {
                    "weight": binary_head["weight"],
                    "bias": binary_head["bias"],
                }
                self.queue_put("binary head", message)

        # Done
        self.queue.put("done")

    def load(self):
        """
        Orchestrate the entire flow of loading the Megatron checkpoint.
        """
        # 1) Parse Megatron arguments
        self.parse_megatron_args()

        # 2) Ensure required arguments are present
        self.ensure_required_arguments()

        # 3) Import the correct model provider (GPT or BERT)
        model_provider = self.import_model_provider()

        # 4) Initialize the Megatron environment
        self.initialize_megatron_env()

        # 5) Determine the true vocab size and verify if both sources match
        true_vocab_size = self.compute_true_vocab_size()
        if not self.verify_vocabs_match(true_vocab_size):
            self.queue.put("exit")
            sys.exit(1)

        # 6) Build metadata
        self.md = self.build_checkpoint_metadata(true_vocab_size)

        # 7) Load all model shards
        self.all_models, self.consumed_train_samples, self.consumed_valid_samples = self.load_model_shards(
            model_provider,
            self.md.params_dtype
        )

        # 8) Send model over the queue        
        self.send_model_over_queue()

    def build_checkpoint_metadata(self, true_vocab_size):
        """
        Construct a simple namespace for all relevant model metadata.
        """
        norm_has_bias = True
        if hasattr(self.checkpoint_args, 'normalization'):
            # For older models, normalization was always "LayerNorm".
            norm_has_bias = (self.checkpoint_args.normalization == "LayerNorm")

        md = types.SimpleNamespace()
        md.model_type = self.args.model_type
        md.num_layers = self.margs.num_layers
        md.hidden_size = self.margs.hidden_size
        md.seq_length = self.margs.seq_length
        md.num_attention_heads = self.margs.num_attention_heads
        md.max_position_embeddings = self.margs.max_position_embeddings
        md.tokenizer_type = self.margs.tokenizer_type
        md.iteration = self.margs.iteration
        md.params_dtype = self.margs.params_dtype
        md.bert_binary_head = self.margs.bert_binary_head
        md.output_layer = self.margs.untie_embeddings_and_output_weights
        md.position_embedding_type = self.margs.position_embedding_type
        md.linear_bias = self.margs.add_bias_linear
        md.qkv_bias = self.margs.add_qkv_bias
        md.norm_has_bias = norm_has_bias
        md.swiglu = self.margs.swiglu
        md.xielu = self.margs.xielu
        md.qk_layernorm = self.margs.qk_layernorm
        md.previous_tensor_parallel_size = self.margs.tensor_model_parallel_size
        md.previous_pipeline_parallel_size = self.margs.pipeline_model_parallel_size
        md.true_vocab_size = true_vocab_size
        md.make_vocab_size_divisible_by = self.margs.make_vocab_size_divisible_by
        md.checkpoint_args = self.checkpoint_args
        md.use_legacy_models = self.margs.use_legacy_models
        return md

    def build_sys_argv(self):
        """
        Construct a sys.argv list for Megatron's argument parser.
        This centralizes the hack of overwriting sys.argv.
        """

        return [
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
            '--no-initialization',
            '--mock-data',  # To pass the "blend data checks" in arguments.py
            '--load', self.args.load_dir,
            '--exit-on-missing-checkpoint',
            '--use-mp-args-from-checkpoint-args',
            '--no-one-logger',
        ]

    def import_model_provider(self):
        """Return the correct model_provider function depending on GPT vs. BERT."""
        raise NotImplementedError

    def send_model_over_queue(self):
        """Creates model schema and sends the model over the queue"""
        raise NotImplementedError
