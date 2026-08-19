# Copyright (c) 2026, Swiss AI. All rights reserved.

"""VPP-aware checkpoint saver plugin.

Sibling of ``saver_core``. Used when the target topology has
``virtual_pipeline_model_parallel_size > 1`` so the on-disk per-Python-object
shard descriptors match a training runtime that constructs ``vp_size`` model
chunks per pipeline rank. VPP routing is intentionally kept out of
``saver_base.py`` / ``saver_core.py`` because it is one-time conversion logic
that does not need to live in shared infrastructure used by
LLaVA/BERT/retro/vision savers.

Most of ``receive_lm`` here is text-duplicated from
``MegatronCheckpointSaverBase.receive_lm`` because it is tightly coupled to the
loader's queue-message protocol. Only the layer-iteration shape and the
post-process placement differ:

  - layers iterate over ``topology.build_layer_route(num_layers, pp, vp)``
    instead of a nested ``for pp: for layer_within_pp:`` loop.
  - layer placement targets ``self.get_local_models(pp, ep, tp)[vp_rank]``
    instead of a single model.
  - final_norm / output_layer / pooler / lm_head / binary_head are placed on
    the ``(last_pp, last_vp)`` chunk (where ``post_process=True`` lives) once
    after the layer loop, instead of inside the per-pp conditional.

If the loader's message protocol changes, both copies must be updated in
lockstep. Schema source-of-truth is ``saver_base.MegatronCheckpointSaverBase.receive_lm``.
"""

import sys
from functools import cached_property

import torch

from saver_core import MegatronCheckpointSaverLLM, add_arguments as _core_add_arguments
from topology import (
    assert_uniform_pipeline_layout,
    build_layer_route,
    derive_vp_size,
    is_post_process_chunk,
    is_pre_process_chunk,
    post_process_chunk_index,
)
from utils import chunk_bias, chunk_weight, pop_xielu_params


def add_arguments(parser):
    """Register every flag the LLM saver accepts plus VPP-specific flags."""
    _core_add_arguments(parser)
    group = parser.add_argument_group(title='VPP saver')
    group.add_argument(
        '--target-num-layers-per-virtual-pipeline-stage',
        type=int,
        default=None,
        help='Number of transformer layers each virtual pipeline chunk owns at '
             'the target topology. When set, the saver builds vp_size model '
             'chunks per pipeline rank so the resulting checkpoint loads '
             'correctly into a training run that uses the same flag.',
    )


class MegatronCheckpointSaverWithVPP(MegatronCheckpointSaverLLM):
    """LLM saver that materializes virtual-pipeline chunks on disk."""

    @cached_property
    def _vp_size(self):
        margs_vp = getattr(self.margs, "virtual_pipeline_model_parallel_size", None) or 1
        target_n = getattr(
            self.args, "target_num_layers_per_virtual_pipeline_stage", None
        )
        if target_n is not None:
            derived = derive_vp_size(
                num_layers=self.md.num_layers,
                pp_size=self.args.target_pipeline_parallel_size,
                num_layers_per_virtual_pipeline_stage=target_n,
            )
            if derived != margs_vp:
                raise RuntimeError(
                    f"Derived vp_size ({derived}) disagrees with "
                    f"margs.virtual_pipeline_model_parallel_size ({margs_vp}). "
                    f"This usually means the source checkpoint or target args use a "
                    f"non-uniform pipeline layout (custom pipeline_model_parallel_layout, "
                    f"uneven first/last pipeline stages, or "
                    f"account_for_embedding/loss_in_pipeline_split). saver_vpp only "
                    f"supports the simple uniform layout — see "
                    f"topology.UNSUPPORTED_LAYOUT_MARGS."
                )
        return margs_vp

    @cached_property
    def _layer_route(self):
        return build_layer_route(
            num_layers=self.md.num_layers,
            pp_size=self.args.target_pipeline_parallel_size,
            vp_size=self._vp_size,
        )

    def _require_head_attr(self, target, attr, kind):
        """Fail fast if ``target`` doesn't have the head we received."""
        if not hasattr(target, attr):
            print(f"ERROR: got a {kind}, but model does not have one")
            sys.exit(1)

    def initialize_megatron_env(self):
        super().initialize_megatron_env()
        try:
            from megatron.core import mpu
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            sys.exit(1)
        vpp_world_size = getattr(
            self.margs, "virtual_pipeline_model_parallel_size", None
        )
        if vpp_world_size is not None:
            mpu.set_virtual_pipeline_model_parallel_world_size(vpp_world_size)

    def build_sys_argv(self):
        argv = super().build_sys_argv()
        n = getattr(self.args, 'target_num_layers_per_virtual_pipeline_stage', None)
        if n is not None:
            argv.extend(['--num-layers-per-virtual-pipeline-stage', str(n)])
        return argv

    def get_local_models(self, pp_rank, ep_rank, tp_rank):
        """Return the list of vp model chunks for ``(pp, ep, tp)``.

        Length is ``vp_size`` (1 if VPP is disabled). Each chunk is constructed
        via ``model_provider`` with the chunk's ``vp_stage`` so layer numbering
        and ``pre_process`` / ``post_process`` flags follow Megatron's native
        VPP semantics.
        """
        if self.models[pp_rank][ep_rank][tp_rank] is None:
            from megatron.core import mpu

            self._set_converter_fake_group_ranks(mpu, tp_rank, pp_rank, ep_rank)
            vp_size = self._vp_size
            pp_size = self.args.target_pipeline_parallel_size
            chunks = []
            for vp_rank in range(vp_size):
                pre_process = is_pre_process_chunk(pp_rank, vp_rank)
                post_process = is_post_process_chunk(pp_rank, vp_rank, pp_size=pp_size, vp_size=vp_size)
                if vp_size > 1:
                    mpu.set_virtual_pipeline_model_parallel_rank(vp_rank)
                    chunks.append(
                        self.model_provider(
                            pre_process=pre_process,
                            post_process=post_process,
                            vp_stage=vp_rank,
                        ).to(self.md.params_dtype)
                    )
                else:
                    # Preserve the exact non-VPP calling convention used by
                    # MegatronCheckpointSaverBase.get_local_model so anything
                    # that ran with ``--saver core`` still runs with
                    # ``--saver vpp`` when no VPP flag is set.
                    chunks.append(
                        self.model_provider(pre_process, post_process).to(self.md.params_dtype)
                    )
            if vp_size > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(0)
            self.models[pp_rank][ep_rank][tp_rank] = chunks
        return self.models[pp_rank][ep_rank][tp_rank]

    def get_local_model(self, pp_rank, ep_rank, tp_rank):
        """Backward-compatible accessor used by the embedding placement path."""
        return self.get_local_models(pp_rank, ep_rank, tp_rank)[0]

    def save_local_models_to_checkpoint(self):
        try:
            from megatron.training.checkpointing import save_checkpoint
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            sys.exit(1)

        # PP rank is set inside get_local_models via _set_converter_fake_group_ranks;
        # no separate mpu.set_pipeline_model_parallel_rank call needed here.
        for pp_rank in range(self.args.target_pipeline_parallel_size):
            for ep_rank in range(self.args.target_expert_parallel_size):
                for tp_rank in range(self.args.target_tensor_parallel_size):
                    save_checkpoint(
                        self.md.iteration,
                        self.get_local_models(pp_rank, ep_rank, tp_rank),
                        None, None,
                        num_floating_point_operations_so_far=0,
                        pipeline_rank=pp_rank,
                        pipeline_parallel=self.args.target_pipeline_parallel_size > 1,
                        expert_rank=ep_rank,
                        expert_parallel=self.args.target_expert_parallel_size > 1,
                        tensor_rank=tp_rank,
                    )
                    self.models[pp_rank][ep_rank][tp_rank] = None

    def receive_lm(self, schema, prefix=None):
        """Route-driven scatter against ``topology.build_layer_route``.

        Mirrors ``MegatronCheckpointSaverBase.receive_lm`` for everything below
        the layer-iteration outer loop. See module docstring for what changes.
        """
        assert_uniform_pipeline_layout(self.margs)
        self._vocab_extension_config = self._resolve_vocab_extension_config()

        pp_size = self.args.target_pipeline_parallel_size
        ep_size = self.args.target_expert_parallel_size
        tp_size = self.args.target_tensor_parallel_size
        vp_size = self._vp_size
        last_pp, last_vp = post_process_chunk_index(pp_size=pp_size, vp_size=vp_size)

        embeddings_msg = self.queue_get("embeddings")
        pos_embed = None
        if self.md.position_embedding_type == 'learned_absolute':
            pos_embed = embeddings_msg.pop("position embeddings")
        orig_word_embed = embeddings_msg.pop("word embeddings")
        self.check_message(embeddings_msg)

        full_word_embed, padded_vocab_size = self._pad_vocab_table(
            orig_word_embed,
            true_vocab_size=self.md.true_vocab_size,
            table_name="word embeddings",
        )
        self.margs.padded_vocab_size = padded_vocab_size
        if self._vocab_extension_config is not None and hasattr(self.margs, "vocab_size"):
            self.margs.vocab_size = self._vocab_extension_config.target_vocab_size

        out_word_embed = torch.chunk(full_word_embed, tp_size, dim=0)

        for ep_rank in range(ep_size):
            for tp_rank in range(tp_size):
                embed_chunk = self.get_local_models(0, ep_rank, tp_rank)[0]
                if pos_embed is None:
                    assert not schema.has_position_embeddings(embed_chunk)
                schema.set("embeddings", embed_chunk, {
                    "pos": pos_embed,
                    "word": out_word_embed[tp_rank],
                })

        for global_layer, (pp_rank, vp_rank, local_layer) in enumerate(self._layer_route):
            msg = self.queue_get(f"transformer layer {global_layer}")

            input_norm_weight = msg.pop("input norm weight")
            post_norm_weight = msg.pop("post norm weight")
            if self.md.norm_has_bias:
                input_norm_bias = msg.pop("input norm bias")
                post_norm_bias = msg.pop("post norm bias")

            qkv_weight = chunk_weight(msg.pop("qkv weight"), "column", tp_size)
            dense_weight = chunk_weight(msg.pop("dense weight"), "row", tp_size)
            mlp_l1_weight = chunk_weight(msg.pop("mlp l1 weight"), "row", tp_size, ep_size)

            if self.margs.num_experts:
                router = msg.pop("router weight")

            if self.md.swiglu:
                mlp_l0_weight_W = chunk_weight(msg.pop("mlp l0 weight W"), "column", tp_size, ep_size)
                mlp_l0_weight_V = chunk_weight(msg.pop("mlp l0 weight V"), "column", tp_size, ep_size)
                mlp_l0_weight = torch.cat((mlp_l0_weight_W, mlp_l0_weight_V), dim=-2)
            else:
                mlp_l0_weight = chunk_weight(msg.pop("mlp l0 weight"), "column", tp_size, ep_size)

            if self.md.qkv_bias:
                qkv_bias = chunk_bias(msg.pop("qkv bias"), 'column', tp_size)
            if self.md.linear_bias:
                dense_bias = msg.pop("dense bias")
                mlp_l1_bias = chunk_bias(msg.pop("mlp l1 bias"), 'row', tp_size, ep_size)
                if self.md.swiglu:
                    mlp_l0_bias_W = chunk_bias(msg.pop("mlp l0 bias W"), 'column', tp_size, ep_size)
                    mlp_l0_bias_V = chunk_bias(msg.pop("mlp l0 bias V"), 'column', tp_size, ep_size)
                    mlp_l0_bias = torch.cat((mlp_l0_bias_W, mlp_l0_bias_V), dim=-1)
                else:
                    mlp_l0_bias = chunk_bias(msg.pop("mlp l0 bias"), 'column', tp_size, ep_size)

            if getattr(self.md, 'qk_layernorm', False):
                q_layernorm_weight = msg.pop("q norm weight")
                k_layernorm_weight = msg.pop("k norm weight")

            # See saver_base.receive_lm: beta is drained but not propagated because
            # megatron.core.activations.XIELU.beta is a Python float (not a tensor).
            xielu_eps = None
            if getattr(self.md, 'xielu', False):
                xielu_alpha_p, xielu_alpha_n, _xielu_beta, xielu_eps = (
                    pop_xielu_params(msg)
                )

            for ep_rank in range(ep_size):
                for tp_rank in range(tp_size):
                    params_dict = {
                        "self_attn_norm_weight": input_norm_weight,
                        "self_attn_qkv_weight": qkv_weight[tp_rank],
                        "self_attn_proj_weight": dense_weight[tp_rank],
                        "mlp_norm_weight": post_norm_weight,
                    }
                    if self.margs.num_experts:
                        params_dict.update({
                            "mlp_fc1_weight": mlp_l0_weight[ep_rank][tp_rank],
                            "mlp_fc2_weight": mlp_l1_weight[ep_rank][tp_rank],
                        })
                    else:
                        params_dict.update({
                            "mlp_fc1_weight": mlp_l0_weight[tp_rank],
                            "mlp_fc2_weight": mlp_l1_weight[tp_rank],
                        })
                    params_dict.update({
                        "self_attn_norm_bias": input_norm_bias if self.md.norm_has_bias else None,
                        "mlp_norm_bias": post_norm_bias if self.md.norm_has_bias else None,
                    })
                    if self.md.qkv_bias:
                        params_dict.update({"self_attn_qkv_bias": qkv_bias[tp_rank]})
                    if self.md.linear_bias:
                        params_dict.update({"self_attn_proj_bias": dense_bias})
                        if self.margs.num_experts:
                            params_dict.update({
                                "mlp_fc1_bias": mlp_l0_bias[ep_rank][tp_rank],
                                "mlp_fc2_bias": mlp_l1_bias[ep_rank],
                            })
                        else:
                            params_dict.update({
                                "mlp_fc1_bias": mlp_l0_bias[tp_rank],
                                "mlp_fc2_bias": mlp_l1_bias,
                            })
                    if self.margs.num_experts:
                        params_dict.update({"router_weight": router})
                    if getattr(self.md, 'qk_layernorm', False):
                        params_dict.update({
                            "self_attn_q_layernorm_weight": q_layernorm_weight,
                            "self_attn_k_layernorm_weight": k_layernorm_weight,
                        })
                    if getattr(self.md, 'xielu', False):
                        params_dict.update({
                            "mlp_xielu_alpha_p": xielu_alpha_p,
                            "mlp_xielu_alpha_n": xielu_alpha_n,
                        })
                        if xielu_eps is not None:
                            params_dict["mlp_xielu_eps"] = xielu_eps
                    chunk = self.get_local_models(pp_rank, ep_rank, tp_rank)[vp_rank]
                    schema.set_layer(chunk, local_layer, params_dict)

            self.check_message(msg)

        pp_local_models = [
            self.get_local_models(last_pp, ep_rank, tp_rank)[last_vp]
            for ep_rank in range(ep_size)
            for tp_rank in range(tp_size)
        ]
        head_target = (
            pp_local_models[0] if prefix is None
            else getattr(pp_local_models[0], prefix)
        )

        msg = self.queue_get("final norm")
        final_norm_weight = msg.pop("weight")
        if self.md.norm_has_bias:
            final_norm_bias = msg.pop("bias")
        for eptp_rank, model in enumerate(pp_local_models):
            tp_rank = eptp_rank % tp_size
            schema.set("final_norm", model, {
                "weight": final_norm_weight,
                "bias": final_norm_bias if self.md.norm_has_bias else None,
            })
            if last_pp != 0 and not self.md.output_layer:
                # Tied embeddings: copy word embeddings to the post-process chunk.
                schema.set("output_layer", model, {
                    "weight": out_word_embed[tp_rank],
                })
        del final_norm_weight
        if self.md.norm_has_bias:
            del final_norm_bias
        self.check_message(msg)

        if self.md.output_layer:
            msg = self.queue_get("output layer")
            self._require_head_attr(head_target, 'output_layer', 'output layer')
            output_layer_weight, output_padded_size = self._pad_vocab_table(
                msg.pop("weight"),
                true_vocab_size=self.md.true_vocab_size,
                table_name="output layer",
                seed_offset=1,
            )
            assert output_padded_size == padded_vocab_size, (
                f"output layer padded size ({output_padded_size}) disagrees "
                f"with word embeddings ({padded_vocab_size})"
            )
            output_layer_weight = torch.chunk(output_layer_weight, tp_size, dim=0)
            for eptp_rank, model in enumerate(pp_local_models):
                tp_rank = eptp_rank % tp_size
                schema.set("output_layer", model, {
                    "weight": output_layer_weight[tp_rank],
                })
            self.check_message(msg)

        msg = self.queue_get()
        if msg != "done" and msg["name"] == "pooler":
            self._require_head_attr(head_target, 'pooler', 'pooler')
            print("received pooler")
            pooler_weight = msg.pop("weight")
            pooler_bias = msg.pop("bias")
            for model in pp_local_models:
                schema.set("pooler", model, {
                    "weight": pooler_weight,
                    "bias": pooler_bias,
                })
            del pooler_weight
            del pooler_bias
            self.check_message(msg)
            msg = self.queue_get()

        if msg != "done" and msg["name"] == "lm head":
            self._require_head_attr(head_target, 'lm_head', 'lm head')
            print("received lm head")
            lm_head_dense_weight = msg.pop("dense weight")
            lm_head_dense_bias = msg.pop("dense bias")
            lm_head_norm_weight = msg.pop("norm weight")
            if self.md.norm_has_bias:
                lm_head_norm_bias = msg.pop("norm bias")
            for model in pp_local_models:
                schema.set("lm_head", model, {
                    "dense_weight": lm_head_dense_weight,
                    "dense_bias": lm_head_dense_bias,
                    "norm_weight": lm_head_norm_weight,
                    "norm_bias": lm_head_norm_bias if self.md.norm_has_bias else None,
                })
            self.check_message(msg)
            msg = self.queue_get()

        if msg != "done" and msg["name"] == "binary head":
            self._require_head_attr(head_target, 'binary_head', 'binary head')
            print("received binary head")
            binary_head_weight = msg.pop("weight")
            binary_head_bias = msg.pop("bias")
            for model in pp_local_models:
                schema.set("binary_head", model, {
                    "weight": binary_head_weight,
                    "bias": binary_head_bias,
                })
            self.check_message(msg)
            msg = self.queue_get()

        if msg != "done":
            print("ERROR: got some more data but was expecting to be done")


def save_checkpoint(queue, args):
    """Required top-level function that creates the saver and calls its .save()."""
    saver = MegatronCheckpointSaverWithVPP(args, queue)
    try:
        saver.save()
    except Exception as e:
        raise e
