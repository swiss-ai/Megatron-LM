# Copyright (c) 2026, Swiss AI. All rights reserved.

"""Pure pipeline-topology helpers for VPP-aware checkpoint scatter.

This module owns every fact about where a transformer layer lives and which
chunk owns each "shared" tensor (embeddings, final norm, output layer) under
Megatron's pipeline + virtual-pipeline scheduling.

LIMITATION — only the **simple uniform** Megatron layout is supported. The
exact set of margs flags that diverge from this layout is enumerated in
``UNSUPPORTED_LAYOUT_MARGS`` below; ``assert_uniform_pipeline_layout`` is the
runtime guard.

This matches the simple-case branch of
``megatron.core.transformer.transformer_layer.get_transformer_layer_offset``.
If you need any of the unsupported layouts, extend ``build_layer_route`` to
delegate to the upstream helper and update ``UNSUPPORTED_LAYOUT_MARGS``.

A chunk at ``(pp_rank, vp_rank)`` holds ``num_layers // (pp_size * vp_size)``
consecutive global layers starting at::

    offset(pp, vp) = vp * (num_layers // vp_size) + pp * (num_layers // (pp_size * vp_size))

Centralizing these decisions keeps the saver mechanical: it scatters against
a route table and queries ``is_pre_process_chunk`` / ``is_post_process_chunk``
instead of duplicating the formulas.
"""


# (margs_attr_name, sentinel_meaning_unset). build_layer_route is invalid
# whenever any of these has a non-sentinel value.
#
# Names match argparse / margs (see megatron/training/arguments.py), NOT the
# downstream TransformerConfig field names. arguments.py:1434-1435 maps
# ``args.decoder_first_pipeline_num_layers``  -> ``num_layers_in_first_pipeline_stage``
# ``args.decoder_last_pipeline_num_layers``   -> ``num_layers_in_last_pipeline_stage``
# Inside the saver we only see margs, so we must guard on the CLI side names.
UNSUPPORTED_LAYOUT_MARGS = (
    ("pipeline_model_parallel_layout", None),
    ("decoder_first_pipeline_num_layers", None),
    ("decoder_last_pipeline_num_layers", None),
    ("account_for_embedding_in_pipeline_split", False),
    ("account_for_loss_in_pipeline_split", False),
)


def assert_uniform_pipeline_layout(margs):
    """Fail loudly if margs uses any pipeline-layout option this module doesn't model.

    See module docstring for the contract. The list of disqualifying flags is
    ``UNSUPPORTED_LAYOUT_MARGS``; any one of them set to a non-sentinel value
    means ``build_layer_route``'s simple-uniform formula is not authoritative.
    """
    incompatible = [
        name for name, sentinel in UNSUPPORTED_LAYOUT_MARGS
        if getattr(margs, name, sentinel) != sentinel
    ]
    if incompatible:
        raise NotImplementedError(
            "topology.build_layer_route only supports the simple uniform "
            "Megatron pipeline layout. The following margs would silently "
            f"change layer placement: {incompatible}. Extend "
            "build_layer_route to delegate to "
            "megatron.core.transformer.transformer_layer.get_transformer_layer_offset "
            "if you need these layouts."
        )


def derive_vp_size(*, num_layers, pp_size, num_layers_per_virtual_pipeline_stage):
    """Convert Megatron's user-facing knob into ``vp_size`` and validate.

    Megatron exposes ``--num-layers-per-virtual-pipeline-stage`` (the number of
    layers each virtual chunk owns). The internal ``vp_size`` (number of
    virtual chunks per PP rank) is derived as
    ``layers_per_pp // num_layers_per_virtual_pipeline_stage``.

    Returns the derived ``vp_size`` and raises ``ValueError`` for any topology
    that would produce uneven chunks.
    """
    if pp_size <= 0:
        raise ValueError(f"pp_size={pp_size} must be positive")
    if num_layers <= 0:
        raise ValueError(f"num_layers={num_layers} must be positive")
    if num_layers_per_virtual_pipeline_stage <= 0:
        raise ValueError(
            f"num_layers_per_virtual_pipeline_stage="
            f"{num_layers_per_virtual_pipeline_stage} must be positive"
        )
    if num_layers % pp_size != 0:
        raise ValueError(
            f"num_layers={num_layers} must be divisible by pp_size={pp_size}"
        )
    layers_per_pp = num_layers // pp_size
    if layers_per_pp % num_layers_per_virtual_pipeline_stage != 0:
        raise ValueError(
            f"layers per pipeline rank ({layers_per_pp}) must be divisible by "
            f"num_layers_per_virtual_pipeline_stage="
            f"{num_layers_per_virtual_pipeline_stage}"
        )
    return layers_per_pp // num_layers_per_virtual_pipeline_stage


def build_layer_route(*, num_layers, pp_size, vp_size):
    """Return ``[(pp_rank, vp_rank, local_idx)] * num_layers`` indexed by global layer.

    Args:
        num_layers: Total transformer layers in the model.
        pp_size: Pipeline parallel world size.
        vp_size: Virtual pipeline size (1 = no VPP).

    Raises:
        ValueError: if inputs are not positive or ``num_layers`` is not divisible
            by ``pp_size * vp_size``.
    """
    if pp_size <= 0 or vp_size <= 0:
        raise ValueError(
            f"pp_size={pp_size} and vp_size={vp_size} must be positive"
        )
    if num_layers <= 0:
        raise ValueError(f"num_layers={num_layers} must be positive")
    if num_layers % (pp_size * vp_size) != 0:
        raise ValueError(
            f"num_layers={num_layers} must be divisible by pp_size * vp_size = "
            f"{pp_size * vp_size}; got remainder "
            f"{num_layers % (pp_size * vp_size)}"
        )

    chunk_size = num_layers // (pp_size * vp_size)
    total_per_vp = num_layers // vp_size
    route = [None] * num_layers
    for vp in range(vp_size):
        for pp in range(pp_size):
            base = vp * total_per_vp + pp * chunk_size
            for local in range(chunk_size):
                route[base + local] = (pp, vp, local)
    return route


def is_pre_process_chunk(pp_rank, vp_rank):
    """The embedding (pre-process) chunk is always ``(pp=0, vp=0)``."""
    return pp_rank == 0 and vp_rank == 0


def is_post_process_chunk(pp_rank, vp_rank, *, pp_size, vp_size):
    """The final-norm / output-layer (post-process) chunk is the last vp chunk
    on the last pp rank."""
    return pp_rank == pp_size - 1 and vp_rank == vp_size - 1


def post_process_chunk_index(*, pp_size, vp_size):
    """``(pp_rank, vp_rank)`` of the chunk that owns post-process tensors."""
    if pp_size <= 0 or vp_size <= 0:
        raise ValueError(
            f"pp_size={pp_size} and vp_size={vp_size} must be positive"
        )
    return pp_size - 1, vp_size - 1
