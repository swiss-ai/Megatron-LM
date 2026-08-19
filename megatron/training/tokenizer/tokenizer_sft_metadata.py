from typing import Any

from megatron.training.tokenizer.tokenizer_omni_metadata import extract_tokenizer_init_kwargs


def populate_sft_information_from_tokenizer(args, tokenizer: Any):
    """
    Loads sft information from tokenizer config and populates tokenizer with it. Assumes tokenizer contains this information.
    """
    init_kwargs = extract_tokenizer_init_kwargs(tokenizer)

    sft_assistant_begin_sequence = init_kwargs.get("sft_assistant_begin_sequence")
    sft_assistant_end_sequence = init_kwargs.get("sft_eot_token")

    if sft_assistant_begin_sequence is not None:
        tokenizer.sft_assistant_begin_sequence = sft_assistant_begin_sequence
        if getattr(args, "rank", None) == 0:
            print(
                f" > loaded sft_assistant_begin_sequence: {sft_assistant_begin_sequence}",
                flush=True,
            )

    if sft_assistant_end_sequence is not None:
        tokenizer.sft_assistant_end_sequence = sft_assistant_end_sequence
        if getattr(args, "rank", None) == 0:
            print(
                f" > loaded sft_assistant_end_sequence: {sft_assistant_end_sequence}",
                flush=True,
            )