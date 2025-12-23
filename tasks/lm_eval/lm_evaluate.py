# Copyright (c) 2023 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified from https://github.com/alibaba/Pai-Megatron-Patch

# NOTE: This code was implented at version megatron 250908, ngc 25.06 container and lm-eval 0.4.9.2

import types
import os
import numpy as np
from typing import List, Optional, Union
from functools import partial
from collections.abc import Iterator, Sequence

import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

# Megatron imports
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
# from megatron.core.pipeline_parallel.p2p_communication import recv_forward, send_forward
from megatron.core.dist_checkpointing import load, load_common_state_dict, load_plain_tensors
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel as LocalDDP
from megatron.core.transformer.module import Float16Module
from megatron.training import initialize_megatron, get_model, get_args, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.tokenizer import build_tokenizer
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer

# lm-eval imports
try:
    from lm_eval import utils
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import TemplateLM
    from lm_eval.api.registry import register_model
    from lm_eval.models.huggingface import HFLM, eval_logger
    from lm_eval.models.utils import (
        Collator,
        _add_special_kwargs,
        clear_torch_cache,
        configure_pad_token,
        get_dtype,
        handle_stop_sequences,
        has_bos_prefix,
        pad_and_concat,
        postprocess_generated_text,
        stop_sequences_criteria,
    )

except ImportError:
    raise ImportError("Please install the lm-eval-harness package")

try:
    from accelerate import (
        find_executable_batch_size,
    )
except ImportError:
    raise ImportError("Accelerate should have been installed when isntalling lm-eval-harness")


# Local imports
from model_provider import model_provider
from gpt_builders import gpt_builder


# NOTE: Seems megatron outputs are non-deterministc as there can be it seems a 1 or 2 point difference between runs (but to just get early signals (not final reported resutls) is enouogh, with this one can iterate faster than having to do a conversion to huggingface always and then running that)
class EvalHarnessAdaptor(HFLM):

    def __init__(
        self,
        pretrained: Optional[Union[str, transformers.PreTrainedModel]] = "gpt2", # gpt2 is used just to set the transformer info that lm_eval will use to "causal"
        max_length: Optional[int] = None,
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = False,
        **kwargs,
    ) -> None:
        # TODO: We should grab tokenizer from the checkpoint not from passing the name with arguments when calling the eval script
        self.args = get_args()
        build_tokenizer(self.args)
        self.tokenizer = get_tokenizer()

        self.tokenizer.pad_token = self.tokenizer.pad # from _Hugginfacetokenizer megatron
        self.tokenizer.pad_token_id = self.tokenizer.pad # from _Hugginfacetokenizer megatron
        self.tokenizer.eos_token_id = self.tokenizer.eod # from _Hugginfacetokenizer megatron
        self.tokenizer.bos_token_id = self.tokenizer.bos # from _Hugginfacetokenizer megatron

        self.is_main = torch.distributed.get_rank() == 0
        # self.adaptive_seq_len = self.args.adaptive_seq_len
        self.adaptive_seq_len = True
        self.model_provider = kwargs['model_provider']

        super().__init__(pretrained=pretrained,
                         batch_size=batch_size,
                         trust_remote_code=trust_remote_code,
                         max_length=max_length,
                         tokenizer=self.tokenizer,
                         **kwargs)

    def _create_model(
        self,
        pretrained: str,
        **kwargs,
    ) -> None:  
        args = get_args()
        
        # Build model (no DDP wrapper for inference)
        model = get_model(
            model_provider_func=partial(model_provider, gpt_builder),
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=False  # No DDP wrapper for inference (but i don't understand becaause we would want ddp no?, but it seems it setups gradient sync and other stuff so I don't know)
        )
        
        # Load checkpoint weights (no optimizer, no scheduler)
        iteration, num_flops, tokens = load_checkpoint(model, None, None) # Here is where the model weights are loaded and also the argument that where used to train the model (the arguments are in the common.pt state dict)

        self._model = model[0]

        def tie_weights(self):
            pass
        self._model.tie_weights = types.MethodType(tie_weights, self._model) # Because this is called in the superclass init, so we do a dummy

        self.model.eval()

        self.args = get_args()
        build_tokenizer(self.args)
        self.tokenizer = get_tokenizer()

        return None

    def _create_tokenizer(
        self,
        pretrained: str | transformers.PreTrainedModel,
        tokenizer: str
        | transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast
        | None,
        revision: str | None = "main",
        trust_remote_code: bool | None = False,
        use_fast_tokenizer: bool | None = True,
        gguf_file: str | None = None,
        add_bos_token: bool | None = None,
        subfolder: str | None = "",
    ) -> None:
        pass

    def create_model_inputs(self, tokens):
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.eot_token_id,
            self.args.reset_position_ids,
            self.args.reset_attention_mask,
            self.args.eod_mask_loss,
            eod_mask_loss=True, # this are only for training (to say if we don't compute the loss for this tokens or if yes, True means don't compute)
            pad_mask_loss=True
            )

        return (tokens, position_ids, attention_mask), (tokens, loss_mask)

    def _model_call(self, inps, attn_mask=None, labels=None):
        # Since the shape of the micro-batch will change
        # We need set the correct shapes here
        # So that latter pipeline stages knows which shapes to expect.
        # Otherwise we will deadlock.
        args = get_args()
        
        args.micro_batch_size = len(inps)
        args.seq_length = len(inps[0])
        args.max_position_embeddings = args.seq_length
        
        # Get model inputs
        (tokens, position_ids, attention_mask), _ = self.create_model_inputs(inps)
        
        # Forward pass through the model
        output = self.model(tokens, position_ids, attention_mask)
        
        if mpu.is_pipeline_last_stage():
            return gather_from_tensor_model_parallel_region(output)[..., :self.tokenizer.vocab_size]
        else:
            return None

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(
                    map(
                        utils.make_disjoint_window, utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                        )
                    )
                )

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]

                # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for that
                string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)

                # discard is_greedy result
                string_nll = [x[0] for x in string_nll]

                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        self.model.eval()
    
        with torch.no_grad():
            def _collate(x):
                """Defines the key to sort items by length."""
                toks = x[1] + x[2] # two sequences (input and output?)

                # (longest to shortest, tuple)
                return (-len(toks), tuple(toks))

            def _lookup_one_token_cont(req: tuple[tuple[str, str], list[int], list[int]]):
                """Defines the key to group and lookup one-token continuations."""
                # Use with group_by="contexts" (optional)"
                # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
                # speeds up some multiple-choice tasks proportionally to the number of choices.
                # groups requests by context+continuation[:-1] and infer on one request/group.

                # req[-2] input context, req[-1] output context (options or text to predict)
                return req[-2] + req[-1][:-1]

            re_ord = Collator(
                requests,
                sort_fn=_collate, # sort requests from longest to shortest
                group_by=None,
                # group_by="contexts"
                # if self.backend == "causal" and self.logits_cache
                # else None, # this part was causing errors where we were having empty results for here re_ord.get_original(res), this basically needs a cache so we can reause the input context (prefix) logits I think
                group_fn=_lookup_one_token_cont, # the function is not used
            )

            # automatic (variable) batch size detection for vectorization
            # pull longest context sample from request
            n_reordered_requests = len(re_ord)
            batch_size = self.batch_size
            # batch_fn = (
            #     self._batch_scheduler
            #     if self.batch_size == "auto"
            #     and n_reordered_requests > 0
            #     and not override_bs
            #     else None
            # )
            batch_fn = None

            chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
            pbar = tqdm(
                total=len(requests),
                disable=(disable_tqdm or (self.rank != 0)),
                desc="Running loglikelihood requests",
            )
            for chunk in chunks:
                inputs, continuation_lens, input_lens, padding_length = [], [], [], None
                for _, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left (remove left part)
                    input = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1] # Dont' grab the last token
                        , dtype=torch.long).to(self.device)
                    input_len, = input.shape

                    continuation = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else input_len
                    if not self.adaptive_seq_len:
                        padding_length = self.max_length

                    # pad to length
                    input = torch.cat([
                        input,  # [seq]
                        torch.zeros(padding_length - input_len, dtype=torch.long).to(input.device)  # [padding_length - seq]
                    ], dim=0)
                    inputs.append(input.unsqueeze(0))

                    continuation_lens.append(continuation)
                    input_lens.append(input_len)

                logits = self._model_call(torch.cat(inputs, dim=0)) # [batch, seq, vocab]
                res_len += len(chunk) # to do the normalization of logits by length of text
                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1, dtype=self.softmax_dtype).cpu()  # [batch, seq, vocab]

                    for (cache_key, _, _), logits, inp, input_len, continuation_toks in zip(chunk, multi_logits, inputs, input_lens, continuation_lens):
                        contlen = len(continuation_toks)
                        logits = self._select_cont_toks(logits, contlen=contlen, inplen=input_len).unsqueeze(0)  # [1, seq, vocab]

                        greedy_tokens = logits.argmax(dim=-1) # chose from vocab token with highest prob
                        # cont_toks :: [1, seq]
                        continuation_toks = torch.tensor(continuation_toks, dtype=torch.long).unsqueeze(0)
                        max_equal = (greedy_tokens == continuation_toks).all() # how many greedy tokens where equal to answer
                        # last_token_slice = logits[:, -1, :].squeeze(0).tolist()

                        logits = torch.gather(logits, 2, continuation_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                        answer = (float(logits.sum()), bool(max_equal))
                        # partial caching
                        res.append(answer)

                        pbar.update(1)

        pbar.close()

        if not mpu.is_pipeline_last_stage():
            # @HACK: To make the eval harness happy on threads that don't have access to the results.
            #        We just randomly generate some data.
            res = [(np.random.rand(), np.random.rand()>0.5) for _ in requests]

        return re_ord.get_original(res)

    def tok_encode(
        self, 
        string: str,
        add_special_tokens: bool | None = None,
        left_truncate_len: int | None = None,
    ) -> List[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = _add_special_kwargs(
            add_special_tokens, self.add_bos_token
        )
        # set add_special_tokens=False if the string already starts with BOS token (in loglikelihood_rolling is added with the prefix_token_id option)
        if add_special_tokens is None and has_bos_prefix(
            string, self.tokenizer.detokenize([self.prefix_token_id])
        ):
            special_tokens_kwargs["add_special_tokens"] = False

        # We are using the huggingface tokenizer from megatron
        encoding = self.tokenizer.tokenize(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int | None = None,
        truncation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: This one hasn't been tested yet!

        # Encode each string
        encoded = [self.tokenizer.tokenize(s) for s in strings]
        
        # Apply truncation if needed
        if left_truncate_len:
            encoded = [seq[-left_truncate_len:] for seq in encoded]
        
        # Find max length for padding
        max_len = max(len(seq) for seq in encoded)
        
        # Pad sequences
        input_ids = []
        attention_mask = []
        pad_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else 0
        
        for seq in encoded:
            seq_len = len(seq)
            padding_len = max_len - seq_len
            
            if padding_side == "left":
                padded_seq = [pad_id] * padding_len + seq
                mask = [0] * padding_len + [1] * seq_len
            else:  # right padding
                padded_seq = seq + [pad_id] * padding_len
                mask = [1] * seq_len + [0] * padding_len
            
            input_ids.append(padded_seq)
            attention_mask.append(mask)
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)

    def tok_decode(self, tokens: Iterator[List[int]], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.detokenize(tokens)

    def _model_generate(
        self,
        context,
        max_length: int,
        stop: list[str],
        **generation_kwargs,
    ) -> torch.Tensor:
         raise NotImplementedError("Model generate hasn't been implemented yet") 

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        raise NotImplementedError("Generate until hasn't been implemented yet") 

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Method to apply a chat template to a list of chat history between user and model."""

        # TODO: this will only work if megatron is using the huggingface tokenizer (_HuggingFaceTokenizer not the new one HuggingFaceTokenizer)
        
        assert isinstance(self.tokenizer, _HuggingFaceTokenizer)

        try:
            chat_templated = self.tokenizer._tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                **self.chat_template_args,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer._tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                **self.chat_template_args,
            )

        return chat_templated

    def _get_accelerate_args(
        self,
        **kwargs
    ) -> dict:
        # NOTE: we don't use accelerate with megatron

        # Get the dictionary from the parent class
        args = super()._get_accelerate_args(**kwargs)
  
        return args


    def _detect_batch_size(self, requests: Sequence | None = None, pos: int = 0):
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len(
                (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
            )
            max_context_enc = len(context_enc[-(self.max_length + 1) :])
            max_cont_enc = len(continuation_enc[-(self.max_length + 1) :])
        else:
            max_length = self.max_length
            max_context_enc = max_length
            max_cont_enc = max_length

        # if OOM, then halves batch_size and tries again
        @find_executable_batch_size(starting_batch_size=self.max_batch_size)
        def forward_batch(batch_size: int):
            if self.backend == "seq2seq":
                length = max(max_context_enc, max_cont_enc)
                batched_conts = torch.ones(
                    (batch_size, length), device=self.device
                ).long()
                test_batch = torch.ones((batch_size, length), device=self.device).long()
                call_kwargs = {
                    "attn_mask": test_batch,
                    "labels": batched_conts,
                }
            else:
                call_kwargs = {}
                test_batch = torch.ones(
                    (batch_size, max_length), device=self.device
                ).long()
            for _ in range(5):
                out = F.log_softmax(  # noqa: F841
                    self._model_call(test_batch, **call_kwargs),
                    dim=-1,
                    dtype=self.softmax_dtype,
                )

            return batch_size

        try:
            batch_size = forward_batch()
        except RuntimeError as e:
            if "No executable batch size found" in str(e):
                batch_size = 1
            else:
                raise

        if self.world_size > 1:
            raise NotImplementedError("More than one gpu implementation to get batch size hasn't been implemented")

        clear_torch_cache()
        return batch_size

    def _select_cont_toks(
        self,
        logits: torch.Tensor,
        contlen: int | None = None,
        inplen: int | None = None,
    ) -> torch.Tensor:
        if self.backend == "causal":
            assert contlen and inplen, (
                "Must pass input len and cont. len to select scored logits for causal LM"
            )
            # discard right-padding.
            # also discard the input/context tokens. we'll only score continuations.
            logits = logits[inplen - contlen : inplen]
        elif self.backend == "seq2seq":
            # NOTE: For now we have only checked the cauasal LM case
            assert False, "Seq2SeqLM loglikelihood hasn't been tested yet"
            assert contlen and not inplen, (
                "Selecting scored logits for Seq2SeqLM requires only cont. len"
            )
            # only discard right-padding.
            # the logits input to this fn only contain decoder-side tokens.
            logits = logits[:contlen]

        return logits