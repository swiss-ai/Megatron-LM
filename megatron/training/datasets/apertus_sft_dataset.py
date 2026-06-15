# Copyright (c) 2025, SwissAI.  All rights reserved.

from typing import Dict, Optional, List, Tuple

import bisect
import time
import os
import logging

import numpy as np
import torch
import torch.nn.functional as F

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, _PAD_TOKEN_ID, GPTDataset
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


def _pad_sequence_if_needed(document, target_length: int, padding_value):
    """
    Pad a sequence to max length if needed. Returns unchanged if not.
    """
    if len(document) >= target_length:
        return document

    padding_length = target_length - len(document)
    return np.concatenate([document, np.full(padding_length, padding_value, dtype=document.dtype)])



def _load_bfd_c_library():
    """Load (or compile then load) the C/C++ BFD packing library.

    Uses a segment-tree + min-heap structure for O(n log C) best-fit lookup,
    ~10-15x faster than pure-Python bisect at million-doc scale. Thread-safe.
    Returns the loaded ctypes library, or None if unavailable.
    """
    import ctypes
    import subprocess

    _dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'core', 'datasets'))
    so_path  = os.path.join(_dir, "libbfd_pack.so")
    cpp_path = os.path.join(_dir, "bfd_pack.cpp")

    if os.path.isfile(so_path):
        try:
            lib = ctypes.CDLL(so_path)
            lib.bfd_pack.argtypes = [
                ctypes.POINTER(ctypes.c_int), # sorted_positions
                ctypes.POINTER(ctypes.c_long), # doc_lengths
                ctypes.c_int, # num_docs
                ctypes.c_int, # capacity
                ctypes.c_int, # max_docs_per_bin
                ctypes.POINTER(ctypes.c_int), # document_index
                ctypes.POINTER(ctypes.c_int), # doc_idx_out
                ctypes.POINTER(ctypes.c_int), # boundaries_out
                ctypes.POINTER(ctypes.c_int), # num_bins_out
            ]
            lib.bfd_pack.restype = None
            return lib
        except OSError:
            pass

    # Compile from source (prefer C++, fall back to C)
    for src_path, compiler in [(cpp_path, "g++")]:
        if os.path.isfile(src_path):
            try:
                subprocess.check_call(
                    [compiler, "-O3", "-shared", "-fPIC", "-o", so_path, src_path],
                    stderr=subprocess.DEVNULL,
                )
                return _load_bfd_c_library()
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

    return None

_bfd_c_lib = _load_bfd_c_library()

def _build_sample_idx_bfd(
    sequence_lengths: np.ndarray,
    document_index: np.ndarray,
    seq_length: int,
    add_extra_token: int,
    max_docs_per_bin: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Best-Fit Decreasing bin packing for whole documents.

    Sorts documents by decreasing length and assigns each to the bin with the
    least remaining capacity that still fits. Produces fewer bins (less padding)
    than greedy sequential packing when document lengths vary.

    Uses the C-accelerated implementation when available, otherwise a pure-Python
    bisect-based fallback.

    Args:
        sequence_lengths: Array of document lengths indexed by document ID.
        document_index: Shuffled document IDs for one epoch.
        seq_length: Target sequence length.
        add_extra_token: 0 or 1, added to seq_length for the effective bin capacity.

    Returns:
        reordered_document_index: Same document IDs as input, reordered so that
            each bin's documents are contiguous.
        sample_index: Shape (num_bins + 1, 2) boundary array. Column 0 holds
            offsets into reordered_document_index; column 1 is always 0.
    """
    if _bfd_c_lib is not None:
        return _build_sample_idx_bfd_c(sequence_lengths, document_index, seq_length, add_extra_token, max_docs_per_bin)
    return _build_sample_idx_bfd_python(sequence_lengths, document_index, seq_length, add_extra_token, max_docs_per_bin)


def _build_sample_idx_bfd_c(sequence_lengths, document_index, seq_length, add_extra_token, max_docs_per_bin=0):
    """C-accelerated BFD bin packing via ctypes. See _build_sample_idx_bfd."""
    import ctypes

    capacity = seq_length + add_extra_token
    num_docs = len(document_index)

    doc_lengths = sequence_lengths[document_index].astype(np.int64)
    sorted_positions = np.argsort(-doc_lengths, kind='stable').astype(np.int32)
    doc_index_i32 = document_index.astype(np.int32)

    doc_idx_out = np.empty(num_docs, dtype=np.int32)
    boundaries_out = np.empty(num_docs + 1, dtype=np.int32)
    num_bins_out = ctypes.c_int(0)

    _bfd_c_lib.bfd_pack(
        sorted_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        doc_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        num_docs, capacity, int(max_docs_per_bin),
        doc_index_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        doc_idx_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        boundaries_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.byref(num_bins_out),
    )

    nb = num_bins_out.value
    reordered = doc_idx_out[:num_docs].astype(document_index.dtype)
    sample_index = np.zeros((nb + 1, 2), dtype=document_index.dtype)
    sample_index[:, 0] = boundaries_out[:nb + 1].astype(document_index.dtype)

    assert boundaries_out[nb] == num_docs, f"BFD placed {boundaries_out[nb]} docs but expected {num_docs}"
    return reordered, sample_index


def _build_sample_idx_bfd_python(
    sequence_lengths: np.ndarray,
    document_index: np.ndarray,
    seq_length: int,
    add_extra_token: int,
    max_docs_per_bin: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-Python BFD fallback using bisect. See _build_sample_idx_bfd."""
    capacity = seq_length + add_extra_token
    num_docs = len(document_index)

    # Gather per-position lengths via vectorized lookup
    doc_lengths = sequence_lengths[document_index].astype(np.int64)

    # Sort positions by decreasing length (stable for determinism on ties)
    sorted_positions = np.argsort(-doc_lengths, kind='stable')

    # BFD assignment: maintain a sorted list of (remaining_capacity, bin_id)
    # and use bisect to find the tightest-fitting bin in O(log B) per doc,
    # where B is the number of open bins (typically much smaller than num_docs).
    bins_sorted: List[Tuple[int, int]] = []  # sorted by remaining capacity ascending
    bin_contents: List[List[int]] = []       # bin_id -> list of positions in document_index

    for pos in sorted_positions:
        length = int(doc_lengths[pos])

        if length > capacity:
            # Oversized document gets its own bin (truncated by downstream code)
            bin_id = len(bin_contents)
            bin_contents.append([int(pos)])
            continue

        if length == 0:
            # Zero-length docs fit anywhere; place in the first open bin or a new one
            if bins_sorted:
                remaining, bin_id = bins_sorted[0]
                bin_contents[bin_id].append(int(pos))
                if max_docs_per_bin > 0 and len(bin_contents[bin_id]) >= max_docs_per_bin:
                    bins_sorted.pop(0)
            else:
                bin_id = len(bin_contents)
                bin_contents.append([int(pos)])
                bisect.insort(bins_sorted, (capacity, bin_id))
            continue

        # Find the bin with the smallest remaining capacity >= length
        idx = bisect.bisect_left(bins_sorted, (length,))

        if idx < len(bins_sorted):
            # Best-fit found: use this bin
            remaining, bin_id = bins_sorted.pop(idx)
            new_remaining = remaining - length
            bin_contents[bin_id].append(int(pos))
            if new_remaining > 0 and not (max_docs_per_bin > 0 and len(bin_contents[bin_id]) >= max_docs_per_bin):
                # Re-insert with updated capacity
                bisect.insort(bins_sorted, (new_remaining, bin_id))
        else:
            # No bin fits: open a new one
            bin_id = len(bin_contents)
            bin_contents.append([int(pos)])
            new_remaining = capacity - length
            if new_remaining > 0 and not (max_docs_per_bin > 0 and len(bin_contents[bin_id]) >= max_docs_per_bin):
                bisect.insort(bins_sorted, (new_remaining, bin_id))

    # Rebuild document_index so each bin's documents are contiguous
    reordered = np.empty(num_docs, dtype=document_index.dtype)
    boundaries = []
    offset = 0
    for contents in bin_contents:
        boundaries.append(offset)
        for pos in contents:
            reordered[offset] = document_index[pos]
            offset += 1
    boundaries.append(offset)

    # Build sample_index: (num_bins + 1, 2), column 1 always 0
    sample_index = np.zeros((len(boundaries), 2), dtype=document_index.dtype)
    sample_index[:, 0] = np.array(boundaries, dtype=document_index.dtype)

    assert offset == num_docs, f"BFD placed {offset} docs but expected {num_docs}"
    return reordered, sample_index



class ApertusSFTDataset(GPTDataset):
    """Apertus SFT dataset for supervised fine-tuning on pre-tokenized data.

    Loads already-tokenized conversation data from Megatron indexed datasets (.bin/.idx).
    Supports single-document and multi-document packing modes (--ap-sft-pack-samples).

    Loss masking operates in two modes:
      1. **From disk** (--ap-sft-load-loss-mask): Each document stores tokens and a pre-computed
         loss mask concatenated together ([tokens, loss_mask]). The dataset splits them at load
         time and uses the mask as-is.
      2. **On the fly** (default): The loss mask is built at runtime by detecting assistant
         response regions. This requires the tokenizer to expose ``sft_assistant_begin_sequence``
         and ``sft_assistant_end_sequence`` attributes (see HuggingFaceTokenizer in
         megatron/training/tokenizer/tokenizer.py). Everything outside assistant regions is
         masked (or weighted by --ap-sft-plw). Special tokens (BOS, EOD, assistant begin) can
         optionally be masked via --ap-sft-mask-special-tokens.

    Note: Goldfish loss (--goldfish-loss) is not supported and will be ignored if enabled.
    Note: Omnimodal weighting during loss mask creation is not supported. We simply have all assistant tokens unmasked.
    """
    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indexed_indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        # Call Megatron Dataset init instead of direct parent, as we initialize index differently
        MegatronDataset.__init__(self, dataset, dataset_path, indexed_indices, num_samples, index_split, config)

        if config.goldfish_loss:
            log_single_rank(logger, logging.WARNING,
                          "Goldfish loss is not supported by ApertusSFTDataset and will be ignored")

        self.sft_plw_value = config.sft_plw
        log_single_rank(logger, logging.INFO, f"SFT PLW: {self.sft_plw_value}", )

        self.truncate_right = config.sft_truncate_right

        self.tokenizer = config.tokenizer
        # Set pad token
        try:
            self._pad_token_id = self.tokenizer.pad
            log_single_rank(logger, logging.INFO, f"Using tokenizer pad token ID: {self._pad_token_id}")
        except (AttributeError, KeyError, TypeError, NotImplementedError) as e:
            self._pad_token_id = _PAD_TOKEN_ID
            log_single_rank(logger, logging.WARNING,
                          f"Tokenizer pad token not available ({type(e).__name__}), using default: {self._pad_token_id}")

        # End of Document token to add end to truncated samples TODO: currently works with HF tokenizers only
        self._eod_token_id = self.tokenizer.eod
        self._bos_token_id = self.tokenizer.bos

        # Load pre-computed SFT sequences from tokenizer config (tokenizer_config.json).
        # These must be set as pre-tokenized token ID lists, e.g. by
        # add_emu3_tokens_llama3_vision_instruct.py. Some models use separate assistant/user
        # end sequences, others share a common eot token.

        special_tokens = {
            "assistant_begin": "<|assistant_start|>",
            "assistant_end": "<|assistant_end|>",
            "system_start": "<|system_start|>",
            "tool_output_start": "<|tool_output_start|>",
            "tool_output_end": "<|tool_output_end|>", 
        }

        for attr, string in special_tokens.items():
            token_id = self.tokenizer._tokenizer.tokenizer.encode(string, add_special_tokens=False)
            token_list = token_id if isinstance(token_id, list) else [token_id]
            setattr(self, f"_sft_{attr}_sequence", torch.tensor(token_list, dtype=torch.long))

        # Configure token (sequences) to remove from loss calculation
        self.tokens_to_mask = []
        if self.config.sft_mask_special_tokens and not self.config.sft_load_loss_mask:
            # add tokenizer special tokens like EOS, BOS and assistant begin to be masked. Never mask End of turn.
            # TODO: in current apertus tokenizer eod is eot by default!!
            self.tokens_to_mask.append(torch.tensor([self._eod_token_id], dtype=torch.long))
            self.tokens_to_mask.append(torch.tensor([self._bos_token_id], dtype=torch.long))
            self.tokens_to_mask.append(self._sft_assistant_begin_sequence)  # already a tensor
            # user begin and end are masked by default as only assistant unmasked
        if self.tokens_to_mask:
            log_single_rank(logger, logging.INFO, f"On the fly masking the following tokens/token-sequences: {[t.tolist() for t in self.tokens_to_mask]}")

        # Set actual model sequence length (config.sequence_length is doubled if loading loss masks from disk)
        if self.config.sft_load_loss_mask:
            self.model_seq_length = self.config.sequence_length // 2
            log_single_rank(logger, logging.INFO,
                          f"Loading loss masks from disk: dataset seq_length={self.config.sequence_length}, "
                          f"model seq_length={self.model_seq_length}")
        else:
            self.model_seq_length = self.config.sequence_length

        # Initialize cache manager
        self.cache_manager = IndexCacheManager(
            config=self.config,
            dataset_path_prefix=self.dataset.path_prefix,
            unique_description=self.unique_description,
            unique_description_hash=self.unique_description_hash,
            dataset_class_name=type(self).__name__,
            split_name=self.index_split.name
        )

        # Build indices based on packing mode
        if self.config.sft_pack_samples:
            # Use multi-document packing with sample_index
            (self.document_index, self.sample_index, self.shuffle_index) = (
                self._build_packing_document_to_sample_indices()
            )
            self._using_packed_samples = True
        else:

            # Use simple single-document indexing
            self.document_index = self._build_single_document_indices()
            self._using_packed_samples = False

    @staticmethod
    def _key_config_attributes() -> List[str]:
        """
        Extend key attributes from Megatron dataset, to include vital sft config attributes.
        """
        return [
            "random_seed",
            "sequence_length",
            "split",
            "split_matrix",
            "tokenizer",
            "add_extra_token_to_sequence",
            "sft_pack_samples",
            "sft_packing_strategy",
            "sft_load_loss_mask",
        ]

    def _log_packing_statistics(self, document_index, sample_index, from_cache=False):
        """
        Log statistics about packed samples (one epoch).

        Args:
            document_index: Array of document IDs
            sample_index: Array of sample boundaries
            from_cache: Whether the indices were loaded from cache
        """
        num_samples_available = sample_index.shape[0] - 1
        sequence_length = self.config.sequence_length
        capacity = sequence_length + self.config.add_extra_token_to_sequence

        raw_lengths = self.dataset.sequence_lengths[self.indices]
        raw_tokens_per_epoch = int(raw_lengths.sum())
        # Oversized docs get their own bin and are truncated at `capacity` in
        # _get_packed_sample, so only `capacity` of each contributes to a sample.
        effective_tokens_per_epoch = int(np.minimum(raw_lengths, capacity).sum())
        truncated_tokens = raw_tokens_per_epoch - effective_tokens_per_epoch

        total_capacity = num_samples_available * capacity
        avg_tokens_per_sample = (
            effective_tokens_per_epoch / num_samples_available if num_samples_available > 0 else 0
        )
        avg_documents_per_sample = (
            len(document_index) / num_samples_available if num_samples_available > 0 else 0
        )
        packing_efficiency = (
            100 * effective_tokens_per_epoch / total_capacity if total_capacity > 0 else 0
        )

        # When loss masks are loaded from disk, each stored "doc" is [tokens, loss_mask]
        # concatenated, so config.sequence_length and raw lengths are both 2x. Halve
        # displayed absolute counts so the reader sees real model-token quantities.
        display_divisor = 2 if self.config.sft_load_loss_mask else 1
        display_seq_len = sequence_length // display_divisor

        cache_suffix = " (loaded from cache)" if from_cache else ""

        log_single_rank(logger, logging.INFO, f"> ===== SFT Packing Statistics (ONE EPOCH){cache_suffix} =====")
        log_single_rank(logger, logging.INFO, f" > #docs in epoch:                       {len(document_index):>12}")
        log_single_rank(logger, logging.INFO, f" > #tokens in epoch (raw):               {raw_tokens_per_epoch // display_divisor:>12,}")
        log_single_rank(logger, logging.INFO, f" > #tokens lost to truncation:           {truncated_tokens // display_divisor:>12,}")
        log_single_rank(logger, logging.INFO, f" > Model sequence length:                {display_seq_len:>12}")
        log_single_rank(logger, logging.INFO, f" > #packed samples (per epoch):          {num_samples_available:>12,}")
        log_single_rank(logger, logging.INFO, f" > Total bin capacity:                   {total_capacity // display_divisor:>12,}")
        log_single_rank(logger, logging.INFO, f" > Avg #tokens/sample (effective):       {avg_tokens_per_sample / display_divisor:>12.1f}")
        log_single_rank(logger, logging.INFO, f" > Avg #documents/sample:                {avg_documents_per_sample:>12.2f}")
        log_single_rank(logger, logging.INFO, f" > Packing efficiency:                   {packing_efficiency:>11.2f}%\n\n")

    def _build_packing_document_to_sample_indices(self):
        """
        Build indices for packed document sampling. Always packs exactly one epoch so that
        every document is used. If more samples are requested than one epoch provides, the
        shuffle index is tiled with independent permutations (no re-packing).

        Returns a tuple of three indices:
        - document_index: Shuffled document IDs for one epoch
        - sample_index: Maps sample boundaries to (document_index position, offset) pairs
        - shuffle_index: Permutation indices, possibly tiled to meet num_samples

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                - document_index: Shape (num_documents,) - shuffled document IDs for one epoch
                - sample_index: Shape (num_epoch_samples + 1, 2) - sample boundaries as [doc_idx_index, offset=0]
                - shuffle_index: Shape (num_total_samples,) - permutation indices, tiled if needed
        """
        from megatron.core.datasets import helpers

        index_names = ["document_index", "sample_index", "shuffle_index"]
        cache_hit = self.cache_manager.cache_exists(index_names)

        if self.cache_manager.get_cache_path():
            log_single_rank(logger, logging.WARNING, f"path_to_cache exists! Search for indices in: {self.cache_manager.get_cache_path()}")

        if not self.cache_manager.get_cache_path() or (
            not cache_hit
            and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
        ):
            log_single_rank(
                logger,
                logging.INFO,
                f"No cached indices! Build and save the {type(self).__name__} {self.index_split.name} packed indices",
            )

            t_beg = time.time()

            sequence_length = self.config.sequence_length
            numpy_random_state = np.random.RandomState(self.config.random_seed)

            # Always pack exactly one epoch (all documents used once)
            document_index = _build_document_index(1, self.indices.copy().astype(np.int32), numpy_random_state)

            assert document_index.dtype == np.int32
            assert self.dataset.sequence_lengths.dtype == np.int32

            # Copy sequence lengths for C++ if access density is high
            if len(document_index) * 2 > len(self.dataset.sequence_lengths):
                sequence_lengths_for_cpp = self.dataset.sequence_lengths.copy()
            else:
                sequence_lengths_for_cpp = self.dataset.sequence_lengths

            # Build the sample index using the configured packing strategy
            if self.config.sft_packing_strategy == "bfd":
                log_single_rank(logger, logging.INFO, "Using Best-Fit Decreasing packing strategy")
                document_index, sample_index = _build_sample_idx_bfd(
                    sequence_lengths_for_cpp,
                    document_index,
                    sequence_length,
                    add_extra_token=self.config.add_extra_token_to_sequence,
                    max_docs_per_bin=self.config.max_docs_per_bin_sft,
                )
            else:
                sample_index = helpers.build_sample_idx_packed_whole_docs(
                    sequence_lengths_for_cpp,
                    document_index,
                    sequence_length,
                    add_extra_token_to_sequence=self.config.add_extra_token_to_sequence,
                )

            # Log packing statistics for the single epoch
            self._log_packing_statistics(document_index, sample_index, from_cache=False)

            # Build shuffle index, tiling if more samples are requested than one epoch provides
            num_epoch_samples = sample_index.shape[0] - 1
            shuffle_index = self._build_shuffle_index_with_tiling(
                num_epoch_samples, numpy_random_state
            )

            self.cache_manager.save_indices({
                "description": self.unique_description,
                "document_index": document_index,
                "sample_index": sample_index,
                "shuffle_index": shuffle_index
            })

            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            return document_index, sample_index, shuffle_index

        # Load from cache
        log_single_rank(
            logger, logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} packed indices"
        )
        indices = self.cache_manager.load_indices(index_names)
        document_index = indices["document_index"]
        sample_index = indices["sample_index"]
        shuffle_index = indices["shuffle_index"]
        self._log_packing_statistics(document_index, sample_index, from_cache=True)

        return document_index, sample_index, shuffle_index

    def _build_shuffle_index_with_tiling(
        self, num_epoch_samples: int, numpy_random_state: np.random.RandomState
    ) -> np.ndarray:
        """Build a shuffle index over epoch samples. If num_samples exceeds one epoch,
        tile the index with independent permutations and truncate to num_samples.

        Args:
            num_epoch_samples: Number of packed samples from one epoch of packing.
            numpy_random_state: RNG for reproducible shuffling.

        Returns:
            np.ndarray: Shuffle index of length min(num_samples, num_epoch_samples) or
                        num_epoch_samples if num_samples is None.
        """
        from megatron.core.datasets.gpt_dataset import _build_shuffle_index

        if not self.num_samples or self.num_samples <= num_epoch_samples:
            # One epoch is enough (or no target set) — single shuffle
            n = self.num_samples if self.num_samples else num_epoch_samples
            return _build_shuffle_index(num_epoch_samples, n, numpy_random_state)

        # Need more samples than one epoch provides — tile with fresh permutations
        num_repeats = int(np.ceil(self.num_samples / num_epoch_samples))
        log_single_rank(
            logger, logging.WARNING,
            f"> Requested {self.num_samples} samples but one epoch only provides "
            f"{num_epoch_samples} packed samples. Tiling shuffle index {num_repeats}x "
            f"with independent permutations."
        )

        tiles = []
        for _ in range(num_repeats):
            perm = _build_shuffle_index(num_epoch_samples, num_epoch_samples, numpy_random_state)
            tiles.append(perm)

        shuffle_index = np.concatenate(tiles)[:self.num_samples]
        return shuffle_index

    def _build_single_document_indices(self) -> np.ndarray:
        """
        Build a document index for single-document sampling. Only one document is used per sample.
        Caches the generated index to disk if path_to_cache is specified.

        Returns:
            numpy.ndarray: The document index (Shape: (num_samples,))
        """
        # Check cache
        index_names = ["document_index"]
        cache_hit = self.cache_manager.cache_exists(index_names)

        if self.cache_manager.get_cache_path():
            log_single_rank(logger, logging.WARNING, f"path_to_cache exists! Search for indices in: {self.cache_manager.get_cache_path()}")

        # Build indices if cache miss
        if not self.cache_manager.get_cache_path() or (
                not cache_hit
                and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
        ):
            log_single_rank(
                logger,
                logging.INFO,
                f"Build and save the {type(self).__name__} {self.index_split.name} indices",
            )

            t_beg = time.time()

            numpy_random_state = np.random.RandomState(self.config.random_seed)

            # Each document maps to exactly one sample
            if self.num_samples is None:
                # Use all documents once
                self.num_samples = len(self.indices)
                num_epochs = 1
            else:
                # Calculate how many epochs needed
                docs_per_epoch = len(self.indices)
                num_epochs = (self.num_samples + docs_per_epoch - 1) // docs_per_epoch

            # Build document index by repeating indices for each epoch (shuffle per epoch)
            document_index = _build_document_index(num_epochs, self.indices.copy().astype(np.int32), numpy_random_state)

            # Truncate to exact number of samples if specified (ex. If last epoch is partial)
            document_index = document_index[:self.num_samples]

            # Save to cache
            self.cache_manager.save_indices({
                "description": self.unique_description,
                "document_index": document_index
            })

            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")
            log_single_rank(logger, logging.INFO, f"> total number of samples: {len(document_index)}")
            log_single_rank(logger, logging.INFO, f"> total number of epochs: {num_epochs}")

            return document_index

        # Load from cache
        log_single_rank(
            logger, logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} indices"
        )

        indices = self.cache_manager.load_indices(index_names)
        document_index = indices["document_index"]

        return document_index

    def _truncate_sequence_if_needed(self, document, target_length: int, added_token_on_right_truncate):
        """
        Truncate a document to max length if needed(exceeds model seq len). Returns unchanged if not.
        Depending on left and right truncation:
            - right: keep seq-len -1 tokens in beginning + add eod in the end
            - left: keep seq-len tokens in the end of document. No need to append eod as it's assumed to exist.

        added_token_on_right_truncate: added to right truncated sequence can be eod or other value.
        """
        if len(document) <= target_length:
            return document

        if self.truncate_right:
            return np.concatenate([document[:target_length - 1], np.array([added_token_on_right_truncate], dtype=document.dtype)])
        else:
            # by default do left truncation (keep tokens in the end)
            return document[-target_length:]

    def _get_packed_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load and concatenate multiple whole documents for a packed sample.
        Packs documents as defined in pre-computed index.
            - sample-index: defines ranges of documents to be packed together ex. sample i has interval ( doc-idx[sample-idx[i]], doc-idx[sample-idx[i+1]] (
            - shuffle-index: randomizes packed samples (positions in sample index)
            - document-index: maps documents to low-level-dset documents (randomize raw doc order)

        Args:
            idx (int): Index of sample to retrieve

        Returns:
            Tuple containing:
                - np.ndarray: Concatenated tokens from multiple documents, padded to sequence length
                - np.ndarray: End-of-sequence indices for each document (list of size n for n docs in sequence, so min size = 1)
                - Optional[np.ndarray]: Preloaded loss masks if sft_load_loss_mask is True, else None
        """
        shuffled_idx = self.shuffle_index[idx]

        # Get sample boundaries from sample_index (first(inclusive) and last(exclusive) sample to be packed)
        doc_index_beg, _ = self.sample_index[shuffled_idx]
        doc_index_end, _ = self.sample_index[shuffled_idx + 1]
        assert doc_index_beg < doc_index_end  # the way we create the index, the end idx doc is always excluded => for same doc begin & end idx are different

        target_length = self.model_seq_length + self.config.add_extra_token_to_sequence

        document_tokens = []
        document_loss_masks = [] if self.config.sft_load_loss_mask else None
        doc_end_indices = [] # store positions of sample ends (use to reset pos id and attn mask)

        for i in range(doc_index_beg, doc_index_end):
            # Get the actual document ID from document_index & load whole document
            doc_id = self.document_index[i]
            document = self.dataset.get(doc_id)

            # If loading loss masks from disk, split the document into tokens and loss_mask
            if self.config.sft_load_loss_mask:
                # Dataset stores [tokens, loss_mask] concatenated
                doc_len = len(document) // 2
                doc_tokens = document[:doc_len]
                doc_loss_mask = document[doc_len:]

                # Truncate document and end with EOD if too long
                doc_tokens = self._truncate_sequence_if_needed(doc_tokens, target_length, self._eod_token_id)
                doc_loss_mask = self._truncate_sequence_if_needed(doc_loss_mask, target_length, 0.0)

                document_tokens.append(doc_tokens)
                document_loss_masks.append(doc_loss_mask)
                doc_end_indices.append(doc_tokens.size)
            else:
                # Original behavior: no loss mask splitting
                document = self._truncate_sequence_if_needed(document, target_length, self._eod_token_id)
                document_tokens.append(document)
                doc_end_indices.append(document.size)

        # Concatenate all documents
        if len(document_tokens) > 0:
            text = np.concatenate(document_tokens)
            eos_idx = np.array(doc_end_indices).cumsum() - 1
            if self.config.sft_load_loss_mask:
                loss_mask_data = np.concatenate(document_loss_masks)
            else:
                loss_mask_data = None
        else:
            raise RuntimeError("Encountered empty packed sample. This should not happen!")

        text = _pad_sequence_if_needed(text, target_length, self._pad_token_id)
        if self.config.sft_load_loss_mask:
            # Pad loss_mask with 0.0 (padding tokens should have 0 loss)
            loss_mask_data = _pad_sequence_if_needed(loss_mask_data, target_length, 0.0)
        if len(text) > target_length:
            # This should never happen with correct packing - raise error
            raise RuntimeError(
                f"Packed sample {idx} exceeded target length ({len(text)} > {target_length}). "
                f"This indicates a bug in build_sample_idx_packed_whole_docs. "
                f"Sample contains {len(document_tokens)} documents."
            )

        return text, eos_idx, loss_mask_data

    def __len__(self) -> int:
        if self._using_packed_samples:
            return len(self.shuffle_index)
        else:
            return len(self.document_index)

    def _get_single_sample(self, idx: Optional[int]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Retrieve a single sample as raw numpy arrays. The sample has len=model-seq-len: [sample][padding]

        Args:
            idx (Optional[int]): The sample index. If None, returns a padding sequence.

        Returns:
            Tuple containing:
                - np.ndarray: Token sequence
                - np.ndarray: End-of-sequence indices (single number for case of single sample)
                - Optional[np.ndarray]: Preloaded loss masks if sft_load_loss_mask is True, else None
        """
        actual_doc_id = self.document_index[idx]
        document = self.dataset.get(actual_doc_id)
        preloaded_loss_mask = None

        # If loading loss masks from disk, split document into tokens and loss_mask
        if self.config.sft_load_loss_mask:
            # Dataset stores [tokens, loss_mask] concatenated
            doc_len = len(document) // 2
            doc_tokens = document[:doc_len]
            doc_loss_mask = document[doc_len:]

            # Truncate or pad to sequence_length
            target_length = self.model_seq_length + self.config.add_extra_token_to_sequence
            text = self._truncate_sequence_if_needed(doc_tokens, target_length, self._eod_token_id)
            preloaded_loss_mask = self._truncate_sequence_if_needed(doc_loss_mask, target_length, 0.0)

            # Compute eos_idx after truncation but before padding
            eos_idx = np.array([len(text) - 1], dtype=np.int64)

            text = _pad_sequence_if_needed(text, target_length, self._pad_token_id)
            preloaded_loss_mask = _pad_sequence_if_needed(preloaded_loss_mask, target_length, 0.0)
        else:
            # Normal mode
            # Truncate or pad to sequence_length
            target_length = self.model_seq_length + self.config.add_extra_token_to_sequence
            text = self._truncate_sequence_if_needed(document, target_length, self._eod_token_id)

            # Compute eos_idx after truncation but before padding
            eos_idx = np.array([len(text) - 1], dtype=np.int64)

            # Pad on right side with pad token
            text = _pad_sequence_if_needed(text, target_length, self._pad_token_id)

        return text, eos_idx, preloaded_loss_mask

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        For non-packed mode: Each sample is a single document padded to sequence length OR truncated if too long.
        For packed mode: Each sample contains multiple whole documents concatenated together.

        Args:
            idx (Optional[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        if idx is None:
            # Batch padding sequence
            text = np.array(
                [self._pad_token_id] * (self.model_seq_length + self.config.add_extra_token_to_sequence),
                dtype=np.int64)
            eos_idx = np.array([], dtype=np.int64)
            preloaded_loss_mask = None
        elif self._using_packed_samples:
            # Packed mode: load and concatenate multiple whole documents, return indices of document borders additionally
            text, eos_idx, preloaded_loss_mask = self._get_packed_sample(idx)
        else:
            # Single-document mode: Get document. index is already shuffled
            text, eos_idx, preloaded_loss_mask = self._get_single_sample(idx)

        text = torch.from_numpy(text).long()

        # Create tokens and labels
        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id

        # Generate loss-mask, position-ids, assistant mask and optionally attention mask. If PLW activated, loss mask will have partial weight for user input tokens
        attention_mask, loss_mask, position_ids, assistant_mask = self._get_ltor_masks_and_position_ids(
            labels, # labels are used to create the loss and assistant mask
            eos_idx, # eos_idx is based on tokens(NOT labels) and controls position-ids and attn mask reset
            torch.from_numpy(preloaded_loss_mask).float() if preloaded_loss_mask is not None else None
        )

        # Map pad tokens to valid embedding indices
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        # Return sample dict
        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                "assistant_mask": assistant_mask,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                "assistant_mask": assistant_mask,
            }

    def _get_ltor_masks_and_position_ids(self, data: torch.Tensor, eos_indices: np.ndarray, preloaded_loss_mask: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Build masks and position id for SFT data. Possibility to mask arbitrary (also special) token(sequences).
            1. Can mask full user prompts or with prompt-loss-weight (plw)
            2. Can mask arbitrary token sequences (e.g. assistant begin, assistant end, BOS, EOS)
            3. Creates attention mask if configured. The attention mask will exclude padding tokens from attention.
            4. Can equalize sample loss for packed and non-packed sequences (loss = 1 for each sample in seq)

        For packed samples (when sft_pack_samples=True):
            - Position IDs are reset at each EOD token (document boundary)
            - Attention mask blocks cross-document attention at EOD boundaries
            - ASSUMES NO WRONG PLACED EOD (=ONLY PROPERLY BOUNDARY OF SAMPLES)

        Also creates an assistant_mask to identify assistant response tokens for separate loss tracking.

        Args:
            data:                   labels
            eos_indices:            indices of document boundaries (calculated based on sample loading from low level dataset as
                                    sft data can be contaminated with eod or eod missing).
            preloaded_loss_mask:    Optional pre-computed loss mask loaded from disk. If provided, user prompt masking
                                    and special token masking are skipped for loss_mask.
        """

        position_ids = torch.arange(self.model_seq_length, dtype=torch.long)
        loss_mask = preloaded_loss_mask.to(device=data.device) if preloaded_loss_mask is not None else torch.zeros(self.model_seq_length, dtype=torch.float, device=data.device)

        # 0) For packed samples: reset position IDs at document boundaries
        if self._using_packed_samples:
            if eos_indices.size > 0:
                # Reset position IDs after each EOD token
                for eod_idx in eos_indices:
                    if eod_idx + 1 < len(position_ids):
                        # Subtract the position value at EOD+1 from all subsequent positions
                        to_subtract = position_ids[eod_idx].clone() + 1
                        position_ids[(eod_idx + 1):] -= to_subtract

        # 1) unmask assistant parts and set rest to plw value (if not loaded from disk) otherwise assistant loss needed
        #    to keep track of assistant loss
        # NOTE: Left truncation can cut into an assistant response, leaving an orphaned end_seq
        # with no preceding begin_seq. In that case tokens before the first end_seq are incorrectly
        # treated as non-assistant. A fix would detect orphaned end markers and mask 0..first_end
        # as assistant. Same applies to right truncation of pre-packed data cutting through a begin_seq.
        begin_seq = self._sft_assistant_begin_sequence.to(dtype=data.dtype, device=data.device)
        end_seq = self._sft_assistant_end_sequence.to(dtype=data.dtype, device=data.device)
        assistant_mask = get_matching_mask_by_start_end(data, begin_seq, end_seq)

        # Only apply to loss_mask if NOT loading from disk
        if preloaded_loss_mask is None:
            loss_mask[assistant_mask] = 1
            if self.sft_plw_value > 0:
                loss_mask[~assistant_mask] = self.sft_plw_value # value is 0 by default for full masking

            # Also unmask tokens between BOS and <|system_start|> (pre-system content). Relevant for Long Context Tasks
            bos_seq = torch.tensor([self._bos_token_id], dtype=data.dtype, device=data.device)
            sys_start_seq = self._sft_system_start_sequence.to(dtype=data.dtype, device=data.device)
            loss_mask[get_matching_mask_by_start_end(data, bos_seq, sys_start_seq)] = 1

            # First-doc fallback: if <|system_start|> appears with no preceding <s>, unmask [0..system_start].
            sys_pos = torch.where(get_matching_mask(data, sys_start_seq, only_begin=True))[0]
            if sys_pos.numel() > 0 and not (data[: sys_pos[0]] == self._bos_token_id).any():
                loss_mask[: sys_pos[0].item() + sys_start_seq.numel()] = 1

            loss_mask[get_matching_mask(data, sys_start_seq, only_begin=False)] = 0

            # 1b) Mask tool output tokens from both loss_mask and assistant_mask.
            # Tool output spans (<|tool_output_start|> ... <|tool_output_end|>)
            tool_output_start_seq = self._sft_tool_output_start_sequence.to(dtype=data.dtype, device=data.device)
            tool_output_end_seq   = self._sft_tool_output_end_sequence.to(dtype=data.dtype, device=data.device)
            tool_output_mask = get_matching_mask_by_start_end(data, tool_output_start_seq, tool_output_end_seq)

            loss_mask[tool_output_mask] = 0.0
            if assistant_mask is not None:
                assistant_mask[tool_output_mask] = False

        # 2) Mask loss for special tokens (if activated) - only if not load loss from disk
        if preloaded_loss_mask is None:
            for t in self.tokens_to_mask:
                t_tensor = t.to(dtype=data.dtype, device=data.device)
                if len(t_tensor) == 1:
                    mask = (data == t_tensor[0])
                elif len(t_tensor) > 1:
                    mask = get_matching_mask(data, t_tensor, only_begin=False)
                else:
                    raise ValueError(f"Invalid token to mask: {t}")
                loss_mask[mask] = 0.0

        # 3) Create attention mask: mask attention from/to padding tokens
        if self.config.create_attention_mask:
            attention_mask = torch.tril(
                torch.ones((self.model_seq_length, self.model_seq_length), device=data.device)
            )
            no_padding_mask = (data != self._pad_token_id).float() # 1=real, 0=padding

            # Row masking: padding tokens shouldn't attend to anything
            attention_mask = attention_mask * no_padding_mask.unsqueeze(1)
            # Column masking: nothing should attend to padding tokens
            attention_mask = attention_mask * no_padding_mask.unsqueeze(0)

            # For packed samples: block cross-document attention at EOD boundaries
            if self._using_packed_samples:
                if eos_indices.size > 0:
                    for eod_idx in eos_indices:
                        if eod_idx + 1 < self.model_seq_length:
                            # Zero out attention from all tokens after EOD to all tokens up to and including EOD
                            attention_mask[(eod_idx + 1):, :(eod_idx + 1)] = 0.0

            # Convert attention mask to binary:
            attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask < 0.5
        else:
            attention_mask = None

        # 4) Make sure padding tokens are masked even if they are part of an assistant answer somehow TODO: check!
        loss_mask[data == self._pad_token_id] = 0.0
        if assistant_mask is not None:
            assistant_mask[data == self._pad_token_id] = False

        # 5) Equalize sample loss
        if self.config.sft_equalize_sample_loss:
            # Add small epsilon to prevent division by very small numbers
            eps = 1e-10

            if eos_indices.size > 0: # in sane data this should always be the case as every sample packed or not has >=1 doc
                # Process each sample segment (between EOD tokens)
                start_idx = 0
                for eod_idx in eos_indices:
                    segment_mask = loss_mask[start_idx:eod_idx+1]
                    segment_loss_sum = segment_mask.sum()

                    # Normalize so total sample contribution = 1.0
                    if segment_loss_sum > eps:
                        loss_mask[start_idx:eod_idx+1] = segment_mask / segment_loss_sum

                    start_idx = eod_idx + 1

                # Handle the last segment (from last EOD to end of sequence, can be truncated or padding)
                if start_idx < len(loss_mask):
                    segment_mask = loss_mask[start_idx:]
                    segment_loss_sum = segment_mask.sum()
                    # only do if not just padding (sum = 0)
                    if segment_loss_sum > eps:
                        loss_mask[start_idx:] = segment_mask / segment_loss_sum

        return attention_mask, loss_mask, position_ids, assistant_mask


def get_matching_mask(sequence, query: torch.Tensor, only_begin:bool=True):
    """
    Given a sequence and a query, return a mask indicating which positions in the sequence match the query.
    If the query has len > 1, only_begin arg will determine whether the mask is true only where
    the query begins in the sequence. Otherwise, full query is masked.
    """
    query_len = len(query)
    # Vectorized pattern matching using unfold
    if query_len == 1:
        matches = (sequence == query[0])
    else:
        # Create sliding windows
        windows = sequence.unfold(0, query_len, 1)
        # Compare all windows at once
        matches = (windows == query).all(dim=1)
        # Pad to original length
        matches = F.pad(matches, (0, query_len - 1), value=False)
        if not only_begin:
            matches_float = matches.float().unsqueeze(0).unsqueeze(0)  # (1, 1, N)
            kernel = torch.ones(1, 1, query_len, device=sequence.device)
            expanded = F.conv1d(matches_float, kernel, padding=query_len - 1)
            matches = (expanded.squeeze(0).squeeze(0)[:len(sequence)] > 0)
    return matches


def get_matching_mask_by_start_end(sequence, begin_seq: torch.Tensor, end_seq: torch.Tensor):
    """
    Given a sequence and a start and end query, return a mask indicating which positions in the sequence
    are between the start and end queries (inclusive).

    Limitation: If the sequence starts mid-region (e.g. due to left truncation), an orphaned end_seq
    without a preceding begin_seq will be ignored, leaving those leading tokens unmasked.
    """
    mask = torch.zeros(len(sequence), dtype=torch.bool, device=sequence.device)
    begin_len = len(begin_seq)
    end_len = len(end_seq)

    if 0 < begin_len <= len(sequence):
        matches_begin = get_matching_mask(sequence, begin_seq, only_begin=True)

        if 0 < end_len <= len(sequence):
            matches_end = get_matching_mask(sequence, end_seq, only_begin=True)
            end_indices = torch.where(matches_end)[0]
        else:
            end_indices = torch.empty(0, dtype=torch.long, device=sequence.device)

        begin_indices = torch.where(matches_begin)[0]

        # Vectorized masking
        if len(begin_indices) > 0 and len(end_indices) > 0:
            # For each begin, find the next ends (vectorized)
            end_matrix = end_indices.unsqueeze(0) > begin_indices.unsqueeze(1)
            has_valid_end = end_matrix.any(dim=1)
            first_end_idx = end_matrix.int().argmax(dim=1)

            # Compute end positions for each begin
            end_positions = torch.where(
                has_valid_end,
                end_indices[first_end_idx] + end_len,
                len(mask)
            )

            # Create ranges and mask in one go, Shape: (num_begins, max_range_len)
            max_len = (end_positions - begin_indices).max().item()
            ranges = torch.arange(max_len, device=sequence.device).unsqueeze(0)
            lengths = (end_positions - begin_indices).unsqueeze(1)

            # Get all indices to mask
            mask_positions = begin_indices.unsqueeze(1) + ranges
            valid_mask = ranges < lengths
            indices_to_mask = mask_positions[valid_mask]

            mask[indices_to_mask] = True
        elif len(begin_indices) > 0:
            # No end sequences, mask from each begin to the end
            max_len = len(mask) - begin_indices.min().item()
            ranges = torch.arange(max_len, device=sequence.device).unsqueeze(0)
            mask_positions = begin_indices.unsqueeze(1) + ranges
            valid = mask_positions < len(mask)
            mask[mask_positions[valid]] = True
    return mask


class IndexCacheManager:
    """Manages cache file I/O for dataset indices."""

    def __init__(
        self,
        config,
        dataset_path_prefix: str,
        unique_description: str,
        unique_description_hash: str,
        dataset_class_name: str,
        split_name: str
    ):
        self.config = config
        self.dataset_path_prefix = dataset_path_prefix
        self.unique_description = unique_description
        self.unique_description_hash = unique_description_hash
        self.dataset_class_name = dataset_class_name
        self.split_name = split_name

        self._cache_dir = self._determine_cache_path()

        if self._cache_dir:
            base = f"{unique_description_hash}-{dataset_class_name}-{split_name}"
            self._get_path_to = lambda affix: os.path.join(self._cache_dir, f"{base}-{affix}")

    def _determine_cache_path(self) -> Optional[str]:
        """Determine cache directory path from config."""
        path_to_cache = self.config.path_to_cache
        if path_to_cache is None and not self.config.mock:
            path_to_cache = os.path.join(
                self.dataset_path_prefix, "cache", f"{self.dataset_class_name}_indices"
            )
        return path_to_cache

    def get_cache_path(self) -> Optional[str]:
        """Return cache directory path."""
        return self._cache_dir

    def get_index_path(self, index_name: str) -> str:
        """Return full path for an index file."""
        if not self._cache_dir:
            raise ValueError("Cache path is not configured")

        if index_name == "description":
            return self._get_path_to("description.txt")
        else:
            return self._get_path_to(f"{index_name}.npy")

    def cache_exists(self, index_names: List[str]) -> bool:
        """Check if all required cache files exist."""
        if not self._cache_dir:
            return False

        files_to_check = [self.get_index_path("description")]
        for index_name in index_names:
            if index_name != "description":
                files_to_check.append(self.get_index_path(index_name))

        return all(os.path.isfile(f) for f in files_to_check)

    def save_indices(self, indices: Dict[str, np.ndarray]) -> None:
        """Save indices to cache files."""
        if not self._cache_dir:
            log_single_rank(
                logger,
                logging.WARNING,
                f"Unable to save {self.dataset_class_name} indices because path_to_cache is None",
            )
            return

        os.makedirs(self._cache_dir, exist_ok=True)

        if "description" in indices:
            with open(self.get_index_path("description"), "wt") as writer:
                writer.write(indices["description"])

        for index_name, index_data in indices.items():
            if index_name != "description" and index_data is not None:
                np.save(self.get_index_path(index_name), index_data, allow_pickle=True)

    def load_indices(self, index_names: List[str]) -> Dict[str, np.ndarray]:
        """Load indices from cache files."""
        if not self._cache_dir:
            raise ValueError("Cannot load indices: cache path is not configured")

        indices = {}
        for index_name in index_names:
            index_path = self.get_index_path(index_name)
            log_single_rank(logger, logging.INFO, f"\tLoad the {index_name} from {os.path.basename(index_path)}")

            t_beg = time.time()
            indices[index_name] = np.load(index_path, allow_pickle=True, mmap_mode='r')
            t_end = time.time()

            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        return indices


def _build_document_index(
    num_epochs: int,
    documents: np.ndarray,
    numpy_random_state: np.random.RandomState,
) -> np.ndarray:
    """
    Build document-index with size: num_epochs * len(documents)
    Shuffle within each epoch of documents independently.
    """
    document_index = np.mgrid[0:num_epochs, 0: len(documents)][1]
    document_index[:] = documents
    document_index = document_index.astype(np.int32)

    for epoch in range(num_epochs):
        numpy_random_state.shuffle(document_index[epoch])

    document_index = document_index.reshape(-1)
    return document_index