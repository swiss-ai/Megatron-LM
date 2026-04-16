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


from dataclasses import dataclass

@dataclass
class SpecialTokenIDs:
    """Integer IDs for special tokens used in SFT loss masking."""
    system_start: int     # <|system_start|>
    assistant_start: int  # <|assistant_start|>
    assistant_end: int    # <|assistant_end|>
    eod: int              # </s>
    bos: int              # <s>
    developer_start: int  # <|developer_start|>
    user_start: int       # <|user_start|>


def _get_synthetic_special_ids(tokenizer) -> SpecialTokenIDs:
    """Resolve special token strings to integer IDs via the HF tokenizer."""
    hf_tok = tokenizer._tokenizer.tokenizer
    def _tok_id(token: str) -> int:
        idx = hf_tok.convert_tokens_to_ids(token)
        if idx == hf_tok.unk_token_id:
            raise KeyError(f"Token '{token}' not found in tokenizer vocabulary.")
        return idx
    return SpecialTokenIDs(
        system_start=_tok_id("<|system_start|>"),
        assistant_start=_tok_id("<|assistant_start|>"),
        assistant_end=_tok_id("<|assistant_end|>"),
        developer_start=_tok_id("<|developer_start|>"),
        user_start=_tok_id("<|user_start|>"),
        eod=tokenizer.eod,
        bos=tokenizer.bos,
    )


def _get_subdoc_spans_from_eos(eos_indices: np.ndarray, seq_len: int) -> List[Tuple[int, int]]:
    """Convert per-doc last-token positions into (start, end) inclusive spans.

    Skips degenerate spans (start > end) from duplicate eos indices or eos at seq_len - 1.
    """
    spans: List[Tuple[int, int]] = []
    start = 0
    for eod_idx in eos_indices:
        end = min(int(eod_idx), seq_len - 1)
        if start <= end:
            spans.append((start, end))
        start = end + 1
        if start >= seq_len:
            break
    if start < seq_len:
        spans.append((start, seq_len - 1))
    return spans


def _process_subdoc_masking(
    data: torch.Tensor,
    tokens: torch.Tensor,
    loss_mask: torch.Tensor,
    assistant_mask: torch.Tensor,
    start: int,
    end: int,
    special_ids: SpecialTokenIDs,
) -> None:
    """Apply masking rules in-place for one sub-document [start, end] (inclusive).

    Span does not begin with BOS -> cut fragment; mask each SFT turn from its opening
                              marker (<|system_start|> or <|developer_start|>) through
                              the paired <|assistant_end|> (inclusive), so the model is
                              not trained on answers that reference missing context (e.g.
                              CWE counting tasks). Pre-training tokens before the first
                              marker are left unmasked.
    No SYS, no ASST -> pure pre-training; loss_mask unchanged (stays 1).
    SYS, no ASST    -> broken tail;  zero [SYS .. end].
    Both present    -> normal SFT:   zero [SYS .. ASST] and [AEND];
                       assistant_mask = True for answer tokens (ASST+1 .. AEND-1).
    """

    # Case 1: Document cut
    if tokens[start] != special_ids.bos:
        _sub = data[start : end + 1]
        _SYS = special_ids.system_start
        _DEV = special_ids.developer_start
        _AEND = special_ids.assistant_end
        _USST = special_ids.user_start
        _turn_starts = sorted(
            [int(p) for p in (_sub == _SYS).nonzero(as_tuple=True)[0]] +
            [int(p) for p in (_sub == _DEV).nonzero(as_tuple=True)[0]] +
            [int(p) for p in (_sub == _USST).nonzero(as_tuple=True)[0]]
        )
        _aend_positions = (_sub == _AEND).nonzero(as_tuple=True)[0].tolist()
        for ts_pos in _turn_starts:
            next_aend = next((a for a in _aend_positions if a > ts_pos), None)
            mask_end = next_aend if next_aend is not None else (end - start)
            loss_mask[start + ts_pos : start + mask_end + 1] = 0.0
        return

    sub = data[start : end + 1]

    SYS  = special_ids.system_start
    ASST = special_ids.assistant_start
    AEND = special_ids.assistant_end
    USST  = special_ids.user_start
    DEV  = special_ids.developer_start

    sys_positions = (sub == SYS).nonzero(as_tuple=True)[0].tolist()
    asst_positions = (sub == ASST).nonzero(as_tuple=True)[0].tolist()
    aend_positions = (sub == AEND).nonzero(as_tuple=True)[0].tolist()
    usst_positions = (sub == USST).nonzero(as_tuple=True)[0].tolist()
    dev_positions = (sub == DEV).nonzero(as_tuple=True)[0].tolist()

    has_sys  = bool(sys_positions)
    has_asst = bool(asst_positions)
    has_user = bool(usst_positions)

    # Case 2: pure pre-training
    if not has_user and not has_asst:       
        return
    
    first_turn = min(
        sys_positions + usst_positions + dev_positions
    )
    loss_mask[start + first_turn : end + 1] = 0.0

    # Case 3: broken tail, we mask all the chat
    if has_user and not has_asst:
        return

    # Case 4: both SYS and ASST (possibly multi-turn)
    for asst_pos in asst_positions:
        # Find corresponding AEND
        aend_pos = next((a for a in aend_positions if a > asst_pos), None)

        if aend_pos is not None:
            ans_start = start + asst_pos + 1
            ans_end = start + aend_pos + 1

            if ans_start < ans_end:
                loss_mask[ans_start:ans_end] = 1.0
                assistant_mask[ans_start:ans_end] = True
        else:
            # Broken tail: unmask until end
            ans_start = start + asst_pos + 1
            ans_end = end + 1

            if ans_start < ans_end:
                loss_mask[ans_start:ans_end] = 1.0
                assistant_mask[ans_start:ans_end] = True

def _build_virtual_docs(
    document_index: np.ndarray,
    sequence_lengths: np.ndarray,
    capacity: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split any document longer than *capacity* into capacity-sized chunks.

    Documents that fit within *capacity* are kept as a single virtual entry.
    Oversized documents produce multiple virtual entries, each loadable via
    ``IndexedDataset.get(doc_id, offset=start, length=length)``.

    Returns:
        chunk_map: Shape (num_virtual, 3), int64. Columns:
            [real_doc_id, chunk_start_token, chunk_length_tokens].
        virtual_sizes: Shape (num_virtual,), int32. Length of each virtual chunk.
    """
    real_doc_ids = document_index.astype(np.int64)
    doc_lens = sequence_lengths[real_doc_ids].astype(np.int64)

    # Number of chunks each document produces (ceil division for oversized docs)
    num_chunks = np.where(doc_lens <= capacity, 1, (doc_lens + capacity - 1) // capacity)
    total_virtual = int(num_chunks.sum())

    chunk_map = np.empty((total_virtual, 3), dtype=np.int64)
    virtual_sizes = np.empty(total_virtual, dtype=np.int32)

    # Cumulative output positions: chunk_offsets[i] = first output row for doc i
    chunk_offsets = np.empty(len(num_chunks) + 1, dtype=np.int64)
    chunk_offsets[0] = 0
    np.cumsum(num_chunks, out=chunk_offsets[1:])

    # Fast vectorised path for single-chunk docs (the common case)
    single = num_chunks == 1
    single_out = chunk_offsets[:-1][single]
    chunk_map[single_out, 0] = real_doc_ids[single]
    chunk_map[single_out, 1] = 0
    chunk_map[single_out, 2] = doc_lens[single]
    virtual_sizes[single_out] = doc_lens[single].astype(np.int32)

    # Scalar loop only for oversized docs (rare)
    for pos in np.where(~single)[0]:
        rid = int(real_doc_ids[pos])
        dlen = int(doc_lens[pos])
        base = int(chunk_offsets[pos])
        offset = 0
        ci = 0
        while offset < dlen:
            clen = min(capacity, dlen - offset)
            chunk_map[base + ci] = [rid, offset, clen]
            virtual_sizes[base + ci] = clen
            offset += capacity
            ci += 1

    return chunk_map, virtual_sizes


def _load_bfd_c_library():
    """Load (or compile then load) the C/C++ BFD packing library.

    Uses a segment-tree + min-heap structure for O(n log C) best-fit lookup,
    ~10-15x faster than pure-Python bisect at million-doc scale. Thread-safe.
    Returns the loaded ctypes library, or None if unavailable.
    """
    import ctypes
    import subprocess

    _dir = os.path.dirname(os.path.abspath(__file__))
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
        return _build_sample_idx_bfd_c(sequence_lengths, document_index, seq_length, add_extra_token)
    return _build_sample_idx_bfd_python(sequence_lengths, document_index, seq_length, add_extra_token)


def _build_sample_idx_bfd_c(sequence_lengths, document_index, seq_length, add_extra_token):
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
        num_docs, capacity,
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
            if new_remaining > 0:
                # Re-insert with updated capacity
                bisect.insort(bins_sorted, (new_remaining, bin_id))
        else:
            # No bin fits: open a new one
            bin_id = len(bin_contents)
            bin_contents.append([int(pos)])
            new_remaining = capacity - length
            if new_remaining > 0:
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
    Supports mixed pre-training + SFT binaries: pre-training sub-documents (no
    <|system_start|>) receive full loss; SFT sub-documents have their prompt scaffold
    masked and only assistant answer tokens contribute to the loss.

    Oversized documents (longer than seq_length) are split into capacity-sized virtual
    chunks at index-build time via _build_virtual_docs, eliminating truncation.

    Loss masking operates in two modes:
      1. **From disk** (--ap-sft-load-loss-mask): Each document stores tokens and a pre-computed
         loss mask concatenated together ([tokens, loss_mask]). The dataset splits them at load
         time and uses the mask as-is.
      2. **On the fly** (default): loss_mask starts at 1 for all tokens. Per sub-document,
         _process_subdoc_masking zeros scaffold regions for SFT docs and leaves pre-training
         sub-documents (no <|system_start|>) fully unmasked.

    Note: Goldfish loss (--goldfish-loss) is not supported and will be ignored if enabled.
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

        # Special token IDs for subdoc-based masking. Replaces the old
        # sft_assistant_begin/end_sequence + tokens_to_mask machinery.
        try:
            self._special_ids = _get_synthetic_special_ids(self.tokenizer)
            log_single_rank(logger, logging.INFO,
                f"Synthetic masking IDs — SYS={self._special_ids.system_start}, "
                f"ASST={self._special_ids.assistant_start}, "
                f"AEND={self._special_ids.assistant_end}, "
                f"BOS={self._special_ids.bos}, "
                f"DEV={self._special_ids.developer_start}")
        except KeyError as e:
            raise ValueError(
                f"Tokenizer vocab is missing required special token: {e}. "
                f"ApertusSFTDataset requires <|system_start|>, <|assistant_start|>, "
                f"and <|assistant_end|> in the tokenizer vocabulary."
            ) from e

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
            # Use multi-document packing with sample_index and virtual chunk map
            (self.document_index, self.sample_index, self.shuffle_index, self.chunk_map) = (
                self._build_packing_document_to_sample_indices()
            )
            self._using_packed_samples = True
        else:

            # Use simple single-document indexing
            self.document_index = self._build_single_document_indices()
            self.chunk_map = None
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
        Log statistics about packed samples.

        Args:
            document_index: Array of document IDs (virtual chunk IDs in packing mode)
            sample_index: Array of sample boundaries
            from_cache: Whether the indices were loaded from cache
        """
        num_samples_available = sample_index.shape[0] - 1
        sequence_length = self.config.sequence_length
        num_tokens_per_epoch = int(np.sum(self.dataset.sequence_lengths[self.indices]))
        total_tokens_in_samples = num_samples_available * sequence_length
        avg_tokens_per_sample = num_tokens_per_epoch / num_samples_available if num_samples_available > 0 else 0
        avg_documents_per_sample = len(document_index) / num_samples_available if num_samples_available > 0 else 0

        cache_suffix = " (loaded from cache)" if from_cache else ""
        packing_efficiency = 100 * num_tokens_per_epoch / total_tokens_in_samples if total_tokens_in_samples > 0 else 0

        log_single_rank(logger, logging.INFO, f"> ===== SFT Packing Statistics (ONE EPOCH){cache_suffix} =====")
        log_single_rank(logger, logging.INFO, f" > #docs in epoch:                    {len(document_index):>12}")
        log_single_rank(logger, logging.INFO, f" > #tokens in epoch:                  {num_tokens_per_epoch:>12,}")
        log_single_rank(logger, logging.INFO, f" > Sequence length:                   {sequence_length:>12}")
        log_single_rank(logger, logging.INFO, f" > #packed samples (per epoch):       {num_samples_available:>12,}")
        log_single_rank(logger, logging.INFO, f" > #tokens(incl. padding) in samples: {total_tokens_in_samples:>12,}")
        log_single_rank(logger, logging.INFO, f" > Average #tokens/sample:            {avg_tokens_per_sample:>12.1f}")
        log_single_rank(logger, logging.INFO, f" > Average #documents/sample:         {avg_documents_per_sample:>12.2f}")
        log_single_rank(logger, logging.INFO, f" > Packing efficiency:                {packing_efficiency:>11.2f}%\n\n")

    def _build_packing_document_to_sample_indices(self):
        """
        Build indices for packed document sampling. Always packs exactly one epoch so that
        every document is used. Oversized documents are pre-split into capacity-sized virtual
        chunks via _build_virtual_docs so no tokens are lost to truncation. If more samples
        are requested than one epoch provides, the shuffle index is tiled with independent
        permutations (no re-packing).

        Returns a tuple of four indices:
        - document_index: Permutation of virtual chunk IDs (0..num_virtual-1) in packing order
        - sample_index: Maps sample boundaries to (document_index position, offset) pairs
        - shuffle_index: Permutation indices, possibly tiled to meet num_samples
        - chunk_map: Shape (num_virtual, 3) mapping virtual chunk ID to
          [real_doc_id, chunk_start_token, chunk_length_tokens]
        """
        from megatron.core.datasets import helpers

        index_names = ["document_index", "sample_index", "shuffle_index", "chunk_map"]
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

            # Build virtual document list: oversized docs are split into capacity-sized
            # chunks so they never waste space. Each virtual entry maps to a slice of a
            # real document loadable via dataset.get(doc_id, offset=start, length=length).
            capacity = sequence_length + self.config.add_extra_token_to_sequence
            chunk_map, virtual_sizes = _build_virtual_docs(
                document_index, sequence_lengths_for_cpp, capacity
            )
            num_virtual = len(virtual_sizes)
            virtual_idx = np.arange(num_virtual, dtype=np.int32)

            n_extra = num_virtual - len(document_index)
            log_single_rank(logger, logging.INFO,
                f"  Document chunking: {len(document_index)} docs → {num_virtual} virtual chunks"
                + (f" ({n_extra} extra chunks from oversized docs)" if n_extra > 0 else " (no oversized docs)"))

            # Build the sample index using the configured packing strategy.
            # We pass virtual_sizes as the sequence_lengths lookup table and virtual_idx
            # as the document_index (identity: virtual_idx[i] == i).
            if self.config.sft_packing_strategy == "bfd":
                _accel = "C-accelerated" if _bfd_c_lib is not None else "Python fallback"
                log_single_rank(logger, logging.INFO,
                    f"Using Best-Fit Decreasing packing strategy ({_accel})")
                document_index, sample_index = _build_sample_idx_bfd(
                    virtual_sizes, virtual_idx, sequence_length,
                    add_extra_token=self.config.add_extra_token_to_sequence,
                )
            else:
                sample_index = helpers.build_sample_idx_packed_whole_docs(
                    virtual_sizes, virtual_idx, sequence_length,
                    add_extra_token_to_sequence=self.config.add_extra_token_to_sequence,
                )
                document_index = virtual_idx

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
                "shuffle_index": shuffle_index,
                "chunk_map": chunk_map,
            })

            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            return document_index, sample_index, shuffle_index, chunk_map

        # Load from cache
        log_single_rank(
            logger, logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} packed indices"
        )
        indices = self.cache_manager.load_indices(index_names)
        document_index = indices["document_index"]
        sample_index = indices["sample_index"]
        shuffle_index = indices["shuffle_index"]
        chunk_map = indices["chunk_map"]
        self._log_packing_statistics(document_index, sample_index, from_cache=True)

        return document_index, sample_index, shuffle_index, chunk_map

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
            - document-index: permutation of virtual chunk IDs; each is looked up via chunk_map
              to obtain [real_doc_id, chunk_start, chunk_len]

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
            # chunk_map[virtual_id] = [real_doc_id, chunk_start, chunk_len]. Load exactly the
            # chunk slice; no full-document load or truncation needed (chunks <= capacity).
            virtual_id = int(self.document_index[i])
            real_doc_id = int(self.chunk_map[virtual_id, 0])
            chunk_start = int(self.chunk_map[virtual_id, 1])
            chunk_len   = int(self.chunk_map[virtual_id, 2])
            document = self.dataset.get(real_doc_id, offset=chunk_start, length=chunk_len)

            # If loading loss masks from disk, split the document into tokens and loss_mask
            if self.config.sft_load_loss_mask:
                # Dataset stores [tokens, loss_mask] concatenated
                doc_len = len(document) // 2
                doc_tokens = document[:doc_len]
                doc_loss_mask = document[doc_len:]

                document_tokens.append(doc_tokens)
                document_loss_masks.append(doc_loss_mask)
                doc_end_indices.append(doc_tokens.size)
            else:
                # Original behavior: no loss mask splitting
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
                f"This indicates a bug in _build_virtual_docs or the packing index. "
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

        # Generate loss-mask, position-ids, assistant mask and optionally attention mask.
        # tokens are passed for BOS detection in subdoc masking; labels drive matching.
        attention_mask, loss_mask, position_ids, assistant_mask = self._get_ltor_masks_and_position_ids(
            tokens,
            labels,
            eos_idx,
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

    def _get_ltor_masks_and_position_ids(self, tokens: torch.Tensor, data: torch.Tensor, eos_indices: np.ndarray, preloaded_loss_mask: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Build masks and position ids for mixed pre-training + SFT data.
        loss_mask starts at 1 for all tokens (pre-training default). Per-subdoc scaffold
        masking via _process_subdoc_masking then zeros prompt regions for SFT sub-documents,
        leaving pre-training sub-documents (no <|system_start|>) fully unmasked.
        Also creates an assistant_mask to identify assistant response tokens (always built
        on-the-fly, even when preloaded_loss_mask is used).
            1. Can mask full user prompts (per-subdoc, always)
            2. Special tokens are handled implicitly via _process_subdoc_masking
            3. Creates attention mask if configured. The attention mask will exclude padding tokens from attention.
            4. Can equalize sample loss for packed and non-packed sequences (loss = 1 for each sample in seq)

        For packed samples (when sft_pack_samples=True):
            - Position IDs are reset at each EOD token (document boundary)
            - Attention mask blocks cross-document attention at EOD boundaries
            - ASSUMES NO WRONG PLACED EOD (=ONLY PROPERLY BOUNDARY OF SAMPLES)

        Args:
            tokens:                 tokens (text[:-1]), used only for BOS detection in subdoc masking.
            data:                   labels (text[1:]).
            eos_indices:            indices of document boundaries (calculated based on sample loading from low level dataset as
                                    sft data can be contaminated with eod or eod missing).
            preloaded_loss_mask:    Optional pre-computed loss mask loaded from disk. If provided, scaffold
                                    masking is skipped for loss_mask (but assistant_mask is still populated).
        """

        position_ids = torch.arange(self.model_seq_length, dtype=torch.long)
        # Start at ones (pre-training default). Scaffold regions are zeroed by _process_subdoc_masking.
        loss_mask = preloaded_loss_mask.to(device=data.device) if preloaded_loss_mask is not None else torch.ones(self.model_seq_length, dtype=torch.float, device=data.device)

        # 0) For packed samples: reset position IDs at document boundaries
        if self._using_packed_samples:
            if eos_indices.size > 0:
                # Reset position IDs after each EOD token
                for eod_idx in eos_indices:
                    if eod_idx + 1 < len(position_ids):
                        # Subtract the position value at EOD+1 from all subsequent positions
                        to_subtract = position_ids[eod_idx].clone() + 1
                        position_ids[(eod_idx + 1):] -= to_subtract

        # 1) Per-subdoc scaffold masking + assistant_mask.
        # Pre-training sub-documents pass through Case 1 of _process_subdoc_masking
        # unchanged (loss_mask stays 1). SFT sub-documents have scaffold zeroed.
        # When preloaded_loss_mask is given, scaffold writes go to a scratch tensor so the
        # disk-loaded mask is preserved; assistant_mask is still populated.
        assistant_mask = torch.zeros(self.model_seq_length, dtype=torch.bool, device=data.device)
        scaffold_target = loss_mask if preloaded_loss_mask is None else torch.zeros_like(loss_mask)
        for span_start, span_end in _get_subdoc_spans_from_eos(eos_indices, self.model_seq_length):
            _process_subdoc_masking(data, tokens, scaffold_target, assistant_mask, span_start, span_end, self._special_ids)

        # 2) Create attention mask: mask attention from/to padding tokens
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

        # 3) Make sure padding tokens are masked even if they are part of an assistant answer somehow TODO: check!
        loss_mask[data == self._pad_token_id] = 0.0
        if assistant_mask is not None:
            assistant_mask[data == self._pad_token_id] = False

        # 4) Equalize sample loss
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