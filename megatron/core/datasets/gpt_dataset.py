# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
import os
import time
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional, Tuple

import numpy
import torch

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.object_storage_utils import ObjectStorageConfig, is_object_storage_path
from megatron.core.datasets.utils import Split
from megatron.core.tokenizers import MegatronTokenizerBase
from megatron.core.utils import log_single_rank
import bisect

logger = logging.getLogger(__name__)


_GOLDFISH_TOKEN_ID = -2
_HASH_TABLE_SIZE = 1_000_003
_PAD_TOKEN_ID = -1


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

    # Rebuild if source is newer (e.g. after the max_docs_per_bin signature change).
    so_is_fresh = (
        os.path.isfile(so_path)
        and (not os.path.isfile(cpp_path) or os.path.getmtime(so_path) >= os.path.getmtime(cpp_path))
    )
    if so_is_fresh:
        try:
            lib = ctypes.CDLL(so_path)
            lib.bfd_pack.argtypes = [
                ctypes.POINTER(ctypes.c_int),  # sorted_positions
                ctypes.POINTER(ctypes.c_long), # doc_lengths
                ctypes.c_int,                  # num_docs
                ctypes.c_int,                  # capacity
                ctypes.c_int,                  # max_docs_per_bin (0 = no cap)
                ctypes.POINTER(ctypes.c_int),  # document_index
                ctypes.POINTER(ctypes.c_int),  # doc_idx_out
                ctypes.POINTER(ctypes.c_int),  # boundaries_out
                ctypes.POINTER(ctypes.c_int),  # num_bins_out
            ]
            lib.bfd_pack.restype = None
            return lib
        except OSError:
            pass

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

def _build_virtual_docs(document_index, sequence_lengths, capacity, eod_token_id):
    """Split any document longer than *capacity* into chunks with proper EOD boundaries.

    For oversized docs (L > capacity), chunks are taken at step `capacity - 1` real tokens
    and the loader appends a synthetic EOD to every non-last chunk so each chunk ends on an
    EOD. The last chunk of an oversized doc carries the document's natural EOD. Docs that
    fit within `capacity` are kept whole (single chunk, no synthetic token).

    Returns:
        chunk_map: Shape (num_virtual, 4), int64. Columns:
            [real_doc_id, chunk_start_token, chunk_length_tokens, append_synthetic_eod].
        virtual_sizes: Shape (num_virtual,), int32. Virtual chunk length *including* any
            synthetic EOD that the loader will append.
    """
    real_doc_ids = document_index.astype(numpy.int64)
    doc_lens = sequence_lengths[real_doc_ids].astype(numpy.int64)

    step = capacity - 1
    oversized = doc_lens > capacity
    num_chunks = numpy.where(oversized, (doc_lens + step - 1) // step, 1)
    total_virtual = int(num_chunks.sum())

    chunk_map = numpy.empty((total_virtual, 4), dtype=numpy.int64)
    virtual_sizes = numpy.empty(total_virtual, dtype=numpy.int32)

    chunk_offsets = numpy.empty(len(num_chunks) + 1, dtype=numpy.int64)
    chunk_offsets[0] = 0
    numpy.cumsum(num_chunks, out=chunk_offsets[1:])

    # Single-chunk fast path (L <= capacity): use doc whole, no synthetic EOD
    single_mask = ~oversized
    single_out = chunk_offsets[:-1][single_mask]
    chunk_map[single_out, 0] = real_doc_ids[single_mask]
    chunk_map[single_out, 1] = 0
    chunk_map[single_out, 2] = doc_lens[single_mask]
    chunk_map[single_out, 3] = 0
    virtual_sizes[single_out] = doc_lens[single_mask].astype(numpy.int32)

    # Oversized: step-sized chunks with synth EOD, last chunk uses natural EOD
    for pos in numpy.where(oversized)[0]:
        rid = int(real_doc_ids[pos])
        dlen = int(doc_lens[pos])
        base = int(chunk_offsets[pos])
        nc = int(num_chunks[pos])
        for ci in range(nc):
            off = ci * step
            is_last = (ci == nc - 1)
            if is_last:
                clen = dlen - off
                append_eod = 0
            else:
                clen = step
                append_eod = 1
            chunk_map[base + ci] = [rid, off, clen, append_eod]
            virtual_sizes[base + ci] = clen + append_eod

    return chunk_map, virtual_sizes


def _build_sample_idx_bfd(sequence_lengths, document_index, seq_length, add_extra_token, max_docs_per_bin):
    """Best-Fit Decreasing bin packing for whole documents. C-accelerated when available."""
    if _bfd_c_lib is not None:
        return _build_sample_idx_bfd_c(sequence_lengths, document_index, seq_length, add_extra_token, max_docs_per_bin)
    return _build_sample_idx_bfd_python(sequence_lengths, document_index, seq_length, add_extra_token, max_docs_per_bin)


def _build_sample_idx_bfd_c(sequence_lengths, document_index, seq_length, add_extra_token, max_docs_per_bin=0):
    """C-accelerated BFD bin packing via ctypes. See _build_sample_idx_bfd."""
    import ctypes

    capacity = seq_length + add_extra_token
    num_docs = len(document_index)

    doc_lengths = sequence_lengths[document_index].astype(numpy.int64)
    sorted_positions = numpy.argsort(-doc_lengths, kind='stable').astype(numpy.int32)
    doc_index_i32 = document_index.astype(numpy.int32)

    doc_idx_out = numpy.empty(num_docs, dtype=numpy.int32)
    boundaries_out = numpy.empty(num_docs + 1, dtype=numpy.int32)
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
    sample_index = numpy.zeros((nb + 1, 2), dtype=document_index.dtype)
    sample_index[:, 0] = boundaries_out[:nb + 1].astype(document_index.dtype)

    assert boundaries_out[nb] == num_docs, f"BFD placed {boundaries_out[nb]} docs but expected {num_docs}"
    return reordered, sample_index


def _build_sample_idx_bfd_python(sequence_lengths, document_index, seq_length, add_extra_token, max_docs_per_bin=0):
    """Pure-Python BFD fallback using bisect. See _build_sample_idx_bfd."""
    capacity = seq_length + add_extra_token
    num_docs = len(document_index)

    doc_lengths = sequence_lengths[document_index].astype(numpy.int64)
    sorted_positions = numpy.argsort(-doc_lengths, kind='stable')

    bins_sorted = []
    bin_contents = []

    # A bin is at the cap iff max_docs_per_bin > 0 and it already holds that many docs.
    def _capped(bid):
        return max_docs_per_bin > 0 and len(bin_contents[bid]) >= max_docs_per_bin

    for pos in sorted_positions:
        length = int(doc_lengths[pos])
        if length > capacity:
            bin_contents.append([int(pos)])
            continue
        if length == 0:
            if bins_sorted:
                _, bin_id = bins_sorted[0]
                bin_contents[bin_id].append(int(pos))
                if _capped(bin_id):
                    bins_sorted.pop(0)
            else:
                bin_id = len(bin_contents)
                bin_contents.append([int(pos)])
                if not _capped(bin_id):
                    bisect.insort(bins_sorted, (capacity, bin_id))
            continue
        idx = bisect.bisect_left(bins_sorted, (length,))
        if idx < len(bins_sorted):
            remaining, bin_id = bins_sorted.pop(idx)
            new_remaining = remaining - length
            bin_contents[bin_id].append(int(pos))
            if new_remaining > 0 and not _capped(bin_id):
                bisect.insort(bins_sorted, (new_remaining, bin_id))
        else:
            bin_id = len(bin_contents)
            bin_contents.append([int(pos)])
            new_remaining = capacity - length
            if new_remaining > 0 and not _capped(bin_id):
                bisect.insort(bins_sorted, (new_remaining, bin_id))

    reordered = numpy.empty(num_docs, dtype=document_index.dtype)
    boundaries = []
    offset = 0
    for contents in bin_contents:
        boundaries.append(offset)
        for pos in contents:
            reordered[offset] = document_index[pos]
            offset += 1
    boundaries.append(offset)

    sample_index = numpy.zeros((len(boundaries), 2), dtype=document_index.dtype)
    sample_index[:, 0] = numpy.array(boundaries, dtype=document_index.dtype)
    assert offset == num_docs
    return reordered, sample_index

@dataclass
class GPTDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for Megatron Core GPT datasets"""

    reset_position_ids: Optional[bool] = None
    """Option to reset the position IDs in the dataset at an interval"""

    reset_attention_mask: Optional[bool] = None
    """Option to reset the attention mask from the dataset"""

    eod_mask_loss: Optional[bool] = None
    """Option to enable the EOD mask loss"""

    goldfish_loss: bool = None
    """Option to enable the goldfish loss"""

    goldfish_k: int = None
    """Frequency of ignoring tokens for goldfish loss: 1 / k"""

    goldfish_h: int = None
    """Context width for hashing, everytime the same sequence of h tokens appears, the (h+1)th token is ingored"""

    create_attention_mask: bool = True
    """Option to enable the attention masks generation. Can be disabled if attention kernel
       generates masks by itself.
    """

    drop_last_partial_validation_sequence: bool = True
    """Option to drop the last partial validation sequence"""

    add_extra_token_to_sequence: bool = True
    """Option to draw sequences with one extra token to ensure the sample input tokens and sample
       output tokens are both of the desired sequence length
    """

    object_storage_cache_path: Optional[str] = None
    """Path for caching indices for s3 or msc dataloading."""

    sequences_per_dataset: Optional[Dict[str, int]] = None
    """If provided, the sequence and document counts for each dataset. 
       Check --per-dataset-sequences-path
    """

    vision_weight: float = 1.0
    """Loss weight for vision tokens (between img_start and img_end). Default 1.0 (no change)."""

    audio_weight: float = 1.0
    """Loss weight for audio tokens. Default 1.0 (no change)."""

    modality_weights: Optional[Dict[str, float]] = None
    """Per-modality loss weights keyed by modality name (e.g., vision/audio)."""

    loss_mask_token_ids: Optional[List[int]] = None
    """Token IDs whose predictions should be masked from the loss (loss_mask=0).
    Useful for task-control tokens (e.g. <|stt_transcribe|>, <|tts_continue|>) that
    should condition the model but never be predicted."""

    sft_plw: float = 0.0
    """Prompt loss weight for user tokens (0 = fully masked). Used by ApertusSFTDataset."""

    sft_load_loss_mask: bool = False
    """Load pre-computed loss masks from tokenized data. Used by ApertusSFTDataset."""

    sft_mask_special_tokens: bool = False
    """Mask special tokens (BOS, EOD, assistant begin) from loss. Used by ApertusSFTDataset."""

    sft_pack_samples: bool = False
    """Pack multiple whole documents per sequence. Used by ApertusSFTDataset."""

    sft_equalize_sample_loss: bool = False
    """Normalize loss per sample segment. Used by ApertusSFTDataset."""

    sft_truncate_right: bool = False
    """If sample is too long for context, throw away tokens in the end"""

    sft_packing_strategy: str = "bfd"
    """Packing strategy for SFT: 'greedy' (sequential) or 'bfd' (Best-Fit Decreasing)."""

    pretraining_packing_strategy: str = "greedy"
    """Packing strategy for SFT: 'greedy' (sequential) or 'bfd' (Best-Fit Decreasing)."""

    max_docs_per_bin: int = 0
    """Maximum number of documents allowed per sample in bfd, 0 means no limit"""

    max_docs_per_bin_sft: int = 0
    """Maximum number of documents allowed per sample in bfd for SFT, 0 means no limit"""

    ap_sft_auto_tag: bool = False
    """When True, blend entries without an explicit 'sft:' / 'pretrain:' marker
    are dispatched as ApertusSFTDataset. Set by --ap-sft for backward compatibility
    with launchers that pass a single all-SFT blend."""

    def __post_init__(self) -> None:
        """Do asserts and set fields post init"""
        super().__post_init__()

        assert self.tokenizer is not None

        assert self.reset_position_ids is not None
        assert self.reset_attention_mask is not None
        assert self.eod_mask_loss is not None
        assert self.goldfish_loss is not None

        self.token_dtype_code = (
            None
            if self.tokenizer.vocab_size is None
            else (4 if self.tokenizer.vocab_size > numpy.iinfo(numpy.uint16).max + 1 else 8)
        )
        if self.sequences_per_dataset is not None:
            assert (
                self.token_dtype_code is not None
            ), "Tokenizer vocab size is not set, deactivate --per-dataset-sequences-path or \
            fix the tokenizer."

        if self.modality_weights is None:
            self.modality_weights = {}
        if "vision" not in self.modality_weights and self.vision_weight != 1.0:
            self.modality_weights["vision"] = self.vision_weight
        if "audio" not in self.modality_weights and self.audio_weight != 1.0:
            self.modality_weights["audio"] = self.audio_weight


class GPTDataset(MegatronDataset):
    """The base GPT dataset

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the GPTDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The number of samples to draw from the indexed dataset. When
            None, build as many samples as correspond to one epoch.

        index_split (Split): The indexed_indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: Optional[str],
        indexed_indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )

        self.masks_and_position_ids_are_cached = False
        self.cached_attention_mask = None
        self.cached_loss_mask = None
        self.cached_position_ids = None

        # late import to prevent circular import
        from megatron.training import get_args
        args = get_args()

        # Build weighted modality specs as (name, start_id, end_id, weight).
        self._weighted_modality_specs: List[Tuple[str, int, int, float]] = []
        modality_weights = self.config.modality_weights or {}
        used_modalities = set()
        unresolved_weighted_modalities = []

        omnimodal_config = getattr(args, "omnimodal_config", None)
        if omnimodal_config is not None:
            for modality in omnimodal_config.get("modalities", []):
                name = modality.get("name")
                if not name:
                    continue
                used_modalities.add(name)
                weight = modality_weights.get(name, 1.0)
                offset = modality.get("offset")
                vocab_size = modality.get("vocab_size")
                if weight == 1.0:
                    continue
                if offset is None or vocab_size is None:
                    unresolved_weighted_modalities.append(name)
                    continue
                start = int(offset)
                end = start + int(vocab_size)
                self._weighted_modality_specs.append((name, start, end, float(weight)))

        # Fallback for modality names not listed in omnimodal_config but exposed on args.
        for name, weight in modality_weights.items():
            if name in used_modalities or weight == 1.0:
                continue
            offset = getattr(args, f"{name}_token_offset", None)
            vocab_size = getattr(args, f"{name}_vocab_size", None)
            if offset is None or vocab_size is None:
                unresolved_weighted_modalities.append(name)
                continue
            start = int(offset)
            end = start + int(vocab_size)
            self._weighted_modality_specs.append((name, start, end, float(weight)))

        if unresolved_weighted_modalities:
            missing = ", ".join(sorted(set(unresolved_weighted_modalities)))
            raise ValueError(
                "Configured non-default modality weights without token range metadata for: "
                f"{missing}. Ensure tokenizer omnimodal_config (or args.*_token_offset/args.*_vocab_size) "
                "contains these modalities."
            )

        for name, start, end, weight in self._weighted_modality_specs:
            log_single_rank(
                logger,
                logging.INFO,
                f"{name.upper()} ID RANGE: {start} {end} apply weight {weight}",
            )

        # Token IDs to mask from loss (e.g. task-control tokens like <|stt_transcribe|>).
        self._loss_mask_token_ids = self.config.loss_mask_token_ids or []
        if self._loss_mask_token_ids:
            log_single_rank(
                logger,
                logging.INFO,
                f"Loss-masked token IDs: {self._loss_mask_token_ids}",
            )

        self.masks_and_position_ids_are_cacheable = not any(
            [
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.goldfish_loss,
                bool(self._weighted_modality_specs),
                bool(self._loss_mask_token_ids),
            ]
        )

        # Optional contiguous range of omni special tokens to skip in goldfish masking.
        self._goldfish_exemption_start = None
        self._goldfish_exemption_end = None

        (self.document_index, self.sample_index, self.shuffle_index) = (
            self._build_document_sample_shuffle_indices()
        )

        if self.config.goldfish_loss:
            self._goldfish_k = self.config.goldfish_k
            self._goldfish_h = self.config.goldfish_h
            self._goldfish_token_id = _GOLDFISH_TOKEN_ID
            self._goldfish_hash_table = None
            exempt = getattr(self.config.tokenizer, "goldfish_exemption_range", None)
            if exempt:
                self._goldfish_exemption_start, self._goldfish_exemption_end = exempt
                log_single_rank(
                    logger,
                    logging.INFO,
                    "Goldfish exemption range enabled: "
                    f"[{self._goldfish_exemption_start}, {self._goldfish_exemption_end})",
                )

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: IndexedDataset) -> int:
        """Abstract method implementation

        For GPT, the underlying IndexedDataset should be split by sequence, as opposed to, say,
        BERT, which should be split by document

        Args:
            low_level_dataset (IndexedDataset): The underlying IndexedDataset

        Returns:
            int: The number of unique elements in the underlying IndexedDataset
        """
        return low_level_dataset.sequence_lengths.shape[0]

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> IndexedDataset:
        """Abstract method implementation

        Args:
            dataset_path (str): The real path prefix to the IndexedDataset .bin and .idx files

            config (GPTDatasetConfig): The config

        Returns:
            IndexedDataset: The underlying IndexedDataset
        """
        if is_object_storage_path(dataset_path):
            assert config.object_storage_cache_path is not None
            return IndexedDataset(
                dataset_path,
                multimodal=False,
                mmap=config.mmap_bin_files,
                object_storage_config=ObjectStorageConfig(
                    path_to_idx_cache=config.object_storage_cache_path
                ),
            )
        sequences_per_dataset = None
        if config.sequences_per_dataset:
            sequences_per_dataset = config.sequences_per_dataset[dataset_path]
        return IndexedDataset(
            dataset_path,
            multimodal=False,
            mmap=config.mmap_bin_files,
            fast_cache_load=config.fast_cache_load,
            sequences_per_dataset=sequences_per_dataset,
            dtype_code=config.token_dtype_code,
        )

    def __len__(self) -> int:
        """Abstract method implementation

        Returns:
            int: The length of the dataset
        """
        if self.config.defer_npy_index_mmap:
            # NOTE(asolergi-nv): We need the number of samples of every GPTDataset to build/hit the BlendedDataset cache # pylint: disable=C0301
            # NOTE(asolergi-nv): Uses logic from megatron/core/datasets/helpers.cpp::build_sample_idx to compute the number of samples # pylint: disable=C0301
            num_tokens_per_epoch = self._get_num_tokens_per_epoch()
            num_epochs = self._get_num_epochs(num_tokens_per_epoch)

            drop_last_partial_sequence = True
            if self.index_split == Split.valid:
                drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence

            if drop_last_partial_sequence:
                return (
                    num_epochs * num_tokens_per_epoch - self.config.add_extra_token_to_sequence
                ) // self.config.sequence_length
            else:
                return ceil(
                    float(
                        num_epochs * num_tokens_per_epoch - self.config.add_extra_token_to_sequence
                    )
                    / self.config.sequence_length
                )
        return self.sample_index.shape[0] - 1

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optional[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        if idx is None:
            # Batch padding sequence so the index does not matter
            text, _ = self._query_document_sample_shuffle_indices(0)
        else:
            text, _ = self._query_document_sample_shuffle_indices(idx)

        text = torch.from_numpy(text).long()
        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id

        if (
            not self.masks_and_position_ids_are_cacheable
            or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.create_attention_mask,
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask.clone()
            position_ids = self.cached_position_ids

        # For padded sequences, mask the loss
        loss_mask[labels == self._pad_token_id] = 0.0

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        # Hard-mask task-control tokens (e.g. <|stt_transcribe|>, <|tts_continue|>).
        # Applied early: downstream steps (goldfish, modality weights, weight decay)
        # all guard with `loss_mask > 0` so they cannot re-enable these.
        for tid in self._loss_mask_token_ids:
            loss_mask[labels == tid] = 0.0

        # Goldfish loss masking
        if self.config.goldfish_loss:

            # Init the hash table once only
            if self._goldfish_hash_table is None:
                self._goldfish_hash_table = _create_hash_table(device=labels.device)

            # Apply the goldfish mask
            goldfish_labels = apply_goldfish(
                labels,
                goldfish_token_id=self._goldfish_token_id,
                k=self._goldfish_k,
                goldfish_hash_table=self._goldfish_hash_table,
                goldfish_context_width=self._goldfish_h,
                exemption_start=self._goldfish_exemption_start,
                exemption_end=self._goldfish_exemption_end,
            )

            loss_mask[goldfish_labels == self._goldfish_token_id] = 0.0

        # Modality token loss masking.
        for _, start, end, weight in self._weighted_modality_specs:
            modality_mask = (labels >= start) & (labels < end)
            active_modality_mask = modality_mask & (loss_mask > 0)
            loss_mask[active_modality_mask] = weight

        # Batch padding sequence so we mask the loss
        if idx is None:
            loss_mask = torch.zeros_like(loss_mask)

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                "assistant_mask": torch.zeros_like(tokens)
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                "assistant_mask": torch.zeros_like(tokens)
            }

    def _query_document_sample_shuffle_indices(
        self, idx: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        if self.config.pretraining_packing_strategy == "bfd":
            return self._query_bfd_packed_sample(idx)
        
        if self.shuffle_index is None:
            # NOTE(asolergi-nv): Lazy memmap the indexes
            self.shuffle_index = numpy.load(
                self.path_to_shuffle_index, allow_pickle=True, mmap_mode='r'
            )
            self.sample_index = numpy.load(
                self.path_to_sample_index, allow_pickle=True, mmap_mode='r'
            )
            self.document_index = numpy.load(
                self.path_to_document_index, allow_pickle=True, mmap_mode='r'
            )

        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])

            # Add the entire sample
            sample_parts.append(
                self.dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset
                    - doc_index_beg_offset
                    + self.config.add_extra_token_to_sequence,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the document id
                document_ids.append(self.document_index[i])

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = (
                    None
                    if i < doc_index_end
                    else doc_index_end_offset + self.config.add_extra_token_to_sequence
                )
                sample_parts.append(
                    self.dataset.get(self.document_index[i], offset=offset, length=length)
                )
        assert len(document_ids) == len(
            sample_parts
        ), f"len(document_ids) ({len(document_ids)}) != len(sample_parts) ({len(sample_parts)})"

        length = sum(map(len, sample_parts))

        # Pad the sample if necessary
        if length < (self.config.sequence_length + self.config.add_extra_token_to_sequence):
            sample_parts.append(
                [self._pad_token_id]
                * (self.config.sequence_length + self.config.add_extra_token_to_sequence - length)
            )

        return (
            numpy.concatenate(sample_parts, dtype=numpy.int64),
            numpy.array(document_ids, dtype=numpy.int64),
        )

    def _build_bfd_packing_indices(self):
        """BFD pack virtual chunks; oversized docs are split so no tokens are truncated.

        document_index holds virtual chunk IDs; self.chunk_map maps each virtual chunk
        to [real_doc_id, chunk_start, chunk_length]. Samples are loaded via
        _query_bfd_packed_sample, which uses chunk_map to fetch exact slices.
        """
        path_to_cache = self.config.path_to_cache
        if path_to_cache is None and not self.config.mock:
            path_to_cache = os.path.join(
                self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
            )

        if path_to_cache:
            base = f"{self.unique_description_hash}-{type(self).__name__}-{self.index_split.name}"
            get_path_to = lambda affix: os.path.join(path_to_cache, f"{base}-{affix}")
            path_to_description = get_path_to("description.txt")
            path_to_document_index = get_path_to("document_index.npy")
            path_to_sample_index = get_path_to("sample_index.npy")
            path_to_shuffle_index = get_path_to("shuffle_index.npy")
            path_to_chunk_map = get_path_to("chunk_map.npy")
            cache_hit = all(
                map(os.path.isfile, [
                    path_to_description, path_to_document_index,
                    path_to_sample_index, path_to_shuffle_index, path_to_chunk_map,
                ])
            )
        else:
            cache_hit = False

        if not path_to_cache or (
            not cache_hit
            and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
        ):
            log_single_rank(logger, logging.INFO,
                f"Build and save the {type(self).__name__} {self.index_split.name} BFD indices")
            t_beg = time.time()

            sequence_length = self.config.sequence_length
            numpy_random_state = numpy.random.RandomState(self.config.random_seed)

            # One shuffled epoch of real doc IDs — BFD always packs one epoch; shuffle tiles after.
            real_doc_index = _build_document_index(self.indices, 1, numpy_random_state, False)
            assert real_doc_index.dtype == numpy.int32
            assert self.dataset.sequence_lengths.dtype == numpy.int32

            if len(real_doc_index) * 2 > len(self.dataset.sequence_lengths):
                seq_lens = self.dataset.sequence_lengths.copy()
            else:
                seq_lens = self.dataset.sequence_lengths

            # Virtual chunking: split oversized docs into capacity-sized pieces.
            capacity = sequence_length + self.config.add_extra_token_to_sequence
            chunk_map, virtual_sizes = _build_virtual_docs(
                real_doc_index, seq_lens, capacity, self.config.tokenizer.eod
            )
            num_virtual = len(virtual_sizes)
            virtual_idx = numpy.arange(num_virtual, dtype=numpy.int32)

            n_extra = num_virtual - len(real_doc_index)
            log_single_rank(logger, logging.INFO,
                f"  Document chunking: {len(real_doc_index)} docs → {num_virtual} virtual chunks"
                + (f" ({n_extra} extra from oversized docs)" if n_extra > 0 else ""))

            _accel = "C-accelerated" if _bfd_c_lib is not None else "Python fallback"
            log_single_rank(logger, logging.INFO,
                f"Using Best-Fit Decreasing packing strategy ({_accel})")

            document_index, sample_index = _build_sample_idx_bfd(
                virtual_sizes, virtual_idx, sequence_length,
                add_extra_token=self.config.add_extra_token_to_sequence,
                max_docs_per_bin=self.config.max_docs_per_bin
            )
            self._log_bfd_packing_statistics(
                num_docs=len(real_doc_index),
                num_virtual=num_virtual,
                sample_index=sample_index,
                from_cache=False,
            )

            # Shuffle packed samples; tile with fresh permutations if num_samples > one epoch.
            num_epoch_samples = sample_index.shape[0] - 1
            if self.num_samples is None or self.num_samples <= num_epoch_samples:
                n = self.num_samples if self.num_samples is not None else num_epoch_samples
                shuffle_index = _build_shuffle_index(num_epoch_samples, n, numpy_random_state)
            else:
                num_repeats = int(numpy.ceil(self.num_samples / num_epoch_samples))
                log_single_rank(logger, logging.WARNING,
                    f"> Requested {self.num_samples} samples but one epoch provides only "
                    f"{num_epoch_samples}. Tiling shuffle index {num_repeats}x.")
                tiles = [_build_shuffle_index(num_epoch_samples, num_epoch_samples, numpy_random_state)
                         for _ in range(num_repeats)]
                shuffle_index = numpy.concatenate(tiles)[:self.num_samples]

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)
                with open(path_to_description, "wt") as writer:
                    writer.write(self.unique_description)
                numpy.save(path_to_document_index, document_index, allow_pickle=True)
                numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
                numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
                numpy.save(path_to_chunk_map, chunk_map, allow_pickle=True)

            self.chunk_map = chunk_map
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {time.time() - t_beg:4f} s")
            log_single_rank(logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}")

            self._log_bfd_packing_statistics(
                num_docs=len(self.indices),
                num_virtual=len(self.chunk_map),
                sample_index=sample_index,
                from_cache=True,
            )
            
            return document_index, sample_index, shuffle_index

        # Cache hit
        log_single_rank(logger, logging.INFO,
            f"Load the {type(self).__name__} {self.index_split.name} BFD indices")
        document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode='r')
        sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode='r')
        shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode='r')
        self.chunk_map = numpy.load(path_to_chunk_map, allow_pickle=True, mmap_mode='r')
        log_single_rank(logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}")

        self._log_bfd_packing_statistics(
            num_docs=len(self.indices),
            num_virtual=len(self.chunk_map),
            sample_index=sample_index,
            from_cache=True,
        )
        
        return document_index, sample_index, shuffle_index

    def _log_bfd_packing_statistics(self, num_docs, num_virtual, sample_index, from_cache=False):
        num_samples = sample_index.shape[0] - 1
        sequence_length = self.config.sequence_length
        num_tokens_per_epoch = int(numpy.sum(self.dataset.sequence_lengths[self.indices]))
        total_tokens_in_samples = num_samples * sequence_length
        avg_docs_per_sample = num_virtual / num_samples if num_samples > 0 else 0
        packing_efficiency = (
            100 * num_tokens_per_epoch / total_tokens_in_samples
            if total_tokens_in_samples > 0 else 0
        )
        suffix = " (loaded from cache)" if from_cache else ""
        log_single_rank(logger, logging.INFO, f"> ===== BFD Packing Statistics (ONE EPOCH){suffix} =====")
        log_single_rank(logger, logging.INFO, f" > #real docs in epoch:               {num_docs:>12,}")
        log_single_rank(logger, logging.INFO, f" > #virtual chunks (after split):     {num_virtual:>12,}")
        log_single_rank(logger, logging.INFO, f" > #tokens in epoch:                  {num_tokens_per_epoch:>12,}")
        log_single_rank(logger, logging.INFO, f" > Sequence length:                   {sequence_length:>12,}")
        log_single_rank(logger, logging.INFO, f" > #packed samples (per epoch):       {num_samples:>12,}")
        log_single_rank(logger, logging.INFO, f" > #tokens(incl. padding) in samples: {total_tokens_in_samples:>12,}")
        log_single_rank(logger, logging.INFO, f" > Average #chunks/sample:            {avg_docs_per_sample:>12.2f}")
        log_single_rank(logger, logging.INFO, f" > Packing efficiency:                {packing_efficiency:>11.2f}%")

    def _query_bfd_packed_sample(self, idx):
        """Load one BFD-packed sample: concatenate virtual chunks, synthesize EOD on
        non-last oversized-doc chunks, pad to target length.
        """
        shuffled = int(self.shuffle_index[idx])
        doc_index_beg = int(self.sample_index[shuffled][0])
        doc_index_end = int(self.sample_index[shuffled + 1][0])

        target_length = self.config.sequence_length + self.config.add_extra_token_to_sequence
        eod_token_id = self.config.tokenizer.eod

        sample_parts = []
        document_ids = []
        for i in range(doc_index_beg, doc_index_end):
            virtual_id = int(self.document_index[i])
            real_doc_id = int(self.chunk_map[virtual_id, 0])
            chunk_start = int(self.chunk_map[virtual_id, 1])
            chunk_len   = int(self.chunk_map[virtual_id, 2])
            append_eod  = int(self.chunk_map[virtual_id, 3])

            data = self.dataset.get(real_doc_id, offset=chunk_start, length=chunk_len)
            if append_eod:
                data = numpy.concatenate([data, numpy.array([eod_token_id], dtype=data.dtype)])
            sample_parts.append(data)
            document_ids.append(real_doc_id)

        length = sum(map(len, sample_parts))
        if length < target_length:
            sample_parts.append(
                [self._pad_token_id] * (target_length - length)
            )

        return (
            numpy.concatenate(sample_parts, dtype=numpy.int64),
            numpy.array(document_ids, dtype=numpy.int64),
        )

    def _build_document_sample_shuffle_indices(
        self,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Build the document index, the sample index, and the shuffle index

        The document index:
            -- 1-D
            -- An ordered array of document ids

        The sample index:
            -- 2-D
            -- The document indices and offsets which mark the start of every sample

        The shuffle index:
            -- 1-D
            -- A random permutation of index range of the sample index

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The document index, the sample
            index, and the shuffle index
        """

        if self.config.pretraining_packing_strategy == "bfd":
            return self._build_bfd_packing_indices()
        if self.config.defer_npy_index_mmap:
            # NOTE(asolergi-nv): Direct path to lazy memmap the indexes
            base = f"{self.unique_description_hash}-{type(self).__name__}-{self.index_split.name}"
            get_path_to = lambda affix: os.path.join(self.config.path_to_cache, f"{base}-{affix}")
            self.path_to_document_index = get_path_to("document_index.npy")
            self.path_to_sample_index = get_path_to("sample_index.npy")
            self.path_to_shuffle_index = get_path_to("shuffle_index.npy")
            return None, None, None

        path_to_cache = self.config.path_to_cache
        if path_to_cache is None and not self.config.mock:
            path_to_cache = os.path.join(
                self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
            )

        if path_to_cache:
            base = f"{self.unique_description_hash}-{type(self).__name__}-{self.index_split.name}"
            get_path_to = lambda affix: os.path.join(path_to_cache, f"{base}-{affix}")
            path_to_description = get_path_to("description.txt")
            path_to_document_index = get_path_to("document_index.npy")
            path_to_sample_index = get_path_to("sample_index.npy")
            path_to_shuffle_index = get_path_to("shuffle_index.npy")
            cache_hit = (
                True
                if self.config.fast_cache_load
                else all(
                    map(
                        os.path.isfile,
                        [
                            path_to_description,
                            path_to_document_index,
                            path_to_sample_index,
                            path_to_shuffle_index,
                        ],
                    )
                )
            )
        else:
            cache_hit = False

        if not path_to_cache or (
            not cache_hit
            and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
        ):
            log_single_rank(
                logger,
                logging.INFO,
                f"Build and save the {type(self).__name__} {self.index_split.name} indices",
            )
            t_beg = time.time()

            sequence_length = self.config.sequence_length
            num_tokens_per_epoch = self._get_num_tokens_per_epoch()
            num_epochs = self._get_num_epochs(num_tokens_per_epoch)

            if num_epochs == 1:
                separate_final_epoch = False
            else:
                # Get the number of samples for the last epoch
                num_samples_sans_final_epoch = (
                    (num_epochs - 1) * num_tokens_per_epoch
                    - self.config.add_extra_token_to_sequence
                ) // sequence_length
                num_samples_from_final_epoch = self.num_samples - num_samples_sans_final_epoch
                num_samples_per_epoch = (
                    num_tokens_per_epoch - self.config.add_extra_token_to_sequence
                ) // sequence_length

                # num_samples_from_final_epoch should be non-negative
                assert num_samples_from_final_epoch >= 0

                # num_samples_from_final_epoch should not exceed max value
                assert num_samples_from_final_epoch <= num_samples_per_epoch + 1

                # Separate the final epoch if it falls below the threshold
                threshold = 0.80
                separate_final_epoch = num_samples_from_final_epoch < int(
                    threshold * num_samples_per_epoch
                )

                log_single_rank(
                    logger,
                    logging.DEBUG,
                    f"> num_samples_from_final_epoch: {num_samples_from_final_epoch}",
                )
                log_single_rank(logger, logging.DEBUG, f"> threshold: {threshold}")
                log_single_rank(
                    logger, logging.DEBUG, f"> num_samples_per_epoch: {num_samples_per_epoch}"
                )

            log_single_rank(
                logger, logging.DEBUG, f"> separate_final_epoch: {separate_final_epoch}"
            )

            numpy_random_state = numpy.random.RandomState(self.config.random_seed)

            # Build the document index
            document_index = _build_document_index(
                self.indices, num_epochs, numpy_random_state, separate_final_epoch
            )

            # Build the sample index
            from megatron.core.datasets import helpers

            if self.index_split == Split.valid:
                drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence
            else:
                drop_last_partial_sequence = True

            assert document_index.dtype == numpy.int32
            assert self.dataset.sequence_lengths.dtype == numpy.int32
            if len(document_index) * 2 > len(self.dataset.sequence_lengths):
                # If "access density" of sequence_lengths is high, force load the mmap-ed array
                # into memory by making a copy.
                #
                # System performance benefits come from two aspects:
                #   1. We sequentially pre-load the whole file, most of which we expect to read
                #   2. The GIL is held when entering the c++ program, improving the speed of which
                #      improves parallelism
                sequence_lengths_for_cpp = self.dataset.sequence_lengths.copy()
            else:
                sequence_lengths_for_cpp = self.dataset.sequence_lengths
            sample_index = helpers.build_sample_idx(
                sequence_lengths_for_cpp,
                document_index,
                sequence_length,
                num_epochs,
                num_tokens_per_epoch,
                drop_last_partial_sequence,
                self.config.add_extra_token_to_sequence,
            )

            # Build the shuffle index
            if separate_final_epoch:
                shuffle_index = _build_shuffle_index(
                    num_samples_sans_final_epoch, sample_index.shape[0] - 1, numpy_random_state
                )
            else:
                shuffle_index = _build_shuffle_index(
                    sample_index.shape[0] - 1, sample_index.shape[0] - 1, numpy_random_state
                )

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)
                # Write the description
                with open(path_to_description, "wt") as writer:
                    writer.write(self.unique_description)
                numpy.save(path_to_document_index, document_index, allow_pickle=True)
                numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
                numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unable to save {type(self).__name__} indexes because path_to_cache is None",
                )

            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            log_single_rank(
                logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
            )
            log_single_rank(logger, logging.INFO, f"> total number of epochs: {num_epochs}")

            return document_index, sample_index, shuffle_index

        log_single_rank(
            logger, logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} indices"
        )

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the document index from {os.path.basename(path_to_document_index)}",
        )
        t_beg = time.time()
        document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}",
        )
        t_beg = time.time()
        sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}",
        )
        t_beg = time.time()
        shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
        )

        return document_index, sample_index, shuffle_index

    def _get_num_tokens_per_epoch(self) -> int:
        """Calculate the number of tokens in a single epoch

        Returns:
            int: The number of tokens in a single epoch
        """
        return int(numpy.sum(self.dataset.sequence_lengths[self.indices]))

    def _get_num_epochs(self, num_tokens_per_epoch: int) -> int:
        """Calculate the number of epochs

        Args:
            num_tokens_per_epoch (int): The number of tokens in a single epoch

        Returns:
            int: The number of epochs
        """
        num_epochs = 1
        num_tokens = num_tokens_per_epoch
        if self.num_samples is None:
            return num_epochs
        else:
            num_tokens_requested = (
                self.num_samples * self.config.sequence_length
            ) + self.config.add_extra_token_to_sequence
            while num_tokens < num_tokens_requested:
                num_epochs += 1
                num_tokens += num_tokens_per_epoch
        return num_epochs


def _build_document_index(
    documents: numpy.ndarray,
    num_epochs: int,
    numpy_random_state: numpy.random.RandomState,
    separate_final_epoch: bool,
) -> numpy.ndarray:
    """Build an array with length = num epochs * num documents

    Args:
        documents (numpy.ndarray): the subset of exposed document indices

        num_epochs (int): The number of epochs

        numpy_random_state (numpy.random.RandomState): The NumPy random state

        separate_final_epoch (bool): Whether to exclude the last epoch from the global shuffle

    Returns:
        numpy.ndarray: The document index
    """

    if not separate_final_epoch or num_epochs == 1:
        document_index = numpy.mgrid[0:num_epochs, 0 : len(documents)][1]
        document_index[:] = documents
        document_index = document_index.reshape(-1)
        document_index = document_index.astype(numpy.int32)
        numpy_random_state.shuffle(document_index)
        return document_index

    doc_idx_first = _build_document_index(documents, num_epochs - 1, numpy_random_state, False)
    doc_idx_last = _build_document_index(documents, 1, numpy_random_state, False)
    return numpy.concatenate((doc_idx_first, doc_idx_last))


def _build_shuffle_index(
    num_samples: int, total_size: int, numpy_random_state: numpy.random.RandomState
) -> numpy.ndarray:
    """Build the range [0, size) and shuffle

    Args:
        num_samples (int): The size of the first shuffle range [0, num_samples)

        total_size (int): The size of the entire index. If larger than 'num_samples', it defines
            the second shuffle range [num_samples, total_size)

        numpy_random_state (numpy.random.RandomState): The NumPy random state

    Returns:
        numpy.ndarray: The shuffle index
    """

    dtype_ = numpy.uint32
    if total_size >= (numpy.iinfo(numpy.uint32).max - 1):
        dtype_ = numpy.int64

    shuffle_idx_first = numpy.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = numpy.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_last)

    return numpy.concatenate((shuffle_idx_first, shuffle_idx_last))


def _get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
    create_attention_mask: bool,
):
    """Build masks and position id for left to right model.

    Args:
        data (torch.Tensor): The data tenor that holds the tokens from the dataset

        eod_token (int): ID of the token to that is considered the EOD

        reset_position_ids (bool): Switch to reset the document position ID's

        reset_attention_mask (bool): Switch to reset the attention mask

        eod_mask_loss (bool): Switch to enable the EOD mask loss

        create_attention_mask (bool): Switch to enable the attention masks generation. Can be
            disabled if attention kernel generates masks by itself.

    Returns:
        torch.Tensor: Attention mask needed to be used for Attention

        torch.Tensor: The mask used for loss value during training

        torch.Tensor: The position ID's of the token
    """
    seq_length = data.numel()

    if create_attention_mask:
        attention_mask = torch.tril(
            torch.ones((seq_length, seq_length), device=data.device)
        ).unsqueeze(0)
    else:
        attention_mask = None

    # Loss mask.
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Find indices where EOD token is.
        eod_index = position_ids[data == eod_token]
        # Detach indices from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()

        # Loop through EOD indices:
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            # Mask attention loss.
            if reset_attention_mask and attention_mask is not None:
                attention_mask[0, (i + 1) :, : (i + 1)] = 0
            # Reset positions.
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1

    if attention_mask is not None:
        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def _create_hash_table(device):
    """Goldfish Loss Pre-computed Hash Table"""
    rng = torch.Generator(device=device)
    rng.manual_seed(2971215073)
    hash_table = torch.rand(_HASH_TABLE_SIZE, device=device, generator=rng)

    return hash_table


def apply_goldfish(
    labels: torch.Tensor,
    goldfish_token_id: int,
    k: int,
    goldfish_hash_table: torch.Tensor,
    goldfish_context_width: int=4,
    exemption_start: Optional[int] = None,
    exemption_end: Optional[int] = None,
):
    """
    Apply a mask to a tensor to skip every k-th token, using only the 'hash-table' strategy from the original Goldfish Loss implementation.
    This is utilized within GPTDataset.__getitem__.

    Original implementation: https://github.com/ahans30/goldfish-loss


    Args:
        labels (torch.Tensor):          The label tensor to apply the goldfish mask to.
        goldfish_token_id (int):        The token ID to use for the goldfish mask.
        k (int):                        The frequency with which tokens are ignored.
        goldfish_context_width (int):   Context width for hashing.
    """
    assert labels.ndim == 1, "Expected 1D tensor as used within GPTDataset.__getitem__"
    masked_labels = labels.clone()

    hashed_keys = goldfish_hash_table[
        labels.unfold(0, goldfish_context_width, 1).prod(dim=-1) % _HASH_TABLE_SIZE
    ]

    dropped_token_indices = (hashed_keys < 1 / k)

    if exemption_start is not None:
        # Exemption is in token-id space; cancel drops where labels are in that range.
        exempt_mask = (labels >= exemption_start) & (labels < exemption_end)
        exempt_tail = exempt_mask[goldfish_context_width - 1 :]
        if os.getenv("GOLDFISH_EXEMPT_LOG") == "1":
            exempt_dropped = (dropped_token_indices & exempt_tail).nonzero(as_tuple=True)[0]
            if exempt_dropped.numel() > 0:
                label_idx = exempt_dropped + (goldfish_context_width - 1)
                sample = torch.stack(
                    (label_idx[:5], labels[label_idx[:5]]), dim=1
                ).cpu().tolist()
                log_single_rank(
                    logger,
                    logging.INFO,
                    f"Goldfish exemption cancels {exempt_dropped.numel()} drops; "
                    f"sample (idx, token_id){sample}",
                )
        dropped_token_indices &= ~exempt_tail

    masked_labels[goldfish_context_width-1:][dropped_token_indices] = goldfish_token_id

    return masked_labels


class MockGPTLowLevelDataset:
    """The mock GPT low level dataset

    This class is meant to generate tokenized data in the classic "Megatron-LM" GPT style. Notably,
    we add the end of document token to each element indexed in __getitem__

    Args:
        tokenizer (MegatronTokenizerBase): The tokenizer the special token information of which
        we use to augment the mock data.
    """

    seed: int = 0
    """The hard-coded random seed to use to set the NumPy RNG"""

    size: int = 100000
    """The hard-coded number of samples to generate"""

    max_sequence_length: int = 4096
    """The hard-coded max sequence length to generate"""

    def __init__(self, tokenizer: MegatronTokenizerBase) -> None:
        self.tokenizer = tokenizer
        rng = numpy.random.default_rng(seed=self.seed)
        self.sequence_lengths = rng.integers(
            low=1, high=self.max_sequence_length, size=self.size, dtype=numpy.int32
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> numpy.number:
        length = self.sequence_lengths[idx]
        sample = numpy.int64(
            numpy.concatenate([numpy.arange(length - 1) + 1, [self.tokenizer.eod]])
        )
        return sample

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        """This function is an abstraction over __getitem__ with support for slicing

        Args:
            idx (int): The index into the dataset

            offset (int): The integer token offset in the sequence

            length (Optional[int]): The number of tokens to grab from the sequence

        Returns:
            numpy.ndarray: The sequence tokens at the index
        """
        if length is None:
            length = self.sequence_lengths[idx] - offset
        return self[idx][offset : offset + length]


class MockGPTDataset(GPTDataset):
    """The mock GPT dataset

    Args:
        dataset (MockGPTLowLevelDataset): The MockGPTLowLevelDataset around which to build
            the MockGPTDataset

        dataset_path (Optional[str]): This argument is of no consequence for the MockGPTDataset

        indices (numpy.ndarray): The set of the dataset indices to expose

        num_samples (int): The number of samples to draw from the dataset

        index_split (Split): The indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        dataset: MockGPTLowLevelDataset,
        dataset_path: Optional[str],
        indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        assert config.mock

        super().__init__(
            dataset,  # type: ignore[arg-type]
            dataset_path,
            indices,
            num_samples,
            index_split,
            config,
        )

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: MockGPTLowLevelDataset) -> int:
        """Abstract method implementation

        Args:
            low_level_dataset (MockGPTLowLevelDataset): The underlying MockGPTLowLevelDataset

        Returns:
            int: The number of unique elements in the underlying MockGPTLowLevelDataset
        """
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(  # type: ignore[override]
        dataset_path: Optional[str], config: GPTDatasetConfig
    ) -> MockGPTLowLevelDataset:
        """Abstract method implementation

        Args:
            dataset_path (Optional[str]): This argument is of no consequence for the
                MockGPTLowLevelDataset

            config (GPTDatasetConfig): The config

        Returns:
            MockGPTLowLevelDataset: The underlying MockGPTLowLevelDataset
        """
        return MockGPTLowLevelDataset(config.tokenizer)