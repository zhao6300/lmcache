# SPDX-License-Identifier: Apache-2.0
"""Foundational types for the KV-cache format layer.

``DiscoverableKVCache`` is the canonical KV-cache structure every ``kv_format``
module operates on; ``LayoutHints`` carries the engine-supplied registration
hints. The ``utils.py`` facade re-exports both for backward-compatible call
sites.
"""

# Standard
from typing import Literal, TypedDict, Union, get_args

# Third Party
import torch

# Canonical recursive type consumed by ``detect_format`` and the
# downstream format-aware helpers. A value is either a single
# :class:`torch.Tensor` (e.g. vLLM cross-layer, TRT-LLM) or a list of
# nested ``DiscoverableKVCache`` values (per-layer lists or deeper nesting).
# Engine adapters that hand us other
# containers (e.g. vLLM's ``dict[str, torch.Tensor]``) are responsible
# for unwrapping to this form before calling the helpers.
DiscoverableKVCache = Union[torch.Tensor, list["DiscoverableKVCache"]]

KVLayoutName = Literal["NHD", "HND", "BLHNC", "BLNHC"]
KV_LAYOUT_NAMES: tuple[str, ...] = get_args(KVLayoutName)


class LayoutHints(TypedDict, total=False):
    """Hints passed from a serving engine to LMCache during KV cache
    registration (``REGISTER_KV_CACHE``).

    Serving engines may pass a plain ``dict`` that satisfies this
    schema -- importing this type is optional.

    Keys:
        kv_layout: Physical ordering of the KV cache dimensions.
            ``"NHD"`` -- heads after block-size (default for most
            vLLM builds).
            ``"HND"`` -- heads before block-size (``VLLM_KV_CACHE_LAYOUT=HND``).
            ``"BLHNC"`` / ``"BLNHC"`` -- blocks outermost, all layers
            packed per block (vLLM standardized layouts).
        num_kv_heads: Number of KV heads per layer. Used by TRT-LLM to
            reshape its 4-D pool tensor into the canonical 6-D form.
        tokens_per_block: Tokens per paged block. Used by TRT-LLM to reshape
            its pool tensor.
        head_dim: Per-head dimension. Used by TRT-LLM (same).
    """

    kv_layout: KVLayoutName
    num_kv_heads: int
    tokens_per_block: int
    head_dim: int
    blocked_scale: bool
    """The engine wrote blocked-scale pages whose row width is ambiguous.

    A 528-byte DeepSeek-V4.1 record is per-token by default and page-footer
    (``[BS x values][BS x scales]``) only under the SM120 ``cake`` backend.
    Set when the engine selected the footer form.
    """

    num_head_slots: int
    """Head slots the engine packs into each stored state (vLLM
    ``AttentionSpec.num_head_slots``); absent means one slot per KV head."""

    kv_head_major_slots: bool
    """Whether the packed head slots are grouped by KV head (``H2ND``): head
    ``h``'s K and V slots are adjacent. The slots still form one contiguous
    run per token, so the whole token row is transferred as opaque bytes."""


# uint8 rows vLLM pages as [BS x values][BS x scales] (NL_X_NB_BSV_BSS):
# fp8 / MXFP4 indexer K, the DeepSeek-V4 SM90 main KV (8 B scales) and the
# DeepSeek-V4.1 SM100/120 page-footer record (16 B of UE8M0 scales). The value
# is the per-token scale width in bytes; the native kernels mirror this table
# in ``csrc/engine_kv_format.h``.
BLOCKED_SCALE_BYTES: dict[int, int] = {132: 4, 68: 4, 584: 8, 528: 16}
BLOCKED_SCALE_ROWS: frozenset[int] = frozenset(BLOCKED_SCALE_BYTES)


def transfer_word_bytes(row_bytes: int) -> int:
    """The transfer word the native kernels pick for a ``row_bytes`` row."""
    for word in (8, 4, 2):
        if row_bytes % word == 0:
            return word
    return 1


def scale_bytes_for_row(row_bytes: int) -> int | None:
    """Return the per-token scale width of a blocked-scale row, or ``None``."""
    return BLOCKED_SCALE_BYTES.get(int(row_bytes))


def is_blocked_scale_row(row_bytes: int) -> bool:
    """Whether ``row_bytes``-wide uint8 rows are a blocked-scale record."""
    return int(row_bytes) in BLOCKED_SCALE_ROWS
