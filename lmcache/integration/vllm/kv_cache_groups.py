# SPDX-License-Identifier: Apache-2.0
"""Build LMCache engine group infos from vLLM KV cache group metadata."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.common.utils import LayoutHints

# First Party
from lmcache.logging import init_logger
from lmcache.v1.common.group_view import EngineGroupInfo

logger = init_logger(__name__)


def _leaf_spec(spec: Any) -> Any:
    """Unwrap ``UniformTypeKVCacheSpecs`` to one leaf spec.

    Its layers share one spec type, so one leaf suffices. Any other spec is
    returned unchanged.
    """
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict) and inner:
        spec = next(iter(inner.values()))
    return spec


def _is_attention_spec(spec: Any) -> bool:
    """Return whether the KV cache spec is a vLLM attention spec.

    Checked by class name so this module stays importable without vLLM.
    ``UniformTypeKVCacheSpecs`` is unwrapped first (same-typed layers, one
    leaf suffices); it does not derive from ``AttentionSpec`` itself.
    """
    spec = _leaf_spec(spec)
    return any(cls.__name__ == "AttentionSpec" for cls in type(spec).__mro__)


def _is_mla_spec(spec: Any) -> bool:
    """Return whether the KV cache spec is a vLLM MLA spec (one latent head,
    held in full on every TP rank). Checked by class name like
    :func:`_is_attention_spec`."""
    return any(cls.__name__ == "MLAAttentionSpec" for cls in type(spec).__mro__)


def has_prefix_replay(spec: Any) -> bool:
    """Return whether vLLM recomputes the spec's window by prefix replay.

    Under bounded replay (e.g. DeepSeek V4.1 ``SlidingWindowMLASpec``) vLLM
    does not prefix-cache the sliding-window blocks; after a hit it re-runs
    the last ``prefix_replay_tokens`` tokens to rebuild the window, unless a
    connector declares the window loaded. ``UniformTypeKVCacheSpecs`` is
    unwrapped first.

    Args:
        spec: A vLLM KV cache spec, or a ``UniformTypeKVCacheSpecs`` container.

    Returns:
        ``True`` if the spec reports ``prefix_replay_tokens > 0``.
    """
    return getattr(_leaf_spec(spec), "prefix_replay_tokens", 0) > 0


def is_scratch_spec(spec: Any, *, store_replay_windows: bool = False) -> bool:
    """Return whether vLLM marks the spec as non-prefix-cacheable.

    vLLM never hashes or restores the blocks of such a scratch group, so they
    carry no token range LMCache could store (e.g. the per-request rings
    ``CircularBufferSpec`` and ``KpoolTailSpec``). ``prefix_cacheable`` takes
    precedence over the older ``participates_in_prefix_caching`` marker.
    Specs with neither marker are prefix-cacheable.
    ``UniformTypeKVCacheSpecs`` is unwrapped first.

    Args:
        spec: A vLLM KV cache spec, or a ``UniformTypeKVCacheSpecs`` container.
        store_replay_windows: Treat a prefix-replay spec (see
            :func:`has_prefix_replay`) as storable rather than scratch, so a
            connector can restore its window and spare vLLM the replay.

    Returns:
        ``True`` for a scratch spec, ``False`` for a prefix-cacheable one, or
        for a prefix-replay one when ``store_replay_windows`` is set.
    """
    if store_replay_windows and has_prefix_replay(spec):
        return False
    spec = _leaf_spec(spec)
    if hasattr(spec, "prefix_cacheable"):
        return not spec.prefix_cacheable
    return not getattr(spec, "participates_in_prefix_caching", True)


def is_per_request_state_spec(spec: Any) -> bool:
    """Return whether the spec is a per-request ring LMCache can transfer.

    vLLM's ``CircularBufferSpec`` (``KpoolTailSpec`` on older vLLM) gives each
    request one block for its whole lifetime, written at ``position % capacity``
    and holding the raw keys of the compression group that is still open. It
    is a scratch spec (see :func:`is_scratch_spec`), but unlike the other
    scratch specs its content is needed to resume a request mid-group, e.g.
    from an unfull chunk or on a prefill-to-decode handoff. Checked by class
    name so this module stays
    importable without vLLM; ``UniformTypeKVCacheSpecs`` is unwrapped first.

    Args:
        spec: A vLLM KV cache spec, or a ``UniformTypeKVCacheSpecs`` container.

    Returns:
        ``True`` for a per-request ring spec.
    """
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict) and inner:
        spec = next(iter(inner.values()))
    return is_scratch_spec(spec) and any(
        cls.__name__ in ("CircularBufferSpec", "KpoolTailSpec")
        for cls in type(spec).__mro__
    )


def is_transferable_group(group: Any) -> bool:
    """Return whether vLLM publishes this cache group to external transfer.

    vLLM after RFC #42449 marks ephemeral accelerator-side replicas with
    ``KVCacheGroup.enable_kv_transfer = False``: they are rebuilt from their
    transferable source group after a connector load, so storing them would
    duplicate state and loading them would target the wrong pool. Older vLLM
    has no such marker and every group is transferable.

    Args:
        group: A vLLM ``KVCacheGroup`` (either interface generation).

    Returns:
        ``True`` when the group may be transferred.
    """
    return bool(getattr(group, "enable_kv_transfer", True))


def get_group_kv_cache_spec(group: Any) -> Any:
    """Return one spec representing a vLLM cache group's layers.

    vLLM after RFC #42449 part 2 stores per-layer specs in
    ``KVCacheGroup.kv_cache_specs``; older versions expose a single
    ``kv_cache_spec`` for the group (possibly a ``UniformTypeKVCacheSpecs``
    container). Group-level facts such as scratch state and tokens per block
    are shared by every layer of a group, so the first leaf spec represents
    the group.

    Args:
        group: A vLLM ``KVCacheGroup`` (either interface generation).

    Returns:
        The group's representative spec, or ``None`` for an empty group.
    """
    per_layer = getattr(group, "kv_cache_specs", None)
    if isinstance(per_layer, dict) and per_layer:
        return next(iter(per_layer.values()))
    return getattr(group, "kv_cache_spec", None)


def get_layer_kv_cache_spec(group: Any, layer_name: str) -> Any:
    """Return ``layer_name``'s KV cache spec from a vLLM cache group.

    Handles both the per-layer ``KVCacheGroup.kv_cache_specs`` mapping and the
    older single ``kv_cache_spec`` (which may be a
    ``UniformTypeKVCacheSpecs`` container keyed by layer name).

    Args:
        group: A vLLM ``KVCacheGroup`` (either interface generation).
        layer_name: Registered layer name.

    Returns:
        The layer's spec.
    """
    per_layer = getattr(group, "kv_cache_specs", None)
    if isinstance(per_layer, dict) and per_layer:
        return per_layer[layer_name]
    spec = getattr(group, "kv_cache_spec", None)
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict) and inner:
        return inner[layer_name]
    return spec


def get_tokens_per_block(
    kv_cache_spec: Any, dcp_size: int, *, store_replay_windows: bool = False
) -> int:
    """Global tokens covered by one block id of ``kv_cache_spec``.

    Attention blocks span ``block_size * dcp_size`` tokens under DCP
    (vLLM's ``resolve_kv_cache_block_sizes`` rule); recurrent state is
    replicated, not sharded, and stays at ``block_size``. Scratch specs
    (see :func:`is_scratch_spec`, which also defines
    ``store_replay_windows``) cover no tokens and yield ``0``.
    """
    if is_scratch_spec(kv_cache_spec, store_replay_windows=store_replay_windows):
        return 0
    block_size = kv_cache_spec.block_size
    if dcp_size <= 1:
        return block_size
    if _is_attention_spec(kv_cache_spec):
        return block_size * dcp_size
    return block_size


def is_align_mode_mamba_spec(spec: Any) -> bool:
    """Return whether the spec is a Mamba/linear-attention spec in align mode.

    Align mode keeps state snapshots only at chosen boundaries and may move
    them between blocks, so a block-table position does not identify one;
    vLLM hands each snapshot's block to connectors instead
    (``SchedulerOutput.kv_connector_block_state.boundary_state_offloads``).
    Checked by class name so this module stays importable without vLLM.

    Args:
        spec: A vLLM KV cache spec.

    Returns:
        True for a ``MambaSpec`` with ``mamba_cache_mode == "align"``.
    """
    return (
        any(cls.__name__ == "MambaSpec" for cls in type(spec).__mro__)
        and getattr(spec, "mamba_cache_mode", "none") == "align"
    )


def _is_sliding_window_spec(spec: Any) -> bool:
    """Return whether the KV cache spec is a vLLM sliding-window spec.

    Checked by class name so this module stays importable without vLLM.
    Subclasses such as ``SlidingWindowMLASpec`` count.
    """
    return any(cls.__name__ == "SlidingWindowSpec" for cls in type(spec).__mro__)


def _is_cachable_mamba_spec(spec: Any) -> bool:
    """Return whether the spec is a snapshotting Mamba/linear-attention spec.

    Align mode snapshots only the last block; all mode snapshots every block
    boundary. Either way a restore consumes only the last matched block's
    page, so both behave like a cross-chunk sliding window of one block.
    Checked by class name (like :func:`_is_sliding_window_spec`) so this
    module stays importable without vLLM.
    """
    return any(cls.__name__ == "MambaSpec" for cls in type(spec).__mro__) and getattr(
        spec, "mamba_cache_mode", "none"
    ) in ("align", "all")


def _resolve_per_layer_sw_sizes(
    vllm_groups: Sequence[Any],
    layer_to_idx: Mapping[str, int],
    num_layers: int,
) -> list[int]:
    """Resolve the sliding window size in tokens for each registered KV tensor.

    Will resolve -1 for non-sliding-window layers.

    Args:
        vllm_groups: vLLM ``KVCacheGroupSpec`` instances.
        layer_to_idx: Layer name to registered tensor index mapping.
        num_layers: Number of registered KV tensors.

    Returns:
        A list of length ``num_layers`` mapping each registered tensor index
        to its cross-chunk window size in tokens: the sliding window for
        sliding-window attention, the engine-side block size for align-mode
        Mamba/linear layers (a one-block window), or ``-1`` for full-attention
        layers.
    """
    per_layer_sw_size = [-1] * num_layers
    for group in vllm_groups:
        for name in group.layer_names:
            layer_spec = get_layer_kv_cache_spec(group, name)
            if layer_spec is None:
                continue
            if _is_sliding_window_spec(layer_spec):
                per_layer_sw_size[layer_to_idx[name]] = layer_spec.sliding_window
            elif _is_cachable_mamba_spec(layer_spec):
                per_layer_sw_size[layer_to_idx[name]] = layer_spec.block_size
    return per_layer_sw_size


#: Reserved layer-name prefix for CacheBlend fused-aux page pools:
#: ``cb.aux_pool.<tokens_per_block>[.<label>]`` — the first suffix part is
#: the logical block size; the optional label distinguishes multiple pools
#: sharing one block size (they then share an engine group).
CB_AUX_POOL_LAYER_PREFIX = "cb.aux_pool."


def cb_aux_pool_entries(kv_caches) -> "list[tuple[str, int]]":
    """CacheBlend fused-aux pool entries among the registered tensors.

    Presence-gated: any layer name starting with
    :data:`CB_AUX_POOL_LAYER_PREFIX` is a connector-owned aux page pool
    served by a synthetic engine group (one group per block size).

    Args:
        kv_caches: Registered tensors keyed by layer name.

    Returns:
        ``(layer_name, tokens_per_block)`` per aux pool, in registration
        order; empty for models without one.

    Raises:
        ValueError: If a marker name's block-size suffix does not parse.
    """
    entries: list[tuple[str, int]] = []
    for name in kv_caches:
        if not name.startswith(CB_AUX_POOL_LAYER_PREFIX):
            continue
        suffix = name[len(CB_AUX_POOL_LAYER_PREFIX) :].split(".", 1)[0]
        try:
            tokens_per_block = int(suffix)
        except ValueError as exc:
            raise ValueError(
                f"aux pool layer name {name!r}: block-size suffix "
                f"{suffix!r} is not an integer"
            ) from exc
        if tokens_per_block <= 0:
            raise ValueError(
                f"aux pool layer name {name!r}: tokens_per_block must be "
                f"positive, got {tokens_per_block}"
            )
        entries.append((name, tokens_per_block))
    return entries


def _resolve_per_layer_recurrent(
    vllm_groups: Sequence[Any],
    layer_to_idx: Mapping[str, int],
    num_layers: int,
) -> list[bool]:
    """Resolve whether each registered KV tensor holds recurrent state pages.

    Args:
        vllm_groups: vLLM ``KVCacheGroupSpec`` instances.
        layer_to_idx: Layer name to registered tensor index mapping.
        num_layers: Number of registered KV tensors.

    Returns:
        A list of length ``num_layers``: ``True`` for Mamba/linear-attention
        layers in a snapshotting cache mode (see :func:`_is_cachable_mamba_spec`),
        ``False`` for attention layers.
    """
    per_layer_recurrent = [False] * num_layers
    for group in vllm_groups:
        for name in group.layer_names:
            layer_spec = get_layer_kv_cache_spec(group, name)
            if layer_spec is None:
                continue
            if _is_cachable_mamba_spec(layer_spec):
                per_layer_recurrent[layer_to_idx[name]] = True
    return per_layer_recurrent


def _resolve_per_layer_tp_replicated(
    vllm_groups: Sequence[Any],
    layer_to_idx: Mapping[str, int],
    num_layers: int,
) -> list[bool]:
    """Resolve whether vLLM leaves each registered KV tensor unsharded across
    tensor-parallel ranks: its spec declares ``max_tp_shards == 1``, or is an
    MLA spec that leaves it unset. Every rank of a TP group then holds the
    same KV for the layer, or the same token shard of it under DCP.

    Args:
        vllm_groups: vLLM ``KVCacheGroupSpec`` instances.
        layer_to_idx: Layer name to registered tensor index mapping.
        num_layers: Number of registered KV tensors.

    Returns:
        A list of length ``num_layers``, ``True`` for TP-replicated layers.
    """
    per_layer_tp_replicated = [False] * num_layers
    for group in vllm_groups:
        for name in group.layer_names:
            layer_spec = get_layer_kv_cache_spec(group, name)
            if layer_spec is None:
                continue
            max_tp_shards = getattr(layer_spec, "max_tp_shards", None)
            if max_tp_shards == 1 or (
                max_tp_shards is None and _is_mla_spec(layer_spec)
            ):
                per_layer_tp_replicated[layer_to_idx[name]] = True
    return per_layer_tp_replicated


def _merge_layer_recurrent(per_layer_recurrent: list[bool], indices: list[int]) -> bool:
    """Merge the per-layer recurrent-state flags of one LMCache group.

    Args:
        per_layer_recurrent: Recurrent-state flag per registered tensor index.
        indices: Registered tensor indices of the group's layers.

    Returns:
        The group's common flag.

    Raises:
        ValueError: If the group mixes recurrent and attention layers (vLLM
            groups layers by KV cache spec, so a mix indicates inconsistent
            metadata).
    """
    flags = {per_layer_recurrent[idx] for idx in indices}
    if len(flags) != 1:
        raise ValueError(
            f"Layers with indices {indices} mix recurrent-state and attention "
            "layers in one group. This should not happen because vLLM only "
            "groups layers with the same KV cache spec."
        )
    return flags.pop()


def _merge_layer_sw_sizes(per_layer_sw_size: list[int], indices: list[int]) -> int:
    """Merge the per-layer sliding window sizes of one LMCache group.

    Args:
        per_layer_sw_size: Sliding window size per registered tensor index.
        indices: Registered tensor indices of the group's layers.

    Returns:
        The group's common sliding window size in tokens, or ``-1`` when the
        layers are not sliding-window attention.

    Raises:
        ValueError: If the layers have different non-negative sliding window sizes.
    """
    sw_sizes = {per_layer_sw_size[idx] for idx in indices}
    if len(sw_sizes) != 1:
        raise ValueError(
            f"Layers with indices {indices} have different sliding window sizes "
            f"{sw_sizes}, but they are in the same group. This should "
            "not happen because vLLM should only group layers with the same "
            "KV cache spec, but got inconsistent metadata or registered tensors."
        )
    return sw_sizes.pop()


def create_engine_group_infos_from_vllm(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
    layout_hints: "LayoutHints | None" = None,
    dcp_size: int = 1,
    kv_tp_size: int = 1,
    *,
    store_replay_windows: bool = False,
    include_per_request_state: bool = False,
) -> list[EngineGroupInfo]:
    """Build the LMCache engine group infos from vLLM metadata and registered tensors.

    This is the single entry point for the vLLM -> LMCache conversion. It reads
    the vLLM-specific fields (``KVCacheConfig.kv_cache_groups`` and
    ``KVCacheGroupSpec.layer_names`` from the v1 KV cache interface), maps each
    engine KV cache group's layer names to registered tensor indices, then
    splits the layers by physical transfer identity using the real tensors (via
    the shared :func:`lmcache.v1.kv_layer_groups.group_layers_by_identity`).
    vLLM-specific field access is intentionally confined to this function.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig`` describing the engine KV cache
            groups (or ``None`` / no groups, which yields a single-group spec).
        kv_caches: Registered KV tensors keyed by layer name, in registration
            order. Keys provide the layer-name -> tensor-index mapping; values
            are inspected for physical shape and dtype.
        layout_hints: Optional engine-provided layout hints forwarded to format
            detection (e.g. ``NHD``/``HND`` and compression metadata).
        dcp_size: Decode context parallel size.
        kv_tp_size: Tensor-parallel ranks attached to one LMCache server. A
            group vLLM replicates on every TP rank (``max_tp_shards == 1``, or
            an MLA spec that leaves it unset) is tagged
            ``EngineGroupInfo.tp_replicas = kv_tp_size`` and
            ``dcp_shards = dcp_size``; every other group stays ``(1, 1)``.
        store_replay_windows: Register prefix-replay groups instead of
            excluding them as scratch, flagged ``prefix_replay`` (see
            :func:`is_scratch_spec`).
        include_per_request_state: Register per-request ring groups (see
            :func:`is_per_request_state_spec`) as ``per_request_state`` infos
            instead of excluding them with the other scratch groups.

    Note:
        Under DCP each attention group's ``tokens_per_block`` is scaled by
        ``dcp_size`` to stay in the scheduler's coordinate space; its ratio
        to the physical slot count is what sizes each rank's memory object.
        Mamba groups are replicated per rank and stay unscaled. Layers of
        scratch groups (see :func:`is_scratch_spec`) are excluded and never
        form an info, except per-request rings when
        ``include_per_request_state`` is set.

    Returns:
        The list of ``EngineGroupInfo`` in protocol order, i.e. the LMCache group
        order used by store/retrieve block IDs.
    """
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.common.group_view import EXCLUDED_ENGINE_GROUP
    from lmcache.v1.common.utils import (
        normalize_and_discover_per_layer_formats,
    )
    from lmcache.v1.kv_layer_groups import group_layers_by_identity

    # vLLM-specific field access (confined to this function): map each
    # registered KV tensor to its vLLM engine KV cache group index. vLLM places
    # every registered layer in exactly one group; layers in different groups
    # have disjoint block-id spaces and must not share an LMCache group. ``None``
    # means a single (non-hybrid) group, i.e. every layer shares one block-id
    # space.
    per_layer_discoverable_kv_caches = list(kv_caches.values())
    layer_to_idx = {name: idx for idx, name in enumerate(kv_caches.keys())}
    vllm_groups = (
        getattr(kv_cache_config, "kv_cache_groups", ()) or ()
        if kv_cache_config is not None
        else ()
    )

    def is_registered_per_request_state(spec: Any) -> bool:
        return include_per_request_state and is_per_request_state_spec(spec)

    # Scratch groups skip format discovery: their layers are excluded below and
    # never transferred, so a layout the transfer kernels reject must not fail
    # registration.
    layer_index_groups = [
        [layer_to_idx[name] for name in group.layer_names]
        for group in vllm_groups
        if is_transferable_group(group)
        and (
            not is_scratch_spec(
                get_group_kv_cache_spec(group),
                store_replay_windows=store_replay_windows,
            )
            or is_registered_per_request_state(get_group_kv_cache_spec(group))
        )
    ]

    # CacheBlend fused-aux (presence-gated): the pool joins detection as
    # its own group so its rank-3 layout is classified independently.
    aux_entries = cb_aux_pool_entries(kv_caches)
    layer_index_groups += [[layer_to_idx[name]] for name, _ in aux_entries]
    normalized_kv_caches, engine_kv_formats = normalize_and_discover_per_layer_formats(
        per_layer_discoverable_kv_caches,
        layer_index_groups,
        EngineType.VLLM,
        layout_hints,
    )
    num_layers = len(engine_kv_formats)
    # Layers absent from every engine group's ``layer_names`` are cross-layer
    # KV-sharing layers (e.g. google/gemma-4-E4B-it): vLLM aliases them to a
    # target owner's KV tensor, so the owner's group already covers them. Tag
    # them EXCLUDED_ENGINE_GROUP so they form no group of their own (a
    # wrong-block-size group would corrupt the per-group block-id counts).
    # Scratch groups (tokens_per_block 0) are excluded the same way.
    per_layer_group_idx: list[int] | None = None
    group_tokens_per_block: dict[int, int] = {}
    per_request_state_groups: set[int] = set()
    per_layer_sw_size = [-1] * num_layers
    per_layer_recurrent = [False] * num_layers
    per_layer_tp_replicated = [False] * num_layers
    replay_engine_groups = {
        engine_group_id
        for engine_group_id, group in enumerate(vllm_groups)
        if store_replay_windows and has_prefix_replay(get_group_kv_cache_spec(group))
    }
    if vllm_groups:
        per_layer_group_idx = [EXCLUDED_ENGINE_GROUP] * num_layers
        for engine_group_id, group in enumerate(vllm_groups):
            if not is_transferable_group(group):
                continue
            group_spec = get_group_kv_cache_spec(group)
            # The spec's block_size is the logical tokens covered by one of
            # this group's paged chunks (block IDs); the physical slot count
            # per chunk is discovered later from the registered tensors.
            # Under DCP the two diverge (see get_tokens_per_block).
            group_tokens_per_block[engine_group_id] = get_tokens_per_block(
                group_spec,
                dcp_size,
                store_replay_windows=store_replay_windows,
            )
            if is_registered_per_request_state(group_spec):
                per_request_state_groups.add(engine_group_id)
            elif group_tokens_per_block[engine_group_id] == 0:
                continue
            for name in group.layer_names:
                per_layer_group_idx[layer_to_idx[name]] = engine_group_id
        per_layer_sw_size = _resolve_per_layer_sw_sizes(
            vllm_groups, layer_to_idx, num_layers
        )
        per_layer_recurrent = _resolve_per_layer_recurrent(
            vllm_groups, layer_to_idx, num_layers
        )
        per_layer_tp_replicated = _resolve_per_layer_tp_replicated(
            vllm_groups, layer_to_idx, num_layers
        )

    # Aux pools form synthetic engine groups after the vLLM groups (an
    # unassigned marker layer would fall to EXCLUDED_ENGINE_GROUP and never
    # store), bucketed by tokens_per_block: pools sharing a block size share
    # one engine group — and thereby one kernel group when their tensor
    # identities also match. Tags are 1-based per bucket.
    aux_group_tags: dict[int, int] = {}
    if aux_entries:
        if per_layer_group_idx is None:
            # Non-hybrid engine config: all real layers share group 0.
            per_layer_group_idx = [0] * num_layers
        next_group_id = len(vllm_groups) if vllm_groups else 1
        group_by_tpb: dict[int, int] = {}
        for name, tokens_per_block in aux_entries:
            group_id = group_by_tpb.get(tokens_per_block)
            if group_id is None:
                group_id = next_group_id
                next_group_id += 1
                group_by_tpb[tokens_per_block] = group_id
                group_tokens_per_block[group_id] = tokens_per_block
                aux_group_tags[group_id] = len(group_by_tpb)
            per_layer_group_idx[layer_to_idx[name]] = group_id

    # Within one vLLM engine group, layers can have different hidden dimensions
    # (e.g. a different head count), which require different GPU copy kernels.
    # ``group_layers_by_identity`` splits each engine group further by physical
    # transfer identity (kv_size, num_heads, head_size, block_size,
    # engine_group_idx, dtype, engine_kv_format, block_stride_elems), so every
    # resulting LMCache group can be served by a single copy kernel. It is the
    # shared, engine-neutral primitive the server reuses to reproduce the same
    # grouping from the registered tensors.
    infos: list[EngineGroupInfo] = []
    for identity, indices in group_layers_by_identity(
        normalized_kv_caches,
        engine_kv_formats,
        per_layer_group_idx,
    ):
        # A group's object is stored once per TP group only if every layer in
        # it is replicated: one sharded layer makes the object differ per rank.
        replicated = all(per_layer_tp_replicated[idx] for idx in indices)
        infos.append(
            EngineGroupInfo(
                engine_group_id=identity.engine_group_idx,
                layer_indices=tuple(indices),
                tokens_per_block=group_tokens_per_block.get(
                    identity.engine_group_idx, 0
                ),
                sw_size_tokens=_merge_layer_sw_sizes(per_layer_sw_size, indices),
                # Connector-private pools bucket by tag under
                # --separate-object-groups, after the regular groups.
                extra_object_group_tag=aux_group_tags.get(identity.engine_group_idx, 0),
                recurrent_state=_merge_layer_recurrent(per_layer_recurrent, indices),
                tp_replicas=kv_tp_size if replicated else 1,
                # DCP splits a replicated group's tokens across its replicas.
                dcp_shards=dcp_size if replicated else 1,
                prefix_replay=identity.engine_group_idx in replay_engine_groups,
                per_request_state=identity.engine_group_idx in per_request_state_groups,
            )
        )
    return infos
