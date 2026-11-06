# Unfull Chunks and Per-Request Ring State

## Summary

LMCache stores KV cache in chunks of `C` tokens. Before this feature it threw
away the last, partial chunk of every prompt (the *unfull chunk*). A later
request with the same prefix then recomputed up to `C - 1` tokens.

That cost is small for attention-only models but large for hybrid models:

- The chunk must be a multiple of vLLM's block size, which is 400 to ~1152
  tokens for Qwen3.8, GLM-5.3 and DeepSeek-V4.1. So the thrown-away part is
  large.
- A linear-attention (Mamba/GDN) layer can only resume where a state snapshot
  exists, and a compressed-attention layer (QSA, DSA kpool, DSV4.1
  compressor) can only resume mid-compression-group if the raw keys of the
  still-open group are available. vLLM keeps those keys in a per-request ring
  (`CircularBufferSpec`) that LMCache skipped (#5042), which was correct only
  because LMCache never resumed mid-chunk.

This feature lets LMCache store and load the unfull chunk `[kC, P)` of a
request together with the ring, so a later request (a decode instance in PD
disaggregation, or the next round of a chat) resumes at `P` instead of `kC`.
It is the shared groundwork for PD handoff and multi-round reuse; neither
policy lives here.

Enabled per connector with `lmcache.mp.save_unfull_chunk: true`
(`kv_connector_extra_config`). Off by default: nothing changes for existing
deployments.

## Concepts

- **Unfull chunk**: the token range `[kC, P)` with `kC = floor(P / C) * C`
  and `0 < P - kC < C`, where `C` is the LMCache chunk size. It is stored as
  one object per object group with the regular chunk layout, plus one block
  per narrow sliding-window group (see below); only the blocks that cover
  `[kC, P)` are copied.
- **Per-request state group**: a vLLM KV cache group whose request owns one
  block for its whole lifetime (`CircularBufferSpec`; `KpoolTailSpec` on older
  vLLM). It is not addressed by token position.
- **Temporary object**: an L1 object deleted when its last read lock is
  released (`L1ObjectState.is_temporary`), i.e. read once.

## Ring registration

With the feature on, `create_engine_group_infos_from_vllm` registers ring
groups as `EngineGroupInfo(per_request_state=True)` instead of excluding them
with the other scratch groups (HiSparse pools and bounded-replay SWA stay
excluded). Before format discovery, `_PerRequestStateOpaqueViewEdit`
re-views each ring layer as `[NB, 1, 1, row]` where `row` is the layer's
whole per-block span: GLM-5.3's kpool tail pads between its two heads, which
format discovery rejects, and LMCache never needs per-token addressing of a
ring.

The server sizes a per-request group as **one block per chunk**
(`tokens_per_block = chunk_size`). The connector passes the request's single
ring block once per chunk of every store and retrieve, so the ring rides in
every object:

```text
object for chunk i  =  [attention blocks of chunk i] + [ring block]
```

Why store it with every chunk instead of only with unfull chunks: the object
layout stays one layout per object group, so lookup, prefetch, eviction and
L2 need no change. The ring content stored with a full chunk is stale, and
that is harmless: at a chunk boundary the open compression group is empty
(block size is a multiple of the compression width, chunk size a multiple of
block size), so nothing reads the ring before it is overwritten. The cost is
the ring bytes per chunk (largest for DeepSeek-V4.1's fp32 compressor ring),
paid only when the feature is on.

## Unfull chunk key

```text
prompt, P tokens   |  chunk 0  |  chunk 1  |  chunk 2  | tail |
                   0           C           2C          3C     B   P
before:            [------ stored, hit ---------------]  recomputed
after:             [------ stored, hit ----------------------]  recomputed
                                                         ^ unfull chunk [3C, B)
```

The feature is off by default. With it off, nothing changes.

## How to turn it on

| What | Where | Example |
|---|---|---|
| Turn the feature on | vLLM `kv_connector_extra_config` | `"lmcache.mp.save_unfull_chunk": true` |
| Make vLLM's hit grid finer than the chunk (Mamba hybrids) | `vllm serve` flag | `--prefix-match-unit 32` |
| Make one request's unfull chunk read-once (optional) | the request's `kv_transfer_params` | `"lmcache.temporary_unfull_chunk": true` |

Why `--prefix-match-unit`: LMCache stores the tail at a point on vLLM's hit
grid `g` (see [Which tails are stored](#which-tails-are-stored)). For a Mamba
hybrid, vLLM sets `g` to its block size by default, and the chunk is a
multiple of that. With a one-block chunk (Qwen3.8-27B on vLLM 0.33: block 832,
chunk 832) every grid point is a chunk boundary, so no tail is ever stored.
With `--prefix-match-unit 32` on Qwen3.8-27B and a 4,903-token prompt, a
repeated prompt hits 4,896 tokens instead of 4,160.

Requirements, checked at startup:

- LMCache-driven transfer mode (the CUDA default).
- No speculative decoding, no `lmcache.mp.lazy_offload`.
- For Mamba hybrids, a vLLM whose `SchedulerOutput` has
  `kv_connector_block_state`.

## Parameters: what is sent and where it goes

```text
 user / deployment                       vLLM scheduler process              LMCache server
 -----------------                       ----------------------              --------------
 extra_config                            LMCacheMPConnector
   lmcache.mp.save_unfull_chunk ──────>    reads once at startup
 vllm serve --prefix-match-unit ──────>    grid g = vLLM hash_block_size
 request kv_transfer_params                tracker.request_configs
   lmcache.temporary_unfull_chunk ────>    (copied like any lmcache.* key)
 vLLM (Mamba only)                         │
   kv_connector_block_state ──────────>    picks tail boundaries + state blocks
                                           │
                                           ├─ LOOKUP key ──────────────────────> LookupModule
                                           │    end = lookup length L (unaligned)    probes tails, locks one
                                           │    request_configs:                     replies in tokens
                                           │      lmcache.hint.unfull_chunk_grid = g
                                           │
                                           └─ LoadStoreOp ─> vLLM worker process
                                                end = B (unaligned)   │
                                                                      ├─ STORE / RETRIEVE key ─> LMCacheDrivenTransfer
                                                                      │    end = B (unaligned)       copies the unfull object
                                                                      │    request_configs:          ─> StorageManager
                                                                      │      temporary flag              .reserve_write(is_temporary)
The connector-side store / lookup / retrieve plan:

store (P = prompt end)                     load (lookup end L)
------------------------------------       -----------------------------------------
STORE [s, P), include_unfull_chunk         LOOKUP [0, L), include_unfull_chunk
  full chunks [s, kC) as today               chunk prefetch -> k full chunks hit
  + unfull chunk [kC, P):                    if k == floor(L/C): L1-only probe of the
    blocks: ceil(P/T_g) slices per group       unfull key, read-locked like a chunk
    ring: the request's ring block           reply k (+1 if the unfull chunk hit)
    is_temporary from request config       client: hit = min(reply * C, L)
                                           RETRIEVE [s, L), include_unfull_chunk

```

No new field travels on the wire. A tail is addressed by an `end` that is not
a multiple of `C`: the key's range is then `[start, kC)` in full chunks plus
the unfull chunk `[kC, end)`.

| Parameter | Set by | Sent as | Read by |
|---|---|---|---|
| `lmcache.mp.save_unfull_chunk` | deployment | vLLM `kv_connector_extra_config` | `LMCacheMPConnector` (scheduler and worker) at startup. Turns on everything below. |
| `--prefix-match-unit` | deployment | `vllm serve` flag | vLLM. It sets `hash_block_size`, which the connector uses as the grid `g`. |
| `lmcache.temporary_unfull_chunk` | client, per request | `kv_transfer_params` → `request_configs` on the STORE key | `LMCacheDrivenTransfer`, which passes `is_temporary=True` to `StorageManager.reserve_write`. |
| `lmcache.hint.unfull_chunk_grid` | connector | `request_configs` on every LOOKUP key | `LookupModule`, to know which tail lengths to probe. Without it the server probes only the lookup length. |
| unaligned `end` | connector (LOOKUP, FREE_LOOKUP_LOCKS), worker adapter (STORE, RETRIEVE) | the existing `IPCCacheServerKey.end` | `LookupModule` (probe and release the tail), `LMCacheDrivenTransfer` (copy the tail). |
| `kv_connector_block_state.boundary_state_offloads` | vLLM | `SchedulerOutput` | Connector, Mamba hybrids only: which boundaries have a saved state and in which block. |

The unfull chunk is a single object per object group. Its per-group block
list is trimmed to the last `min(n, keep)` blocks, where `keep` is the
group's per-chunk window in blocks (all blocks for full attention, the last
window for sliding window, one block for Mamba state and for the ring).

A connector with the feature off sends an aligned LOOKUP `end` and no grid
hint, so the server never hands it a tail it cannot load. The other transfer
paths (engine-driven, qstore, server transfer) still floor an unaligned `end`.

## Interface changes

Wire / IPC:

| Interface | Change |
|---|---|
| `IPCCacheServerKey` | No new field. New method `has_unfull_chunk(chunk_size)`: `end % chunk_size != 0`. |
| QUERY_PREFETCH_STATUS / WAIT_PREFETCH_STATUS reply | **Always a token count** now, for every lookup (was a chunk count). It is a multiple of `C` unless the hit ends with a tail. Every MP client is updated (vLLM `LookupState`, SGLang, TensorRT-LLM, ATOM, SDK, `lmcache bench server`); server and clients must be upgraded together. |
| STORE / RETRIEVE on the LMCache-driven path | An unaligned `end` now also stores or loads the tail `[kC, end)`; its block IDs follow the full chunks' in every group. vLLM, SGLang, TensorRT-LLM, the SDK and the bench send aligned ends unless they mean a tail; ATOM's ranges come from ATOM. |
| FREE_LOOKUP_LOCKS | An unaligned `end` also frees the tail ending exactly there, if this request's lookup locked it. |

Server:

| Interface | Change |
|---|---|
| `MPCacheServerContext` | New `resolve_unfull_chunk_obj_keys(key, object_group_ids)`: the tail's object key per group. `resolve_obj_keys` is unchanged (it floors `end`). |
| `Session` | New `get_unfull_chunk_hash(boundary)`. No tail lookup state: the tail's L1 owners join the chunks' owners through the existing `record_prefetch_result`. |
| `LookupModule.query_prefetch_status` / `wait_prefetch_status` / `free_lookup_locks` | Token reply and tail release as above. |
| `LayoutDescRegistry.register` | New argument `group_unfull_chunk_layout_descs`. New `find_group_unfull_chunk_layout_descs()`, which falls back to the full-chunk layouts. |
| `StorageManager` | `reserve_write` gets `is_temporary: bool = False`. New `contains_l1_keys(keys)`: whether every key is in some L1, without locking. |
| `L1ManagerInterface` | New `contains_keys(keys) -> list[bool]`: whether each key is resident and readable, without locking. |
| `group_view.slice_block_ids_per_group` | New keyword `include_unfull_chunk`: `end` may be unaligned, and the slice keeps the block holding the last token (ceil instead of floor). |
| `BaseCacheContext` (CPU, CUDA, NPU, MUSA) | New `get_temp_staging_buffer(num_bytes)`: the start of the transfer staging buffer as one flat region (GDS-registered where the buffer is). Tail objects with a lead block stage there. |
| `object_group_transfer` | New functions: `has_unfull_chunk_lead_block`, `get_unfull_chunk_layout_desc`, `trim_and_stage_unfull_chunk_block_ids`, `transfer_unfull_chunk`, `split_unfull_chunk_block_ids`. The CUDA copy kernel is **unchanged**. |
| `MP_LOOKUP_PREFETCH_END` event | New fields `unfull_chunk_boundary`, `unfull_chunk_probes` (tail lengths checked). |

vLLM integration:

| Interface | Change |
|---|---|
| `LMCacheMPSchedulerAdapter.maybe_submit_lookup_request` | New argument `include_unfull_chunk`: send the unaligned lookup length instead of flooring it. The only adapter change. |
| `LMCacheMPRequestTracker` | New fields `unfull_chunk_boundaries` (tails already stored or hit) and `ends_at_prompt` (`max_tokens == 1`). |
| `LMCacheMPRequestMetadata` | New `GetUnfullChunkStoreMetadata(...)`: builds the tail store ops. |
| `kv_cache_groups` | New `is_per_request_state_spec`, `is_align_mode_mamba_spec`. |

## How it works

Terms:

- `C`: LMCache chunk size. `P`: prompt length. `T`: tokens per block.
- **Boundary `B`**: a token count. "The tail at `B`" covers the first `B`
  tokens. vLLM uses the same word (`boundary_tokens`).
- **Unfull chunk**: tokens `[kC, B)` with `kC = floor(B / C) * C` and
  `0 < B - kC < C`.
- **Hit grid `g`**: the token counts where vLLM's prefix cache can resume.
  For Mamba hybrids it is vLLM's `hash_block_size`; otherwise the scheduler
  block. A tail exists only when `g < C`.

### Which tails are stored

| Tail | Boundary | Stored by | Used by |
|---|---|---|---|
| Grid | `B = floor((P - 1) / g) * g` | every request | a later request with the same prefix: the same prompt again, or the next round of a chat |
| Exact | `B = P` | requests that end at the prompt (`max_tokens == 1`, e.g. a disaggregated prefill) | the disaggregated decode, which continues exactly where the prefill stopped |

Why the grid tail stops below `P`:

- vLLM always recomputes the last prompt token to get its logits, so a later
  request resumes at `P - 1` at most.
- A Mamba state exists only where vLLM saved one, and vLLM saves states on the
  grid. It hands the saved block to the connector as a *boundary-state block*:
  a copy that later steps do not overwrite, so storing it while the request
  keeps decoding is safe.
- The lookup does not know where an earlier request's tail ended. Keeping all
  tails on the grid means it has at most `C / g` lengths to try.

For PD with Mamba, the decode looks up `P - 1` but the exact tail holds the
state at `P`, so they do not match yet. The decode hits the grid tail instead
and recomputes `P - 1 - B < g` extra tokens. They will match once the prefill
stops at `P - 1`, like NIXL does (a separate P/D change).

### Key

```text
unfull_hash = H(tokens[kC:B], prefix = chunk_hash[k-1])
```

The tail is hashed like the next full chunk, chained from the hash of the
chunks before it. All supported hash functions include the token count, so a
tail never has the same key as a full chunk. This matters: a temporary tail
with a full chunk's key would make `reserve_write` fail with
`KEY_NOT_WRITABLE`. A new hash function must keep this property.

### Store

```text
connector (scheduler)                 worker                     server
STORE [kC, B)                         blocks: ceil(B / T)        one object per object group,
  Mamba: boundary-state block           per group                temporary if requested
  as that group's last block
```

- **Mamba hybrids**: vLLM lists the boundary-state blocks it offers this step
  in `kv_connector_block_state`. Each offered boundary inside the prompt's last
  chunk (`kC < B <= P - 1`) that every Mamba group has is stored as a tail.
  The connector uses vLLM's choice and does not re-derive it.
- **Other models**: `B = floor((P - 1) / g) * g`, stored once the step that
  computes it is scheduled. Blocks come from the block table.
- **Requests that end at the prompt**: also store the exact tail `[kC, P)`
  once the prompt is computed, all blocks from the block table. Skipped when
  `P` is a multiple of `C`.
- A tail is a store of its own, with no full chunk in it. vLLM hands over the
  Mamba state one step after computing `B`, usually after the full chunks
  were stored. So one request can send two stores in one step. The worker
  keeps only the latest store future, which is safe because one worker's
  stores run in order on the server.
- **Pinning**: a boundary-state block is not in the request's block table and
  vLLM would free it once the request moves on. The connector pins it
  (`BlockPool.touch`) when it queues the store and unpins it when the request
  shows up in `finished_sending`. One block per Mamba group per request.
- Each tail is stored once (`tracker.unfull_chunk_boundaries`), and not at
  all if the request's own lookup already hit it.
- Blocks below `B` must not change after the store is sent. Attention rows
  below `B` are never rewritten, and the Mamba block is a copy, so this
  holds.

### Lookup

```text
LOOKUP [0, L), L unaligned, hint unfull_chunk_grid = g
  stage 1, full chunks (main lookup): one prefetch      -> k full chunks hit
  stage 2, tail (when stage 1 finishes), only inside chunk k:
       candidate lengths, longest first:
         L itself (if inside chunk k)
         floor(min(L, kC + C - 1) / g) * g, then - g, ... while > kC
       check each in L1 without locking; stop at the first one found
       one L1-only prefetch read-locks that tail
  reply in tokens: the tail length e, or k * C if no tail was locked
```

Example, `C = 128`, `g = 16`, `P = 337`: the prompt stores chunks `[0, 256)`
and the tail `[256, 336)`. The same prompt looks up `L = 336`, hits both
chunks, tries `336` first and hits. A next round with 900 tokens hits the same
two chunks (round 1 stored no chunk `[256, 384)`), tries `368, 352, 336` and
hits `336`.

- Stage 2 searches only the chunk after the full-chunk hit. With 256 tokens
  of chunks and a 32-token tail stored, a 1000-token lookup tries tail lengths
  between 256 and the next chunk boundary, never near 1000.
- The candidates are checked with `StorageManager.contains_l1_keys`, which
  takes no lock. Only the chosen tail is locked, by a prefetch that never goes
  to L2 (`allowed_l2_memory_tags` is empty), so it finishes right away. It
  locks only when every object group and every KV rank hit, and takes
  `num_kv_readers` read locks per key, like a chunk. If the tail is evicted
  between the check and the lock, the hit stops at `kC`.
- The tail's L1 owners are recorded with the chunks' owners. FREE_LOOKUP_LOCKS
  frees a tail only if all its keys are among them, so a key ending anywhere
  else releases nothing extra.
- **Several servers**: if they report different hits, the client keeps only
  the full chunks every server hit and frees each server's locks above that,
  its tail included.
- **Overlap with vLLM's prefix cache**: vLLM may already hold `n` tokens
  with `kC < n < B`. The lookup starts at `kC` (vLLM's hit floored to a
  chunk, #5383), so it still locks the tail, like any chunk that straddles
  vLLM's hit. The retrieve then skips `[kC, n)` (see Retrieve), and the hit
  stays `B`.

### Retrieve

A hit that ends mid-chunk is loaded as one RETRIEVE `[start, e)`, `e`
unaligned. The tail's block IDs follow the full chunks' in every
group. The server copies the full chunks as before, then the tail, in the same
transaction: same wait on the producer event, all-or-nothing commit, same
lock release.

- Sliding-window groups need the window that ends at `B`, which can start in
  chunk `k - 1`. The chunk retrieve already restores that earlier part; the
  tail restores the rest.
- Mamba groups need only the block with the state at `B`. On retrieve it is
  the last block of the tail range, where vLLM expects a resumed request's
  state.
- `skip_first_n_tokens` (vLLM's hit minus `start`) protects blocks vLLM
  already shares with other requests. It counts from `start`, so it reaches
  the tail only when the range has no full chunk (`start = kC`). The tail
  then does not write its blocks below `n`: they are added to the
  kernel's `skip_prefix_n_blocks`, and a lead block below `n` is not copied.
  `n` is aligned to every group's block, so the skip is whole blocks. The
  Mamba block (state at `B > n`) is always written.

### Object layout

A tail is one object per object group with the **same layout as a full
chunk**, so no new kernel is needed. It just has fewer real blocks.

The copy kernel expects exactly `keep` blocks per object (`keep` = all blocks
of a chunk for full attention, the window's blocks for sliding window, one
block for Mamba). The block list is cut to its last `min(n, keep)` blocks and
then padded at the front with placeholder IDs. The kernel's existing
`skip_prefix_n_blocks` argument skips the padding, so it is never read or
written:

```text
full attention, keep = 16, 3 real blocks:
  block ids : [b0 b0 b0 ... b0 | b0 b1 b2]     (13 placeholders + 3 real)
  skip      : 13
  object    : slots 0..12 untouched, slots 13..15 <- b0 b1 b2
```

Store and load pad the same way, so block `i` lands in the same slot both
times. Tails always use the staging copy path (`transfer_unfull_chunk`), never
the direct-copy path, which assumes full chunks. An object group whose tail
blocks are all null stores nothing.

#### Window lead block

This case is only for sliding-window groups with window `W < C`.

Such a group keeps `keep = W / T` blocks per chunk. The next token after `B`
looks at tokens `[B - W + 1, B)`. A full chunk ends on a block edge, so that
range fits in `keep` blocks. A tail can end inside a block, and then the range
touches `keep + 1` blocks. Example with `T = 4`, `W = 8`, `keep = 2`, a tail
of `n` tokens:

```text
n = 9   window [2, 9)   blocks 0 1 2   -> keep + 1, lead block 0
n = 11  window [4, 11)  blocks 1 2     -> keep
n = 13  window [6, 13)  blocks 1 2 3   -> keep + 1, lead block 1
```

The extra first block is the *lead block*. Both sides work out whether it is
needed from `B mod C`, so no extra metadata is sent. (NIXL covers the same
overlap for P/D with `cdiv(W, T) + 1` window blocks. This assumes `W` is a
multiple of `T`.)

- Layout: the tail object uses `get_unfull_chunk_layout_desc`: the full-chunk
  layout, plus one single-block region per such group, **appended at the
  end**. All regular regions keep their offsets, and full-chunk objects are
  unchanged. The server registers these layouts next to the full-chunk ones
  for the lookup's L1 check.
- Copy: the regular copy runs as before; the lead block is one more launch of
  the same kernel with one block. The object is larger than one
  (batch, object-group) slot of the staging buffer, so it stages at the
  start of that buffer (`BaseCacheContext.get_temp_staging_buffer`),
  spanning the following slots. Transfers on the stream run in order, so
  nothing else uses those slots meanwhile. The buffer is registered for GDS,
  so a GDS-backed L1 can hold the object like any other.
- On the decode side the lead block is the first block vLLM keeps
  (`get_num_skipped_tokens = B - W + 1`), so it is allocated, not null.
- A sliding-window model's grid is its scheduler block, which is a whole
  number of window blocks, so its grid tails end on block edges and never
  need a lead block. The lead block matters for models whose grid is finer
  than their window blocks, and for exact tails.

## Temporary objects

`request_configs["lmcache.temporary_unfull_chunk"] = True` stores the
request's unfull chunk with `is_temporary`. L1 then deletes it when the last
read lock of a lookup is released (a retrieve on every worker, or
`free_lookup_locks` for an unused hit), and never offloads it to L2. An
unread temporary object is evicted like any other.

## Lookup, locks and sessions

The unfull probe uses `submit_prefetch_task(..., skip_l2=True)`, which reads
L1 synchronously and takes `num_kv_readers` read locks per key like the chunk
prefetch. The session records the probed keys and their L1 owners
separately from the chunk prefetch result:

- each worker's unfull retrieve releases one lock per key;
- `FREE_LOOKUP_LOCKS` with `include_unfull_chunk` and the lookup's own
  unaligned `end` releases the remaining locks once (vLLM already holds the
  tokens, or the servers disagree on hits);
- anything else is left to the read-lock TTL, like a chunk lock.

## Connector behavior (vLLM MP connector)

- **Store**: the step whose forward computes the prompt's last token `P`
  (`computed == P`, so the ring and Mamba state describe exactly `P`)
  extends its store to `[s, P)` when `P % C != 0` and every full chunk up to
  `kC` is stored by then. At most once per request; a request whose lookup
  already found the unfull chunk does not store it again.
  `num_stored_tokens` stays chunk-aligned.
- **Store invariant**: the unfull chunk's blocks must not be written after
  its store is submitted. Unlike a full chunk, they are live: the next decode
  step writes the ring at `position % capacity` and rewrites the Mamba
  running-state block in place, while the store copies asynchronously on the
  server stream (nothing orders the next forward after it). The connector
  therefore stores the unfull chunk only for a request that ends at the
  prompt (`max_tokens == 1`, e.g. a disaggregated prefill, or a pooling
  request); the request's blocks are held until the store completes.
- **Lookup**: sends `end = L` (the lookup token count, after the Mamba
  reserve-last-token rule) with `include_unfull_chunk`, so a hit can end
  mid-chunk.
- **Retrieve**: a hit that ends mid-chunk is retrieved as one operation
  `[start, L)` with `include_unfull_chunk`.
- **Overlap with vLLM's prefix cache**: if vLLM already holds tokens past
  `kC`, loading the unfull chunk would overwrite blocks other requests read,
  so the connector releases the unfull chunk's locks and keeps the hit at
  `kC`.
- **Ring blocks**: every store and retrieve carries the request's ring block
  once per chunk (`slice_block_ids_per_group(per_request_state_groups=...)`).
  The scheduler-side geometry (`group_tokens_per_block`, LCM, hit alignment)
  keeps the ring at `0`: it never constrains chunk alignment.
- **Startup checks**: the option is rejected with speculative decoding (draft
  rows sit past `P` in the ring and in relocated Mamba blocks), with
  `lmcache.mp.lazy_offload`, and when the worker resolves to an
  engine-driven transfer mode.

## Limits

- Served from L1 only (a persistent unfull object offloaded to L2 is not
  looked up there); LMCache-driven transfer mode only (the CUDA default).
- A prompt shorter than one chunk has no full chunk to look up, so its
  unfull chunk is never probed.
- A failed unfull-chunk retrieve, or a lookup abandoned for an APC-shrink
  re-lookup (`apc_covered_lookup.md`), leaves the unfull chunk's read locks
  (and a temporary object) to the read-lock TTL: those releases resolve full
  chunks only.
- The copy path is chosen per object group: when the ring's opaque layout is
  not eligible for the direct-copy path, the object group holding it (the
  attention one) takes the staging-kernel path for full chunks too.
- A model with a sliding window narrower than a chunk stores no unfull
  chunk when an L1 is GDS-backed: GDS copies only through its registered
  staging buffer, and the lead-block layout does not fit it. Supporting it
  means a registered unfull staging region per platform.
- Only the prompt-end tail is stored. Finding the tail of an earlier round
  (multi-round chat) needs a separate lookup policy; resuming a decode
  instance at the exact prompt end (PD) additionally needs the prefill side
  to stop where the decoder resumes.
- vLLM must accept a non-block-aligned external hit (current vLLM does). DSV4.1
  with bounded-replay SWA still replays its window after a load because
  LMCache reports only the prefix-cacheable groups as loaded.
- A lookup probes up to `C / g` unfull-chunk lengths in L1 (no locks, one
  hash each); the count is in `MP_LOOKUP_PREFETCH_END`
  (`unfull_chunk_probes`).
- Requests with `max_tokens == 1` always store an unfull chunk. Without PD
  nobody reads it, so it only costs L1 space.
- Mamba backends that save the prompt-tail state inside a prefill chunk offer
  no boundary-state block for it, so only block-end boundaries are stored.
- Sliding-window blocks below the window can be freed while an unfull-chunk
  store is in flight, the same risk as for full-chunk sliding-window stores.
- Unfull chunks are not shared over P2P L1; P2P lookup of them is a follow-up
  for LMCache-native PD.
