# NUMA-aware L1 for the multiprocess server

Status: proposal, not implemented.

## Problem

The multiprocess (MP) server keeps all L1 host memory in one pinned pool. A
host with several NUMA nodes and GPUs spread over them (for example 8 GPUs on
2 sockets) gets pages wherever the first-touching thread ran. A GPU on node 1
that stores into or loads from a page on node 0 crosses the CPU interconnect
for every transfer.

Today nothing in the MP path controls placement:

- `create_memory_allocator` (`memory_manager/l1_memory_manager.py`) never
  passes `numa_mapping`, so the NUMA allocation code in
  `memory_allocators/{mixed,lazy}_memory_allocator.py` and
  `csrc/cuda/mem_alloc.cpp` (`mbind` + `cudaHostRegister`) is unreachable.
- `NUMAMapping` (`v1/system_detection.py`) is an empty dataclass. No code
  detects which node a GPU belongs to.
- The store handler knows the requesting GPU (`cache_context.device`) but does
  not pass it to `StorageManager.reserve_write`.
- The `AffinityThreadPool` workers are not pinned to any CPU.

`--numa-bind` in vLLM covers vLLM's own processes only, not the LMCache server.

## Goals

- GPUs on one NUMA node share one L1 region that lives on that node.
- A store from a GPU goes to its own node's L1 first.
- When that L1 is out of memory, the store may use another node's L1. The
  placement is a preference, not a hard limit.
- Worker threads of a node serve only the GPUs of that node, and run on that
  node's CPUs.
- Disabled by default. Without the new option, behavior is unchanged.

## Non-goals

- Making a load from another node's L1 fast. A GPU that hits a block stored on
  another node (a block stored by another node's GPU, or one that spilled)
  still crosses the interconnect.
- Moving blocks between nodes after they are stored.
- Changing what is cached, the cache keys, or eviction policy.

## Design

```
            store from GPU g
                  |
         node(g) = topology.node_of_gpu(g)
                  |
   NumaAwareWritePolicy: [L1(node(g)), nearest other L1s ...]
                  |
   reserve_write tries each L1 in order, next one only on OUT_OF_MEMORY
                  |
      transfer runs on a worker pinned to node(g)'s CPUs
```

### 1. Topology detection (new module)

A small module under `lmcache/v1/` that builds, once at server start:

- GPU to node: read `/sys/bus/pci/devices/<bus id>/numa_node` for the PCI bus
  id of each visible GPU. When that file reports `-1`, fall back to NVML CPU
  affinity.
- Node to CPUs: `/sys/devices/system/node/node<N>/cpulist`, intersected with
  the CPUs the process may run on (`os.sched_getaffinity(0)`).
- Node to node distance: `/sys/devices/system/node/node<N>/distance`, used to
  order spill targets nearest first.
- Result: a populated `NUMAMapping` plus the CPU and distance tables.

GPUs are keyed by PCI bus id and resolved to the CUDA ordinals the server sees.
`CUDA_VISIBLE_DEVICES` can reorder devices (for example `0,4,1,5,...`), so the
ordinal of the server's device is never assumed to match the physical index.

### 2. One L1 manager per NUMA node

- `L1MemoryManagerConfig` gains `numa_node: int | None` (default `None`).
- `create_memory_allocator` passes it to both `MixedMemoryAllocator` and
  `LazyMemoryAllocator`. `_resolve_pinned_alloc_free` and the lazy allocator
  take an explicit node instead of deriving it from `torch_dev.current_device()`
  of the calling thread, because the server thread that creates the pool is not
  a GPU thread.
- New option `--l1-numa-nodes auto|<list>`. `auto` creates one L1 for every node
  that has at least one GPU. The total size is split evenly by default.
  Each manager gets the tag `numa<N>`.
- The existing multi-L1 machinery is reused (`create_l1_managers`, per-manager
  eviction controllers, L1 owners recorded on `finish_write`, tags).
- Shared memory (`shm_name`) is already disabled when more than one L1 exists
  (`_compute_shm_pool_info`). That affects the engine-driven path only, not
  LMCache-driven transfer.

### 3. Write placement with spill

- `StorageManager.reserve_write` and `WritePolicy.select_write_targets` get an
  optional `origin_device` argument, default `None`. Existing callers and
  policies are unaffected. The LMCache-driven store handler
  (`multiprocess/modules/lmcache_driven_transfer.py`, the two `reserve_write`
  call sites) passes `cache_context.device`.
- New `NumaAwareWritePolicy` returns, for each key, an ordered list of L1
  managers: the node of `origin_device` first, then the other nodes by
  distance. With no origin it behaves like the default policy.
- `reserve_write` walks that list and moves to the next manager only when the
  allocation returns `OUT_OF_MEMORY`. It fails only when every manager is full.
  The current "an OOM does not fall back" behavior stays for the existing
  policies.
- Each L1 still evicts on its own watermark, so a node normally frees space
  before neighbors are used. Spilling happens when eviction cannot keep up.

### 4. Worker threads per node, pinned to the node's CPUs

- `AffinityThreadPool` gets an optional node partition: one group of workers
  per node. Each worker calls `os.sched_setaffinity` on its own thread with
  that node's CPU list when it starts.
- A task is routed to the group of the node that owns the instance's GPU. The
  instance to node mapping is recorded when the instance registers its KV
  caches (`register_kv_cache`), where `cache_context.device` becomes known.
- Requests that arrive before registration, or whose instance has no mapping,
  use the existing path, so nothing is blocked.
- With the option on, `--max-gpu-workers` means workers per node.
- ZMQ (`transport/zmq_impl`) and gRPC (`transport/grpc_impl`) each create their
  own pool. Both need the same hook.

The GPU copy itself is a DMA or kernel launch. Pinning the threads mainly
removes cross-socket cost in the Python and CPU-copy paths. Placing the pages
(section 2) is what changes DMA locality.

## Open points to check while implementing

- The ZMQ routing key (client identity) and the `instance_id` used at
  registration might not be the same value. If they differ, add an explicit
  identity to instance mapping.
- How a load picks among several L1 managers when more than one holds the same
  key. The owner tracking exists, but its exact selection rule has not been
  read in detail.
- Whether the lazy allocator's chunked pinning keeps the `mbind` policy on the
  chunks it pins after the first one.
- Behavior when a node has memory but no GPU: `auto` skips it. An explicit list
  may name it and is allowed.

## Compatibility

- Off by default. Without `--l1-numa-nodes`, no new code path runs and the
  server has one L1, as today.
- New arguments are optional. Existing write policies ignore `origin_device`.

## Test plan

CPU-only tests that can run here:

- Topology parsing against a fake sysfs tree, including `-1` fallback and
  `CUDA_VISIBLE_DEVICES` reordering.
- The allocator receives the configured node; the C++ allocation functions are
  mocked at the device-ops boundary.
- `NumaAwareWritePolicy` ordering with and without an origin, and spill order by
  distance.
- `reserve_write` falls through only on `OUT_OF_MEMORY` and fails when all
  managers are full; existing policies keep their behavior.
- Worker pool: workers of each node run with the expected CPU affinity (the
  real `sched_setaffinity` works on any Linux host when pinned to a CPU
  subset), tasks reach the right group, unmapped keys use the existing path.
- Config validation and unchanged defaults.

Needs a multi-node machine, not testable in the CI sandbox:

- Pages actually land on the target node. Check with
  `numastat -p <server pid>` or `/proc/<pid>/numa_maps`.
- Transfer bandwidth and time to first token with the option on and off.

## Risks

- This touches the L1 allocation core and the write path. The default-off
  option limits the exposure to deployments that enable it.
- Splitting L1 into per-node pools lowers the capacity each node can hold.
  Spill reduces the effect but moves data off the preferred node.
- Hits on blocks stored by another node's GPU are still remote reads. The
  benefit shrinks as DP replicas share more prefixes.
