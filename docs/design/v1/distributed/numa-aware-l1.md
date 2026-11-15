# NUMA-aware L1 for the multiprocess server

Status: implemented, off by default (`--l1-numa-aware`). User-facing options are
in `docs/source/mp/configuration.rst` (*NUMA-aware L1*).

## Problem

The multiprocess (MP) server keeps all L1 host memory in one pinned pool. A
host with several NUMA nodes and GPUs spread over them (for example 8 GPUs on
2 sockets) gets pages wherever the first-touching thread ran. A GPU on node 1
that stores into or loads from a page on node 0 crosses the CPU interconnect
for every transfer.

vLLM's `--numa-bind` covers vLLM's own processes only, not the LMCache server.

## Goals

- GPUs on one NUMA node share one L1 region that lives on that node.
- A store from a GPU goes to its own node's L1 first.
- When that L1 is out of memory, the store may use another node's L1. The
  placement is a preference, not a hard limit.
- The thread serving a GPU runs on that GPU's node.
- Disabled by default. Without the switch, behavior is unchanged.

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
   NumaWritePolicy: L1(node(g)) first
                  |
   OUT_OF_MEMORY -> next L1 by node distance, until one has room
                  |
   the thread serving g runs on node(g)'s CPUs
```

### Switch and options

- `--l1-numa-aware` / `--no-l1-numa-aware` (default off) is the switch.
- `--l1-numa-nodes` (default `auto`) splits `--l1-size-gb`:
  - `auto`: every node that has a GPU, split evenly.
  - `0,1`: the named nodes, split evenly.
  - `0:60,1:40`: the named nodes with their share in percent.
- Share rules: either every node has a percentage or none does; each is above
  0 and they sum to 100. An unknown node, a repeated node, or a node whose
  share is smaller than `--l1-align-bytes` fails at startup. Sizes are rounded
  down to `--l1-align-bytes` and the remainder goes to the first node. The lazy
  allocator's initial size is split with the same percentages.
- Not combinable with `--l1-manager`, `--l1-devdax-path`, `--gds-l1-path` or
  `--shm-name`.

### 1. Topology detection (`lmcache/v1/system_detection.py`)

`NumaTopology` is built once per process from sysfs:

- GPU to node: `/sys/bus/pci/devices/<bus id>/numa_node` for the PCI bus id
  from `device_ops.get_gpu_pci_bus_id` (CUDA runtime, so the id is what this
  process sees and `CUDA_VISIBLE_DEVICES` reordering does not matter). A GPU
  whose node is `-1` or unreadable is left out.
- Node to CPUs: `/sys/devices/system/node/node<N>/cpulist`, intersected with
  `os.sched_getaffinity(0)`.
- Node distances: `/sys/devices/system/node/node<N>/distance`, used to order
  spill targets nearest first.

There is no NVML fallback: a GPU sysfs cannot place is treated as unknown.

### 2. One L1 manager per NUMA node

`--l1-numa-aware` builds one DRAM `L1ManagerConfig` per node with
`L1MemoryManagerConfig.numa_node` set and the tag `numa<N>`, and selects the
`numa` write policy. The multi-L1 machinery is reused unchanged (per-manager
eviction, L1 owners on `finish_write`, tags).

- Eager pools (`--no-l1-use-lazy`, hugepages) allocate through
  `alloc_pinned_numa_ptr` / `alloc_hugepage_pinned_numa_ptr`, which `mbind` the
  mapping to the node and then register it.
- Lazy pools (the default) set only the node's memory policy
  (`MPOL_BIND` through `mbind(2)`) on the untouched buffer, so pages are
  faulted in on the node while the expansion thread pins them. The existing
  `alloc_numa_ptr` is not used there because it touches every page at startup,
  which would defeat lazy allocation.
- A pool bound to a node cannot be a shared-memory segment, so `numa_node`
  requires `shm_name=""`. The server's engine-driven `--shm-name` step skips
  a NUMA-bound L1.

### 3. Write placement with spill

- `StorageManager.reserve_write` takes an optional `origin_device`
  (redacted from traces). The LMCache-driven store handler passes
  `cache_context.device` for both the full-chunk and the unfull-chunk
  reservation. Other callers pass nothing.
- `WritePolicy` gains two non-abstract methods with backwards-compatible
  defaults: `select_write_targets_from(keys, l1_managers, origin_device)`
  (default: ignore the origin) and `select_spill_targets(l1_manager_id,
  origin_device, l1_managers)` (default: no spill).
- `NumaWritePolicy` returns the L1 whose `numa_node` is the origin GPU's node,
  and the first L1 when the origin or its node is unknown. Its spill order is
  the other L1s sorted by node distance from the full one.
- `StorageManager` retries only keys that failed with `OUT_OF_MEMORY`, as one
  batch, on each spill target in order. A key that failed for another reason
  (for example a write conflict) is not retried. All pools full gives the same
  empty result as before.
- Each L1 still evicts on its own watermark, so a node normally frees space
  before its neighbors are used.

### 4. Worker threads follow the GPU they serve

The pool of an affinity worker is chosen by a key derived from the transport
client (the ZMQ identity or a gRPC client id), which is not the `instance_id`
the module sees. The node of a client is therefore not known when its first
task is routed, and moving a key to another worker later could run its queued
tasks out of order.

So the thread is moved instead of the key: `store` and `retrieve` call
`pin_current_thread_near_gpu` once the request's `cache_context` is resolved. It
sets the calling thread's CPU affinity to the GPU's node and skips the syscall
when the thread is already there. A worker serving one client stays on one
node; a worker shared by clients on different nodes moves with each request.
Nothing changes when the L1 is not NUMA-bound, or when the GPU's node is
unknown.

The GPU copy is a DMA or a kernel launch. Pinning the thread mainly removes
cross-socket cost in the Python and CPU-copy paths; the page placement from
section 2 is what changes DMA locality.

## Compatibility

- Off by default; without `--l1-numa-aware` no new code path runs.
- New `WritePolicy` methods and `StorageManager.reserve_write(origin_device=)`
  are optional; existing policies and callers are unaffected.
- Several pools mean no single L1 memory region, so L2 adapters that need one
  (P2P, NIXL, Mooncake over RDMA) are rejected as they are for any multi-L1
  setup.

## Tests

- `tests/v1/test_numa_topology.py`: sysfs parsing against a fake tree, share
  parsing, thread pinning with the real `sched_setaffinity`, and
  `bind_memory_to_node` checked against `/proc/self/numa_maps`.
- `tests/v1/test_numa_allocators.py`: the eager allocator asks for its node;
  a lazy allocator's pages report `bind:0` in `/proc/self/numa_maps`.
- `tests/v1/distributed/test_numa_l1.py`: placement by GPU node, spill order,
  batch spill, completion and reads of spilled objects.
- `tests/v1/distributed/test_numa_l1_config.py`: the option forms, remainder,
  lazy split, rejections, defaults unchanged.

These run on a CPU-only host with one NUMA node. They cannot show that pages
land on the right node of a multi-node machine, or the bandwidth change. Check
on the target host with `numastat -p <server pid>` or `/proc/<pid>/numa_maps`.

## Risks

- This touches the L1 allocation core and the write path. The default-off
  switch limits the exposure to deployments that enable it.
- Splitting L1 into per-node pools lowers the capacity each node can hold.
  Spill reduces the effect but moves data off the preferred node.
- Hits on blocks stored by another node's GPU are still remote reads. The
  benefit shrinks as DP replicas share more prefixes.
- When several clients share one worker thread, the thread changes its CPU
  mask whenever the node changes between requests.
