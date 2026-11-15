# SPDX-License-Identifier: Apache-2.0
"""Shared system topology types and NUMA detection helpers."""

# Standard
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
import ctypes
import os
import platform
import threading

# First Party
from lmcache import device_ops, torch_dev

_SYS_ROOT = "/sys"
# mbind(2) syscall numbers by machine; the C library has no wrapper.
_MBIND_SYSCALL = {"x86_64": 237, "aarch64": 235}
_MPOL_BIND = 2
# One unsigned long of node mask, matching csrc/cuda/mem_alloc.cpp.
_MAX_BINDABLE_NODES = 8 * ctypes.sizeof(ctypes.c_ulong)

_thread_state = threading.local()


@dataclass
class NUMAMapping:
    """Map a GPU index to its NUMA node."""

    gpu_to_numa_mapping: dict[int, int]


@dataclass(frozen=True)
class NumaTopology:
    """Which NUMA node each GPU and CPU belongs to, and how far nodes are apart."""

    gpu_to_node: Mapping[int, int] = field(default_factory=dict)
    """CUDA ordinal (as this process sees it) to NUMA node. A GPU whose node is
    unknown is absent."""

    node_cpus: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    """NUMA node to the CPUs this process may run on."""

    node_distances: Mapping[int, Mapping[int, int]] = field(default_factory=dict)
    """NUMA node to ``{other node: distance}``, as in the kernel's table."""

    def node_of_gpu(self, gpu_index: int) -> int | None:
        """Return the NUMA node of a GPU, or ``None`` when it is not known.

        Args:
            gpu_index: CUDA ordinal as this process sees it.

        Returns:
            The NUMA node, or ``None``.
        """
        return self.gpu_to_node.get(gpu_index)

    def nodes_with_gpus(self) -> list[int]:
        """Return the NUMA nodes that have at least one GPU, ascending."""
        return sorted(set(self.gpu_to_node.values()))

    def nodes(self) -> list[int]:
        """Return every NUMA node of the machine, ascending."""
        return sorted(self.node_cpus)

    def cpus_of(self, node: int) -> tuple[int, ...]:
        """Return the CPUs of a node this process may run on.

        Args:
            node: NUMA node.

        Returns:
            The CPU ids, empty when the node is unknown or has no allowed CPU.
        """
        return self.node_cpus.get(node, ())

    def nodes_by_distance(self, node: int) -> list[int]:
        """Return the other nodes, nearest to ``node`` first.

        Args:
            node: NUMA node to measure from.

        Returns:
            The other known nodes ordered by distance, ties by node id. Nodes
            without a distance entry come last.
        """
        distances = self.node_distances.get(node, {})
        others = [n for n in self.nodes() if n != node]
        return sorted(others, key=lambda n: (distances.get(n, 1 << 30), n))


def parse_cpu_list(text: str) -> tuple[int, ...]:
    """Parse a kernel CPU list such as ``0-3,8,10-11``.

    Args:
        text: Contents of a sysfs ``cpulist`` file.

    Returns:
        The CPU ids in ascending order.

    Raises:
        ValueError: The text is not a valid CPU list.
    """
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return tuple(sorted(cpus))


def read_numa_topology(
    gpu_bus_ids: Mapping[int, str | None],
    sysfs_root: str = _SYS_ROOT,
    allowed_cpus: Collection[int] | None = None,
) -> NumaTopology:
    """Build a topology from sysfs.

    Args:
        gpu_bus_ids: CUDA ordinal to PCI bus id (``0000:3B:00.0``). A GPU
            without a bus id is left out of the topology.
        sysfs_root: Root of the sysfs tree; a fake tree in tests.
        allowed_cpus: CPUs the process may run on. ``None`` reads the current
            process affinity.

    Returns:
        The topology. It has no nodes on a host that exposes none, and a GPU
        whose ``numa_node`` is ``-1`` or unreadable is left out.
    """
    if allowed_cpus is None:
        allowed_cpus = os.sched_getaffinity(0)
    node_dir = Path(sysfs_root) / "devices" / "system" / "node"
    node_ids = sorted(
        int(path.name[len("node") :])
        for path in node_dir.glob("node[0-9]*")
        if path.is_dir()
    )
    node_cpus: dict[int, tuple[int, ...]] = {}
    node_distances: dict[int, dict[int, int]] = {}
    for node in node_ids:
        cpulist = parse_cpu_list((node_dir / f"node{node}" / "cpulist").read_text())
        node_cpus[node] = tuple(c for c in cpulist if c in allowed_cpus)
        distance_file = node_dir / f"node{node}" / "distance"
        if distance_file.exists():
            row = [int(value) for value in distance_file.read_text().split()]
            node_distances[node] = dict(zip(node_ids, row, strict=False))

    gpu_to_node: dict[int, int] = {}
    pci_dir = Path(sysfs_root) / "bus" / "pci" / "devices"
    for gpu_index, bus_id in gpu_bus_ids.items():
        if bus_id is None:
            continue
        numa_file = pci_dir / bus_id.lower() / "numa_node"
        if not numa_file.exists():
            continue
        node = int(numa_file.read_text().strip())
        if node in node_cpus:
            gpu_to_node[gpu_index] = node
    return NumaTopology(gpu_to_node, node_cpus, node_distances)


@cache
def get_numa_topology() -> NumaTopology:
    """Detect the topology of this host once and reuse it.

    Returns:
        The NUMA topology for the GPUs this process can see.
    """
    gpu_bus_ids: dict[int, str | None] = {}
    if torch_dev.is_available():
        for gpu_index in range(torch_dev.device_count()):
            gpu_bus_ids[gpu_index] = device_ops.get_gpu_pci_bus_id(gpu_index)
    return read_numa_topology(gpu_bus_ids)


def bind_memory_to_node(ptr: int, size: int, node: int) -> None:
    """Set the NUMA policy of an untouched range so its pages come from a node.

    Only the policy is set. Pages that are already resident stay where they
    are, so call this before the range is first touched or pinned.

    Args:
        ptr: Start address of the range.
        size: Length of the range in bytes.
        node: NUMA node to allocate the pages from.

    Raises:
        ValueError: ``node`` cannot be expressed in a one-word node mask.
        OSError: The machine has no known ``mbind`` syscall number, or the
            kernel rejects the request.
    """
    if not 0 <= node < _MAX_BINDABLE_NODES:
        raise ValueError(f"NUMA node {node} is outside 0..{_MAX_BINDABLE_NODES - 1}")
    syscall_number = _MBIND_SYSCALL.get(platform.machine())
    if syscall_number is None:
        raise OSError(f"No mbind syscall number for {platform.machine()}")
    page = os.sysconf("SC_PAGE_SIZE")
    start = -(-ptr // page) * page
    end = (ptr + size) // page * page
    if end <= start:
        return
    node_mask = ctypes.c_ulong(1 << node)
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(
        ctypes.c_long(syscall_number),
        ctypes.c_void_p(start),
        ctypes.c_ulong(end - start),
        ctypes.c_int(_MPOL_BIND),
        ctypes.byref(node_mask),
        ctypes.c_ulong(_MAX_BINDABLE_NODES),
        ctypes.c_uint(0),
    )
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"mbind to node {node} failed: {os.strerror(errno)}")


def pin_current_thread_to_node(topology: NumaTopology, node: int) -> None:
    """Run the calling thread on the CPUs of a NUMA node.

    The request is skipped when this thread is already pinned to the node, so
    calling it for every request is cheap.

    Args:
        topology: Where the node's CPUs come from.
        node: NUMA node whose CPUs the thread should use.
    """
    if getattr(_thread_state, "node", None) == node:
        return
    cpus = topology.cpus_of(node)
    if cpus:
        os.sched_setaffinity(0, cpus)
    _thread_state.node = node


def pin_current_thread_near_gpu(topology: NumaTopology, gpu_index: int) -> None:
    """Run the calling thread on the CPUs of a GPU's NUMA node.

    Leaves the thread alone when the GPU's node is not known.

    Args:
        topology: Where the GPU's node and its CPUs come from.
        gpu_index: CUDA ordinal as this process sees it.
    """
    node = topology.node_of_gpu(gpu_index)
    if node is not None:
        pin_current_thread_to_node(topology, node)


def parse_numa_nodes_spec(spec: str, topology: NumaTopology) -> list[tuple[int, float]]:
    """Resolve ``--l1-numa-nodes`` into nodes and their share of the L1 size.

    Accepted forms: ``auto`` (every node with a GPU, even split), ``0,1``
    (the named nodes, even split) and ``0:60,1:40`` (the named nodes with
    percentages that sum to 100).

    Args:
        spec: The option value.
        topology: Where ``auto`` and node existence are checked.

    Returns:
        ``(node, fraction of the total)`` per node, fractions summing to 1.

    Raises:
        ValueError: The spec is malformed, names a node that does not exist or
            a node twice, mixes percentages with plain nodes, has a percentage
            that is not above 0, percentages that do not sum to 100, or
            ``auto`` finds no node with a GPU.
    """
    spec = spec.strip()
    if spec == "auto":
        nodes = topology.nodes_with_gpus()
        if not nodes:
            raise ValueError(
                "--l1-numa-nodes auto found no GPU with a known NUMA node; "
                "name the nodes explicitly, for example 0,1"
            )
        return [(node, 1.0 / len(nodes)) for node in nodes]

    entries = [entry.strip() for entry in spec.split(",")]
    nodes_in_spec: list[int] = []
    percents: list[float | None] = []
    for entry in entries:
        node_text, has_percent, percent_text = entry.partition(":")
        try:
            node = int(node_text)
            percent = float(percent_text) if has_percent else None
        except ValueError:
            raise ValueError(
                f"Invalid --l1-numa-nodes entry {entry!r}; use auto, 0,1 or 0:60,1:40"
            ) from None
        nodes_in_spec.append(node)
        percents.append(percent)
    if len(set(nodes_in_spec)) != len(nodes_in_spec):
        raise ValueError(f"--l1-numa-nodes names a node twice: {spec!r}")
    unknown = [node for node in nodes_in_spec if node not in topology.node_cpus]
    if unknown:
        raise ValueError(
            f"--l1-numa-nodes names unknown NUMA nodes {unknown}; "
            f"this host has {topology.nodes()}"
        )
    if all(p is None for p in percents):
        return [(node, 1.0 / len(nodes_in_spec)) for node in nodes_in_spec]
    if any(p is None for p in percents):
        raise ValueError(
            "--l1-numa-nodes must give a percentage for every node or for none"
        )
    shares = [p for p in percents if p is not None]
    if any(share <= 0 for share in shares):
        raise ValueError("--l1-numa-nodes percentages must be above 0")
    if abs(sum(shares) - 100.0) > 1e-6:
        raise ValueError(
            f"--l1-numa-nodes percentages must sum to 100, got {sum(shares):g}"
        )
    return [
        (node, share / 100.0) for node, share in zip(nodes_in_spec, shares, strict=True)
    ]
