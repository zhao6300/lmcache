# SPDX-License-Identifier: Apache-2.0
"""NUMA topology detection, node-share parsing and node binding helpers."""

# Standard
from pathlib import Path
import ctypes
import mmap
import os
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.system_detection import (
    NumaTopology,
    bind_memory_to_node,
    parse_cpu_list,
    parse_numa_nodes_spec,
    pin_current_thread_near_gpu,
    pin_current_thread_to_node,
    read_numa_topology,
)

TWO_NODES = NumaTopology(
    gpu_to_node={0: 0, 1: 0, 2: 1, 3: 1},
    node_cpus={0: (0, 1), 1: (2, 3)},
    node_distances={0: {0: 10, 1: 21}, 1: {0: 21, 1: 10}},
)


def write_node(root: Path, node: int, cpulist: str, distances: str) -> None:
    node_dir = root / "devices" / "system" / "node" / f"node{node}"
    node_dir.mkdir(parents=True)
    (node_dir / "cpulist").write_text(cpulist + "\n")
    (node_dir / "distance").write_text(distances + "\n")


def write_gpu(root: Path, bus_id: str, node: int) -> None:
    device_dir = root / "bus" / "pci" / "devices" / bus_id
    device_dir.mkdir(parents=True)
    (device_dir / "numa_node").write_text(f"{node}\n")


@pytest.mark.parametrize(
    "text, expected",
    [("0-3,8,10-11", (0, 1, 2, 3, 8, 10, 11)), ("5\n", (5,)), ("", ())],
)
def test_parse_cpu_list(text: str, expected: tuple[int, ...]) -> None:
    assert parse_cpu_list(text) == expected


def test_topology_reads_gpus_cpus_and_distances(tmp_path: Path) -> None:
    write_node(tmp_path, 0, "0-3", "10 21")
    write_node(tmp_path, 1, "4-7", "21 10")
    write_gpu(tmp_path, "0000:3b:00.0", 0)
    write_gpu(tmp_path, "0000:af:00.0", 1)

    topology = read_numa_topology(
        # Bus ids from the CUDA runtime are upper case, sysfs names are not.
        {0: "0000:3B:00.0", 1: "0000:AF:00.0"},
        sysfs_root=str(tmp_path),
        allowed_cpus={0, 1, 2, 4, 5},
    )

    assert topology.node_of_gpu(0) == 0
    assert topology.node_of_gpu(1) == 1
    assert topology.nodes_with_gpus() == [0, 1]
    # Only CPUs the process may use are listed.
    assert topology.cpus_of(0) == (0, 1, 2)
    assert topology.cpus_of(1) == (4, 5)
    assert topology.nodes_by_distance(0) == [1]


def test_gpu_without_a_known_node_is_left_out(tmp_path: Path) -> None:
    write_node(tmp_path, 0, "0-3", "10")
    write_gpu(tmp_path, "0000:3b:00.0", -1)

    topology = read_numa_topology(
        {0: "0000:3B:00.0", 1: None, 2: "0000:ff:00.0"},
        sysfs_root=str(tmp_path),
        allowed_cpus={0},
    )

    assert topology.gpu_to_node == {}
    assert topology.nodes_with_gpus() == []


def test_nodes_by_distance_orders_nearest_first() -> None:
    topology = NumaTopology(
        node_cpus={n: (n,) for n in range(4)},
        node_distances={0: {0: 10, 1: 32, 2: 21, 3: 21}},
    )
    assert topology.nodes_by_distance(0) == [2, 3, 1]


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("auto", [(0, 0.5), (1, 0.5)]),
        ("1", [(1, 1.0)]),
        ("0,1", [(0, 0.5), (1, 0.5)]),
        ("0:60,1:40", [(0, 0.6), (1, 0.4)]),
        (" 1:25 , 0:75 ", [(1, 0.25), (0, 0.75)]),
    ],
)
def test_numa_nodes_spec_resolves_shares(
    spec: str, expected: list[tuple[int, float]]
) -> None:
    assert parse_numa_nodes_spec(spec, TWO_NODES) == expected


@pytest.mark.parametrize(
    "spec, message",
    [
        ("0:60,1", "every node or for none"),
        ("0:60,1:30", "sum to 100"),
        ("0:100,1:0", "above 0"),
        ("0,0", "twice"),
        ("0,5", "unknown NUMA nodes"),
        ("zero", "Invalid --l1-numa-nodes entry"),
        ("0:x", "Invalid --l1-numa-nodes entry"),
    ],
)
def test_numa_nodes_spec_rejects_bad_input(spec: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_numa_nodes_spec(spec, TWO_NODES)


def test_numa_nodes_auto_needs_a_gpu_with_a_known_node() -> None:
    no_gpus = NumaTopology(node_cpus={0: (0,)})
    with pytest.raises(ValueError, match="no GPU"):
        parse_numa_nodes_spec("auto", no_gpus)
    assert parse_numa_nodes_spec("0", no_gpus) == [(0, 1.0)]


def test_pin_current_thread_runs_the_thread_on_the_nodes_cpus() -> None:
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("Needs two CPUs to tell a pinned thread from an unpinned one")
    topology = NumaTopology(node_cpus={0: (allowed[0],), 1: (allowed[1],)})
    seen: dict[str, set[int]] = {}

    def run() -> None:
        pin_current_thread_to_node(topology, 1)
        seen["pinned"] = os.sched_getaffinity(0)
        pin_current_thread_to_node(topology, 0)
        seen["moved"] = os.sched_getaffinity(0)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()

    assert seen["pinned"] == {allowed[1]}
    assert seen["moved"] == {allowed[0]}
    # Pinning one thread does not move the caller.
    assert os.sched_getaffinity(0) == set(allowed)


def test_thread_follows_the_gpu_it_serves_and_ignores_an_unknown_gpu() -> None:
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("Needs two CPUs to tell a pinned thread from an unpinned one")
    topology = NumaTopology(
        gpu_to_node={0: 0, 1: 1},
        node_cpus={0: (allowed[0],), 1: (allowed[1],)},
    )
    seen: list[set[int]] = []

    def run() -> None:
        for gpu_index in (1, 0, 7):
            pin_current_thread_near_gpu(topology, gpu_index)
            seen.append(os.sched_getaffinity(0))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()

    # GPU 7 has no known node, so the thread stays where GPU 0 put it.
    assert seen == [{allowed[1]}, {allowed[0]}, {allowed[0]}]


@pytest.mark.skipif(
    not Path("/proc/self/numa_maps").exists(), reason="no NUMA policy reporting"
)
def test_bind_memory_to_node_sets_the_policy_of_an_untouched_range() -> None:
    size = 1 << 22
    region = mmap.mmap(-1, size)
    try:
        address = ctypes.addressof(ctypes.c_char.from_buffer(region))
        bind_memory_to_node(address, size, 0)
        mappings = [
            (int(line.split()[0], 16), line.split()[1])
            for line in Path("/proc/self/numa_maps").read_text().splitlines()
        ]
        assert max(m for m in mappings if m[0] <= address)[1] == "bind:0"
    finally:
        region.close()


def test_bind_memory_to_node_rejects_a_node_it_cannot_express() -> None:
    with pytest.raises(ValueError, match="outside"):
        bind_memory_to_node(0, 4096, 64)
