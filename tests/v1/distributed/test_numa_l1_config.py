# SPDX-License-Identifier: Apache-2.0
"""``--l1-numa-aware`` turns the L1 flags into one DRAM pool per NUMA node."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed import config as distributed_config
from lmcache.v1.distributed.config import StorageManagerConfig, parse_args
from lmcache.v1.system_detection import NumaTopology

GIB = 1 << 30
TOPOLOGY = NumaTopology(
    gpu_to_node={0: 0, 1: 0, 2: 1, 3: 1},
    node_cpus={0: (0,), 1: (1,)},
)
BASE = ["--l1-size-gb", "10", "--eviction-policy", "LRU"]


@pytest.fixture(autouse=True)
def two_node_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(distributed_config, "get_numa_topology", lambda: TOPOLOGY)


def numa_config(*extra: str) -> StorageManagerConfig:
    return parse_args([*BASE, "--l1-numa-aware", *extra])


def test_numa_aware_is_off_by_default() -> None:
    config = parse_args(BASE)
    assert len(config.l1_manager_configs) == 1
    assert config.l1_manager_config.memory_config.numa_node is None
    assert config.write_policy == "default"


def test_auto_gives_each_gpu_node_an_even_pool() -> None:
    config = numa_config()

    pools = [c.memory_config for c in config.l1_manager_configs]
    assert [c.tag for c in config.l1_manager_configs] == ["numa0", "numa1"]
    assert [p.numa_node for p in pools] == [0, 1]
    assert [p.size_in_bytes for p in pools] == [5 * GIB, 5 * GIB]
    assert all(p.shm_name == "" for p in pools)
    assert config.write_policy == "numa"


def test_percentages_split_the_total() -> None:
    config = numa_config("--l1-numa-nodes", "0:60,1:40")
    sizes = [c.memory_config.size_in_bytes for c in config.l1_manager_configs]
    assert sizes == [6 * GIB, 4 * GIB]


def test_rounding_remainder_goes_to_the_first_node() -> None:
    # 1 GiB + 3 bytes is not a multiple of 4096, so the total rounds down.
    config = parse_args(
        [
            "--l1-size-gb",
            str((GIB + 3) / GIB),
            "--eviction-policy",
            "LRU",
            "--l1-numa-aware",
            "--l1-numa-nodes",
            "0:33,1:67",
        ]
    )
    sizes = [c.memory_config.size_in_bytes for c in config.l1_manager_configs]
    assert all(size % 4096 == 0 for size in sizes)
    assert sum(sizes) == GIB
    assert sizes[0] >= int(GIB * 0.33) // 4096 * 4096


def test_lazy_initial_size_follows_the_same_shares() -> None:
    config = numa_config("--l1-numa-nodes", "0:75,1:25", "--l1-init-size-gb", "8")
    inits = [c.memory_config.init_size_in_bytes for c in config.l1_manager_configs]
    assert inits == [6 * GIB, 2 * GIB]


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--l1-numa-nodes", "0:60,1:30"], "sum to 100"),
        (["--l1-numa-nodes", "0,7"], "unknown NUMA nodes"),
        (["--l1-devdax-path", "/dev/dax0.0"], "needs DRAM L1"),
        (["--gds-l1-path", "/mnt/nvme"], "needs DRAM L1"),
        (["--l1-manager", '{"type": "DRAM", "tag": "x", "size_gb": 1}'], "either"),
    ],
)
def test_incompatible_input_is_rejected_at_startup(
    extra: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        numa_config(*extra)


def test_a_node_too_small_for_one_allocation_unit_is_rejected() -> None:
    with pytest.raises(ValueError, match="NUMA node 1 would get"):
        numa_config("--l1-numa-nodes", "0:99.99999,1:0.00001")
