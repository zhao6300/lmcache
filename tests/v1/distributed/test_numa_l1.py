# SPDX-License-Identifier: Apache-2.0
"""NUMA-aware L1: placement by the requesting GPU's node, and spill when full."""

# Standard
from collections.abc import Callable, Iterator
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l1_manager.base import BaseL1Manager
from lmcache.v1.distributed.storage_controllers import write_policy
from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of
from lmcache.v1.system_detection import NumaTopology
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([1024])], dtypes=[torch.float32])
OBJECT_BYTES = 4096

# GPUs 0-1 on node 0, GPU 2 on node 1, GPU 3 on node 2. Node 2 is nearer to
# node 0 than node 1 is.
TOPOLOGY = NumaTopology(
    gpu_to_node={0: 0, 1: 0, 2: 1, 3: 2},
    node_cpus={0: (0,), 1: (1,), 2: (2,)},
    node_distances={
        0: {0: 10, 1: 32, 2: 21},
        1: {0: 32, 1: 10, 2: 21},
        2: {0: 21, 1: 21, 2: 10},
    },
)
NumaStoreFactory = Callable[..., tuple[StorageManager, tuple[BaseL1Manager, ...]]]


def key(index: int) -> ObjectKey:
    """Return a deterministic test key."""
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "numa-test", 0)


@pytest.fixture
def numa_store(monkeypatch: pytest.MonkeyPatch) -> Iterator[NumaStoreFactory]:
    """Create stores with one L1 per node; replace only host pinning and detection."""
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    monkeypatch.setattr(write_policy, "get_numa_topology", lambda: TOPOLOGY)
    stores: list[StorageManager] = []

    def create(
        objects_per_node: tuple[int, ...] = (2, 2, 2),
    ) -> tuple[StorageManager, tuple[BaseL1Manager, ...]]:
        configs = [
            L1ManagerConfig(
                L1MemoryManagerConfig(
                    size_in_bytes=count * OBJECT_BYTES,
                    use_lazy=False,
                    shm_name="",
                    numa_node=node,
                ),
                tag=f"numa{node}",
            )
            for node, count in enumerate(objects_per_node)
        ]
        store = StorageManager(
            StorageManagerConfig(
                l1_manager_config=configs[0],
                eviction_config=EvictionConfig("noop"),
                l1_manager_configs=configs,
                write_policy="numa",
            )
        )
        stores.append(store)
        return store, tuple(cast(BaseL1Manager, m) for m in store.l1_managers())

    yield create
    for store in reversed(stores):
        store.close()


def owner_tag(store: StorageManager, objects: dict) -> set[str]:
    """Return the tags of the L1s that hold the reserved objects."""
    by_id = {m.l1_manager_id: m.config.tag for m in store.l1_managers()}
    return {by_id[owner] for owner in l1_owners_of(objects).values()}


@pytest.mark.parametrize(
    "origin, expected_tag",
    [
        (0, "numa0"),
        (1, "numa0"),
        (2, "numa1"),
        (torch.device("cuda", 3), "numa2"),
        # No origin, or a GPU whose node is unknown, behaves like the default.
        (None, "numa0"),
        (9, "numa0"),
    ],
)
def test_store_goes_to_the_l1_of_the_gpus_node(
    numa_store: NumaStoreFactory, origin: torch.device | int | None, expected_tag: str
) -> None:
    store, _ = numa_store()
    objects = store.reserve_write([key(1)], LAYOUT, origin_device=origin)
    assert owner_tag(store, objects) == {expected_tag}


def test_full_l1_spills_to_the_nearest_node_first(
    numa_store: NumaStoreFactory,
) -> None:
    store, (node0, node1, node2) = numa_store((1, 1, 1))
    placed = []
    for index in range(1, 4):
        objects = store.reserve_write([key(index)], LAYOUT, origin_device=0)
        assert len(objects) == 1
        placed.append(owner_tag(store, objects))

    # Own node first, then node 2 (distance 21), then node 1 (distance 32).
    assert placed == [{"numa0"}, {"numa2"}, {"numa1"}]
    assert store.reserve_write([key(4)], LAYOUT, origin_device=0) == {}
    assert [m.get_memory_usage()[0] for m in (node0, node1, node2)] == [
        OBJECT_BYTES
    ] * 3


def test_a_batch_that_does_not_fit_moves_together(
    numa_store: NumaStoreFactory,
) -> None:
    store, (node0, _, node2) = numa_store((1, 1, 2))
    objects = store.reserve_write([key(1), key(2)], LAYOUT, origin_device=0)

    assert set(objects) == {key(1), key(2)}
    assert owner_tag(store, objects) == {"numa2"}
    assert node0.get_memory_usage()[0] == 0


def test_spilled_objects_complete_and_read_from_the_node_that_holds_them(
    numa_store: NumaStoreFactory,
) -> None:
    store, (node0, _, node2) = numa_store((1, 1, 1))
    for index in (1, 2):
        objects = store.reserve_write([key(index)], LAYOUT, origin_device=0)
        store.finish_write(list(objects), l1_owners_of(objects))

    assert node0.get_object_state(key(1)) is not None
    assert node2.get_object_state(key(2)) is not None
    assert node0.get_object_state(key(2)) is None


def test_numa_aware_reports_whether_l1s_are_bound(
    numa_store: NumaStoreFactory,
) -> None:
    store, _ = numa_store()
    assert store.numa_aware
