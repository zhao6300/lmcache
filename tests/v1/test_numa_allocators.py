# SPDX-License-Identifier: Apache-2.0
"""L1 allocators place their pages on the NUMA node they are given."""

# Standard
from pathlib import Path
import ctypes

# Third Party
import pytest

# First Party
from lmcache import device_ops
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.platform import current_device_spec

MIB = 1 << 20


def memory_policy_of(address: int) -> str:
    """Return the NUMA policy the kernel reports for the mapping at ``address``."""
    mappings = []
    for line in Path("/proc/self/numa_maps").read_text().splitlines():
        start, _, rest = line.partition(" ")
        mappings.append((int(start, 16), rest))
    return max(m for m in mappings if m[0] <= address)[1].split()[0]


def test_mixed_allocator_asks_for_pinned_memory_on_its_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = 4 * MIB
    backing = (ctypes.c_uint8 * size)()
    calls: list[tuple[str, int, int | None]] = []

    def alloc(allocated_size: int, node: int) -> int:
        calls.append(("alloc", allocated_size, node))
        return ctypes.addressof(backing)

    def free(ptr: int, freed_size: int) -> None:
        calls.append(("free", freed_size, None))

    monkeypatch.setattr(device_ops, "alloc_pinned_numa_ptr", alloc)
    monkeypatch.setattr(device_ops, "free_pinned_numa_ptr", free)

    allocator = MixedMemoryAllocator(size, numa_node=1)
    allocator.close()

    assert calls == [("alloc", size, 1), ("free", size, None)]


@pytest.mark.skipif(
    not Path("/proc/self/numa_maps").exists(), reason="no NUMA policy reporting"
)
def test_lazy_allocator_binds_its_pages_to_the_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pinning needs a GPU runtime; the policy under test does not.
    monkeypatch.setattr(
        type(current_device_spec), "is_pin_supported", property(lambda self: True)
    )
    monkeypatch.setattr(current_device_spec, "pin_memory", lambda *args: True)
    monkeypatch.setattr(current_device_spec, "unpin_memory", lambda ptr: True)

    allocator = LazyMemoryAllocator(64 * MIB, 128 * MIB, numa_node=0)
    try:
        address = allocator.get_underlying_buffer().data_ptr()
        assert memory_policy_of(address) == "bind:0"
    finally:
        allocator.close()
