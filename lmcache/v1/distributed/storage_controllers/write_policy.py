# SPDX-License-Identifier: Apache-2.0
"""
Write policy interface and default implementation for L1 placement decisions.

The write policy decides, for every key an engine store reserves, which L1
manager holds the object. ``StorageManager.reserve_write`` asks it once per
batch, before any allocation.
"""

# Standard
from abc import ABC, abstractmethod

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.system_detection import NumaTopology, get_numa_topology


class WritePolicy(ABC):
    """
    Abstract interface for L1 write placement.

    The write policy is called by the StorageManager to decide which L1
    manager each key is reserved in (select_write_targets).
    """

    @abstractmethod
    def select_write_targets(
        self,
        keys: list[ObjectKey],
        l1_managers: list[L1ManagerInterface],
    ) -> dict[int, list[ObjectKey]]:
        """
        Decide which L1 manager reserves each key.

        Args:
            keys: Keys an engine store is about to write.
            l1_managers: The L1 managers in placement order. A policy may
                read their memory tags, memory usage or configuration to
                decide.

        Returns:
            Mapping from ``l1_manager_id`` to the keys to reserve there.
            Every key appears exactly once across the mapping.
        """

    def select_write_targets_from(
        self,
        keys: list[ObjectKey],
        l1_managers: list[L1ManagerInterface],
        origin_device: torch.device | int | None,
    ) -> dict[int, list[ObjectKey]]:
        """
        Decide which L1 manager reserves each key for a store from one GPU.

        Policies that place by origin override this. The default ignores the
        origin and defers to :meth:`select_write_targets`.

        Args:
            keys: Keys an engine store is about to write.
            l1_managers: The L1 managers in placement order.
            origin_device: The GPU the store reads from, or ``None`` when the
                caller does not know it.

        Returns:
            Same contract as :meth:`select_write_targets`.
        """
        return self.select_write_targets(keys, l1_managers)

    def select_spill_targets(
        self,
        l1_manager_id: int,
        origin_device: torch.device | int | None,
        l1_managers: list[L1ManagerInterface],
    ) -> list[int]:
        """
        Name the L1 managers to try, in order, when one runs out of memory.

        Args:
            l1_manager_id: The L1 manager that returned out-of-memory.
            origin_device: The GPU the store reads from, or ``None``.
            l1_managers: The L1 managers in placement order.

        Returns:
            ``l1_manager_id`` values to retry the failed keys on, nearest
            first. Empty, the default, keeps an out-of-memory failure as is.
        """
        return []


# -----------------------------------------------------------------------------
# Registry: write policy name -> policy class
# -----------------------------------------------------------------------------

_WRITE_POLICY_REGISTRY: dict[str, type[WritePolicy]] = {}


def register_write_policy(
    name: str,
    policy_cls: type[WritePolicy],
) -> None:
    """
    Register a write policy class under a name.

    Each policy module should call this at import time.

    Args:
        name: Policy name (e.g. "default").
        policy_cls: A concrete WritePolicy subclass.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    if name in _WRITE_POLICY_REGISTRY:
        raise ValueError(f"Write policy already registered: {name!r}")
    _WRITE_POLICY_REGISTRY[name] = policy_cls


def get_registered_write_policies() -> list[str]:
    """Return the list of registered write policy names."""
    return list(_WRITE_POLICY_REGISTRY)


def create_write_policy(name: str) -> WritePolicy:
    """
    Create a write policy instance by name.

    Args:
        name: Registered policy name.

    Returns:
        A new WritePolicy instance.

    Raises:
        ValueError: If no policy is registered under the given name.
    """
    if name not in _WRITE_POLICY_REGISTRY:
        known = ", ".join(sorted(_WRITE_POLICY_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown write policy {name!r}. Known: {known}")
    return _WRITE_POLICY_REGISTRY[name]()


class DefaultWritePolicy(WritePolicy):
    """
    Default write policy: every key goes to the first L1 in placement order.
    """

    def select_write_targets(
        self,
        keys: list[ObjectKey],
        l1_managers: list[L1ManagerInterface],
    ) -> dict[int, list[ObjectKey]]:
        """Reserve all keys in the first L1 manager."""
        return {l1_managers[0].l1_manager_id: list(keys)}


class NumaWritePolicy(WritePolicy):
    """
    Place a store in the L1 of the NUMA node its GPU is on.

    When that L1 is out of memory the keys spill to the other nodes' L1s,
    nearest node first, so a full node does not fail a store another node can
    hold. A store without a known origin goes to the first L1, as in the
    default policy.
    """

    def __init__(self, topology: NumaTopology | None = None) -> None:
        """
        Args:
            topology: The host's NUMA topology; detected on first use if None.
        """
        self._topology = topology

    def select_write_targets(
        self,
        keys: list[ObjectKey],
        l1_managers: list[L1ManagerInterface],
    ) -> dict[int, list[ObjectKey]]:
        """Reserve all keys in the first L1 manager."""
        return {l1_managers[0].l1_manager_id: list(keys)}

    def select_write_targets_from(
        self,
        keys: list[ObjectKey],
        l1_managers: list[L1ManagerInterface],
        origin_device: torch.device | int | None,
    ) -> dict[int, list[ObjectKey]]:
        """Reserve all keys in the L1 on the origin GPU's NUMA node."""
        return {self._preferred(l1_managers, origin_device): list(keys)}

    def select_spill_targets(
        self,
        l1_manager_id: int,
        origin_device: torch.device | int | None,
        l1_managers: list[L1ManagerInterface],
    ) -> list[int]:
        """Return the other L1s ordered by NUMA distance from the full one."""
        by_id = {m.l1_manager_id: m for m in l1_managers}
        node = by_id[l1_manager_id].config.memory_config.numa_node
        if node is None:
            return []
        nodes_by_distance = self._get_topology().nodes_by_distance(node)
        others = [m for m in l1_managers if m.l1_manager_id != l1_manager_id]
        return [
            m.l1_manager_id
            for m in sorted(
                others,
                key=lambda m: self._rank(m, nodes_by_distance),
            )
        ]

    def _get_topology(self) -> NumaTopology:
        if self._topology is None:
            self._topology = get_numa_topology()
        return self._topology

    def _preferred(
        self,
        l1_managers: list[L1ManagerInterface],
        origin_device: torch.device | int | None,
    ) -> int:
        gpu_index = (
            origin_device.index
            if isinstance(origin_device, torch.device)
            else origin_device
        )
        if gpu_index is None:
            return l1_managers[0].l1_manager_id
        node = self._get_topology().node_of_gpu(gpu_index)
        for manager in l1_managers:
            if manager.config.memory_config.numa_node == node:
                return manager.l1_manager_id
        return l1_managers[0].l1_manager_id

    @staticmethod
    def _rank(manager: L1ManagerInterface, nodes_by_distance: list[int]) -> int:
        node = manager.config.memory_config.numa_node
        if node in nodes_by_distance:
            return nodes_by_distance.index(node)
        return len(nodes_by_distance)


register_write_policy("default", DefaultWritePolicy)
register_write_policy("numa", NumaWritePolicy)
