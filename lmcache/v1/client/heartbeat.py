# SPDX-License-Identifier: Apache-2.0
"""Shared client heartbeat for LMCache multiprocess integrations."""

# Standard
from collections.abc import Callable
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.common.custom_types import PingResponse
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.periodic_thread import PeriodicThread, ThreadLevel, ThreadRunSummary

logger = init_logger(__name__)


def send_ping(
    req_client: RequestClient,
    timeout: float,
    instance_id: int | None = None,
) -> PingResponse | None:
    """Send one MP server health probe.

    Args:
        req_client: Request client used to send the probe.
        timeout: Maximum seconds to wait for the response.
        instance_id: Worker registration or scheduler subscription to refresh,
            or ``None`` for an untracked probe.

    Returns:
        The server's response when it answers, otherwise ``None``.
    """
    try:
        return req_client.ping(instance_id).result(timeout=timeout)
    except TimeoutError:
        return None
    except Exception:
        logger.debug("LMCache heartbeat ping failed", exc_info=True)
        return None


class HeartbeatThread(PeriodicThread):
    """Probe an MP server and publish health transitions to an adapter.

    Each response carries the server's boot epoch. A changed epoch means the
    server restarted and lost every in-memory registration, so the recover
    callback runs even when no PING failed -- closing the window where a
    restart completes between two probes.

    The recover callback runs before health is published, allowing an adapter
    to restore server-side registration before new work is admitted. Adapters
    retain ownership of when the heartbeat starts and how recovery is done.
    """

    def __init__(
        self,
        req_client: RequestClient,
        health_event: threading.Event,
        interval: float = 10.0,
        instance_id: int | None = None,
        *,
        name: str = "lmcache-heartbeat",
        initial_epoch: str | None = None,
    ) -> None:
        """Initialize an MP heartbeat.

        Args:
            req_client: Request client used to send PING requests.
            health_event: Adapter-owned event that represents server health.
            interval: Seconds between probes and the timeout for each probe.
            instance_id: Worker registration or scheduler subscription to
                refresh, or ``None`` for an untracked probe.
            name: Background thread name.
            initial_epoch: Boot epoch already observed before the heartbeat
                started (e.g. a pre-registration probe), so the first PING can
                detect a restart that happened in between. ``None`` adopts the
                first response's epoch as the baseline.
        """
        super().__init__(
            name=name,
            interval=interval,
            level=ThreadLevel.CRITICAL,
        )
        self._req_client = req_client
        self._health_event = health_event
        self._instance_id = instance_id
        self._server_epoch = initial_epoch
        self._recover_callback: Callable[[], bool] = lambda: True
        self._unhealthy_callback: Callable[[], None] = lambda: None
        self._healthy_callback: Callable[[], bool] = self._set_health_event

    def register_recover_callback(self, callback: Callable[[], bool]) -> None:
        """Register work required before publishing a recovered server.

        Args:
            callback: Function returning whether recovery succeeded.
        """
        self._recover_callback = callback

    def register_unhealthy_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback for a healthy-to-unhealthy transition.

        Args:
            callback: Function that publishes adapter-specific unhealthy state.
        """
        self._unhealthy_callback = callback

    def register_healthy_callback(self, callback: Callable[[], bool]) -> None:
        """Register a callback that publishes adapter-specific healthy state.

        Args:
            callback: Function returning whether health was published.
        """
        self._healthy_callback = callback

    def stop_and_wait(self) -> None:
        """Stop and wait for an in-flight callback without a join timeout."""
        self.stop()
        thread = self._thread
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join()

    def _set_health_event(self) -> bool:
        self._health_event.set()
        return True

    def _publish_unhealthy(self) -> None:
        try:
            self._unhealthy_callback()
        finally:
            self._health_event.clear()

    def _execute(self) -> ThreadRunSummary:
        was_healthy = self._health_event.is_set()
        response = send_ping(
            self._req_client,
            timeout=self.interval,
            instance_id=self._instance_id,
        )
        if self.stop_requested:
            return ThreadRunSummary(
                success=True,
                message="stop requested; skipping health update",
            )

        healthy = response is not None
        restarted = False
        if response is not None:
            epoch = response.server_epoch
            previous_epoch = self._server_epoch
            restarted = previous_epoch is not None and epoch != previous_epoch
            self._server_epoch = epoch
            if restarted:
                logger.warning(
                    "LMCache server restarted (epoch %s -> %s); "
                    "re-registering before admitting work",
                    previous_epoch,
                    epoch,
                )
                # The replacement server lost every registration. Clear health
                # before recovery, as on an unhealthy->healthy edge, so no
                # store/retrieve reaches it before re-registration.
                if self._health_event.is_set():
                    self._publish_unhealthy()

        if healthy and (restarted or not was_healthy):
            logger.warning(
                "LMCache server is healthy again, triggering recovery callback"
            )
            try:
                healthy = self._recover_callback()
            except Exception:
                logger.exception("LMCache heartbeat recovery callback failed")
                healthy = False

        if self.stop_requested:
            return ThreadRunSummary(
                success=True,
                message="stop requested; skipping health update",
            )

        if healthy:
            try:
                healthy = self._healthy_callback()
            except Exception:
                logger.exception("LMCache heartbeat healthy callback failed")
                healthy = False

        if not healthy and self._health_event.is_set():
            logger.warning("LMCache server is unhealthy — entering degraded mode")
            self._publish_unhealthy()
        elif healthy and (restarted or not was_healthy):
            logger.warning(
                "LMCache server is healthy again — resuming normal operation"
            )

        return ThreadRunSummary(
            success=True,
            message="healthy" if healthy else "unhealthy",
        )
