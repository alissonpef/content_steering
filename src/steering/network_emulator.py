import asyncio
import subprocess
import logging
from .context import (
    last_client_coords,
    active_spam_targets,
    calculate_haversine_distance,
)

emulator_logger = logging.getLogger("NetworkEmulator")
emulator_logger.setLevel(logging.INFO)
if not emulator_logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
    )
    _handler.setFormatter(_formatter)
    emulator_logger.addHandler(_handler)


class NetworkEmulatorDaemon:
    def __init__(self, monitor, interval=1.0):
        self.monitor = monitor
        self.interval = interval
        self.running = False
        self._task = None
        # Cache of last applied delay per pod to avoid unnecessary tc calls.
        # Key: pod_name, Value: delay_ms (int)
        self._applied_delays: dict[str, int] = {}

    def start(self):
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._loop())
            emulator_logger.info("NetworkEmulator daemon started.")

    def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()

    async def _loop(self):
        while self.running:
            try:
                await self._update_network_delays()
            except asyncio.CancelledError:
                break
            except Exception as e:
                emulator_logger.error(f"Error in NetworkEmulator loop: {e}")
            await asyncio.sleep(self.interval)

    async def _update_network_delays(self):
        if not last_client_coords.get("lat") or not self.monitor:
            return

        if not hasattr(self.monitor, "v1") or not self.monitor.v1:
            return

        try:
            pods = await asyncio.to_thread(
                self.monitor.v1.list_namespaced_pod,
                namespace=self.monitor.namespace,
                label_selector=self.monitor.label_selector,
            )
        except Exception as e:
            emulator_logger.error(f"Failed to list pods: {e}")
            return

        node_coords = self.monitor.get_node_coordinates()

        tasks = []
        for pod in getattr(pods, "items", []):
            if pod.status.phase != "Running":
                continue
            logical_name = self.monitor._pod_logical_name(pod)
            pod_name = pod.metadata.name
            coords = node_coords.get(logical_name, {})
            if not coords:
                continue

            distance_km = calculate_haversine_distance(
                last_client_coords["lat"],
                last_client_coords["lon"],
                coords.get("lat"),
                coords.get("lon"),
            )

            propagation_ms = (distance_km / 200_000.0) * 1000.0 * 2.0
            total_delay = 5.0 + propagation_ms

            is_spam = logical_name in active_spam_targets
            if is_spam:
                total_delay += 150.0

            delay_ms = max(5, int(total_delay))

            # Only issue `tc qdisc replace` when the target delay actually changed.
            # Issuing it every second (even with the same value) momentarily
            # disrupts the netem queue and causes latency measurement spikes.
            prev_delay = self._applied_delays.get(pod_name)
            if prev_delay == delay_ms:
                continue  # Nothing to do for this pod

            self._applied_delays[pod_name] = delay_ms
            emulator_logger.info(
                f"Updating tc delay: {logical_name} ({pod_name}) "
                f"{prev_delay}ms -> {delay_ms}ms (spam={is_spam})"
            )
            tasks.append(self._apply_tc_delay(pod_name, delay_ms, is_spam=is_spam))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _apply_tc_delay(
        self, pod_name: str, delay_ms: int, is_spam: bool = False
    ):
        """Apply tc/netem delay to a single pod container (non-blocking)."""
        if is_spam:
            tc_args = [
                "delay",
                f"{delay_ms}ms",
                "50ms",
                "loss",
                "5%",
                "distribution",
                "normal",
            ]
        else:
            tc_args = ["delay", f"{delay_ms}ms", "1ms", "distribution", "normal"]

        cmd = [
            "kubectl",
            "exec",
            pod_name,
            "-c",
            "caddy",
            "--",
            "tc",
            "qdisc",
            "replace",
            "dev",
            "eth0",
            "root",
            "netem",
        ] + tc_args
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception as e:
            emulator_logger.warning(f"tc failed for {pod_name}: {e}")
