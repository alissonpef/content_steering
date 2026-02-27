import docker
import threading
import time
import logging

monitor_logger = logging.getLogger("ContainerMonitor")
if not monitor_logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(name)s - %(levelname)s: %(message)s")
    _handler.setFormatter(_formatter)
    monitor_logger.addHandler(_handler)
    monitor_logger.setLevel(logging.WARNING)


class ContainerMonitor:
    def __init__(
        self, interval_seconds: int = 2, network_name: str = "video-streaming_default"
    ):
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            monitor_logger.critical(
                f"Could not connect to Docker daemon: {e}. Monitor will not function."
            )
            self.client = None
        self.container_stats = {}
        self.interval = interval_seconds
        self.network_name = network_name
        self._timer_thread = None
        self.running = False

    def start_collecting(self):
        if not self.client:
            monitor_logger.error(
                "Docker client not initialized. Stats collection cannot be started."
            )
            return
        if not self.running:
            self.running = True
            self._timer_thread = threading.Thread(
                target=self._collection_loop, daemon=True
            )
            self._timer_thread.start()
            monitor_logger.info(
                f"Container stats collection started (interval: {self.interval}s)."
            )

    def _collection_loop(self):
        while self.running:
            self.collect_stats()
            for _ in range(self.interval * 10):
                if not self.running:
                    break
                time.sleep(0.1)
        monitor_logger.info("Stats collection loop ended.")

    def stop_collecting(self):
        monitor_logger.info("Requesting stop of stats collection...")
        self.running = False
        if self._timer_thread and self._timer_thread.is_alive():
            self._timer_thread.join(timeout=self.interval + 1)
        if self._timer_thread and self._timer_thread.is_alive():
            monitor_logger.warning(
                "Collection thread did not terminate in the expected time."
            )
        else:
            monitor_logger.info("Stats collection stopped.")
        self._timer_thread = None

    def collect_stats(self):
        if not self.client:
            return
        active_containers_this_cycle = set()
        try:
            for container in self.client.containers.list(all=True):
                if not container.name.startswith("video-streaming-cache-"):
                    continue
                active_containers_this_cycle.add(container.name)
                if container.status != "running":
                    if container.name in self.container_stats:
                        del self.container_stats[container.name]
                    continue
                try:
                    stats = container.stats(stream=False, one_shot=True)
                    attrs = container.attrs
                    networks = attrs.get("NetworkSettings", {}).get("Networks", {})
                    ip_address = networks.get(self.network_name, {}).get(
                        "IPAddress", "N/A"
                    )
                    if ip_address == "N/A" and networks:
                        first_network_key = next(iter(networks), None)
                        if first_network_key:
                            ip_address = networks[first_network_key].get(
                                "IPAddress", "N/A"
                            )
                    latitude, longitude = None, None
                    for env_var in attrs.get("Config", {}).get("Env", []):
                        if env_var.startswith("LATITUDE="):
                            latitude = float(env_var.split("=", 1)[1])
                        elif env_var.startswith("LONGITUDE="):
                            longitude = float(env_var.split("=", 1)[1])
                    prev_run_stats_list = self.container_stats.get(container.name, [])
                    prev_s_dict = prev_run_stats_list[-1] if prev_run_stats_list else {}
                    cpu_delta = stats["cpu_stats"]["cpu_usage"][
                        "total_usage"
                    ] - prev_s_dict.get("precpu_total_usage", 0)
                    sys_cpu_delta = stats["cpu_stats"][
                        "system_cpu_usage"
                    ] - prev_s_dict.get("presystem_cpu_usage", 0)
                    cpus = stats["cpu_stats"].get(
                        "online_cpus",
                        len(stats["cpu_stats"]["cpu_usage"].get("percpu_usage", [1])),
                    )
                    cpu_usage = (
                        (cpu_delta / sys_cpu_delta) * cpus * 100.0
                        if sys_cpu_delta > 0 and cpu_delta > 0
                        else 0.0
                    )
                    mem_stats = stats["memory_stats"]
                    mem_usage = (
                        (mem_stats["usage"] / mem_stats["limit"]) * 100.0
                        if mem_stats.get("limit", 0) > 0
                        else 0.0
                    )
                    net_stats = stats.get("networks", {}).get("eth0", {})
                    rx_bytes, tx_bytes = (
                        net_stats.get("rx_bytes", 0),
                        net_stats.get("tx_bytes", 0),
                    )
                    rate_rx_bytes = rx_bytes - prev_s_dict.get("rx_bytes", 0)
                    rate_tx_bytes = tx_bytes - prev_s_dict.get("tx_bytes", 0)
                    current_s_dict = {
                        "cpu_usage": cpu_usage,
                        "mem_usage": mem_usage,
                        "rx_bytes": rx_bytes,
                        "tx_bytes": tx_bytes,
                        "rate_rx_bytes": rate_rx_bytes,
                        "rate_tx_bytes": rate_tx_bytes,
                        "ip_address": ip_address,
                        "latitude": latitude,
                        "longitude": longitude,
                        "precpu_total_usage": stats["cpu_stats"]["cpu_usage"][
                            "total_usage"
                        ],
                        "presystem_cpu_usage": stats["cpu_stats"]["system_cpu_usage"],
                    }
                    if container.name not in self.container_stats:
                        self.container_stats[container.name] = []
                    self.container_stats[container.name].append(current_s_dict)
                    self.container_stats[container.name] = self.container_stats[
                        container.name
                    ][-10:]
                except Exception as e_inner:
                    monitor_logger.error(
                        f"Failed to process stats for {container.name}: {e_inner}",
                        exc_info=False,
                    )
        except docker.errors.APIError as e_outer:
            monitor_logger.error(
                f"Docker API error while listing containers: {e_outer}"
            )
            self.container_stats.clear()
            return
        for name_in_stats in list(self.container_stats.keys()):
            if name_in_stats not in active_containers_this_cycle:
                monitor_logger.info(
                    f"Container {name_in_stats} no longer active, removing stats."
                )
                del self.container_stats[name_in_stats]

    def getNodes(self) -> list:
        nodes = []
        current_stats_snapshot = dict(self.container_stats)
        for name, stats_history in current_stats_snapshot.items():
            if stats_history:
                latest_stat = stats_history[-1]
                ip = latest_stat.get("ip_address", "N/A")
                if ip != "N/A":
                    nodes.append((name, ip))
        return nodes

    def get_node_coordinates(self) -> dict:
        node_coords = {}
        current_stats_snapshot = dict(self.container_stats)
        for name, stats_history in current_stats_snapshot.items():
            if stats_history:
                latest_stat = stats_history[-1]
                lat = latest_stat.get("latitude")
                lon = latest_stat.get("longitude")
                ip = latest_stat.get("ip_address", "N/A")
                if lat is not None and lon is not None and ip != "N/A":
                    node_coords[name] = {"lat": lat, "lon": lon}
        return node_coords


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    monitor_logger.info("Starting ContainerMonitor test...")
    monitor = ContainerMonitor(interval_seconds=5, network_name="bridge")
    monitor.start_collecting()
    try:
        for i in range(2):
            time.sleep(5)
            monitor_logger.info(f"Active nodes detected: {monitor.getNodes()}")
            monitor_logger.info(f"Node coordinates: {monitor.get_node_coordinates()}")
    except KeyboardInterrupt:
        monitor_logger.info("Test interrupted.")
    finally:
        monitor.stop_collecting()
        monitor_logger.info("ContainerMonitor test finished.")
