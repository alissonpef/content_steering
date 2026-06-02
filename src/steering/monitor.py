import threading
import time
import logging

try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config
except ImportError:
    k8s_client = None
    k8s_config = None

monitor_logger = logging.getLogger("KubernetesMonitor")
if not monitor_logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(name)s - %(levelname)s: %(message)s")
    _handler.setFormatter(_formatter)
    monitor_logger.addHandler(_handler)
    monitor_logger.setLevel(logging.WARNING)


class KubernetesMonitor:
    def __init__(
        self,
        interval_seconds: int = 2,
        namespace: str = "default",
        label_selector: str = "app=delivery-node",
    ):
        self.namespace = namespace
        self.label_selector = label_selector
        self.interval = interval_seconds
        self.container_stats = {}
        self._timer_thread = None
        self.running = False
        self.v1 = None

        if k8s_client is None or k8s_config is None:
            monitor_logger.critical(
                "Kubernetes Python client is not installed. KubernetesMonitor will not function."
            )
            return

        try:
            k8s_config.load_incluster_config()
            monitor_logger.info("Loaded in-cluster Kubernetes configuration.")
        except Exception:
            try:
                k8s_config.load_kube_config()
                monitor_logger.info("Loaded local Kubernetes configuration.")
            except Exception as exc:
                monitor_logger.critical(
                    f"Could not load Kubernetes configuration: {exc}. Monitor will not function."
                )
                return

        self.v1 = k8s_client.CoreV1Api()

    def start_collecting(self):
        if not self.v1:
            monitor_logger.error(
                "Kubernetes client not initialized. Pod discovery cannot be started."
            )
            return
        if not self.running:
            self.running = True
            self.collect_stats()
            self._timer_thread = threading.Thread(
                target=self._collection_loop, daemon=True
            )
            self._timer_thread.start()
            monitor_logger.info(
                f"Kubernetes pod discovery started (interval: {self.interval}s, selector: {self.label_selector})."
            )

    def _collection_loop(self):
        while self.running:
            self.collect_stats()
            for _ in range(self.interval * 10):
                if not self.running:
                    break
                time.sleep(0.1)
        monitor_logger.info("Kubernetes pod discovery loop ended.")

    def stop_collecting(self):
        monitor_logger.info("Requesting stop of Kubernetes pod discovery...")
        self.running = False
        if self._timer_thread and self._timer_thread.is_alive():
            self._timer_thread.join(timeout=self.interval + 1)
        self._timer_thread = None

    @staticmethod
    def _pod_logical_name(pod) -> str:
        labels = pod.metadata.labels or {}
        cache_id = labels.get("cache-id")
        if cache_id:
            return f"delivery-node-{cache_id}"
        return pod.metadata.name

    @staticmethod
    def _container_coordinates(container) -> tuple:
        lat, lon = None, None
        for env_var in container.env or []:
            if env_var.name == "LATITUDE":
                try:
                    lat = float(env_var.value)
                except (TypeError, ValueError):
                    pass
            elif env_var.name == "LONGITUDE":
                try:
                    lon = float(env_var.value)
                except (TypeError, ValueError):
                    pass
        return lat, lon

    def collect_stats(self):
        if not self.v1:
            return
        try:
            pods = self.v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=self.label_selector,
            )
        except Exception as exc:
            monitor_logger.error(f"Error listing Kubernetes pods: {exc}")
            return

        new_stats = {}
        for pod in getattr(pods, "items", []):
            if pod.status.phase != "Running" or not pod.status.pod_ip:
                continue
            lat, lon = None, None
            if pod.spec.containers:
                lat, lon = self._container_coordinates(pod.spec.containers[0])
            logical_name = self._pod_logical_name(pod)
            new_stats[logical_name] = [
                {
                    "cpu_usage": 0.0,
                    "mem_usage": 0.0,
                    "rx_bytes": 0,
                    "tx_bytes": 0,
                    "rate_rx_bytes": 0,
                    "rate_tx_bytes": 0,
                    "ip_address": pod.status.pod_ip,
                    "latitude": lat,
                    "longitude": lon,
                    "pod_name": pod.metadata.name,
                }
            ]
        self.container_stats = new_stats

    def get_nodes(self) -> list:
        nodes = []
        current_stats_snapshot = dict(self.container_stats)
        for name, stats_history in current_stats_snapshot.items():
            if stats_history:
                ip = stats_history[-1].get("ip_address", "N/A")
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
    monitor_logger.info("Starting KubernetesMonitor test...")
    monitor = KubernetesMonitor(interval_seconds=5)
    monitor.start_collecting()
    try:
        for i in range(2):
            time.sleep(5)
            monitor_logger.info(f"Active nodes detected: {monitor.get_nodes()}")
            monitor_logger.info(f"Node coordinates: {monitor.get_node_coordinates()}")
    except KeyboardInterrupt:
        monitor_logger.info("Test interrupted.")
    finally:
        monitor.stop_collecting()
        monitor_logger.info("KubernetesMonitor test finished.")
