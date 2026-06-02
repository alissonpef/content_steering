import socket
import os
import time

import requests


DEFAULT_NODE_NAMES = ("delivery-node-1", "delivery-node-2", "delivery-node-3")
DEFAULT_PROBE_PATH = "/Eldorado/4sec/avc/manifest.mpd"

_dns_cache = {}


def _resolve_host(host: str) -> str:
    if host not in _dns_cache:
        try:
            _dns_cache[host] = socket.gethostbyname(host)
        except Exception:
            return host
    return _dns_cache[host]


def _service_url_and_headers(node_name: str, probe_path: str) -> tuple[str, dict]:
    namespace = os.environ.get("K8S_NAMESPACE", "default")
    host_suffix = os.environ.get(
        "K8S_SERVICE_HOST_SUFFIX", f".{namespace}.svc.cluster.local"
    )
    hostname = f"{node_name}{host_suffix}"
    ip = _resolve_host(hostname)
    path = probe_path if probe_path.startswith("/") else f"/{probe_path}"
    url = f"http://{ip}:80{path}"
    headers = {"Host": hostname}
    return url, headers


def measure_latency_ms(
    node_name: str,
    probe_path: str = DEFAULT_PROBE_PATH,
    timeout_seconds: float = 1.0,
) -> float:
    start = time.perf_counter()
    url, headers = _service_url_and_headers(node_name, probe_path)
    response = requests.get(
        url,
        headers=headers,
        timeout=timeout_seconds,
        stream=True,
    )
    response.raise_for_status()
    response.close()
    return (time.perf_counter() - start) * 1000.0


def get_all_latencies(
    nodes: list[str] | tuple[str, ...] | None = None,
    probe_path: str | None = None,
    timeout_seconds: float = 1.0,
) -> dict[str, float]:
    node_names = nodes or tuple(
        name.strip()
        for name in os.environ.get("DELIVERY_NODE_NAMES", "").split(",")
        if name.strip()
    )
    if not node_names:
        node_names = DEFAULT_NODE_NAMES
    path = probe_path or os.environ.get("REAL_LATENCY_PROBE_PATH", DEFAULT_PROBE_PATH)

    latencies = {}
    for node_name in node_names:
        try:
            latencies[node_name] = measure_latency_ms(
                node_name=node_name,
                probe_path=path,
                timeout_seconds=timeout_seconds,
            )
        except Exception:
            latencies[node_name] = 9999.0
    return latencies
