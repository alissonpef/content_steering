import os
import socket
import time

import requests

DEFAULT_NODE_NAMES = ("delivery-node-1", "delivery-node-2", "delivery-node-3")
DEFAULT_PROBE_PATH = "/Eldorado/4sec/avc/manifest.mpd"
_DNS_CACHE_TTL_SECONDS = 60.0
_dns_cache: dict[str, tuple[str, float]] = {}


def _resolve_host(host: str) -> str:
    now = time.time()
    cached = _dns_cache.get(host)
    if cached is not None:
        ip, ts = cached
        if now - ts < _DNS_CACHE_TTL_SECONDS:
            return ip
    try:
        ip = socket.gethostbyname(host)
        _dns_cache[host] = (ip, now)
        return ip
    except Exception:
        return host


_session = requests.Session()


def _service_url_and_headers(node_name: str, probe_path: str) -> tuple[str, dict]:
    namespace = os.environ.get("K8S_NAMESPACE", "default")
    host_suffix = os.environ.get("K8S_SERVICE_HOST_SUFFIX", f".{namespace}.svc.cluster.local")
    hostname = f"{node_name}{host_suffix}"
    ip = _resolve_host(hostname)
    path = probe_path if probe_path.startswith("/") else f"/{probe_path}"
    url = f"http://{ip}:80{path}"
    headers = {"Host": hostname}
    return (url, headers)


def warmup_nodes(nodes: list[str], probe_path: str = DEFAULT_PROBE_PATH):
    for node_name in nodes:
        url, headers = _service_url_and_headers(node_name, probe_path)
        for _ in range(2):
            try:
                r = _session.get(url, headers=headers, timeout=1.5, stream=True)
                r.close()
            except Exception:
                pass


def measure_latency_ms(
    node_name: str,
    probe_path: str = DEFAULT_PROBE_PATH,
    timeout_seconds: float = 1.0,
    n_samples: int = 3,
) -> float:
    url, headers = _service_url_and_headers(node_name, probe_path)
    samples: list[float] = []
    for i in range(n_samples):
        t0 = time.perf_counter()
        response = _session.get(url, headers=headers, timeout=timeout_seconds, stream=True)
        response.raise_for_status()
        response.close()
        elapsed = (time.perf_counter() - t0) * 1000.0
        if i == 0 and n_samples > 1:
            continue
        samples.append(elapsed)
    return sum(samples) / len(samples) if samples else 9999.0


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
                node_name=node_name, probe_path=path, timeout_seconds=timeout_seconds
            )
        except Exception:
            latencies[node_name] = 9999.0
    return latencies
