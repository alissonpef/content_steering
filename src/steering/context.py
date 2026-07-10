import time
import math
import numpy as np
from .config import MOVEMENT_THRESHOLD_KM, CLIENT_COORDS_UPDATE_INTERVAL_SEC, app_logger

last_client_coords = {"lat": None, "lon": None, "time": 0.0}
active_spam_targets: list[str] = []


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return 0.0
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def update_client_position(lat, lon) -> bool:
    global last_client_coords
    client_is_moving = False
    now = time.time()
    if lat is None or lon is None:
        return False
    if last_client_coords["lat"] is not None and last_client_coords["lon"] is not None:
        elapsed = now - last_client_coords["time"]
        if elapsed >= CLIENT_COORDS_UPDATE_INTERVAL_SEC:
            dist = calculate_haversine_distance(
                last_client_coords["lat"], last_client_coords["lon"], lat, lon
            )
            if dist > MOVEMENT_THRESHOLD_KM:
                client_is_moving = True
            last_client_coords["lat"] = lat
            last_client_coords["lon"] = lon
            last_client_coords["time"] = now
            app_logger.warning(
                f"[CONTEXT] Client coords updated to ({lat}, {lon}) (moving: {client_is_moving})"
            )
        else:
            app_logger.warning(
                f"[CONTEXT] Client coords update throttled. Elapsed: {elapsed:.2f}s < {CLIENT_COORDS_UPDATE_INTERVAL_SEC}s"
            )
    elif last_client_coords["lat"] is None:
        last_client_coords["lat"] = lat
        last_client_coords["lon"] = lon
        last_client_coords["time"] = now
        app_logger.warning(f"[CONTEXT] Client coords initialized to ({lat}, {lon})")
    return client_is_moving


def update_spam_target(target: str | list[str] | None):
    global active_spam_targets
    raw_targets = []
    if isinstance(target, list):
        raw_targets = [t for t in target if t]
    elif target:
        raw_targets = [target]
    normalized = []
    for t in raw_targets:
        if t == "No Spam":
            continue
        tl = t.lower()
        if "cache 1" in tl or "node-1" in tl or "node1" in tl:
            normalized.append("delivery-node-1")
        elif "cache 2" in tl or "node-2" in tl or "node2" in tl:
            normalized.append("delivery-node-2")
        elif "cache 3" in tl or "node-3" in tl or "node3" in tl:
            normalized.append("delivery-node-3")
        else:
            normalized.append(t)
    active_spam_targets.clear()
    active_spam_targets.extend(normalized)
    if active_spam_targets:
        app_logger.warning(f"[CONTEXT] Active spam targets: {active_spam_targets}")


def get_dynamic_penalty(node_name: str, monitor) -> float:
    penalty = 0.0
    if not last_client_coords.get("lat"):
        return penalty
    node_coords = monitor.get_node_coordinates() if monitor else {}
    coords = node_coords.get(node_name, {})
    if coords:
        distance_km = calculate_haversine_distance(
            last_client_coords["lat"],
            last_client_coords["lon"],
            coords.get("lat"),
            coords.get("lon"),
        )
        propagation_ms = distance_km / 200000.0 * 1000.0 * 2.0
        penalty += propagation_ms
        app_logger.warning(
            f"[CONTEXT] Node {node_name} propagation penalty: {penalty:.2f} ms (distance: {distance_km:.2f} km, coords: {coords})"
        )
    else:
        app_logger.warning(
            f"[CONTEXT] Node {node_name} has no coordinates in monitor! Coords dict keys: {list(node_coords.keys())}"
        )
    return penalty


def get_simple_context(
    normalized_name: str,
    latency: float,
    history: list,
    monitor,
    selector_instance,
    last_real_latencies: dict,
):
    recent_avg = float(np.mean(history[-5:])) if history else latency
    recent_std = float(np.std(history[-5:])) if len(history) > 1 else 0.0
    if len(history) >= 4:
        midpoint = len(history) // 2
        old_avg = float(np.mean(history[:midpoint]))
        new_avg = float(np.mean(history[midpoint:]))
        trend = (new_avg - old_avg) / max(1.0, old_avg)
    else:
        trend = 0.0
    node_coords = monitor.get_node_coordinates() if monitor else {}
    coords = node_coords.get(normalized_name, {})
    distance_km = calculate_haversine_distance(
        last_client_coords.get("lat"),
        last_client_coords.get("lon"),
        coords.get("lat"),
        coords.get("lon"),
    )
    propagation_ms = distance_km / 200000.0 * 1000.0 * 2.0
    t = time.localtime()
    time_of_day = (t.tm_hour + t.tm_min / 60.0) / 24.0
    counts = getattr(selector_instance, "counts", {}) or {}
    total_counts = max(1, sum(counts.values()) if counts else 1)
    popularity = counts.get(normalized_name, 0) / total_counts
    observed = 1.0 if normalized_name in last_real_latencies else 0.0
    is_spam_target = 1.0 if normalized_name in active_spam_targets else 0.0
    if is_spam_target:
        app_logger.warning(
            f"[CONTEXT] Node {normalized_name} flagged as spam target in context vector."
        )
    all_latencies = last_real_latencies or {}
    if all_latencies:
        all_lat_values = list(all_latencies.values())
        min_latency = min(all_lat_values)
        max_latency = max(all_lat_values)
        latency_spread_ratio = (
            min(1.0, min_latency / max_latency) if max_latency > 0 else 0.0
        )
        relative_performance = min(1.0, min_latency / max(latency, 1.0))
    else:
        latency_spread_ratio = 0.0
        relative_performance = 0.0
    return np.array(
        [
            1.0,
            min(1.0, propagation_ms / 300.0),
            min(1.0, distance_km / 12000.0),
            min(1.0, latency / 300.0),
            min(1.0, recent_std / 40.0),
            is_spam_target,
            time_of_day,
            min(1.0, get_dynamic_penalty(normalized_name, monitor) / 200.0),
            latency_spread_ratio,
            relative_performance,
            observed,
            popularity,
            min(1.0, recent_avg / 300.0),
            max(-1.0, min(1.0, trend)),
        ],
        dtype=float,
    )
