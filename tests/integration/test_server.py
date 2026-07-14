from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.steering.server import SteeringServer, fastapi_app


@pytest.fixture
def mock_monitor():
    monitor = MagicMock()
    monitor.get_nodes.return_value = [
        ("node1", "ip1"),
        ("node2", "ip2"),
        ("node3", "ip3"),
    ]
    return monitor


@pytest.fixture
def steering_server(mock_monitor):
    server = SteeringServer(monitor_ref=mock_monitor, gateway_mode=False)
    server.current_strategy_name = "random"
    server._initialize_selector_if_needed()
    return server


@pytest.fixture
def client(steering_server):
    return TestClient(fastapi_app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_nodes" in data


def test_strategies_list(client):
    response = client.get("/strategies")
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data
    assert isinstance(data["strategies"], list)


def test_steering_decision(client, steering_server):
    client.post("/reset_simulation", json={"strategy": "random"})
    response = client.get("/node1/manifest.mpd?_DASH_pathway=true")
    assert response.status_code == 200
    data = response.json()
    assert "MEASURED-LATENCIES-MS" in data
    assert "DECISION-ID" in data


def test_coords_post_invalid_json(client):
    response = client.post("/coords", content="not-a-json")
    assert response.status_code == 422


def test_reset_simulation_unknown_strategy(client):
    response = client.post("/reset_simulation", json={"strategy": "invalid_strat"})
    assert response.status_code == 400
    assert "Unknown strategy" in response.json()["error"]
