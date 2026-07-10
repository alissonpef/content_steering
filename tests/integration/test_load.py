import pytest
import asyncio
import httpx
from src.steering.server import fastapi_app, SteeringServer
from unittest.mock import MagicMock
import uuid
import os


@pytest.fixture
def mock_monitor():
    monitor = MagicMock()
    monitor.get_nodes.return_value = [
        ("node1", "ip1"),
        ("node2", "ip2"),
        ("node3", "ip3"),
    ]
    return monitor


@pytest.fixture(autouse=True)
def setup_server(mock_monitor):
    server = SteeringServer(monitor_ref=mock_monitor, gateway_mode=False)
    server.current_strategy_name = "ucb1"
    server.active_log_filename = os.path.devnull
    server._initialize_selector_if_needed()


@pytest.mark.asyncio
async def test_concurrent_steering_requests():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fastapi_app), base_url="http://test"
    ) as ac:
        await ac.post("/reset_simulation", json={"strategy": "ucb1"})
        tasks = []
        for i in range(100):
            tasks.append(ac.get(f"/node1/manifest.mpd?_DASH_pathway=true&req_id={i}"))
        responses = await asyncio.gather(*tasks)
        assert all((r.status_code == 200 for r in responses))
        assert all(("DECISION-ID" in r.json() for r in responses))


@pytest.mark.asyncio
async def test_concurrent_coords_updates():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fastapi_app), base_url="http://test"
    ) as ac:
        await ac.post("/reset_simulation", json={"strategy": "ucb1"})
        payload = {
            "time": 10.0,
            "lat": -23.0,
            "long": -47.0,
            "rt": 15.0,
            "server_used": "node1",
            "decision_id": str(uuid.uuid4()),
            "stall_time": 0,
            "spam_target": "none",
        }
        tasks = []
        for _ in range(50):
            tasks.append(ac.post("/coords", json=payload))
        responses = await asyncio.gather(*tasks)
        assert all((r.status_code == 200 for r in responses))
