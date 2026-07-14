from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.steering.server import SteeringServer, fastapi_app


@pytest.mark.asyncio
@patch("src.steering.server.log_data_to_csv")
async def test_coords_update_does_not_overwrite_oracle_latency(mock_log):
    fastapi_app.routes.clear()
    mock_monitor = MagicMock()
    server = SteeringServer(monitor_ref=mock_monitor)
    server.current_strategy_name = "ucb1"
    server.last_real_latencies = {"delivery-node-1": 15.0}
    mock_selector = MagicMock()
    mock_selector.update = MagicMock()
    server.selector_instance = mock_selector
    server._handle_rl_feedback = AsyncMock(return_value="OK")
    server._build_log_base = MagicMock(return_value={})
    client_rt = 250.0

    class FakeRequest:
        def __init__(self):
            self.client = MagicMock(host="127.0.0.1")

        async def json(self):
            return {
                "time": 1,
                "lat": -23.0,
                "long": -47.0,
                "rt": client_rt,
                "server_used": "delivery-node-1",
                "decision_id": "decision-123",
            }

    coords_route = None
    for route in server.app.routes:
        if route.path == "/coords" and "POST" in route.methods:
            coords_route = route
            break
    assert coords_route is not None
    await coords_route.endpoint(FakeRequest())
    assert server.last_real_latencies["delivery-node-1"] == 15.0
    server._handle_rl_feedback.assert_called_once()
    called_args, called_kwargs = server._handle_rl_feedback.call_args
    assert called_args[0] == "delivery-node-1"
    assert called_args[1] == 250.0
