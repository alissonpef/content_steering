import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.steering.network_emulator import NetworkEmulatorDaemon


@pytest.fixture
def emulator():
    mock_monitor = MagicMock()
    mock_monitor.namespace = "default"
    mock_monitor.label_selector = "app=delivery-node"
    return NetworkEmulatorDaemon(mock_monitor, interval=0.1)


@pytest.mark.asyncio
async def test_emulator_lifecycle(emulator):
    emulator.start()
    assert emulator.running is True
    assert emulator._task is not None

    emulator.stop()
    assert emulator.running is False


@pytest.mark.asyncio
@patch("src.steering.network_emulator.last_client_coords", {"lat": -23.0, "lon": -47.0})
@patch("src.steering.network_emulator.active_spam_targets", set())
@patch("src.steering.network_emulator.calculate_haversine_distance")
@patch("asyncio.create_subprocess_exec")
async def test_update_network_delays(mock_exec, mock_haversine, emulator):
    mock_haversine.return_value = 100.0

    mock_proc = AsyncMock()
    mock_exec.return_value = mock_proc

    mock_pod = MagicMock()
    mock_pod.status.phase = "Running"
    mock_pod.metadata.name = "node-1-pod"

    mock_pods_list = MagicMock()
    mock_pods_list.items = [mock_pod]

    emulator.monitor.v1.list_namespaced_pod = MagicMock(return_value=mock_pods_list)
    emulator.monitor._pod_logical_name = MagicMock(return_value="delivery-node-1")
    emulator.monitor.get_node_coordinates = MagicMock(
        return_value={"delivery-node-1": {"lat": -24.0, "lon": -48.0}}
    )

    await emulator._update_network_delays()

    mock_haversine.assert_called()

    mock_exec.assert_called_once()
    args = mock_exec.call_args[0]
    assert "kubectl" in args
    assert "tc" in args
    
    # Second call should not trigger another kubectl exec because delay hasn't changed
    await emulator._update_network_delays()
    mock_exec.assert_called_once()
