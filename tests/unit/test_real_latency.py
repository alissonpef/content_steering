from unittest.mock import patch, MagicMock
from src.steering.real_latency import (
    measure_latency_ms,
    get_all_latencies,
    _resolve_host,
    warmup_nodes,
)


@patch("src.steering.real_latency.socket.gethostbyname")
def test_resolve_host(mock_gethost):
    mock_gethost.return_value = "10.0.0.1"
    ip = _resolve_host("test.cluster.local")
    assert ip == "10.0.0.1"
    ip2 = _resolve_host("test.cluster.local")
    assert ip2 == "10.0.0.1"
    mock_gethost.assert_called_once()


@patch("src.steering.real_latency._session.get")
@patch("src.steering.real_latency._resolve_host")
def test_measure_latency_ms(mock_resolve, mock_get):
    mock_resolve.return_value = "10.0.0.2"
    mock_response = MagicMock()
    mock_get.return_value = mock_response
    latency = measure_latency_ms("node-1", "/test", n_samples=3)
    assert latency >= 0.0
    assert mock_get.call_count == 3
    assert mock_response.raise_for_status.call_count == 3
    assert mock_response.close.call_count == 3


@patch("src.steering.real_latency.measure_latency_ms")
def test_get_all_latencies(mock_measure):
    mock_measure.side_effect = [15.5, 20.0, Exception("Timeout")]
    nodes = ["node-1", "node-2", "node-3"]
    latencies = get_all_latencies(nodes=nodes)
    assert latencies["node-1"] == 15.5
    assert latencies["node-2"] == 20.0
    assert latencies["node-3"] == 9999.0


@patch("src.steering.real_latency._session.get")
def test_warmup_nodes(mock_get):
    mock_get.side_effect = [MagicMock(), MagicMock(), Exception("Timeout"), MagicMock()]
    warmup_nodes(["node-1", "node-2"])
    assert mock_get.call_count == 4
