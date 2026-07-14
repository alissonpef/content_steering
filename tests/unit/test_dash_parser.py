from unittest.mock import MagicMock

from src.steering.dash_parser import DashParser


def test_dash_parser_build():
    parser = DashParser()
    mock_request = MagicMock()
    mock_request.url.path = "/Eldorado/4sec/avc/manifest.mpd"
    nodes = [("delivery-node-1", 10.0), ("delivery-node-2", 20.0)]
    result = parser.build(
        target="delivery-node-1",
        nodes=nodes,
        uri="http://steering:5000",
        request=mock_request,
        host_suffix=".test.local",
        gateway_mode=False,
    )
    assert result["VERSION"] == 1
    assert result["RELOAD-URI"] == "http://steering:5000/Eldorado/4sec/avc/manifest.mpd"
    assert result["PATHWAY-PRIORITY"] == ["delivery-node-1", "delivery-node-2", "cloud"]
    clones = result["PATHWAY-CLONES"]
    assert len(clones) == 2
    assert clones[0]["ID"] == "delivery-node-1"
    assert clones[0]["URI-REPLACEMENT"]["HOST"] == "https://delivery-node-1.test.local"


def test_dash_parser_gateway_mode():
    parser = DashParser()
    mock_request = MagicMock()
    mock_request.url.path = "/Eldorado/4sec/avc/manifest.mpd"
    nodes = [("delivery-node-1", 10.0)]
    result = parser.build(
        target="delivery-node-1",
        nodes=nodes,
        uri="http://localhost:5000",
        request=mock_request,
        gateway_mode=True,
        request_host="localhost:5000",
    )
    clones = result["PATHWAY-CLONES"]
    assert clones[0]["URI-REPLACEMENT"]["HOST"] == "localhost:5000/node1"
