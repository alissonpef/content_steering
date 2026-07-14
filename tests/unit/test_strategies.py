import uuid
from unittest.mock import MagicMock

import pytest

from src.steering.strategies import (
    BestSelector,
    EpsilonGreedy,
    RandomSelector,
    UCB1Selector,
)


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
def nodes():
    return ["node1", "node2", "node3"]


def test_random_selector_initialization(mock_monitor, nodes):
    selector = RandomSelector(monitor=mock_monitor)
    selector.initialize(nodes)
    assert set(selector.nodes) == set(nodes)


def test_random_selector_select_arm(mock_monitor, nodes):
    selector = RandomSelector(monitor=mock_monitor)
    selector.initialize(nodes)
    decision = selector.select_arm()
    assert isinstance(decision, list)
    assert len(decision) > 0
    assert decision[0] in nodes


def test_epsilon_greedy_initialization(mock_monitor, nodes):
    selector = EpsilonGreedy(epsilon=0.2, counts={}, values={}, monitor=mock_monitor)
    selector.initialize(nodes)
    assert set(selector.nodes) == set(nodes)
    assert all(count == 0 for count in selector.counts.values())


def test_epsilon_greedy_update(mock_monitor, nodes):
    selector = EpsilonGreedy(epsilon=0.2, counts={}, values={}, monitor=mock_monitor)
    selector.initialize(nodes)
    selector.update("node1", 10.0)
    assert selector.counts["node1"] == 1
    assert selector.values["node1"] == 0.05
    selector.update("node1", 20.0)
    assert selector.counts["node1"] == 2
    assert selector.values["node1"] == pytest.approx(0.055)


def test_ucb1_selector_initialization(mock_monitor, nodes):
    selector = UCB1Selector(c=1.0, monitor=mock_monitor)
    selector.initialize(nodes)
    assert set(selector.nodes) == set(nodes)
    assert all(count == 0 for count in selector.counts.values())


def test_ucb1_selector_update_and_select(mock_monitor, nodes):
    selector = UCB1Selector(c=1.0, monitor=mock_monitor)
    selector.initialize(nodes)
    decision = selector.select_arm()
    assert decision[0] in nodes
    for _ in range(5):
        selector.update("node1", 100.0)
    selector.update("node2", 1.0)
    decision = selector.select_arm()
    assert decision[0] in nodes


def test_best_selector_select(mock_monitor, nodes):
    selector = BestSelector(monitor=mock_monitor)
    selector.initialize(nodes)
    contexts = {"node1": [1], "node2": [1], "node3": [1]}
    latencies = {"node1": 150.0, "node2": 50.0, "node3": 200.0}
    decision = selector.select_arm(
        contexts=contexts, latencies=latencies, decision_id=str(uuid.uuid4())
    )
    assert decision[0] == "node2"
