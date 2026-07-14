from unittest.mock import MagicMock

import numpy as np
import pytest

from src.steering.strategies.lin_ucb import LinUCBSelector
from src.steering.strategies.thompson import ThompsonSamplingSelector


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


@pytest.fixture
def dummy_contexts():
    return {
        "node1": np.array([0.1, 0.2]),
        "node2": np.array([0.3, 0.4]),
        "node3": np.array([0.5, 0.6]),
    }


def test_linucb_selector_initialization(mock_monitor, nodes):
    selector = LinUCBSelector(d=2, alpha=0.5, monitor=mock_monitor)
    selector.initialize(nodes)
    assert set(selector.nodes) == set(nodes)
    assert selector.A is not None
    assert selector.b is not None
    assert selector.n_arms == 3
    assert selector.d_total == 5


def test_linucb_selector_select_arm(mock_monitor, nodes, dummy_contexts):
    selector = LinUCBSelector(d=2, alpha=0.5, monitor=mock_monitor)
    selector.initialize(nodes)

    decision = selector.select_arm(contexts=dummy_contexts)
    assert isinstance(decision, list)
    assert len(decision) == 3
    assert decision[0] in nodes


def test_linucb_selector_update(mock_monitor, nodes, dummy_contexts):
    selector = LinUCBSelector(d=2, alpha=0.5, monitor=mock_monitor)
    selector.initialize(nodes)

    selector.update("node1", 50.0, context=dummy_contexts["node1"])
    assert selector.pull_counts["node1"] == 1
    assert selector.total_pulls == 1

    selector.update("node2", 10.0, context=dummy_contexts["node2"])
    assert selector.pull_counts["node2"] == 1
    assert selector.total_pulls == 2


def test_thompson_selector_initialization(mock_monitor, nodes):
    selector = ThompsonSamplingSelector(d=2, monitor=mock_monitor)
    selector.initialize(nodes)
    assert set(selector.nodes) == set(nodes)
    assert len(selector._means) == 3
    assert len(selector._precisions) == 3


def test_thompson_selector_select_arm(mock_monitor, nodes, dummy_contexts):
    selector = ThompsonSamplingSelector(d=2, monitor=mock_monitor)
    selector.initialize(nodes)

    decision = selector.select_arm(contexts=dummy_contexts)
    assert isinstance(decision, list)
    assert len(decision) == 3
    assert decision[0] in nodes


def test_thompson_selector_update(mock_monitor, nodes, dummy_contexts):
    selector = ThompsonSamplingSelector(d=2, monitor=mock_monitor)
    selector.initialize(nodes)

    selector.update("node1", 80.0, context=dummy_contexts["node1"])
    assert selector.counts["node1"] == 1
    assert selector.total_pulls == 1

    selector.update("node2", 20.0, context=dummy_contexts["node2"])
    assert selector.counts["node2"] == 1
    assert selector.total_pulls == 2
