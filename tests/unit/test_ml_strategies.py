import pytest
from unittest.mock import MagicMock
from src.steering.strategies import PPOHybridSelector, SACHybridSelector


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


def get_dummy_contexts():
    return {"node1": [1.0] * 14, "node2": [0.5] * 14, "node3": [0.1] * 14}


def get_dummy_latencies():
    return {"node1": 10.0, "node2": 50.0, "node3": 200.0}


def test_ppo_initialization(mock_monitor, nodes):
    selector = PPOHybridSelector(monitor=mock_monitor, random_state=42)
    selector.initialize(nodes)
    assert set(selector.nodes) == set(nodes)


def test_ppo_determinism(mock_monitor, nodes):
    selector_a = PPOHybridSelector(monitor=mock_monitor, random_state=42)
    selector_a.initialize(nodes)
    selector_b = PPOHybridSelector(monitor=mock_monitor, random_state=42)
    selector_b.initialize(nodes)
    contexts = get_dummy_contexts()
    latencies = get_dummy_latencies()
    decision_a = selector_a.select_arm(
        contexts=contexts, latencies=latencies, decision_id="id1"
    )
    decision_b = selector_b.select_arm(
        contexts=contexts, latencies=latencies, decision_id="id2"
    )
    assert decision_a == decision_b


def test_ppo_nan_inf_sanity(mock_monitor, nodes):
    selector = PPOHybridSelector(monitor=mock_monitor, random_state=42)
    selector.initialize(nodes)
    contexts = get_dummy_contexts()
    latencies = get_dummy_latencies()
    decision_id = "test-nan"
    decision = selector.select_arm(
        contexts=contexts, latencies=latencies, decision_id=decision_id
    )
    selector.update(
        decision[0],
        feedback_value=-9999999999.0,
        decision_id=decision_id,
        context=contexts[decision[0]],
        done=True,
    )
    next_decision = selector.select_arm(
        contexts=contexts, latencies=latencies, decision_id="test-nan2"
    )
    assert next_decision is not None
    assert len(next_decision) > 0


def test_sac_determinism(mock_monitor, nodes):
    selector_a = SACHybridSelector(monitor=mock_monitor, random_state=42)
    selector_a.initialize(nodes)
    selector_b = SACHybridSelector(monitor=mock_monitor, random_state=42)
    selector_b.initialize(nodes)
    contexts = get_dummy_contexts()
    latencies = get_dummy_latencies()
    decision_a = selector_a.select_arm(
        contexts=contexts, latencies=latencies, decision_id="id1"
    )
    decision_b = selector_b.select_arm(
        contexts=contexts, latencies=latencies, decision_id="id2"
    )
    assert decision_a == decision_b
