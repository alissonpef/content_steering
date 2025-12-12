import logging

selector_logger = logging.getLogger("SelectorStrategies")

class Selector:
    def __init__(self, monitor=None, latency_oracle=None):
        self.monitor = monitor
        self.latency_oracle = latency_oracle
        self.nodes = []

    def initialize(self, arms_names: list):
        self.nodes = [str(arm) for arm in arms_names if arm is not None] if arms_names else []
        selector_logger.debug(f"Selector {self.__class__.__name__} initialized with nodes: {self.nodes}")

    def select_arm(self, **kwargs) -> list:
        raise NotImplementedError

    def update(self, chosen_arm_name: str, feedback_value: float, **kwargs):
        pass
