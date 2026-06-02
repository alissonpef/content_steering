import logging

selector_logger = logging.getLogger("SelectorStrategies")


class Selector:
    def __init__(self, monitor=None):
        self.monitor = monitor
        self.nodes = []

    def initialize(self, arms_names: list[str]):
        self.nodes = list(arms_names) if arms_names else []
        selector_logger.debug(
            f"Selector {self.__class__.__name__} initialized with nodes: {self.nodes}"
        )

    def select_arm(self, **kwargs) -> list[str]:
        raise NotImplementedError

    def update(self, chosen_arm_name: str, feedback_value: float, **kwargs):
        pass
