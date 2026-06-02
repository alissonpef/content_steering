import math

import numpy as np

from .base import Selector, selector_logger


class ThompsonSamplingSelector(Selector):
    def __init__(
        self,
        d: int = 14,
        alpha: float = 0.8,
        reward_scale: float = 10.0,
        prior_precision: float = 1.0,
        learning_rate: float = 0.75,
        update_steps: int = 1,
        min_precision: float = 1e-3,
        random_state: int | None = None,
        monitor=None,
    ):
        super().__init__(monitor=monitor)
        self.context_dim = max(1, d)
        self.alpha = alpha
        self.reward_scale = max(1e-6, reward_scale)
        self.prior_precision = max(min_precision, prior_precision)
        self.learning_rate = learning_rate
        self.update_steps = max(1, update_steps)
        self.min_precision = max(1e-6, min_precision)
        self.rng = np.random.default_rng(random_state)
        self.counts: dict = {}
        self.values: dict = {}
        self.total_pulls = 0
        self._means: dict[str, np.ndarray] = {}
        self._precisions: dict[str, np.ndarray] = {}

    def initialize(self, arms_names: list[str]):
        super().initialize(arms_names)
        self.counts = {arm: self.counts.get(arm, 0) for arm in self.nodes}
        self.values = {arm: self.values.get(arm, 0.0) for arm in self.nodes}
        new_means: dict[str, np.ndarray] = {}
        new_precisions: dict[str, np.ndarray] = {}
        for arm in self.nodes:
            mean = self._means.get(arm)
            precision = self._precisions.get(arm)
            if mean is None or mean.shape[0] != self.context_dim:
                mean = np.zeros(self.context_dim, dtype=float)
            if precision is None or precision.shape[0] != self.context_dim:
                precision = np.full(self.context_dim, self.prior_precision, dtype=float)
            new_means[arm] = mean
            new_precisions[arm] = precision
        self._means = new_means
        self._precisions = new_precisions

    def _prepare_context(self, context):
        if context is None:
            return None
        context_vector = np.asarray(context, dtype=float).reshape(-1)
        if context_vector.size == self.context_dim:
            return context_vector
        if context_vector.size < self.context_dim:
            return np.pad(context_vector, (0, self.context_dim - context_vector.size))
        selector_logger.warning(
            f"[ThompsonSampling] Context dimension {context_vector.size} "
            f"truncated to {self.context_dim}."
        )
        return context_vector[: self.context_dim]

    @staticmethod
    def _sigmoid(value: float) -> float:
        clipped = max(-35.0, min(35.0, value))
        return 1.0 / (1.0 + math.exp(-clipped))

    def _reward_to_target(self, reward: float) -> float:
        reward_value = max(0.0, reward)
        return 1.0 - math.exp(-reward_value / self.reward_scale)

    def select_arm(self, **kwargs) -> list[str]:
        contexts = kwargs.get("contexts")
        if not contexts:
            selector_logger.error(
                "[ThompsonSampling] 'contexts' not provided for select_arm."
            )
            return []

        if set(contexts.keys()) != set(self.nodes):
            self.initialize(list(contexts.keys()))

        if not self.nodes:
            return []

        unvisited_arms = [arm for arm in self.nodes if self.counts.get(arm, 0) == 0]
        if unvisited_arms:
            randomized_unvisited = list(self.rng.permutation(unvisited_arms))
            remaining_arms = [arm for arm in self.nodes if arm not in unvisited_arms]
            if remaining_arms:
                remaining_arms = list(self.rng.permutation(remaining_arms))
            return randomized_unvisited + remaining_arms

        sampled_scores = {}
        for arm in self.nodes:
            context_vector = self._prepare_context(contexts.get(arm))
            if context_vector is None:
                continue
            mean = self._means[arm]
            precision = np.maximum(self._precisions[arm], self.min_precision)
            sampled_weights = self.rng.normal(
                loc=mean, scale=np.sqrt(self.alpha / precision)
            )
            sampled_scores[arm] = self._sigmoid(float(sampled_weights @ context_vector))

        if not sampled_scores:
            return []
        return sorted(
            sampled_scores, key=lambda k: float(sampled_scores[k]), reverse=True
        )

    def update(self, chosen_arm_name: str, feedback_value: float, **kwargs):
        context = kwargs.get("context")
        if context is None:
            selector_logger.error(
                f"[ThompsonSampling] 'context' not provided for update of arm {chosen_arm_name}."
            )
            return

        if chosen_arm_name not in self.nodes:
            if self.monitor:
                nodes = [name for name, _ in self.monitor.get_nodes() if name]
                if chosen_arm_name in nodes:
                    self.initialize(nodes)
                if chosen_arm_name not in self.nodes:
                    selector_logger.warning(
                        f"[ThompsonSampling] Update: Arm {chosen_arm_name} not in self.nodes. Ignoring."
                    )
                    return
            else:
                selector_logger.warning(
                    f"[ThompsonSampling] Update: Arm {chosen_arm_name} not in self.nodes (no monitor). Ignoring."
                )
                return

        context_vector = self._prepare_context(context)
        if context_vector is None:
            return

        reward_value = max(0.0, feedback_value)
        target = self._reward_to_target(reward_value)
        mean = self._means[chosen_arm_name]
        precision = self._precisions[chosen_arm_name]
        x_sq = np.square(context_vector)
        updated_mean = mean.copy()
        curvature = 0.0

        for _ in range(self.update_steps):
            prediction = self._sigmoid(float(updated_mean @ context_vector))
            curvature = prediction * (1.0 - prediction)
            denominator = np.maximum(precision + curvature * x_sq, self.min_precision)
            updated_mean = (
                updated_mean
                - (self.learning_rate * (prediction - target) * context_vector)
                / denominator
            )

        updated_precision = np.maximum(precision + curvature * x_sq, self.min_precision)
        self._means[chosen_arm_name] = updated_mean
        self._precisions[chosen_arm_name] = updated_precision
        self.counts[chosen_arm_name] = self.counts.get(chosen_arm_name, 0) + 1
        self.total_pulls += 1
        count = self.counts[chosen_arm_name]
        previous_value = self.values.get(chosen_arm_name, 0.0)
        self.values[chosen_arm_name] = (
            ((count - 1) * previous_value) + reward_value
        ) / count

    @property
    def real_counts(self):
        return dict(self.counts)
