import math
import random
import numpy as np
from .base import Selector, selector_logger


class LinUCBSelector(Selector):
    def __init__(
        self,
        d: int,
        alpha: float,
        gamma: float = 1.0,
        monitor=None,
        latency_oracle=None,
    ):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)
        self.d_env = d
        self.alpha = alpha
        self.n_arms: int = 0
        self.d_total: int = 0
        self.A = None
        self.b = None
        self.arm_index: dict = {}
        self.pull_counts: dict = {}
        self.total_pulls: int = 0
        selector_logger.info(
            f"LinUCBSelector initialised (shared+arm-indicator): "
            f"d_env={d}, alpha={alpha}"
        )

    def initialize(self, arms_names: list):
        super().initialize(arms_names)
        n = len(self.nodes)
        if n != self.n_arms or self.A is None:
            self.n_arms = n
            self.d_total = n + self.d_env
            self.A = np.identity(self.d_total)
            self.b = np.zeros((self.d_total, 1))
            self.arm_index = {arm: i for i, arm in enumerate(self.nodes)}
            selector_logger.info(
                f"[LinUCB] Model initialised: {n} arms, "
                f"d_total={self.d_total} ({n}+{self.d_env})"
            )
        for arm in self.nodes:
            if arm not in self.pull_counts:
                self.pull_counts[arm] = 0

    def _augmented_context(self, arm: str, env_ctx: np.ndarray) -> np.ndarray:
        one_hot = np.zeros(self.n_arms)
        one_hot[self.arm_index[arm]] = 1.0
        return np.concatenate([one_hot, env_ctx]).reshape(-1, 1)

    def select_arm(self, **kwargs) -> list:
        contexts = kwargs.get("contexts")
        if not contexts:
            selector_logger.error("[LinUCB] 'contexts' not provided for select_arm.")
            return []

        if set(contexts.keys()) != set(self.nodes):
            self.initialize(list(contexts.keys()))

        for arm in self.nodes:
            if arm in contexts and self.pull_counts.get(arm, 0) == 0:
                other_arms = [a for a in self.nodes if a != arm]
                random.shuffle(other_arms)
                selector_logger.info(f"[LinUCB] Exploring untested arm: {arm}")
                return [arm] + other_arms

        ucb_scores: dict = {}
        for arm in self.nodes:
            if arm not in contexts:
                continue
            x = self._augmented_context(arm, contexts[arm])
            try:
                theta_hat = np.linalg.solve(self.A, self.b)
                v = np.linalg.solve(self.A, x)
            except np.linalg.LinAlgError:
                selector_logger.warning(
                    f"[LinUCB] Singular A; fallback for arm '{arm}'."
                )
                theta_hat = self.b.copy()
                v = x.copy()
            predicted_reward = float((theta_hat.T @ x).item())
            confidence = self.alpha * math.sqrt(max(0.0, float((x.T @ v).item())))
            ucb_scores[arm] = predicted_reward + confidence

        if not ucb_scores:
            return []
        return sorted(ucb_scores, key=ucb_scores.get, reverse=True)

    def update(self, chosen_arm_name: str, reward: float, **kwargs):
        context = kwargs.get("context")
        if context is None:
            selector_logger.error(
                f"[LinUCB] 'context' not provided for update of arm {chosen_arm_name}."
            )
            return
        if chosen_arm_name not in self.nodes:
            selector_logger.warning(
                f"[LinUCB] Update: Arm '{chosen_arm_name}' unknown. Ignoring."
            )
            return

        x = self._augmented_context(chosen_arm_name, context)
        self.A = self.A + x @ x.T
        self.b = self.b + reward * x

        self.pull_counts[chosen_arm_name] = self.pull_counts.get(chosen_arm_name, 0) + 1
        self.total_pulls += 1
        selector_logger.debug(
            f"[LinUCB] Updated arm '{chosen_arm_name}': "
            f"reward={reward:.4f}, pulls={self.pull_counts[chosen_arm_name]}, "
            f"total={self.total_pulls}"
        )

    @property
    def counts(self):
        return dict(self.pull_counts)

    @property
    def real_counts(self):
        return dict(self.pull_counts)

    @property
    def values(self):
        return dict(self.pull_counts)
