import math
import random
import numpy as np
from .base import Selector, selector_logger


class LinUCBSelector(Selector):
    def __init__(
        self,
        d: int,
        alpha: float,
        gamma: float = 0.95,
        monitor=None,
        latency_oracle=None,
    ):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)
        self.d = d
        self.alpha = alpha
        self.gamma = gamma
        self.A_shared = np.identity(self.d)
        self.b_shared = np.zeros((self.d, 1))
        self.pull_counts = {}
        self.total_pulls = 0
        selector_logger.info(
            f"LinUCBSelector initialised (shared model): d={d}, alpha={alpha}, gamma={gamma}"
        )

    def initialize(self, arms_names: list):
        super().initialize(arms_names)
        for arm in self.nodes:
            if arm not in self.pull_counts:
                self.pull_counts[arm] = 0
                selector_logger.debug(f"[LinUCB] Arm '{arm}' registered.")

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

        ucb_scores = {}
        for arm in self.nodes:
            if arm not in contexts:
                continue
            x_a = contexts[arm].reshape(-1, 1)
            try:
                theta_hat = np.linalg.solve(self.A_shared, self.b_shared)
                v = np.linalg.solve(self.A_shared, x_a)
            except np.linalg.LinAlgError:
                selector_logger.warning(
                    f"Shared A matrix singular — falling back to identity."
                )
                theta_hat = self.b_shared.copy()
                v = x_a.copy()
            predicted_reward = float(theta_hat.T.dot(x_a).item())
            confidence_bonus = self.alpha * math.sqrt(
                max(0.0, float(x_a.T.dot(v).item()))
            )
            ucb_scores[arm] = predicted_reward + confidence_bonus
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
                f"[LinUCB] Update: Arm {chosen_arm_name} unknown. Ignoring."
            )
            return
        x_chosen = context.reshape(-1, 1)

        if self.gamma < 1.0:
            eye = np.identity(self.d)
            self.A_shared = self.gamma * self.A_shared + (1.0 - self.gamma) * eye
            self.b_shared = self.gamma * self.b_shared

        self.A_shared += x_chosen.dot(x_chosen.T)
        self.b_shared += reward * x_chosen
        self.pull_counts[chosen_arm_name] = self.pull_counts.get(chosen_arm_name, 0) + 1
        self.total_pulls += 1
        selector_logger.debug(
            f"[LinUCB] Shared model updated via arm '{chosen_arm_name}', "
            f"reward={reward:.2f}, total_pulls={self.total_pulls}, gamma={self.gamma}."
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
