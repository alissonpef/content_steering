from __future__ import annotations

import json
import math
import os
import typing
from dataclasses import dataclass

import numpy as np

from .base import Selector, selector_logger


@dataclass
class _Transition:
    state: np.ndarray
    abr_action: int
    steering_action: np.ndarray
    old_logprob: float
    value: float
    reward: float = 0.0
    done: bool = False


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    total = float(np.sum(exps))
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(values, 1.0 / max(1, values.size))
    return exps / total


def _normal_log_prob(sample: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    variance = np.square(std)
    log_std = np.log(std)
    return float(
        np.sum(
            -0.5 * (np.square(sample - mean) / variance + 2.0 * log_std + math.log(2.0 * math.pi))
        )
    )


class PPOHybridSelector(Selector):
    def __init__(
        self,
        hidden_dim: int = 64,
        learning_rate: float = 0.005,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        batch_size: int = 8,
        update_epochs: int = 10,
        reward_scale: float = 10.0,
        min_std: float = 0.1,
        max_std: float = 1.5,
        max_grad_norm: float = 1.0,
        random_state: int | None = None,
        quality_levels: list[int] | None = None,
        policy_path: str | None = None,
        monitor=None,
    ):
        super().__init__(monitor=monitor)
        self.hidden_dim = max(8, hidden_dim)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = max(1, batch_size)
        self.update_epochs = max(1, update_epochs)
        self.reward_scale = max(1e-06, reward_scale)
        self.min_std = max(0.001, min_std)
        self.max_std = max(self.min_std, max_std)
        self.max_grad_norm = max(1e-06, max_grad_norm)
        self.rng = np.random.default_rng(random_state)
        self.quality_levels = quality_levels or [0, 1, 2, 3, 4, 5]
        self.policy_path = policy_path
        self.counts: dict[str, int] = {}
        self.values: dict[str, float] = {}
        self.total_pulls = 0
        self._params: dict[str, np.ndarray] = {}
        self._adam_m: dict[str, np.ndarray] = {}
        self._adam_v: dict[str, np.ndarray] = {}
        self._adam_t = 0
        self._input_dim = 0
        self._context_dim = 14
        self._pending_transitions: dict[str, list[_Transition]] = {}
        self._buffer: list[_Transition] = []
        self._last_contexts: dict[str, np.ndarray] = {}
        self._last_latencies: dict[str, float] = {}
        self._last_action: dict[str, object] = {}
        self._last_selected_index = -1
        self._last_quality_index = 0
        self._last_reward_norm = 0.0
        if self.policy_path and os.path.exists(self.policy_path):
            self.load_policy(self.policy_path)

    def initialize(self, arms_names: list[str]):
        super().initialize(arms_names)
        self.counts = {arm: self.counts.get(arm, 0) for arm in self.nodes}
        self.values = {arm: self.values.get(arm, 0.0) for arm in self.nodes}
        if self._last_selected_index >= len(self.nodes):
            self._last_selected_index = -1

    def _context_dim_for_nodes(self) -> int:
        if self._last_contexts:
            first_ctx = next(iter(self._last_contexts.values()))
            return np.asarray(first_ctx, dtype=float).reshape(-1).size
        return self._context_dim

    def _ensure_model(self, input_dim: int, n_arms: int):
        if self._params:
            if (
                self._params["W1"].shape[0] == input_dim
                and self._params["W_abr"].shape[1] == len(self.quality_levels)
                and (self._params["W_mu"].shape[1] == n_arms)
            ):
                return
            selector_logger.info(
                f"[PPO] Reinitializing model for input_dim={input_dim}, arms={n_arms}."
            )
        scale = 1.0 / math.sqrt(max(1, input_dim))
        self._input_dim = input_dim
        self._params = {
            "W1": self.rng.normal(0.0, scale, size=(input_dim, self.hidden_dim)),
            "b1": np.zeros(self.hidden_dim, dtype=float),
            "W_abr": self.rng.normal(
                0.0,
                1.0 / math.sqrt(max(1, self.hidden_dim)),
                size=(self.hidden_dim, len(self.quality_levels)),
            ),
            "b_abr": np.zeros(len(self.quality_levels), dtype=float),
            "W_mu": self.rng.normal(
                0.0,
                1.0 / math.sqrt(max(1, self.hidden_dim)),
                size=(self.hidden_dim, n_arms),
            ),
            "b_mu": np.zeros(n_arms, dtype=float),
            "W_log_std": self.rng.normal(
                0.0,
                1.0 / math.sqrt(max(1, self.hidden_dim)),
                size=(self.hidden_dim, n_arms),
            ),
            "b_log_std": np.full(n_arms, math.log(0.5), dtype=float),
            "W_value": self.rng.normal(
                0.0, 1.0 / math.sqrt(max(1, self.hidden_dim)), size=(self.hidden_dim, 1)
            ),
            "b_value": np.zeros(1, dtype=float),
        }
        self._adam_m = {name: np.zeros_like(value) for name, value in self._params.items()}
        self._adam_v = {name: np.zeros_like(value) for name, value in self._params.items()}
        self._adam_t = 0

    def _refresh_nodes(self):
        if self.monitor:
            current_nodes = [name for name, _ in self.monitor.get_nodes() if name]
            if set(current_nodes) != set(self.nodes):
                self.initialize(current_nodes)

    def _build_context_bundle(self, kwargs):
        contexts = kwargs.get("contexts")
        latencies = kwargs.get("latencies")
        if contexts is not None:
            contexts = {
                str(arm): np.asarray(context, dtype=float).reshape(-1)
                for arm, context in contexts.items()
            }
        else:
            contexts = {
                arm: np.zeros(self._context_dim_for_nodes(), dtype=float) for arm in self.nodes
            }
            latencies = {arm: None for arm in self.nodes}
        return (contexts, latencies or {})

    def _build_state(
        self,
        contexts: dict[str, np.ndarray],
        latencies: dict[str, float | None] | dict[str, None] | dict[str, float],
    ) -> np.ndarray:
        context_dim = self._context_dim_for_nodes()
        self._context_dim = context_dim
        pieces: list[float] = []
        total_pulls = max(1, self.total_pulls)
        for arm in self.nodes:
            context = np.asarray(contexts.get(arm, np.zeros(context_dim)), dtype=float).reshape(-1)
            if context.size < context_dim:
                context = np.pad(context, (0, context_dim - context.size))
            elif context.size > context_dim:
                context = context[:context_dim]
            pieces.extend(context.tolist())
            latency = latencies.get(arm)
            if latency is None or not np.isfinite(latency):
                pieces.append(0.0)
            else:
                pieces.append(min(1.0, latency / 300.0))
            pieces.append(self.counts.get(arm, 0) / total_pulls)
        pieces.append(self._last_reward_norm)
        pieces.append(self._last_quality_index / max(1, len(self.quality_levels) - 1))
        last_selected = np.zeros(len(self.nodes), dtype=float)
        if 0 <= self._last_selected_index < len(self.nodes):
            last_selected[self._last_selected_index] = 1.0
        pieces.extend(last_selected.tolist())
        state = np.asarray(pieces, dtype=float)
        self._ensure_model(state.size, len(self.nodes))
        return state

    def _forward(self, state: np.ndarray):
        z1 = state @ self._params["W1"] + self._params["b1"]
        h1 = np.tanh(z1)
        abr_logits = h1 @ self._params["W_abr"] + self._params["b_abr"]
        mu = h1 @ self._params["W_mu"] + self._params["b_mu"]
        log_std = h1 @ self._params["W_log_std"] + self._params["b_log_std"]
        log_std = np.clip(log_std, math.log(self.min_std), math.log(self.max_std))
        value = (h1 @ self._params["W_value"] + self._params["b_value"])[0]
        return {
            "state": state,
            "z1": z1,
            "h1": h1,
            "abr_logits": abr_logits,
            "abr_probs": _softmax(abr_logits),
            "mu": mu,
            "log_std": log_std,
            "std": np.exp(log_std),
            "value": value,
        }

    def _logprob(
        self,
        abr_action: int,
        steering_action: np.ndarray,
        forward: dict[str, typing.Any],
    ) -> float:
        abr_probs = forward["abr_probs"]
        abr_prob = float(abr_probs[abr_action]) if 0 <= abr_action < abr_probs.size else 1e-12
        abr_prob = max(1e-12, abr_prob)
        return math.log(abr_prob) + _normal_log_prob(steering_action, forward["mu"], forward["std"])

    def _select_quality_index(self, abr_probs: np.ndarray, explore: bool) -> int:
        if explore:
            return self.rng.choice(len(abr_probs), p=abr_probs)
        return int(np.argmax(abr_probs))

    def select_arm(self, **kwargs) -> list[str]:
        self._refresh_nodes()
        contexts, latencies = self._build_context_bundle(kwargs)
        if not self.nodes:
            return []
        for arm in self.nodes:
            contexts.setdefault(arm, np.zeros(self._context_dim_for_nodes(), dtype=float))
            latencies.setdefault(arm, None)
        state = self._build_state(contexts, latencies)
        forward = self._forward(state)
        explore = bool(kwargs.get("explore", True))
        abr_action = self._select_quality_index(forward["abr_probs"], explore=explore)
        if explore:
            steering_action = forward["mu"] + forward["std"] * self.rng.normal(size=len(self.nodes))
        else:
            steering_action = forward["mu"].copy()
        steering_weights = _softmax(np.asarray(steering_action, dtype=float))
        ordered_indices = list(np.argsort(-steering_weights))
        ordered_nodes = [self.nodes[idx] for idx in ordered_indices]
        logprob = self._logprob(abr_action, np.asarray(steering_action, dtype=float), forward)
        self._last_selected_index = ordered_indices[0]
        self._last_quality_index = abr_action
        self._last_action = {
            "abr_action": abr_action,
            "abr_probability": float(forward["abr_probs"][abr_action]),
            "quality_level": self.quality_levels[abr_action],
            "steering_action": np.asarray(steering_action, dtype=float),
            "steering_weights": steering_weights,
            "ordered_nodes": ordered_nodes,
            "value": forward["value"],
            "logprob": logprob,
        }
        transition = _Transition(
            state=state,
            abr_action=abr_action,
            steering_action=np.asarray(steering_action, dtype=float),
            old_logprob=logprob,
            value=forward["value"],
        )
        decision_id = kwargs.get("decision_id")
        if decision_id:
            self._pending_transitions.setdefault(decision_id, []).append(transition)
        else:
            self._pending_transitions.setdefault(ordered_nodes[0], []).append(transition)
        return ordered_nodes

    def _discounted_returns(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        returns = np.zeros_like(rewards, dtype=float)
        running = 0.0
        for index in range(rewards.size - 1, -1, -1):
            if dones[index]:
                running = 0.0
            running = rewards[index] + self.gamma * running
            returns[index] = running
        return returns

    def _zero_grads(self):
        return {name: np.zeros_like(value) for name, value in self._params.items()}

    def _clip_gradients(self, grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        total_norm = math.sqrt(sum(float(np.sum(np.square(grad))) for grad in grads.values()))
        if total_norm <= self.max_grad_norm or total_norm == 0.0:
            return grads
        scale = self.max_grad_norm / total_norm
        return {name: grad * scale for name, grad in grads.items()}

    def _apply_adam(self, grads: dict[str, np.ndarray]):
        self._adam_t += 1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-08
        for name, grad in grads.items():
            self._adam_m[name] = beta1 * self._adam_m[name] + (1.0 - beta1) * grad
            self._adam_v[name] = beta2 * self._adam_v[name] + (1.0 - beta2) * np.square(grad)
            m_hat = self._adam_m[name] / (1.0 - beta1**self._adam_t)
            v_hat = self._adam_v[name] / (1.0 - beta2**self._adam_t)
            self._params[name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def _backprop_sample(self, sample: _Transition, adv: float, ret: float):
        forward = self._forward(sample.state)
        new_logprob = self._logprob(sample.abr_action, sample.steering_action, forward)
        ratio = math.exp(max(-20.0, min(20.0, new_logprob - sample.old_logprob)))
        use_unclipped = not (
            adv > 0.0
            and ratio > 1.0 + self.clip_ratio
            or (adv < 0.0 and ratio < 1.0 - self.clip_ratio)
        )
        policy_scale = adv * ratio if use_unclipped else 0.0
        probs = forward["abr_probs"]
        entropy_grad_discrete = self.entropy_coef * (
            math.log(max(1e-12, float(probs[sample.abr_action]))) + 1.0
        )
        abr_grad_logits = policy_scale * probs
        abr_grad_logits[sample.abr_action] -= policy_scale + entropy_grad_discrete
        diff = sample.steering_action - forward["mu"]
        std_sq = np.maximum(np.square(forward["std"]), self.min_std**2)
        mean_grad = policy_scale * (forward["mu"] - sample.steering_action) / std_sq
        log_std_grad = policy_scale * (1.0 - np.square(diff) / std_sq) - self.entropy_coef
        value_grad = self.value_coef * (forward["value"] - ret)
        d_h = (
            self._params["W_abr"] @ abr_grad_logits
            + self._params["W_mu"] @ mean_grad
            + self._params["W_log_std"] @ log_std_grad
            + self._params["W_value"] @ np.asarray([value_grad])
        )
        d_z1 = d_h * (1.0 - np.square(forward["h1"]))
        grads = self._zero_grads()
        grads["W_abr"] = np.outer(forward["h1"], abr_grad_logits)
        grads["b_abr"] = abr_grad_logits
        grads["W_mu"] = np.outer(forward["h1"], mean_grad)
        grads["b_mu"] = mean_grad
        grads["W_log_std"] = np.outer(forward["h1"], log_std_grad)
        grads["b_log_std"] = log_std_grad
        grads["W_value"] = np.outer(forward["h1"], np.asarray([value_grad]))
        grads["b_value"] = np.asarray([value_grad])
        grads["W1"] = np.outer(forward["state"], d_z1)
        grads["b1"] = d_z1
        return grads

    def _train_buffer(self):
        if not self._buffer:
            return
        rewards = np.asarray(
            [sample.reward / self.reward_scale for sample in self._buffer], dtype=float
        )
        dones = np.asarray([sample.done for sample in self._buffer], dtype=bool)
        returns = self._discounted_returns(rewards, dones)
        values = np.asarray([sample.value for sample in self._buffer], dtype=float)
        advantages = returns - values
        advantages = advantages - float(np.mean(advantages))
        adv_std = float(np.std(advantages))
        if adv_std > 1e-08:
            advantages = advantages / adv_std
        for _ in range(self.update_epochs):
            grads = self._zero_grads()
            for sample, adv, ret in zip(self._buffer, advantages, returns, strict=False):
                sample_grads = self._backprop_sample(sample, float(adv), float(ret))
                for name in grads:
                    grads[name] += sample_grads[name]
            scale = 1.0 / max(1, len(self._buffer))
            for name in grads:
                grads[name] *= scale
            grads = self._clip_gradients(grads)
            self._apply_adam(grads)
        self._buffer.clear()
        if self.policy_path:
            self.save_policy(self.policy_path)

    def update(self, chosen_arm_name: str, feedback_value: float, **kwargs):
        if chosen_arm_name not in self.nodes and self.monitor:
            current_nodes = [name for name, _ in self.monitor.get_nodes() if name]
            if chosen_arm_name in current_nodes:
                self.initialize(current_nodes)
        reward_value = feedback_value
        self._last_reward_norm = reward_value / self.reward_scale
        if chosen_arm_name in self.nodes:
            self.counts[chosen_arm_name] = self.counts.get(chosen_arm_name, 0) + 1
            self.total_pulls += 1
            count = self.counts[chosen_arm_name]
            previous = self.values.get(chosen_arm_name, 0.0)
            self.values[chosen_arm_name] = ((count - 1) * previous + reward_value) / count
        decision_id = kwargs.get("decision_id")
        if decision_id and decision_id in self._pending_transitions:
            pending_list = self._pending_transitions.pop(decision_id)
        else:
            pending_list = self._pending_transitions.get(chosen_arm_name)
        if not pending_list:
            return
        transition = pending_list.pop(0)
        transition.reward = reward_value
        transition.done = bool(kwargs.get("done", False))
        self._buffer.append(transition)
        if len(self._buffer) >= self.batch_size or kwargs.get("force_train", False):
            self._train_buffer()

    @property
    def real_counts(self):
        return dict(self.counts)

    @property
    def last_action(self):
        return dict(self._last_action)

    def policy_snapshot(self, explore: bool = False):
        if not self.nodes:
            return {}
        contexts = (
            self._last_contexts
            if self._last_contexts
            else {arm: np.zeros(self._context_dim_for_nodes(), dtype=float) for arm in self.nodes}
        )
        latencies = (
            self._last_latencies if self._last_latencies else {arm: None for arm in self.nodes}
        )
        state = self._build_state(contexts, latencies)
        forward = self._forward(state)
        abr_probs = forward["abr_probs"]
        abr_index = self._select_quality_index(abr_probs, explore=explore)
        steering_action = (
            forward["mu"]
            if not explore
            else forward["mu"] + forward["std"] * self.rng.normal(size=len(self.nodes))
        )
        steering_weights = _softmax(np.asarray(steering_action, dtype=float))
        ordered_nodes = [self.nodes[idx] for idx in np.argsort(-steering_weights)]
        return {
            "abr_index": abr_index,
            "quality_level": self.quality_levels[abr_index],
            "abr_probabilities": abr_probs.tolist(),
            "steering_mean": forward["mu"].tolist(),
            "steering_std": forward["std"].tolist(),
            "steering_weights": steering_weights.tolist(),
            "ordered_nodes": ordered_nodes,
            "value": forward["value"],
        }

    def save_policy(self, path: str | None = None):
        target_path = path or self.policy_path
        if not target_path:
            return None
        directory = os.path.dirname(target_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {name: value for name, value in self._params.items()}
        payload["metadata"] = np.asarray(
            json.dumps(
                {
                    "quality_levels": self.quality_levels,
                    "hidden_dim": self.hidden_dim,
                    "input_dim": self._input_dim,
                    "nodes": self.nodes,
                }
            ),
            dtype=object,
        )
        np.savez_compressed(target_path, **payload)
        return target_path

    def load_policy(self, path: str | None = None):
        target_path = path or self.policy_path
        if not target_path or not os.path.exists(target_path):
            return None
        data = np.load(target_path, allow_pickle=True)
        self._params = {name: data[name] for name in data.files if name != "metadata"}
        self._adam_m = {name: np.zeros_like(value) for name, value in self._params.items()}
        self._adam_v = {name: np.zeros_like(value) for name, value in self._params.items()}
        self._adam_t = 0
        if "metadata" in data.files:
            try:
                metadata = json.loads(str(data["metadata"].item()))
                self.quality_levels = metadata.get("quality_levels", self.quality_levels)
                self.hidden_dim = int(metadata.get("hidden_dim", self.hidden_dim))
                self._input_dim = int(metadata.get("input_dim", self._input_dim))
            except Exception:
                selector_logger.warning(
                    "[PPO] Failed to parse policy metadata while loading weights."
                )
        return target_path
