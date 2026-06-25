from __future__ import annotations
import typing

import json
import math
import os
from collections import deque
from dataclasses import dataclass

import numpy as np

from .base import Selector, selector_logger


@dataclass
class _Transition:
    state: np.ndarray
    abr_action: int
    steering_raw: np.ndarray
    logprob: float
    reward: float | None = None
    next_state: np.ndarray | None = None
    done: bool = False


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    total = float(np.sum(exps))
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(values, 1.0 / max(1, values.size))
    return exps / total


def _one_hot(size: int, index: int) -> np.ndarray:
    vector = np.zeros(size, dtype=float)
    if 0 <= index < size:
        vector[index] = 1.0
    return vector


def _entropy(probs: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _normal_log_prob(sample: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    variance = np.square(std)
    log_std = np.log(std)
    return float(
        np.sum(
            -0.5
            * (
                np.square(sample - mean) / variance
                + 2.0 * log_std
                + math.log(2.0 * math.pi)
            )
        )
    )


class SACHybridSelector(Selector):
    def __init__(
        self,
        hidden_dim: int = 64,
        critic_hidden_dim: int = 64,
        actor_learning_rate: float = 5e-3,
        critic_learning_rate: float = 5e-3,
        gamma: float = 0.99,
        tau: float = 0.02,
        entropy_coef: float = 0.05,
        batch_size: int = 16,
        replay_size: int = 4096,
        update_steps: int = 10,
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
        self.critic_hidden_dim = max(8, critic_hidden_dim)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.entropy_coef = entropy_coef
        self.batch_size = max(1, batch_size)
        self.replay_size = max(64, replay_size)
        self.update_steps = max(1, update_steps)
        self.reward_scale = max(1e-6, reward_scale)
        self.min_std = max(1e-3, min_std)
        self.max_std = max(self.min_std, max_std)
        self.max_grad_norm = max(1e-6, max_grad_norm)
        self.rng = np.random.default_rng(random_state)
        self.quality_levels = quality_levels or [0, 1, 2, 3, 4, 5]
        self.policy_path = policy_path
        self.counts: dict[str, int] = {}
        self.values: dict[str, float] = {}
        self.total_pulls = 0
        self._actor_params: dict[str, np.ndarray] = {}
        self._critic1_params: dict[str, np.ndarray] = {}
        self._critic2_params: dict[str, np.ndarray] = {}
        self._critic1_target: dict[str, np.ndarray] = {}
        self._critic2_target: dict[str, np.ndarray] = {}
        self._actor_m: dict[str, np.ndarray] = {}
        self._actor_v: dict[str, np.ndarray] = {}
        self._critic1_m: dict[str, np.ndarray] = {}
        self._critic1_v: dict[str, np.ndarray] = {}
        self._critic2_m: dict[str, np.ndarray] = {}
        self._critic2_v: dict[str, np.ndarray] = {}
        self._actor_step = 0
        self._critic1_step = 0
        self._critic2_step = 0
        self._state_dim = 0
        self._critic_input_dim = 0
        self._context_dim = 14
        self._pending_transitions: dict[str, list[_Transition]] = {}
        self._replay = deque(maxlen=self.replay_size)
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

    def _ensure_params(self, state_dim: int, n_arms: int):
        critic_input_dim = state_dim + len(self.quality_levels) + n_arms
        if self._actor_params:
            if (
                self._actor_params["W1"].shape[0] == state_dim
                and self._actor_params["W_abr"].shape[1] == len(self.quality_levels)
                and self._actor_params["W_mu"].shape[1] == n_arms
                and self._critic1_params["W1"].shape[0] == critic_input_dim
            ):
                return
            selector_logger.info(
                f"[SAC] Reinitializing model for state_dim={state_dim}, arms={n_arms}."
            )
        actor_scale = 1.0 / math.sqrt(max(1, state_dim))
        critic_scale = 1.0 / math.sqrt(max(1, critic_input_dim))
        self._state_dim = state_dim
        self._critic_input_dim = critic_input_dim
        self._actor_params = {
            "W1": self.rng.normal(0.0, actor_scale, size=(state_dim, self.hidden_dim)),
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
        }

        def _critic_init():
            return {
                "W1": self.rng.normal(
                    0.0, critic_scale, size=(critic_input_dim, self.critic_hidden_dim)
                ),
                "b1": np.zeros(self.critic_hidden_dim, dtype=float),
                "W2": self.rng.normal(
                    0.0,
                    1.0 / math.sqrt(max(1, self.critic_hidden_dim)),
                    size=(self.critic_hidden_dim, 1),
                ),
                "b2": np.zeros(1, dtype=float),
            }

        self._critic1_params = _critic_init()
        self._critic2_params = _critic_init()
        self._critic1_target = {
            name: value.copy() for name, value in self._critic1_params.items()
        }
        self._critic2_target = {
            name: value.copy() for name, value in self._critic2_params.items()
        }
        self._actor_m = {
            name: np.zeros_like(value) for name, value in self._actor_params.items()
        }
        self._actor_v = {
            name: np.zeros_like(value) for name, value in self._actor_params.items()
        }
        self._critic1_m = {
            name: np.zeros_like(value) for name, value in self._critic1_params.items()
        }
        self._critic1_v = {
            name: np.zeros_like(value) for name, value in self._critic1_params.items()
        }
        self._critic2_m = {
            name: np.zeros_like(value) for name, value in self._critic2_params.items()
        }
        self._critic2_v = {
            name: np.zeros_like(value) for name, value in self._critic2_params.items()
        }
        self._actor_step = 0
        self._critic1_step = 0
        self._critic2_step = 0

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
                arm: np.zeros(self._context_dim_for_nodes(), dtype=float)
                for arm in self.nodes
            }
            latencies = {arm: None for arm in self.nodes}
        return contexts, latencies or {}

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
            context = np.asarray(
                contexts.get(arm, np.zeros(context_dim)), dtype=float
            ).reshape(-1)
            if context.size < context_dim:
                context = np.pad(context, (0, context_dim - context.size))
            elif context.size > context_dim:
                context = context[:context_dim]
            pieces.extend(context.tolist())
            latency = latencies.get(arm)
            pieces.append(
                0.0
                if latency is None or not np.isfinite(latency)
                else min(1.0, latency / 300.0)
            )
            pieces.append(self.counts.get(arm, 0) / total_pulls)
        pieces.append(self._last_reward_norm)
        pieces.append(self._last_quality_index / max(1, len(self.quality_levels) - 1))
        last_selected = np.zeros(len(self.nodes), dtype=float)
        if 0 <= self._last_selected_index < len(self.nodes):
            last_selected[self._last_selected_index] = 1.0
        pieces.extend(last_selected.tolist())
        state = np.asarray(pieces, dtype=float)
        self._ensure_params(state.size, len(self.nodes))
        return state

    def _actor_forward(self, state: np.ndarray):
        z1 = state @ self._actor_params["W1"] + self._actor_params["b1"]
        h1 = np.tanh(z1)
        abr_logits = h1 @ self._actor_params["W_abr"] + self._actor_params["b_abr"]
        mu = h1 @ self._actor_params["W_mu"] + self._actor_params["b_mu"]
        log_std = h1 @ self._actor_params["W_log_std"] + self._actor_params["b_log_std"]
        log_std = np.clip(log_std, math.log(self.min_std), math.log(self.max_std))
        return {
            "state": state,
            "z1": z1,
            "h1": h1,
            "abr_logits": abr_logits,
            "abr_probs": _softmax(abr_logits),
            "mu": mu,
            "log_std": log_std,
            "std": np.exp(log_std),
        }

    def _critic_forward(
        self,
        params: dict[str, np.ndarray],
        state: np.ndarray,
        abr_index: int,
        steering_raw: np.ndarray,
    ):
        action = np.concatenate(
            [
                _one_hot(len(self.quality_levels), abr_index),
                np.asarray(steering_raw, dtype=float).reshape(-1),
            ]
        )
        x = np.concatenate([state, action])
        z1 = x @ params["W1"] + params["b1"]
        h1 = np.tanh(z1)
        q = float((h1 @ params["W2"] + params["b2"])[0])
        return {"x": x, "z1": z1, "h1": h1, "q": q, "action": action}

    def _critic_backward(
        self,
        params: dict[str, np.ndarray],
        forward: typing.Dict[str, typing.Any],
        grad_output: float,
    ):
        grad_W2 = np.outer(forward["h1"], np.asarray([grad_output]))
        grad_b2 = np.asarray([grad_output])
        d_hidden = params["W2"][:, 0] * grad_output
        d_z1 = d_hidden * (1.0 - np.square(forward["h1"]))
        grad_W1 = np.outer(forward["x"], d_z1)
        grad_b1 = d_z1
        dx = params["W1"] @ d_z1
        action_grad = dx[self._state_dim :]
        return {
            "W1": grad_W1,
            "b1": grad_b1,
            "W2": grad_W2,
            "b2": grad_b2,
        }, action_grad

    @staticmethod
    def _clip_gradients(grads: dict[str, np.ndarray], max_norm: float):
        total_norm = math.sqrt(
            sum(float(np.sum(np.square(grad))) for grad in grads.values())
        )
        if total_norm <= max_norm or total_norm == 0.0:
            return grads
        scale = max_norm / total_norm
        return {name: grad * scale for name, grad in grads.items()}

    def _adam_step(
        self,
        params: dict[str, np.ndarray],
        grads: dict[str, np.ndarray],
        moments_m: dict[str, np.ndarray],
        moments_v: dict[str, np.ndarray],
        step: int,
        learning_rate: float,
    ):
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        step += 1
        for name, grad in grads.items():
            moments_m[name] = beta1 * moments_m[name] + (1.0 - beta1) * grad
            moments_v[name] = beta2 * moments_v[name] + (1.0 - beta2) * np.square(grad)
            m_hat = moments_m[name] / (1.0 - beta1**step)
            v_hat = moments_v[name] / (1.0 - beta2**step)
            params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        return step

    def _soft_update_targets(self):
        for name in self._critic1_params:
            self._critic1_target[name] = (
                self.tau * self._critic1_params[name]
                + (1.0 - self.tau) * self._critic1_target[name]
            )
            self._critic2_target[name] = (
                self.tau * self._critic2_params[name]
                + (1.0 - self.tau) * self._critic2_target[name]
            )

    def _finalize_transition(
        self, transition: _Transition, next_state: np.ndarray, terminal: bool = False
    ):
        if transition.reward is None:
            return
        finalized = _Transition(
            state=transition.state,
            abr_action=transition.abr_action,
            steering_raw=transition.steering_raw,
            logprob=transition.logprob,
            reward=transition.reward,
            next_state=np.asarray(next_state, dtype=float).reshape(-1),
            done=terminal or transition.done,
        )
        self._replay.append(finalized)
        if len(self._replay) >= self.batch_size:
            self._train_from_replay()

    def _target_value(self, next_state: np.ndarray):
        next_actor = self._actor_forward(next_state)
        next_abr_probs = next_actor["abr_probs"]
        next_steering_raw = next_actor["mu"] + next_actor["std"] * self.rng.normal(
            size=len(self.nodes)
        )
        next_logprob_gauss = _normal_log_prob(
            next_steering_raw, next_actor["mu"], next_actor["std"]
        )
        q_values = []
        for abr_index, abr_prob in enumerate(next_abr_probs):
            q1 = self._critic_forward(
                self._critic1_target, next_state, abr_index, next_steering_raw
            )["q"]
            q2 = self._critic_forward(
                self._critic2_target, next_state, abr_index, next_steering_raw
            )["q"]
            q_values.append(
                min(q1, q2)
                - self.entropy_coef
                * (math.log(max(1e-12, float(abr_prob))) + next_logprob_gauss)
            )
        return float(np.sum(next_abr_probs * np.asarray(q_values)))

    def _train_from_replay(self):
        if len(self._replay) < self.batch_size:
            return
        batch_size = min(self.batch_size, len(self._replay))
        indices = self.rng.choice(len(self._replay), size=batch_size, replace=False)
        batch = [list(self._replay)[index] for index in indices]
        for _ in range(self.update_steps):
            critic1_grads = {
                name: np.zeros_like(value)
                for name, value in self._critic1_params.items()
            }
            critic2_grads = {
                name: np.zeros_like(value)
                for name, value in self._critic2_params.items()
            }
            actor_grads = {
                name: np.zeros_like(value) for name, value in self._actor_params.items()
            }
            for sample in batch:
                reward = (
                    float(sample.reward if sample.reward is not None else 0.0)
                    / self.reward_scale
                )
                next_state = (
                    sample.next_state if sample.next_state is not None else sample.state
                )
                target_value = (
                    reward
                    if sample.done
                    else reward + self.gamma * self._target_value(next_state)
                )

                q1_forward = self._critic_forward(
                    self._critic1_params,
                    sample.state,
                    sample.abr_action,
                    sample.steering_raw,
                )
                q2_forward = self._critic_forward(
                    self._critic2_params,
                    sample.state,
                    sample.abr_action,
                    sample.steering_raw,
                )
                q1_error = q1_forward["q"] - target_value
                q2_error = q2_forward["q"] - target_value
                q1_sample_grads, _ = self._critic_backward(
                    self._critic1_params, q1_forward, q1_error
                )
                q2_sample_grads, _ = self._critic_backward(
                    self._critic2_params, q2_forward, q2_error
                )
                for name in critic1_grads:
                    critic1_grads[name] += q1_sample_grads[name]
                    critic2_grads[name] += q2_sample_grads[name]

                actor_forward = self._actor_forward(sample.state)
                steering_raw = actor_forward["mu"] + actor_forward[
                    "std"
                ] * self.rng.normal(size=len(self.nodes))
                abr_probs = actor_forward["abr_probs"]
                q_values = []
                q_action_grads = []
                for abr_index in range(len(self.quality_levels)):
                    q1_eval = self._critic_forward(
                        self._critic1_params, sample.state, abr_index, steering_raw
                    )
                    q2_eval = self._critic_forward(
                        self._critic2_params, sample.state, abr_index, steering_raw
                    )
                    if q1_eval["q"] <= q2_eval["q"]:
                        q_values.append(q1_eval["q"])
                        _, action_grad = self._critic_backward(
                            self._critic1_params, q1_eval, 1.0
                        )
                    else:
                        q_values.append(q2_eval["q"])
                        _, action_grad = self._critic_backward(
                            self._critic2_params, q2_eval, 1.0
                        )
                    q_action_grads.append(action_grad)
                q_values_arr = np.asarray(q_values, dtype=float)
                log_probs = np.log(np.clip(abr_probs, 1e-12, 1.0))
                entropy_cat = _entropy(abr_probs)
                expected_q = float(np.sum(abr_probs * q_values_arr))
                grad_logits = -abr_probs * (
                    q_values_arr - expected_q
                ) + self.entropy_coef * abr_probs * (log_probs + entropy_cat)
                grad_q_raw = np.sum(
                    [
                        abr_probs[index] * q_action_grads[index]
                        for index in range(len(q_action_grads))
                    ],
                    axis=0,
                )
                grad_q_raw_steer = grad_q_raw[len(self.quality_levels) :]
                eps = (steering_raw - actor_forward["mu"]) / actor_forward["std"]
                grad_mu = -grad_q_raw_steer
                grad_log_std = (
                    -grad_q_raw_steer * (eps * actor_forward["std"]) - self.entropy_coef
                )
                d_hidden = (
                    self._actor_params["W_abr"] @ grad_logits
                    + self._actor_params["W_mu"] @ grad_mu
                    + self._actor_params["W_log_std"] @ grad_log_std
                )
                d_z1 = d_hidden * (1.0 - np.square(actor_forward["h1"]))
                actor_sample_grads = {
                    "W1": np.outer(actor_forward["state"], d_z1),
                    "b1": d_z1,
                    "W_abr": np.outer(actor_forward["h1"], grad_logits),
                    "b_abr": grad_logits,
                    "W_mu": np.outer(actor_forward["h1"], grad_mu),
                    "b_mu": grad_mu,
                    "W_log_std": np.outer(actor_forward["h1"], grad_log_std),
                    "b_log_std": grad_log_std,
                }
                for name in actor_grads:
                    actor_grads[name] += actor_sample_grads[name]

            scale = 1.0 / max(1, batch_size)
            for grads in (critic1_grads, critic2_grads, actor_grads):
                for name in grads:
                    grads[name] *= scale
            critic1_grads = self._clip_gradients(critic1_grads, self.max_grad_norm)
            critic2_grads = self._clip_gradients(critic2_grads, self.max_grad_norm)
            actor_grads = self._clip_gradients(actor_grads, self.max_grad_norm)
            self._critic1_step = self._adam_step(
                self._critic1_params,
                critic1_grads,
                self._critic1_m,
                self._critic1_v,
                self._critic1_step,
                self.critic_learning_rate,
            )
            self._critic2_step = self._adam_step(
                self._critic2_params,
                critic2_grads,
                self._critic2_m,
                self._critic2_v,
                self._critic2_step,
                self.critic_learning_rate,
            )
            self._actor_step = self._adam_step(
                self._actor_params,
                actor_grads,
                self._actor_m,
                self._actor_v,
                self._actor_step,
                self.actor_learning_rate,
            )
            self._soft_update_targets()
        if self.policy_path:
            self.save_policy(self.policy_path)

    def select_arm(self, **kwargs) -> list[str]:
        self._refresh_nodes()
        contexts, latencies = self._build_context_bundle(kwargs)
        if not self.nodes:
            return []
        for arm in self.nodes:
            contexts.setdefault(
                arm, np.zeros(self._context_dim_for_nodes(), dtype=float)
            )
            latencies.setdefault(arm, None)
        state = self._build_state(contexts, latencies)
        actor_forward = self._actor_forward(state)
        explore = bool(kwargs.get("explore", True))
        if explore:
            abr_action = self.rng.choice(
                len(self.quality_levels), p=actor_forward["abr_probs"]
            )
            steering_raw = actor_forward["mu"] + actor_forward["std"] * self.rng.normal(
                size=len(self.nodes)
            )
        else:
            abr_action = int(np.argmax(actor_forward["abr_probs"]))
            steering_raw = actor_forward["mu"].copy()
        steering_weights = _softmax(np.asarray(steering_raw, dtype=float))
        ordered_nodes = [self.nodes[idx] for idx in np.argsort(-steering_weights)]
        logprob = math.log(
            max(1e-12, float(actor_forward["abr_probs"][abr_action]))
        ) + _normal_log_prob(
            np.asarray(steering_raw, dtype=float),
            actor_forward["mu"],
            actor_forward["std"],
        )
        self._last_selected_index = int(np.argmax(steering_weights))
        self._last_quality_index = abr_action
        self._last_action = {
            "abr_action": abr_action,
            "abr_probability": float(actor_forward["abr_probs"][abr_action]),
            "quality_level": self.quality_levels[abr_action],
            "steering_raw": np.asarray(steering_raw, dtype=float),
            "steering_weights": steering_weights,
            "ordered_nodes": ordered_nodes,
            "logprob": logprob,
        }
        transition = _Transition(
            state=state,
            abr_action=abr_action,
            steering_raw=np.asarray(steering_raw, dtype=float),
            logprob=logprob,
        )
        decision_id = kwargs.get("decision_id")
        if decision_id:
            self._pending_transitions.setdefault(decision_id, []).append(transition)
        else:
            self._pending_transitions.setdefault(ordered_nodes[0], []).append(
                transition
            )
        return ordered_nodes

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
            self.values[chosen_arm_name] = (
                (count - 1) * previous + reward_value
            ) / count
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
        if transition.done or kwargs.get("force_train", False):
            next_state = (
                transition.state
                if kwargs.get("next_state") is None
                else np.asarray(kwargs.get("next_state"), dtype=float)
            )
            self._finalize_transition(transition, next_state, terminal=transition.done)

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
            else {
                arm: np.zeros(self._context_dim_for_nodes(), dtype=float)
                for arm in self.nodes
            }
        )
        latencies = (
            self._last_latencies
            if self._last_latencies
            else {arm: None for arm in self.nodes}
        )
        state = self._build_state(contexts, latencies)
        actor_forward = self._actor_forward(state)
        if explore:
            abr_action = self.rng.choice(
                len(self.quality_levels), p=actor_forward["abr_probs"]
            )
            steering_raw = actor_forward["mu"] + actor_forward["std"] * self.rng.normal(
                size=len(self.nodes)
            )
        else:
            abr_action = int(np.argmax(actor_forward["abr_probs"]))
            steering_raw = actor_forward["mu"].copy()
        steering_weights = _softmax(np.asarray(steering_raw, dtype=float))
        ordered_nodes = [self.nodes[idx] for idx in np.argsort(-steering_weights)]
        return {
            "abr_index": abr_action,
            "quality_level": self.quality_levels[abr_action],
            "abr_probabilities": actor_forward["abr_probs"].tolist(),
            "steering_mean": actor_forward["mu"].tolist(),
            "steering_std": actor_forward["std"].tolist(),
            "steering_weights": steering_weights.tolist(),
            "ordered_nodes": ordered_nodes,
        }

    def save_policy(self, path: str | None = None):
        target_path = path or self.policy_path
        if not target_path:
            return None
        directory = os.path.dirname(target_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "actor_W1": self._actor_params.get("W1"),
            "actor_b1": self._actor_params.get("b1"),
            "actor_W_abr": self._actor_params.get("W_abr"),
            "actor_b_abr": self._actor_params.get("b_abr"),
            "actor_W_mu": self._actor_params.get("W_mu"),
            "actor_b_mu": self._actor_params.get("b_mu"),
            "actor_W_log_std": self._actor_params.get("W_log_std"),
            "actor_b_log_std": self._actor_params.get("b_log_std"),
            "critic1_W1": self._critic1_params.get("W1"),
            "critic1_b1": self._critic1_params.get("b1"),
            "critic1_W2": self._critic1_params.get("W2"),
            "critic1_b2": self._critic1_params.get("b2"),
            "critic2_W1": self._critic2_params.get("W1"),
            "critic2_b1": self._critic2_params.get("b1"),
            "critic2_W2": self._critic2_params.get("W2"),
            "critic2_b2": self._critic2_params.get("b2"),
            "metadata": np.asarray(
                json.dumps(
                    {
                        "quality_levels": self.quality_levels,
                        "hidden_dim": self.hidden_dim,
                        "critic_hidden_dim": self.critic_hidden_dim,
                        "state_dim": self._state_dim,
                        "critic_input_dim": self._critic_input_dim,
                        "nodes": self.nodes,
                    }
                ),
                dtype=object,
            ),
        }
        np.savez_compressed(target_path, **payload)  # type: ignore
        return target_path

    def load_policy(self, path: str | None = None):
        target_path = path or self.policy_path
        if not target_path or not os.path.exists(target_path):
            return None
        data = np.load(target_path, allow_pickle=True)
        self._actor_params = {
            "W1": data["actor_W1"],
            "b1": data["actor_b1"],
            "W_abr": data["actor_W_abr"],
            "b_abr": data["actor_b_abr"],
            "W_mu": data["actor_W_mu"],
            "b_mu": data["actor_b_mu"],
            "W_log_std": data["actor_W_log_std"],
            "b_log_std": data["actor_b_log_std"],
        }
        self._critic1_params = {
            "W1": data["critic1_W1"],
            "b1": data["critic1_b1"],
            "W2": data["critic1_W2"],
            "b2": data["critic1_b2"],
        }
        self._critic2_params = {
            "W1": data["critic2_W1"],
            "b1": data["critic2_b1"],
            "W2": data["critic2_W2"],
            "b2": data["critic2_b2"],
        }
        self._critic1_target = {
            name: value.copy() for name, value in self._critic1_params.items()
        }
        self._critic2_target = {
            name: value.copy() for name, value in self._critic2_params.items()
        }
        self._actor_m = {
            name: np.zeros_like(value) for name, value in self._actor_params.items()
        }
        self._actor_v = {
            name: np.zeros_like(value) for name, value in self._actor_params.items()
        }
        self._critic1_m = {
            name: np.zeros_like(value) for name, value in self._critic1_params.items()
        }
        self._critic1_v = {
            name: np.zeros_like(value) for name, value in self._critic1_params.items()
        }
        self._critic2_m = {
            name: np.zeros_like(value) for name, value in self._critic2_params.items()
        }
        self._critic2_v = {
            name: np.zeros_like(value) for name, value in self._critic2_params.items()
        }
        self._actor_step = 0
        self._critic1_step = 0
        self._critic2_step = 0
        if "metadata" in data.files:
            try:
                metadata = json.loads(str(data["metadata"].item()))
                self.quality_levels = metadata.get(
                    "quality_levels", self.quality_levels
                )
                self.hidden_dim = int(metadata.get("hidden_dim", self.hidden_dim))
                self.critic_hidden_dim = int(
                    metadata.get("critic_hidden_dim", self.critic_hidden_dim)
                )
                self._state_dim = int(metadata.get("state_dim", self._state_dim))
                self._critic_input_dim = int(
                    metadata.get("critic_input_dim", self._critic_input_dim)
                )
            except Exception:
                selector_logger.warning(
                    "[SAC] Failed to parse policy metadata while loading weights."
                )
        return target_path
