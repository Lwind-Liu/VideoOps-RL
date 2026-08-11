"""Rule and trainable Coordinator policies for the business highlight environment."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from .business_env import ACTIONS, HighlightTask, RealHighlightEnv


class CoordinatorPolicy(nn.Module):
    def __init__(self, feature_dim: int = 9, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(ACTIONS)),
        )

    def forward(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        logits = self.network(features)
        return logits.masked_fill(~masks, -1e9)


@dataclass
class Episode:
    reward: float
    transitions: list[dict[str, Any]]
    trajectory: list[dict[str, Any]]
    result: dict[str, Any]


def run_sequence(env: RealHighlightEnv, actions: list[str]) -> Episode:
    total = 0.0
    for action in actions:
        reward, done, _ = env.step(action)
        total += reward
        if done:
            break
    if not env.state.done:
        reward, _, _ = env.step("submit")
        total += reward
    return Episode(total, [], env.trajectory, env.final_result())


def rollout_policy(
    policy: CoordinatorPolicy,
    env: RealHighlightEnv,
    device: torch.device,
    deterministic: bool = False,
) -> Episode:
    transitions: list[dict[str, Any]] = []
    total_reward = 0.0
    while not env.state.done:
        features = torch.tensor(env.features(), dtype=torch.float32, device=device)
        mask = torch.tensor(env.valid_action_mask(), dtype=torch.bool, device=device)
        with torch.no_grad():
            logits = policy(features.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
            distribution = Categorical(logits=logits)
            action_index = torch.argmax(logits) if deterministic else distribution.sample()
            old_log_prob = distribution.log_prob(action_index)
        action = ACTIONS[int(action_index.item())]
        reward, _, _ = env.step(action)
        total_reward += reward
        transitions.append(
            {
                "features": features.cpu(),
                "mask": mask.cpu(),
                "action": int(action_index.item()),
                "old_log_prob": float(old_log_prob.item()),
            }
        )
    return Episode(total_reward, transitions, env.trajectory, env.final_result())


def train_group_relative_policy(
    policy: CoordinatorPolicy,
    env_factory,
    tasks: list[HighlightTask],
    device: torch.device,
    epochs: int = 120,
    group_size: int = 8,
    learning_rate: float = 3e-3,
    clip_ratio: float = 0.2,
    seed: int = 7,
) -> list[dict[str, float]]:
    """GRPO-style update over groups of complete tool-use trajectories.

    This is a structured policy network, not LLM token-level GRPO. It uses the
    same core mechanism: grouped rollouts, relative advantages, clipped policy
    ratios, and a frozen reference regularizer.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    policy.to(device)
    reference = copy.deepcopy(policy).to(device).eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        old_policy = copy.deepcopy(policy).to(device).eval()
        samples: list[dict[str, Any]] = []
        episode_rewards: list[float] = []
        for task in tasks:
            episodes = [
                rollout_policy(old_policy, env_factory(task), device, deterministic=False)
                for _ in range(group_size)
            ]
            rewards = torch.tensor([episode.reward for episode in episodes], dtype=torch.float32)
            advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
            episode_rewards.extend(rewards.tolist())
            for episode, advantage in zip(episodes, advantages.tolist()):
                for transition in episode.transitions:
                    samples.append({**transition, "advantage": advantage})

        if not samples:
            raise RuntimeError("training produced no transitions")
        features = torch.stack([sample["features"] for sample in samples]).to(device)
        masks = torch.stack([sample["mask"] for sample in samples]).to(device)
        actions = torch.tensor([sample["action"] for sample in samples], device=device)
        old_log_probs = torch.tensor([sample["old_log_prob"] for sample in samples], device=device)
        advantages = torch.tensor([sample["advantage"] for sample in samples], device=device)

        logits = policy(features, masks)
        distribution = Categorical(logits=logits)
        new_log_probs = distribution.log_prob(actions)
        ratios = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratios * advantages
        clipped = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        entropy = distribution.entropy().mean()
        with torch.no_grad():
            reference_logits = reference(features, masks)
            reference_log_probs = Categorical(logits=reference_logits).log_prob(actions)
        reference_penalty = ((new_log_probs - reference_log_probs) ** 2).mean()
        loss = policy_loss - 0.01 * entropy + 0.002 * reference_penalty

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    "mean_group_reward": sum(episode_rewards) / len(episode_rewards),
                    "loss": float(loss.item()),
                    "entropy": float(entropy.item()),
                }
            )
    return history
