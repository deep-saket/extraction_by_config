from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class GRPOAgent:
    """
    GRPO objective tailored for auto-regressive text generation.

    The agent expects:
      - `model_outputs`: logits from the language model.
      - `labels`: target token ids.
      - `old_log_probs` (optional): reference policy log probs.
    """

    def __init__(self, rl_cfg: Dict):
        self.rl_cfg = dict(rl_cfg)
        self.kl_beta = float(self.rl_cfg.get("kl_beta", 0.1))
        self.reward_scale = float(self.rl_cfg.get("reward_scale", 1.0))
        self.ignore_index = int(self.rl_cfg.get("ignore_index", -100))
        self.baseline_momentum = float(self.rl_cfg.get("baseline_momentum", 0.95))
        self._running_baseline = None
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="none")

    def supervised_loss(self, model_outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = model_outputs[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss.mean()

    def compute_loss(
        self,
        model_outputs: torch.Tensor,
        labels: torch.Tensor,
        old_log_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_logits = model_outputs[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        log_probs = self._gather_log_probs(shift_logits, shift_labels)
        token_rewards = self._token_rewards(log_probs)
        baseline = self._update_baseline(token_rewards)
        advantages = token_rewards - baseline

        policy_loss = -(log_probs * advantages.detach() * self.reward_scale)
        policy_loss = policy_loss.mean()

        if old_log_probs is not None:
            kl = torch.mean(log_probs - old_log_probs)
            policy_loss = policy_loss + self.kl_beta * kl

        return policy_loss

    def _gather_log_probs(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        one_hot = torch.nn.functional.one_hot(
            torch.clamp(labels, min=0),
            num_classes=log_probs.size(-1),
        ).float()
        selected = (one_hot * log_probs).sum(dim=-1)
        mask = (labels != self.ignore_index).float()
        selected = selected * mask
        lengths = mask.sum(dim=-1).clamp(min=1.0)
        return selected.sum(dim=-1) / lengths

    def _token_rewards(self, log_probs: torch.Tensor) -> torch.Tensor:
        return torch.exp(log_probs)

    def _update_baseline(self, rewards: torch.Tensor) -> torch.Tensor:
        mean_reward = rewards.mean()
        if self._running_baseline is None:
            self._running_baseline = mean_reward
        else:
            self._running_baseline = (
                self.baseline_momentum * self._running_baseline + (1 - self.baseline_momentum) * mean_reward
            )
        return self._running_baseline.detach()

