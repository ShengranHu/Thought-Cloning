from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn
import pdb

from babyai.rl.algos.DQN import DQN


class AWAC(DQN):
    def __init__(
        self,
        temperature: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temperature = temperature

    def compute_critic_loss(
        self,
        qa_values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_qa_values: torch.Tensor,
        dones: torch.Tensor,
        *args,
    ):
        with torch.no_grad():
            # TODO(student): compute the actor distribution, then use it to compute E[Q(s, a)]
            # Use the actor to compute a critic backup
            next_qs = rewards + torch.max(next_qa_values, dim=-1)[0] * (
                1 - dones.float()
            )

            # TODO(student): Compute the TD target
            target_values = rewards + self.discount * next_qs

        # TODO(student): Compute Q(s, a) and loss similar to DQN
        q_values = torch.gather(qa_values, -1, actions.unsqueeze(1)).squeeze()
        assert q_values.shape == target_values.shape

        loss = self.critic_loss(q_values, target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def compute_advantage(
        self,
        qa_values: torch.Tensor,
        actions: torch.Tensor,
        values: torch.Tensor = None,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): compute the advantage of the actions compared to E[Q(s, a)]
        if action_dist is not None:
            action_dist = action_dist.sample()
        q_values = torch.gather(qa_values, -1, actions.unsqueeze(1)).squeeze()
        values = torch.max(qa_values, dim=-1)[0]
        advantages = q_values - values
        return advantages

    def update_actor(
        self,
        qa_values: torch.Tensor,
        actor_dist: torch.distributions.Distribution,
        actions: torch.Tensor,
        values=None,
    ):
        # TODO(student): update the actor using AWAC
        adv = self.compute_advantage(qa_values, actions, values)
        loss = -actor_dist.log_prob(actions) * torch.exp(adv / self.temperature)
        loss = loss.mean()
        return loss

    def update(
        self,
        qa_values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_qa_values: torch.Tensor,
        dones: torch.Tensor,
        actor_dist: torch.distributions.Distribution,
    ):
        critic_loss, metrics = super().update(
            qa_values,
            actions,
            rewards,
            next_qa_values,
            dones,
        )

        # Update the actor.
        actor_loss = self.update_actor(qa_values, actor_dist, actions)
        metrics["actor_loss"] = actor_loss
        metrics["critic_loss"] = critic_loss

        return metrics


class LAWAC(DQN):
    def __init__(
        self,
        temperature: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temperature = temperature

    def compute_critic_loss(
        self,
        qa_values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_qa_values: torch.Tensor,
        dones: torch.Tensor,
        *args,
    ):
        with torch.no_grad():
            # TODO(student): compute the actor distribution, then use it to compute E[Q(s, a)]
            # Use the actor to compute a critic backup
            next_qs = rewards + torch.max(next_qa_values, dim=-1)[0] * (
                1 - dones.float()
            )

            # TODO(student): Compute the TD target
            target_values = rewards + self.discount * next_qs

        # TODO(student): Compute Q(s, a) and loss similar to DQN
        q_values = torch.gather(qa_values, -1, actions.unsqueeze(1)).squeeze()
        assert q_values.shape == target_values.shape

        loss = self.critic_loss(q_values, target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def compute_advantage(
        self,
        qa_values: torch.Tensor,
        actions: torch.Tensor,
        values=None,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): compute the advantage of the actions compared to E[Q(s, a)]
        if action_dist is not None:
            action_dist = action_dist.sample()
        q_values = torch.gather(qa_values, -1, actions.unsqueeze(1)).squeeze()
        values = torch.max(qa_values, dim=-1)[0]
        advantages = q_values - values
        return advantages

    def update_actor(
        self,
        qa_values: torch.Tensor,
        actor_dist: torch.distributions.Distribution,
        values: torch.Tensor,
        actions: torch.Tensor,
    ):
        # TODO(student): update the actor using AWAC
        adv = self.compute_advantage(qa_values, actions, values)
        loss = (
            -actor_dist.log_prob(actions)
            * torch.exp(adv / self.temperature)
            * self.temperature
        )
        loss = loss.mean()
        return loss

    def update(
        self,
        qa_values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_qa_values: torch.Tensor,
        dones: torch.Tensor,
        actor_dist: torch.distributions.Distribution,
    ):
        critic_loss, metrics = super().update(
            qa_values,
            actions,
            rewards,
            next_qa_values,
            dones,
        )

        # Update the actor.
        actor_loss = self.update_actor(qa_values, actor_dist, actions)
        metrics["actor_loss"] = actor_loss
        metrics["critic_loss"] = critic_loss

        return metrics
