from typing import Optional
import torch
from torch import nn
from babyai.rl.algos.AWAC import AWAC

from typing import Callable, Optional, Sequence, Tuple, List


class IQL(AWAC):
    def __init__(self, expectile: float, **kwargs):
        super().__init__(**kwargs)
        self.expectile = expectile

    def compute_advantage(
        self,
        qa_values: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): Compute advantage with IQL

        # qa_values = self.target_critic(observations)
        qa_values = self.critic(observations)
        q_values = qa_values.gather(1, actions.unsqueeze(1)).squeeze(-1)
        values = self.value_critic(observations).squeeze(-1)

        advantages = q_values - values
        return advantages

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(student): Update Q(s, a) to match targets (based on V)
        with torch.no_grad():
            target_values = rewards + self.discount * self.target_value_critic(
                next_observations
            ).squeeze(-1) * (1 - dones.float())
            # trying = self.target_value_critic(next_observations)
            # print(rewards.shape, trying.shape, ((1-dones.float()).unsqueeze(1)).shape)

        # qa_values = self.target_critic(observations)
        qa_values = self.critic(observations)
        q_values = qa_values.gather(1, actions.unsqueeze(1)).squeeze(-1)
        # print(q_values.shape, target_values.shape)
        loss = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        metrics = {
            "q_loss": self.critic_loss(q_values, target_values).item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "q_grad_norm": grad_norm.item(),
        }

        return metrics

    @staticmethod
    def iql_expectile_loss(
        expectile: float, vs: torch.Tensor, target_qs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the expectile loss for IQL
        """
        # TODO(student): Compute the expectile loss
        inside = target_qs - vs
        loss = torch.where(
            inside <= 0, (1 - expectile) * (inside**2), expectile * (inside**2)
        )
        return loss

    def update_v(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the value network V(s) using targets Q(s, a)
        """
        # TODO(student): Compute target values for V(s)

        qa_values = self.target_critic(observations)
        # qa_values = self.critic(observations)
        q_values = qa_values.gather(1, actions.unsqueeze(1)).squeeze(-1)
        target_values = q_values
        # TODO(student): Update V(s) using the loss from the IQL paper

        vs = self.value_critic(observations).squeeze(-1)

        loss = self.iql_expectile_loss(self.expectile, vs, q_values).mean()

        self.value_critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.value_critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.value_critic_optimizer.step()

        return {
            "v_loss": loss.item(),
            "vs_adv": (vs - target_values).mean().item(),
            "vs": vs.mean().item(),
            "target_values": target_values.mean().item(),
            "v_grad_norm": grad_norm.item(),
        }

    def update_critic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update both Q(s, a) and V(s)
        """

        metrics_q = self.update_q(
            observations, actions, rewards, next_observations, dones
        )
        metrics_v = self.update_v(observations, actions)

        return {**metrics_q, **metrics_v}

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics = self.update_critic(
            observations, actions, rewards, next_observations, dones
        )
        metrics["actor_loss"] = self.update_actor(observations, actions)

        if step % self.target_update_period == 0:
            self.update_target_critic()
            self.update_target_value_critic()

        return metrics

    def update_target_value_critic(self):
        self.target_value_critic.load_state_dict(self.value_critic.state_dict())
