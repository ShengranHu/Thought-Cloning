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
        values: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): Compute advantage with IQL

        # qa_values = self.target_critic(observations)
        q_values = qa_values.gather(1, actions.unsqueeze(1)).squeeze(-1)
        advantages = q_values - values
        return advantages

    def update_q(
        self,
        qa_values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_qa_values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(student): Update Q(s, a) to match targets (based on V)
        with torch.no_grad():
            target_values = rewards + self.discount * next_values.squeeze(-1) * (
                1 - dones.float()
            )
        q_values = qa_values.gather(1, actions.unsqueeze(1)).squeeze(-1)
        # print(q_values.shape, target_values.shape)
        loss = self.critic_loss(q_values, target_values)

        metrics = {
            "critic_loss": loss,
            "q_loss": self.critic_loss(q_values, target_values).item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
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
        qa_values: torch.Tensor,
        values: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the value network V(s) using targets Q(s, a)
        """
        # TODO(student): Compute target values for V(s)

        # qa_values = self.critic(observations)
        q_values = qa_values.gather(1, actions.unsqueeze(1)).squeeze(-1)
        target_values = q_values
        # TODO(student): Update V(s) using the loss from the IQL paper

        loss = self.iql_expectile_loss(self.expectile, values, q_values).mean()

        return {
            "value_loss": loss,
            "v_loss": loss.item(),
            "vs_adv": (values - target_values).mean().item(),
            "vs": values.mean().item(),
            "target_values": target_values.mean().item(),
        }

    def update_critic(
        self,
        qa_values: torch.Tensor,
        values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_qa_values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update both Q(s, a) and V(s)
        """

        metrics_q = self.update_q(
            qa_values, actions, rewards, next_qa_values, next_values, dones
        )
        metrics_v = self.update_v(qa_values, values, actions)

        return {**metrics_q, **metrics_v}

    def update(
        self,
        qa_values: torch.Tensor,
        values: torch.Tensor,
        actions: torch.Tensor,
        action_dist,
        rewards: torch.Tensor,
        next_qa_values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ):
        metrics = self.update_critic(
            qa_values, values, actions, rewards, next_qa_values, next_values, dones
        )
        metrics["actor_loss"] = self.update_actor(
            qa_values, action_dist, actions, values
        )

        return metrics

    def update_target_value_critic(self):
        self.target_value_critic.load_state_dict(self.value_critic.state_dict())
